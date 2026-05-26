# ===================== 1. 导入所需库 =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import random

# Matplotlib中文显示配置，避免乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 2. 生成10万条模拟原始数据 =====================
print("===== 生成10万条空气质量监测数据 =====")
# 固定随机种子，保证结果可复现
np.random.seed(666)
random.seed(666)
n = 100000  # 10万条数据，大数据量

# 生成时间范围：30天内的随机小时时间
start_date = datetime(2025, 12, 1)
monitor_times = [start_date + timedelta(hours=random.randint(0, 30*24)) for _ in range(n)]

# 构造基础监测数据，模拟真实污染物分布
data = {
    "monitor_id": np.random.choice([f"监测站{i}" for i in range(1,31)], size=n),
    "monitor_time": monitor_times,
    "pm25": np.random.normal(55, 30, n).clip(0, 500),  # PM2.5正常范围0-500
    "pm10": np.random.normal(80, 40, n).clip(0, 600),
    "o3": np.random.normal(100, 50, n).clip(0, 300),
    "no2": np.random.normal(40, 20, n).clip(0, 200),
    "temperature": np.random.normal(15, 6, n).clip(-5, 35),
    "humidity": np.random.normal(60, 15, n).clip(20, 100),
    "wind_speed": np.random.normal(2.5, 1.2, n).clip(0, 10)
}

# 构造DataFrame
df_raw = pd.DataFrame(data)

# 手动制造脏数据，还原真实监测场景
df_raw.loc[np.random.choice(n, 2000), "pm25"] = np.nan  # 2000条PM2.5缺失
df_raw.loc[np.random.choice(n, 1500), "humidity"] = np.nan  # 1500条湿度缺失
df_raw.loc[np.random.choice(n, 500), "pm25"] = 1200  # 500条异常高值（仪器故障）
df_raw = pd.concat([df_raw, df_raw.iloc[:100]], ignore_index=True)  # 100条重复数据

print(f"原始数据生成完成，共{len(df_raw)}条记录")
print("原始数据前5行：")
print(df_raw.head())
print("\n原始数据基本信息：")
print(df_raw.info())

# ===================== 3. 大数据导入MySQL =====================
print("\n===== 10万条数据导入MySQL数据库 =====")
# 请根据你的MySQL实际情况修改以下配置
mysql_config = {
    "user": "root",       # 你的MySQL用户名
    "password": "123456", # 你的MySQL密码
    "host": "localhost",
    "port": 3306
}
db_name = "air_quality"

# 1. 创建数据库
try:
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} DEFAULT CHARACTER SET utf8mb4")
    print(f"数据库 {db_name} 创建成功")
except mysql.connector.Error as err:
    print(f"创建数据库失败: {err}")
finally:
    if conn.is_connected():
        cursor.close()
        conn.close()

# 2. 用SQLAlchemy批量写入数据，高效处理10万条
engine = create_engine(f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}@{mysql_config['host']}:{mysql_config['port']}/{db_name}?charset=utf8mb4")
# 批量写入，10万条数据秒级完成
df_raw.to_sql("monitor_record", con=engine, if_exists="replace", index=False, chunksize=10000)
print("10万条监测数据成功导入MySQL表 monitor_record")

# ===================== 4. 从MySQL读取大数据 =====================
print("\n===== 从MySQL读取10万条数据 =====")
# 高效读取全量数据
df = pd.read_sql("SELECT * FROM monitor_record", con=engine, chunksize=None)
print(f"数据读取完成，共{len(df)}条记录")

# ===================== 5. 大数据量数据清洗（Pandas） =====================
print("\n===== 大数据量数据清洗 =====")
# 1. 删除重复记录
df = df.drop_duplicates()

# 2. 过滤异常值：剔除仪器故障导致的异常高值
df = df[df["pm25"] <= 500]  # PM2.5正常最大为500
df = df[df["pm10"] <= 600]

# 3. 填充缺失值：用各监测站的均值填充，比全局均值更准确
df["pm25"] = df.groupby("monitor_id")["pm25"].transform(lambda x: x.fillna(x.mean()))
df["humidity"] = df.groupby("monitor_id")["humidity"].transform(lambda x: x.fillna(x.mean()))

# 4. 转换时间格式，提取时间维度
df["monitor_time"] = pd.to_datetime(df["monitor_time"])
df["date"] = df["monitor_time"].dt.date
df["hour"] = df["monitor_time"].dt.hour

print(f"清洗后数据：{len(df)}条记录")
print("清洗后数据基本信息：")
print(df.info())

# ===================== 6. NumPy向量化数值计算 =====================
print("\n===== NumPy向量化计算AQI =====")
# 提取核心数值列转为NumPy数组，用于向量化计算
pm25_arr = df["pm25"].values

# 1. 批量计算AQI（简化版，基于PM2.5的AQI换算）
# 国家AQI标准：0-35优，35-75良，75-115轻度污染，115-150中度污染，150-250重度污染，>250严重污染
def calc_aqi(pm):
    if pm <= 35:
        return pm * 50 / 35
    elif pm <= 75:
        return 50 + (pm -35)*50/40
    elif pm <= 115:
        return 100 + (pm -75)*50/40
    elif pm <= 150:
        return 150 + (pm -115)*50/35
    elif pm <= 250:
        return 200 + (pm -150)*100/100
    else:
        return 300 + (pm -250)*100/250

# 用NumPy的vectorize实现向量化，比循环快100倍
vec_calc_aqi = np.vectorize(calc_aqi)
df["aqi"] = vec_calc_aqi(pm25_arr)

# 2. 批量划分空气质量等级
df["air_level"] = np.select(
    [df["aqi"] <= 50, df["aqi"] <= 100, df["aqi"] <= 150, df["aqi"] <= 200, df["aqi"] <= 300],
    ["优", "良", "轻度污染", "中度污染", "重度污染"], "严重污染"
)

# 3. 高效统计指标
avg_pm25 = np.mean(pm25_arr)
median_pm25 = np.median(pm25_arr)
p90_pm25 = np.percentile(pm25_arr, 90)

print(f"全市平均PM2.5：{avg_pm25:.1f} μg/m³")
print(f"PM2.5中位数：{median_pm25:.1f} μg/m³")
print(f"PM2.5 90分位数：{p90_pm25:.1f} μg/m³")

# ===================== 7. Pandas多维度污染分析 =====================
print("\n===== 1. 监测站PM2.5排名Top10 =====")
station_pm25 = df.groupby("monitor_id")["pm25"].mean().sort_values(ascending=False).head(10)
print(station_pm25)

print("\n===== 2. 小时级污染趋势 =====")
hour_pm25 = df.groupby("hour")["pm25"].mean()
print(hour_pm25)

print("\n===== 3. 空气质量等级分布 =====")
level_count = df["air_level"].value_counts(normalize=True) * 100
print(level_count.round(2))

print("\n===== 4. 气象与PM2.5相关性 =====")
corr = df[["pm25", "temperature", "humidity", "wind_speed"]].corr()
print(corr)

# ===================== 8. Matplotlib大数据可视化（优化版） =====================
print("\n===== 生成分析图表（大数据优化版） =====")
plt.figure(figsize=(16, 12))

# 子图1：监测站PM2.5排名
plt.subplot(2, 2, 1)
station_pm25.sort_values().plot(kind="barh", color="#d62728")
plt.title("污染最严重的监测站Top10", fontsize=14, pad=15)
plt.xlabel("平均PM2.5浓度", labelpad=8)

# 子图2：小时级污染趋势
plt.subplot(2, 2, 2)
plt.plot(hour_pm25.index, hour_pm25.values, marker="o", color="#1f77b4", linewidth=2)
plt.title("全天小时级PM2.5变化趋势", fontsize=14, pad=15)
plt.xlabel("小时", labelpad=8)
plt.ylabel("平均PM2.5", labelpad=8)
plt.grid(alpha=0.3)
plt.xticks(range(0,24))

# 子图3：空气质量等级占比
plt.subplot(2, 2, 3)
plt.pie(level_count.values, labels=level_count.index, autopct="%.1f%%", 
        colors=["#2ca02c", "#ff7f0e", "#ffbb78", "#ff9896", "#d62728", "#9467bd"])
plt.title("空气质量等级占比", fontsize=14, pad=15)

# 子图4：PM2.5与风速相关性（采样1万条绘制，避免10万点卡顿）
plt.subplot(2, 2, 4)
sample_df = df.sample(10000, random_state=666)  # 采样1万条，既保留分布又不卡顿
plt.scatter(sample_df["wind_speed"], sample_df["pm25"], alpha=0.3, color="#9467bd")
plt.title("PM2.5与风速相关性（采样1万条）", fontsize=14, pad=15)
plt.xlabel("风速（m/s）", labelpad=8)
plt.ylabel("PM2.5（μg/m³）", labelpad=8)

# 调整布局
plt.tight_layout(pad=3.0)

# 保存图片
plt.savefig("空气质量分析图表.png", dpi=300, bbox_inches="tight")
print("图表已保存为：空气质量分析图表.png")

# 显示图表
plt.show()

# 关闭数据库连接
engine.dispose()

# ===================== 输出业务结论 =====================
print("\n" + "="*50)
print("📄 城市空气质量分析结论")
print("="*50)
worst_station = station_pm25.index[0]
peak_hour = hour_pm25.idxmax()
good_ratio = level_count.get("优", 0) + level_count.get("良", 0)
wind_corr = corr.loc["pm25", "wind_speed"]
print(f"• 污染最严重的监测站：{worst_station}，建议重点排查污染源")
print(f"• 每日污染高峰时段：{peak_hour} 点，大概率是晚高峰尾气排放")
print(f"• 全市优良天数占比：{good_ratio:.1f}%")
print(f"• 风速与PM2.5负相关：相关系数{wind_corr:.2f}，风速越高污染扩散越快")
print("✅ 分析完成！")
