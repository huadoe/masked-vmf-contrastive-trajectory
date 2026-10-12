"""计算7个运动特征和20个时间窗口特征"""
import pandas as pd
from math import radians, degrees, sin, cos, atan2, sqrt
import zipfile
import os
from pyproj import Transformer
import numpy as np
import sys
# 将上一级目录添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_file import Config

config = Config()


# 计算航向角（单位：度）
def calculate_heading(lon1, lat1, lon2, lat2):
    dlon = radians(lon2 - lon1)  # 经度差转换为弧度
    lat1 = radians(lat1)  # 起点纬度转换为弧度
    lat2 = radians(lat2)  # 终点纬度转换为弧度

    x = sin(dlon) * cos(lat2)  # 正确的 x 表达式
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)  # 正确的 y 表达式
    heading = atan2(x, y)  # atan2 使用 (x, y)
    heading_deg = round((degrees(heading) + 360) % 360, 2)  # 转换为角度并归一化到 [0, 360), 保留两位小数
    return heading_deg


# 使用 Haversine 公式计算距离（单位：米）
def calculate_distance(lon1, lat1, lon2, lat2):
    R = 6371  # 地球平均半径，单位为公里
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = round(R * c * 1000, 2)  # 单位转换为米，保留两位小数
    return distance


# 重新计算运动特征
def motion_feature(data):
    data['long_shift'] = data['long'].shift()
    data['lat_shift'] = data['lat'].shift()
    data.dropna(inplace=True)

    # 计算航向角
    data['heading'] = data.apply(lambda row: calculate_heading(row['long'], row['lat'], row['long_shift'],
                                                               row['lat_shift']), axis=1)

    # 计算距离
    data['distance'] = data.apply(lambda row: calculate_distance(row['long'], row['lat'], row['long_shift'],
                                                                 row['lat_shift']), axis=1)

    # 计算速度
    data['speed'] = data['distance'] / data['time_diff']

    # 删除辅助列
    data.drop(columns=['long_shift', 'lat_shift'], inplace=True)

    # 计算运动特征
    data['acceleration'] = data['speed'].diff() / data['timestamp'].diff()
    data['angular_speed'] = data['heading'].diff() / data['timestamp'].diff()
    data['angular_acceleration'] = data['angular_speed'].diff() / data['timestamp'].diff()
    data['angle_diff'] = data['heading'].diff()

    # 将计算出的特征合并回原始数据框
    data.loc[data.index, 'heading'] = data['heading']
    data.loc[data.index, 'distance'] = data['distance']
    data.loc[data.index, 'speed'] = data['speed']
    data.loc[data.index, 'acceleration'] = data['acceleration']
    data.loc[data.index, 'angular_speed'] = data['angular_speed']
    data.loc[data.index, 'angular_acceleration'] = data['angular_acceleration']
    data.loc[data.index, 'angle_diff'] = data['angle_diff']

    return data


# 计算时间窗口特征
# 输入：原始数据 df，时间列名 time_column，特征列名 feature_columns，窗口大小 window_size_seconds
# 输出：原始数据 df，新增了窗口特征列
def custom_window_feature(data, time_column, feature_columns, window_size_seconds):
    data = data.sort_values(by=time_column)

    # 初始化新的特征列
    for col in feature_columns:
        data[f'{col}_med_{window_size_seconds}s'] = None
        data[f'{col}_std_{window_size_seconds}s'] = None

    # 按时间划分滑动窗口
    # 从当前时间往后滑动窗口，窗口大小为 window_size_seconds
    for i in range(len(data)):
        current_time = data.iloc[i][time_column]  # 当前时间戳
        start_time = current_time - window_size_seconds  # 起始时间戳
        window_data = data[(data[time_column] >= start_time) & (data[time_column] <= current_time)]  # 窗口数据
        for col in feature_columns:
            data.at[i, f'{col}_med_{window_size_seconds}s'] = round(window_data[col].median(), 2)  # 中位数
            data.at[i, f'{col}_std_{window_size_seconds}s'] = round(window_data[col].std(), 2)  # 标准差

    return data


def cal_density(data, rect_size=10):
    # 创建坐标转换器（使用新版pyproj语法）
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # 执行坐标转换
    x, y = transformer.transform(data['long'].values, data['lat'].values)

    # 创建坐标矩阵
    coords = np.column_stack((x, y))

    # 计算每个点的矩形边界
    half_size = rect_size / 2
    min_bounds = coords - half_size
    max_bounds = coords + half_size

    # 统计每个矩形内的点数
    densities = []
    for i in range(len(coords)):
        # 找出在当前点矩形范围内的所有点
        in_rect = np.all((coords >= min_bounds[i]) & (coords <= max_bounds[i]), axis=1)
        densities.append(np.sum(in_rect))

    data['density'] = np.array(densities)


def process_files_in_zip(zip_path, output_zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('temp_data')  # 解压到临时文件夹

    # 排序临时文件夹
    files = os.listdir('temp_data')
    sorted_files = sorted(files, key=lambda x: int(x.split('.')[0]))

    processed_files = []
    for file in sorted_files:
        if file.endswith('.xlsx'):
            file_path = os.path.join('temp_data', file)
            df = pd.read_excel(file_path)
            # 处理时间格式
            df['timestamp'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]) * 3600 +
                                                    int(x.split(':')[1]) * 60 + int(x.split(':')[2]))
            df['time_diff'] = (df['timestamp'] - df['timestamp'].shift())
            df['density'] = 0
            try:
                motion_feature(df)
                df = df.dropna().reset_index(drop=True)  # 在提取完运动特征后先删除空值行再进行窗口特征计算，避免出现更多的空值
                feature_column = ['speed', 'acceleration', 'angular_speed', 'angular_acceleration', 'angle_diff']
                # df = custom_window_feature(df, 'timestamp', feature_column, 5)
                # df = custom_window_feature(df, 'timestamp', feature_column, 20)
                df = custom_window_feature(df, 'timestamp', feature_column, 15)
                df = custom_window_feature(df, 'timestamp', feature_column, 50)
                df = df.dropna().reset_index(drop=True)
                cal_density(df)

                # 将处理后的文件保存到临时文件夹
                new_file_path = os.path.join('temp_data', f'{os.path.splitext(file)[0]}.xlsx')
                df.to_excel(new_file_path, index=False)
                processed_files.append(new_file_path)
            except Exception as e:
                print(f"文件 {file} 特征计算过程中出现错误: {e}")

    # 创建新的zip文件并将处理后的文件添加到其中
    with zipfile.ZipFile(output_zip_path, 'w') as zip_out:
        for file in processed_files:
            zip_out.write(file, os.path.basename(file))

    # 清理临时文件夹
    for file in processed_files:
        os.remove(file)
    os.rmdir('temp_data')

    print('特征计算完成，数据已保存到新的zip文件中')


if __name__ == '__main__':
    try:
        zip_path = '../data/raw_data_cau.zip'
        output_zip_path = '../data/cal_features_data.zip'
        process_files_in_zip(zip_path, output_zip_path)
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
