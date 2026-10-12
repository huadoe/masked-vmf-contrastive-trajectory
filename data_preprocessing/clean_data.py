"""数据清洗，包括删除停止点、漂移点以及离群点。"""
import os
import zipfile
from math import radians, degrees, sin, cos, atan2, sqrt

import numpy as np
import pandas as pd
import sys

from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
from pyproj import Transformer

# 将上一级目录添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_file import Config

config = Config()


def delete_stop_points(data):
    # 删除经纬度完全相同且速度为0的样本
    long, lat, speed = data['long'], data['lat'], data['speed']
    delete_samples = []
    for i in range(len(data)):
        if i == 0:
            delete_samples.append(i)
            continue
        if long[i] == long[i-1] and lat[i] == lat[i-1] and speed[i] == 0:
            delete_samples.append(i)
    return data.drop(delete_samples).reset_index(drop=True)


def delete_abnormal_points(data):
    # 异常点有以下三种:
    # 1、漂移，即两个连续点之间的距离大于500m
    distance = data['distance']
    delete_samples = []
    for i in range(len(data)):
        if distance[i] > 500:
            delete_samples.append(i)
            print("删除漂移点")
    data = data.drop(delete_samples).reset_index(drop=True)

    # 2、速度很小但角度变化大于90度的样本
    speed, angle_diff = data['speed'], data['angle_diff']
    delete_samples = []
    for i in range(len(data)):
        if speed[i] <= 0.1 and angle_diff[i] >= 90:
            delete_samples.append(i)
    data = data.drop(delete_samples).reset_index(drop=True)

    # 3、转动角度不处于-60度到60度之间的样本
    angle_diff = data['angle_diff']
    delete_samples = []
    for i in range(len(data)):
        if angle_diff[i] < -60 or angle_diff[i] > 60:
            delete_samples.append(i)
    return data.drop(delete_samples).reset_index(drop=True)


def moving_average(num, lst):
    # 计算样本移动的平均距离
    ma = [0 for x in range(len(lst))]
    ma[0] = np.mean([lst[0]])
    for i in range(1, len(lst)):
        if i <= num:
            ma[i] = np.mean(lst[:i+1])
        else:
            ma[i] = np.mean(lst[i-num:i+1])
    return ma


def delete_scatter_points(data, average_distance):
    # 删除离群点，即样本移动平均距离大于阈值的样本
    delete_samples = []
    for q in range(len(data)):
        if (average_distance[q] <= config.scatter_distance_threshold) and (data['type'][q] == 0):
            delete_samples.append(q)
            continue

    return data.drop(delete_samples).reset_index(drop=True)


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

    count = 0

    processed_files = []
    min_file_number = 10000
    min_count = 0
    for file in sorted_files:
        if file.endswith('.xlsx'):
            file_path = os.path.join('temp_data', file)
            df = pd.read_excel(file_path)
            count += 1
            print(len(df), count)
            df['density'] = 0
            try:
                df = delete_stop_points(df)
                df = delete_abnormal_points(df)
                # distance_list = df['distance']
                # distance_average = moving_average(config.moving_average_num, distance_list)
                # for r in range(len(list(distance_list))):
                #     distance_average[r] = round(distance_average[r], 2)
                # df = delete_scatter_points(df, distance_average)
                cal_density(df)
                print(len(df))
                file_len = len(df)
                if file_len <= min_file_number:
                    min_file_number = file_len
                    min_count = count
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

    print(f'数据清洗完成，数据已保存到新的zip文件中，最小文件数是{min_file_number}，第{min_count}个文件')


if __name__ == '__main__':
    try:
        zip_path = '../data/cal_features_data.zip'
        output_zip_path = '../data/cleaned_data.zip'
        process_files_in_zip(zip_path, output_zip_path)
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
