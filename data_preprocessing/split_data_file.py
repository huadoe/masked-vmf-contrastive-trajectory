import os
import re
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 假设 SignedMinMaxScaler 也在某个地方（比如 scaler_data.py）
from scaler_data import SignedMinMaxScaler 

# 将上一级目录添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config_file import Config

# 获取压缩文件中的文件列表
file_dir = '../data/clustered_data.zip' # 存放压缩文件的目录
output_dir = "../datasets/"
# 创建保存归一化后原始数据的文件夹
normalized_data_dir = os.path.join(output_dir, "normalized_data_now")
os.makedirs(normalized_data_dir, exist_ok=True) # 确保文件夹存在
config = Config()

def create_windows(data, labels, tags, window_size):
    """
    根据标签动态步长划分时间窗口。
    """
    samples = []
    window_labels = []
    window_tags = []
    start = 0

    while start <= len(data) - window_size:
        end = start + window_size
        mid = start + window_size // 2

        # 检查窗口内的 'type' 是否都相同
        if len(np.unique(labels[start:end])) == 1:
            current_label = labels[mid]
            samples.append(data[start:end])
            window_labels.append(current_label)
            window_tags.append(tags[mid])

            if current_label == 0:
                start += 1
            elif current_label == 1:
                start += 10
            else:
                start += 1
        else:
            start += 1
    
    return np.array(samples), np.array(window_labels), np.array(window_tags)

def process_and_save_by_class_per_file_norm(zip_ref, file_list, dataset_name, output_dir, window_size):
    """
    处理文件列表，生成时间窗口，并按类别保存数据集。
    **每个文件单独拟合和转换归一化器。**
    """
    all_x = []
    all_y = []
    all_cluster_tag = []

    for f in file_list:
        with zip_ref.open(f) as file:
            df = pd.read_excel(file).sort_values(by='timestamp')
            
            # --- 关键修改：每个文件单独归一化 ---
            # 1. 实例化一个新的归一化器
            scaler = SignedMinMaxScaler(feature_range=(-1, 1))
            data_to_normalize = df.drop(columns=['type', 'timestamp', 'long', 'lat', 'cluster_tag', 'time_diff'])

            # 2. 拟合归一化器 (仅针对当前文件)
            scaler.fit(data_to_normalize)

            # 3. 使用拟合的归一化器转换数据 (仅针对当前文件)
            normalized_features = scaler.transform(data_to_normalize)
            
            # 划分时间窗口
            x_samples, y_labels, tag_labels = create_windows(normalized_features, df['type'].values, df['cluster_tag'].values, window_size)
            
            all_x.extend(x_samples)
            all_y.extend(y_labels)
            all_cluster_tag.extend(tag_labels)
            
            # 如果是训练集，可以保存归一化器，以便后续检查 (可选)
            # if dataset_name == "train":
            #     scaler_path = os.path.join(output_dir, f"scaler_{f.split('.')[0]}.pkl")
            #     joblib.dump(scaler, scaler_path)
            #     print(f"已保存文件 {f} 的归一化器到 {scaler_path}")
            
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_cluster_tag = np.array(all_cluster_tag)
    
    # 根据标签将数据分离
    class_0_indices = np.where(all_y == 0)[0]
    class_1_indices = np.where(all_y == 1)[0]
    
    # 保存类别 0 的数据
    if len(class_0_indices) > 0:
        dat_dict_0 = {
            "samples": torch.from_numpy(all_x[class_0_indices]).float(), # 转换为 float 类型
            "labels": torch.from_numpy(all_y[class_0_indices]).long(), # 标签使用 long 类型
            "cluster_tags": torch.from_numpy(all_cluster_tag[class_0_indices]).long(), # 标签使用 long 类型
        }
        output_path_0 = os.path.join(output_dir, f"{dataset_name}_class_0.pt")
        torch.save(dat_dict_0, output_path_0)
        print(f"已保存 {dataset_name} 类别 0 的数据到 {output_path_0}")
        print(f"形状: {all_x[class_0_indices].shape}, 长度: {len(class_0_indices)}")
    
    # 保存类别 1 的数据
    if len(class_1_indices) > 0:
        dat_dict_1 = {
            "samples": torch.from_numpy(all_x[class_1_indices]).float(), # 转换为 float 类型
            "labels": torch.from_numpy(all_y[class_1_indices]).long(), # 标签使用 long 类型
            "cluster_tags": torch.from_numpy(all_cluster_tag[class_1_indices]).long(), # 标签使用 long 类型
        }
        output_path_1 = os.path.join(output_dir, f"{dataset_name}_class_1.pt")
        torch.save(dat_dict_1, output_path_1)
        print(f"已保存 {dataset_name} 类别 1 的数据到 {output_path_1}")
        print(f"形状: {all_x[class_1_indices].shape}, 长度: {len(class_1_indices)}")

    print(f"总 {dataset_name} 类别分布: {np.bincount(all_y)}")

# --- 主程序逻辑 ---
with zipfile.ZipFile(file_dir, 'r') as zip_ref:
    file_list = [f for f in zip_ref.namelist() if f.endswith('.xlsx')]
    file_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))

    # 1. 定义训练、验证、测试文件列表
    train_files = []
    val_files = []
    test_files = []
    
    # 使用列表推导式简化文件分类
    for file_name in file_list:
        match = re.fullmatch(r'(\d+)\.xlsx', file_name)
        if match:
            file_number = int(match.group(1))
            if (0 <= file_number <= 71) or (101 <= file_number <= 300):
                train_files.append(file_name)
            elif 72 <= file_number <= 85:
                val_files.append(file_name)
            elif 86 <= file_number <= 100:
                test_files.append(file_name)

    # 2. **跳过全局归一化拟合**：
    # 既然每个文件单独归一化，则不需要在所有训练数据上拟合一个全局归一化器并保存。
    # 原始的 scaler.fit 和 joblib.dump(scaler, '../scaler.pkl') 步骤被移除或注释。
    # scaler = SignedMinMaxScaler(feature_range=(-1, 1))
    # scaler.fit(pd.concat(all_train_data))
    # joblib.dump(scaler, '../scaler.pkl') # 不再保存全局 scaler
    
    # 3. 分别处理并按类别保存训练、验证和测试集
    window_size = config.window_size
    
    # 调用新的按文件独立归一化的处理函数
    process_and_save_by_class_per_file_norm(zip_ref, train_files, "train", output_dir, window_size)
    print("---" * 10)
    process_and_save_by_class_per_file_norm(zip_ref, val_files, "val", output_dir, window_size)
    print("---" * 10)
    process_and_save_by_class_per_file_norm(zip_ref, test_files, "test", output_dir, window_size)

print("\n数据集按类别划分并保存完成！")
