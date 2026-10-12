"""工具文件"""
import torch
import random
import numpy as np
import os
import sys
import logging
from geopy import Point
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from shutil import copy
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx


def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir):
    """
    计算模型的性能指标，并保存到指定目录。
    :param pred_labels:  预测的标签
    :param true_labels:  真实的标签
    :param log_dir:      日志保存目录
    """
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(log_dir, file_name)
    df.to_excel(report_Save_path)

    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    创建日志记录器。
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def copy_Files(destination):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_file.py", os.path.join(destination_dir, "config_file.py"))
    copy("dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader_notype.py", os.path.join(destination_dir, "dataloader_notype.py"))
    copy(f"models/base_model.py", os.path.join(destination_dir, f"base_model.py"))
    copy("models/loss_function.py", os.path.join(destination_dir, "loss_function.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))
    copy("models/ssm.py", os.path.join(destination_dir, "ssm.py"))
    copy("clustering.py", os.path.join(destination_dir, "clustering.py"))


def plot_column_distribution(data, column):
    """
    绘制指定列的分布图。
    :param data: 数据的 pandas DataFrame。
    :param column: 列名。
    """
    plt.hist(data[column], bins=100)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(column + ' Distribution')
    plt.show()


def plot_lat_lon_distribution(data, lat_col='latitude', lon_col='longitude', label_col='label',
                              title='Latitude-Longitude Distribution'):
    """
    根据经纬度列和类别标签绘制样本分布图，不同标签用不同颜色，并有卫星图背景。

    参数:
    - data: 包含经纬度数据和标签的 pandas DataFrame。
    - lat_col: 纬度列名，默认为 'latitude'。
    - lon_col: 经度列名，默认为 'longitude'。
    - label_col: 标签列名，默认为 'label'。
    - title: 图表标题，默认为 'Latitude-Longitude Distribution'。
    """
    # 确保经纬度和标签列为数值型
    data[lat_col] = pd.to_numeric(data[lat_col], errors='coerce')
    data[lon_col] = pd.to_numeric(data[lon_col], errors='coerce')
    data[label_col] = pd.to_numeric(data[label_col], errors='coerce')

    # 统计不同标签的数量
    field_count = len(data[data[label_col] == 1])
    road_count = len(data[data[label_col] == 0])
    print('Field count:', field_count)
    print('Road count:', road_count)

    # 删除包含缺失值的行
    data = data.dropna(subset=[lat_col, lon_col, label_col])

    # 将 DataFrame 转换为 GeoDataFrame
    geometry = [Point(xy) for xy in zip(data[lon_col], data[lat_col])]
    geo_data = gpd.GeoDataFrame(data, geometry=geometry)

    # 转换为 Web Mercator 坐标系
    geo_data = geo_data.set_crs(epsg=4326)
    geo_data = geo_data.to_crs(epsg=3857)

    # 创建基础绘图
    ax = geo_data.plot(figsize=(10, 6), alpha=0.6, edgecolor='k')

    # 添加不同类别的点
    categories = {
        0: {'color': 'orange', 'label': 'Road'},
        1: {'color': 'green', 'label': 'Field'}
    }

    for cat, props in categories.items():
        subset = geo_data[geo_data[label_col] == cat]
        subset.plot(ax=ax, color=props['color'], label=props['label'], markersize=10)

    # 添加卫星图背景
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

    # 设置图表属性
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.show()

