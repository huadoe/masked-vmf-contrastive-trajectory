"""使用训练好的分类器进行预测（修正版）"""
import numpy as np
import torch
import pandas as pd
import joblib  # 用于保存和加载 scaler
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

from models.base_model import base_Model
import torch.nn.functional as F
from config_file import Config


def load_checkpoint_weights(path, model, device):
    """
    兼容两种常见情况：
    1) torch.save(model.state_dict(), path) -> checkpoint 是 state_dict
    2) torch.save({'base_model_state_dict': model.state_dict(), ...}, path)
    """
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        # 尝试常见的 key
        if 'base_model_state_dict' in checkpoint:
            state = checkpoint['base_model_state_dict']
        elif 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        else:
            # 可能整个 dict 就是 state_dict（训练代码有时直接保存 dict）
            # 如果值看起来像 tensor 则当作 state_dict
            # 简单判断：若键都是 str 且值是 Tensor/ndarray -> 视为 state_dict
            # 否则直接报错
            maybe_state = True
            for k, v in checkpoint.items():
                if not isinstance(k, str):
                    maybe_state = False
                    break
            if maybe_state:
                state = checkpoint
            else:
                raise RuntimeError("无法识别 checkpoint 格式，请检查保存时的 key 名称。")
    else:
        raise RuntimeError("checkpoint 不是 dict 格式，无法加载。")

    model.load_state_dict(state)
    return model


def extract_windows_and_centers(data_x, data_y, window_size, step):
    """
    从 numpy data_x(data_y) 中提取窗口与对应 center indices（相对于原序列索引）
    返回：
      windows: list of ndarray (window_size, feat)
      centers: list of int (center index in original sequence)
      labels: list of labels (window center label)
    """
    pred_x = []
    centers = []
    labels = []

    start = 0
    N = len(data_x)
    while start <= N - window_size:
        end = start + window_size
        mid = start + window_size // 2
        if len(np.unique(data_y[start:end])) == 1:
            pred_x.append(data_x[start:end])
            centers.append(mid)
            labels.append(data_y[mid])
            start += step
        else:
            start += 1

    if len(pred_x) == 0:
        return np.zeros((0, window_size, data_x.shape[1])), [], np.array([])
    return np.array(pred_x), centers, np.array(labels)


def predict(data_df, config, checkpoint_path='logs/Exp_19/Run_2/train_linear_seed_42/saved_models/ckp_last.pt',
            scaler_path='scaler.pkl', device=None):
    # device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 scaler
    scaler = joblib.load(scaler_path)

    # prepare data arrays
    raw_features = scaler.transform(data_df.drop(columns=['type']))
    raw_labels = data_df['type'].values

    # 提取窗口 & 中心索引 & labels
    window_size = config.window_size
    step = getattr(config, 'step', 1)
    windows, centers, original_y = extract_windows_and_centers(raw_features, raw_labels, window_size, step)

    if windows.shape[0] == 0:
        raise ValueError("没有生成任何窗口。检查 window_size/step/数据标签一致性。")

    # 转为 tensor 并送 device（模型期望输入 shape: (B, T, C)）
    X_tensor = torch.tensor(windows, dtype=torch.float32, device=device)

    # 实例化模型并加载权重
    clf = base_Model(config).to(device)
    clf = load_checkpoint_weights(checkpoint_path, clf, device)
    clf.eval()

    with torch.no_grad():
        logits, _ = clf(X_tensor)               # logits (B, num_classes)
        # 如果你更喜欢概率可以用 softmax；但 classification_report 用整数标签就够了
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    # 返回：概率、预测标签、真实标签、center indices（用于回映坐标）
    return probs, preds, original_y, centers


if __name__ == '__main__':
    # 读取测试数据
    data = pd.read_excel('data/pred_data/10.xlsx')
    coordinates = data[['long', 'lat']].values
    data = data.drop(columns=['timestamp', 'long', 'lat', 'cluster_tag', 'time_diff'])
    config = Config()

    # 预测
    y_pred_proba, y_pred, y_true, centers = predict(data, config)

    print("样本窗口数:", len(y_true))
    print("分类报告:\n", classification_report(y_true, y_pred, digits=4))

    # 将中心索引映射回坐标
    pred_coordinates = coordinates[centers]  # centers 已经是 center 索引数组

    # 根据经纬度和标签绘图
    plt.figure(figsize=(10, 8))
    plt.scatter(pred_coordinates[:, 0], pred_coordinates[:, 1], c=y_pred, cmap='viridis', s=8)
    plt.colorbar(label='Predicted Label')
    plt.title('Predicted Labels on Coordinates')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

