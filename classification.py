# predict_and_eval.py
import numpy as np
import torch
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import seaborn as sns  # 用于更好画混淆矩阵
from sklearn.preprocessing import StandardScaler

from models.base_model import base_Model
from config_file import Config

# 如果希望使用 DTW 聚类（序列级别），需要 tslearn
try:
    from tslearn.clustering import TimeSeriesKMeans
    tslearn_available = True
except Exception:
    tslearn_available = False


def extract_windows_and_labels(data_df, scaler, window_size, step):
    """
    从 dataframe 提取滑窗序列和对应中间标签（与你原逻辑一致）
    data_df: 已去除不必要列，包含 'type'
    返回:
        X_windows: np.array (n_windows, window_size, feature_dim)
        y_windows: np.array (n_windows,)
    """
    data_x = scaler.transform(data_df.drop(columns=['type']))  # numpy array (N, feat)
    data_y = data_df['type'].values
    pred_x = []
    original_y = []
    start = 0
    N = len(data_x)
    while start <= N - window_size:
        end = start + window_size
        mid = start + window_size // 2
        if len(np.unique(data_y[start:end])) == 1:
            current_label = data_y[mid]
            pred_x.append(data_x[start:end])
            original_y.append(current_label)
            # 保持原来行为：每次只移动1步
            start += step
        else:
            start += 1
    pred_x = np.array(pred_x)          # (n_windows, window_size, feat_dim)
    original_y = np.array(original_y)  # (n_windows,)
    return pred_x, original_y


def predict_and_extract_features(data_df, config, checkpoint_path, scaler_path, device=None):
    # device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    clf = base_Model(config).to(device)
    if 'base_model_state_dict' in checkpoint:
        clf.load_state_dict(checkpoint['base_model_state_dict'])
    else:
        # 兼容性：如果保存的是 state_dict 本身
        clf.load_state_dict(checkpoint)
    clf.eval()

    # 加载 scaler
    scaler = joblib.load(scaler_path)

    # 取窗口
    window_size = config.window_size
    step = config.step
    X_win, y_win = extract_windows_and_labels(data_df, scaler, window_size, step)
    if len(X_win) == 0:
        raise ValueError("没有生成任何窗口。检查 window_size / step / 数据标签一致性。")

    # 转 tensor: base_Model 接受形状 (B, T, C)
    X_tensor = torch.tensor(X_win, dtype=torch.float32).to(device)

    # 推理并提取特征
    with torch.no_grad():
        logits, seq_feat = clf(X_tensor)   # logits: (B, num_classes), seq_feat: (B, C, seq_len)

    # 全局特征：对时间维度做平均池化 -> (B, C)
    global_feat = seq_feat.mean(dim=2).cpu().numpy()
    # scaler2 = StandardScaler()
    # global_feat = scaler2.fit_transform(global_feat)

    # 如果做时序 DTW 聚类：将通道上池化得到单通道时间序列 (B, seq_len)
    seq_for_dtw = seq_feat.mean(dim=1).cpu().numpy()  # shape (B, seq_len)

    return y_win, global_feat, seq_for_dtw, logits.cpu().numpy()


def cluster_and_visualize(y_true, global_feat, seq_for_dtw, use_dtw=False, n_clusters=2, tslearn_max_iter=10):
    """
    - use_dtw=False: 对 global_feat (B, C) 用 KMeans 聚类
    - use_dtw=True: 使用 TimeSeriesKMeans(DTW) 对 seq_for_dtw 做聚类（需要 tslearn）
    返回 aligned_pred, raw_pred
    """
    if use_dtw:
        if not tslearn_available:
            raise RuntimeError("tslearn 未安装，无法做 DTW 聚类。")
        # tslearn 接收形状 (n_ts, sz, dim). 我们的 seq_for_dtw: (B, seq_len) -> expand last dim
        seq_input = seq_for_dtw[..., np.newaxis]
        kmeans_ts = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=tslearn_max_iter, random_state=42, n_jobs=-1)
        raw_pred = kmeans_ts.fit_predict(seq_input)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        raw_pred = kmeans.fit_predict(global_feat)

    # 对齐标签（Hungarian）
    cm = confusion_matrix(y_true, raw_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    aligned_pred = np.zeros_like(raw_pred)
    for i, j in zip(row_ind, col_ind):
        aligned_pred[raw_pred == j] = i

    # 打印评估
    print("Confusion Matrix (aligned):")
    print(confusion_matrix(y_true, aligned_pred))
    print("\nClassification Report (aligned):")
    print(classification_report(y_true, aligned_pred, digits=4))

    # 可视化混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, aligned_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (aligned)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return aligned_pred, raw_pred

import numpy as np
from sklearn.neighbors import NearestNeighbors

def neighborhood_consistency_refine(global_feat, raw_pred, k=15, theta=0.6, T=3):
    """
    基于邻域一致性的聚类结果增强（无监督）。
    
    参数：
        global_feat : ndarray (B, C)
        raw_pred    : ndarray (B,)
        k           : kNN 数
        theta       : 邻域一致比例阈值（如 60%）
        T           : 多轮传播次数
    返回：
        refined_pred (ndarray): 增强后的标签（无监督 refinement）
    """
    refined = raw_pred.copy()
    n = len(refined)

    # 构建 kNN
    knn = NearestNeighbors(n_neighbors=k).fit(global_feat)
    neighbors = knn.kneighbors(return_distance=False)

    for _ in range(T):
        new_refined = refined.copy()

        for i in range(n):
            neigh = neighbors[i]
            neigh_labels = refined[neigh]

            # 统计邻居中每类数量
            unique, counts = np.unique(neigh_labels, return_counts=True)

            # 找到局部多数类
            major_cluster = unique[np.argmax(counts)]
            major_ratio = counts.max() / k

            # 若邻域一致性强 → 覆盖当前标签
            if major_ratio >= theta:
                new_refined[i] = major_cluster

        refined = new_refined

    return refined

def tsne_and_plot(global_feat, y_true, perplexity=30):
    n_samples = global_feat.shape[0]
    # perplexity 必须 < n_samples
    if n_samples - 1 <= 0:
        raise ValueError("样本数过少，无法进行 t-SNE。")
    if perplexity >= n_samples:
        adjusted = max(5, n_samples // 3)
        print(f"[Warning] t-SNE perplexity ({perplexity}) >= n_samples ({n_samples}), 自动调整为 {adjusted}.")
        perplexity = adjusted

    tsne = TSNE(n_components=2, perplexity=perplexity, init='random', random_state=42)
    feat2 = tsne.fit_transform(global_feat)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(feat2[:, 0], feat2[:, 1], c=y_true, cmap='viridis', s=12, alpha=0.8)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE of Global Features (mean pooled over time)")
    plt.show()


if __name__ == '__main__':
    # ========== 配置 ==========
    data_path = 'data/pred_data/10.xlsx'
    checkpoint_path = 'logs/Exp_21/Run_16/self_supervised_seed_42/saved_models/ckp_last.pt'
    # checkpoint_path = 'logs/Exp_Prototype/Run_1/self_supervised_seed_42/saved_models/ckp_last.pt'
    scaler_path = 'scaler.pkl'
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # =======================

    # 读取数据
    raw_df = pd.read_excel(data_path)
    coordinates = raw_df[['long', 'lat']].values
    # 保留 type 列，并删除不需要的列（和你之前一致）
    data_df = raw_df.drop(columns=['timestamp', 'long', 'lat', 'cluster_tag', 'time_diff'])

    # 提取 & 预测特征
    y_true, global_feat, seq_for_dtw, logits = predict_and_extract_features(
        data_df, config, checkpoint_path, scaler_path, device=device
    )

    print("global_feat.shape =", global_feat.shape)
    print("seq_for_dtw.shape =", seq_for_dtw.shape)

    # 聚类（优先用 global embedding + KMeans）
    aligned_pred, raw_pred = cluster_and_visualize(y_true, global_feat, seq_for_dtw, use_dtw=True, n_clusters=len(np.unique(y_true)))

    # ========== 邻域一致性增强（无监督） ==========
    refined_pred = neighborhood_consistency_refine(global_feat, raw_pred, k=15, theta=0.6, T=3)

    #与真值对齐（只用于评估，不用于算法）
    cm = confusion_matrix(y_true, refined_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    aligned_refined = np.zeros_like(refined_pred)
    for i, j in zip(row_ind, col_ind):
        aligned_refined[refined_pred == j] = i

    print("\n===== NCC Refined 聚类结果 =====")
    print(confusion_matrix(y_true, aligned_refined))
    print(classification_report(y_true, aligned_refined, digits=4))

    # 将标签映射回坐标（注意：窗口中心对应的 coordinate index）
    # 你的窗口中心索引是 start + window_size//2；提取时是按 window 顺序做的，所以这里也按顺序取坐标中心
    window_size = config.window_size
    # compute coordinate indices used: first window mid at index window_size//2, then step... but we used step=config.step there.
    # 为简化，直接重算与 extract_windows_and_labels 相同逻辑来取中心坐标
    _, y_centers = extract_windows_and_labels(data_df, joblib.load(scaler_path), window_size, getattr(config, 'step', 1))
    # 上面 y_centers 只是标签占位，实际我们需要对应的 coordinates:
    coords_pred = []
    start = 0
    N = len(raw_df)
    while start <= N - window_size:
        end = start + window_size
        mid = start + window_size // 2
        if len(np.unique(raw_df['type'].values[start:end])) == 1:
            coords_pred.append(coordinates[mid])
            start += getattr(config, 'step', 1)
        else:
            start += 1
    coords_pred = np.array(coords_pred)

    # 绘制预测标签在坐标上的分布
    plt.figure(figsize=(10, 8))
    plt.scatter(coords_pred[:, 0], coords_pred[:, 1], c=aligned_pred, cmap='viridis', s=6)
    plt.colorbar(label='Predicted Label (aligned)')
    plt.title('Predicted Labels on Coordinates (aligned)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # t-SNE 可视化 global features
    tsne_and_plot(global_feat, y_true)

