import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
try:
    import hdbscan  # 可选依赖（仅注释掉的 HDBSCAN 聚类分支使用）
except ImportError:
    hdbscan = None
import torch.nn.functional as F

class ClusterMemory:
    def __init__(self, n_clusters, feature_dim, device):
        self.n_clusters = n_clusters
        self.device = device

        self.centroids = torch.zeros(n_clusters, feature_dim, dtype=torch.float32).to(device)
        self.labels = None  # 每个样本的 cluster id
        self.high_quality_mask = None  # 是否高质量样本

    @torch.no_grad()
    def run_clustering(self, features):
        """
        features: (N, D) tensor
        """
        features_np = features.cpu().numpy()

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = kmeans.fit_predict(features_np)
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

        self.labels = torch.tensor(labels).long().to(self.device)
        self.centroids = F.normalize(centroids, dim=1)

    # def run_clustering(self, features):
    #     """
    #     features: (N, D) tensor
    #     将结果整合为两个簇：噪声(-1)和非噪声(0)
    #     """
    #     features_np = features.cpu().numpy()
    #     # dbscan = DBSCAN(eps=5.0e-6, min_samples=50)
    #     # labels = dbscan.fit_predict(features_np)

    #     cluster = hdbscan.HDBSCAN(min_cluster_size=50, gen_min_span_tree=False)
    #     labels = cluster.fit_predict(features_np)

    #     # 将结果整合为两个簇：噪声(-1)保持为1，非噪声(其他标签)设为0
    #     integrated_labels = np.where(labels == -1, 1, 0)
    #     self.labels = torch.tensor(integrated_labels).long().to(self.device)
        
    #     # 为整合后的两个簇计算中心（使用非噪声样本的均值作为一个中心，噪声样本的均值作为另一个中心）
    #     if len(features_np) > 0:
    #         # 非噪声样本中心
    #         non_noise_features = features_np[labels != -1]
    #         # 噪声样本中心
    #         noise_features = features_np[labels == -1]
            
    #         # 计算中心
    #         centroids = []
    #         if len(non_noise_features) > 0:
    #             non_noise_center = np.mean(non_noise_features, axis=0)
    #             centroids.append(non_noise_center)
    #         else:
    #             # 如果没有非噪声样本，使用随机中心
    #             centroids.append(np.random.rand(features_np.shape[1]))
            
    #         if len(noise_features) > 0:
    #             noise_center = np.mean(noise_features, axis=0)
    #             centroids.append(noise_center)
    #         else:
    #             # 如果没有噪声样本，使用随机中心
    #             centroids.append(np.random.rand(features_np.shape[1]))
            
    #         # 确保有两个中心
    #         while len(centroids) < 2:
    #             centroids.append(np.random.rand(features_np.shape[1]))
            
    #         centroids_np = np.array(centroids, dtype=np.float32)
    #         self.centroids = F.normalize(torch.tensor(centroids_np).to(self.device), dim=1)

    # def run_clustering(self, features):
    #     """
    #     features: (N, D) tensor
    #     使用谱聚类进行聚类，将结果整合为两个簇：噪声(-1)和非噪声(0)
    #     """
    #     features_np = features.cpu().numpy()
        
    #     # 限制样本数量，避免谱聚类计算过慢
    #     max_samples = 5000
    #     if features_np.shape[0] > max_samples:
    #         # 随机采样
    #         indices = np.random.choice(features_np.shape[0], max_samples, replace=False)
    #         sampled_features = features_np[indices]
    #         use_sampling = True
    #     else:
    #         sampled_features = features_np
    #         use_sampling = False
        
    #     # 使用谱聚类
    #     spectral = SpectralClustering(
    #         n_clusters=2,  # 直接聚类为2个簇
    #         affinity='nearest_neighbors',  # 使用最近邻构建亲和矩阵
    #         n_neighbors=5,  # 减少最近邻数量以提高速度
    #         assign_labels='kmeans',  # 使用kmeans分配标签
    #         random_state=42,
    #         n_jobs=-1  # 使用所有CPU核心
    #     )
        
    #     # 对采样数据进行聚类
    #     sampled_labels = spectral.fit_predict(sampled_features)
        
    #     # 如果使用了采样，需要将标签映射回原始数据
    #     if use_sampling:
    #         # 初始化所有标签为0
    #         labels = np.zeros(features_np.shape[0], dtype=int)
    #         # 将采样数据的标签赋值回对应位置
    #         labels[indices] = sampled_labels
    #     else:
    #         labels = sampled_labels

    #     # 谱聚类没有噪声标签，所有点都属于某个簇
    #     # 这里我们将其中一个簇视为"噪声"，另一个视为"非噪声"
    #     # 简单起见，将标签0视为非噪声，标签1视为噪声
    #     integrated_labels = np.where(labels == 1, 1, 0)
    #     self.labels = torch.tensor(integrated_labels).long().to(self.device)
        
    #     # 为整合后的两个簇计算中心
    #     if len(features_np) > 0:
    #         # 非噪声样本中心
    #         non_noise_features = features_np[labels == 0]
    #         # 噪声样本中心
    #         noise_features = features_np[labels == 1]
            
    #         # 计算中心
    #         centroids = []
    #         if len(non_noise_features) > 0:
    #             non_noise_center = np.mean(non_noise_features, axis=0)
    #             centroids.append(non_noise_center)
    #         else:
    #             # 如果没有非噪声样本，使用随机中心
    #             centroids.append(np.random.rand(features_np.shape[1]))
            
    #         if len(noise_features) > 0:
    #             noise_center = np.mean(noise_features, axis=0)
    #             centroids.append(noise_center)
    #         else:
    #             # 如果没有噪声样本，使用随机中心
    #             centroids.append(np.random.rand(features_np.shape[1]))
            
    #         # 确保有两个中心
    #         while len(centroids) < 2:
    #             centroids.append(np.random.rand(features_np.shape[1]))
            
    #         centroids_np = np.array(centroids, dtype=np.float32)
    #         self.centroids = F.normalize(torch.tensor(centroids_np).to(self.device), dim=1)

    @torch.no_grad()
    def compute_prototypes(self, features):
        """
        计算每个簇的原型（簇内特征均值）
        features: (N, D) tensor
        """
        features = F.normalize(features.to(torch.float32), dim=1)
        
        self.prototypes = torch.zeros(self.n_clusters, features.shape[1], 
                                      dtype=torch.float32).to(self.device)
        
        for c in range(self.n_clusters):
            mask = self.labels == c
            if mask.any():
                cluster_features = features[mask]
                self.prototypes[c] = cluster_features.mean(dim=0)
                self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)
            else:
                # 如果某个簇没有样本，用随机初始化
                self.prototypes[c] = F.normalize(torch.randn(features.shape[1]).to(self.device), dim=0)

    @torch.no_grad()
    def compute_high_quality(self, features, threshold=0.5):
        """
        基于"样本-中心相似度"筛选高质量样本
        Run_17 基线语义：绝对阈值（与所属簇中心的余弦相似度 > threshold）
        （v2 曾改为 top30% 分位数，无视 threshold 参数，已恢复绝对阈值）
        """
        features = F.normalize(features.to(torch.float32), dim=1)

        sim = torch.matmul(features, self.centroids.T)  # (N, K)
        assigned_sim = sim[torch.arange(len(features)), self.labels]

        self.high_quality_mask = assigned_sim > threshold

    @torch.no_grad()
    def compute_confidences(self, features, tau=0.1):
        """
        v2: 计算每个样本伪标签的置信度权重（用于 vMF 概率对比损失的加权）
        置信度 = softmax(到各中心的相似度 / tau) 中所属簇的概率
        """
        features = F.normalize(features.to(torch.float32), dim=1)
        sim = torch.matmul(features, self.centroids.T)  # (N, K)
        probs = torch.softmax(sim / tau, dim=1)
        self.confidences = probs[torch.arange(len(features)), self.labels]  # (N,)
        return self.confidences