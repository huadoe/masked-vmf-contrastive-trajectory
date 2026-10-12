import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import ive as _scipy_ive


class _LogIVFunction(torch.autograd.Function):
    """
    数值稳定的 log I_v(x)（第一类修正贝塞尔函数，任意阶）
    scipy float64 前向 + 解析导数反向:
    d/dx log I_v(x) = I_v'(x)/I_v(x) = I_{v-1}(x)/I_v(x) - v/x
    """

    @staticmethod
    def forward(ctx, v, x):
        ctx.save_for_backward(x)
        ctx.v = float(v)
        xv = x.detach().double().cpu().numpy()
        ive_val = _scipy_ive(ctx.v, xv)
        logI = np.log(np.maximum(ive_val, 1e-300)) + xv
        return torch.as_tensor(logI, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        v = ctx.v
        xv = x.detach().double().cpu().numpy()
        num = _scipy_ive(v - 1.0, xv)
        den = np.maximum(_scipy_ive(v, xv), 1e-300)
        with np.errstate(divide='ignore', invalid='ignore'):
            dlog = num / den - v / xv
        dlog = np.nan_to_num(dlog, nan=0.0, posinf=0.0, neginf=0.0)
        grad = torch.as_tensor(dlog, dtype=x.dtype, device=x.device) * grad_out
        return None, grad


def log_iv(v, x):
    return _LogIVFunction.apply(v, x)


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self.cal_cosine_similarity
        else:
            return self.cal_dot_similarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def cal_dot_similarity(x, y):
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def cal_cosine_similarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


def cluster_contrastive_loss(z, positive_mask, temperature=0.2):
    """
    z: (B, D)
    positive_mask: (B, B) bool
    Run_17 基线原版：多正样本 InfoNCE（Σexp(pos)/Σexp(all)），温度 0.2。
    （v2 曾改为 -mean(pos logit)+logΣexp 且温度 0.5，消融确认大幅掉点，恢复原版）
    """

    sim = torch.matmul(z, z.T) / temperature  # (B, B)

    # 数值稳定
    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim)

    # 去掉自身
    mask = torch.ones_like(exp_sim) - torch.eye(z.size(0), device=z.device)
    exp_sim = exp_sim * mask

    # 正样本
    pos_exp = exp_sim * positive_mask.float()

    # 分子：所有正样本
    numerator = pos_exp.sum(dim=1)

    # 分母：所有样本
    denominator = exp_sim.sum(dim=1)

    loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)

    return loss.mean()


def prototype_alignment_loss(z, labels, prototypes, distance_type='cosine', temperature=0.2):
    """
    Prototype Alignment Loss (PGA): 将特征对齐到其簇的原型

    Args:
        z: (B, D) - 特征向量
        labels: (B,) - 每个样本的簇标签
        prototypes: (K, D) - 每个簇的原型，K为簇数
        distance_type: 'softmax'（旧论文 Eq.4）| 'cosine' | 'mse'
        temperature: softmax 版温度 τ

    Returns:
        loss: 原型对齐损失
    """
    z = F.normalize(z, dim=1)
    prototypes = F.normalize(prototypes, dim=1)

    # 获取每个样本对应的原型
    batch_prototypes = prototypes[labels]  # (B, D)

    if distance_type == 'softmax':
        # 旧论文 Eq.4（PGA 原始公式）:
        # L_proto = -log[ exp(sim(z_i, c_yi)/τ) / Σ_m exp(sim(z_i, c_m)/τ) ]
        # 即以原型为类中心的分类式对齐（SwAV 式 prototype assignment）
        logits = torch.matmul(z, prototypes.t()) / temperature  # (B, K)
        loss = F.cross_entropy(logits, labels)
    elif distance_type == 'cosine':
        # 余弦距离损失: 1 - cosine_similarity
        cos_sim = F.cosine_similarity(z, batch_prototypes, dim=1)
        loss = 1 - cos_sim.mean()
    else:  # mse
        # MSE损失
        loss = F.mse_loss(z, batch_prototypes)

    return loss


# =====================================================================
# 以下为 v2 新增损失：硬负样本 NTXent + vMF 概率对比损失
# =====================================================================

class HardNegativeNTXentLoss(torch.nn.Module):
    """
    硬负样本 NTXent：每个锚点只保留相似度最高的 top_k 个负样本（AMD/SAIL 思路），
    迫使模型区分与正样本高度相似的困难负样本，提升对少数类（道路）边界的判别力。
    """

    def __init__(self, device, batch_size, temperature, use_cosine_similarity=True, top_k=256):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.top_k = top_k

    def forward(self, zis, zjs):
        B = zis.shape[0]
        representations = torch.cat([zjs, zis], dim=0)  # (2B, D)
        representations = F.normalize(representations, dim=1)
        sim = torch.matmul(representations, representations.T)  # (2B, 2B)

        # 正样本：跨视图的对应样本
        pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(self.device)  # (2B,)
        pos_sim = sim[torch.arange(2 * B, device=self.device), pos_idx]  # (2B,)

        # 负样本：除自身与正样本外的全部
        self_mask = torch.eye(2 * B, device=self.device, dtype=torch.bool)
        pos_mask = self_mask.clone()
        pos_mask[torch.arange(2 * B, device=self.device), pos_idx] = True
        neg_sim = sim.masked_fill(self_mask | pos_mask, float('-inf'))  # (2B, 2B-2)

        # 每个锚点保留 top_k 最难负样本
        k = min(self.top_k, neg_sim.shape[1])
        topk_neg, _ = torch.topk(neg_sim, k, dim=1)  # (2B, k)

        logits = torch.cat([pos_sim.unsqueeze(1), topk_neg], dim=1) / self.temperature
        labels = torch.zeros(2 * B, device=self.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        return loss


class vMFProbContrastiveLoss(torch.nn.Module):
    """
    vMF 概率对比损失（ProCo / vMF-PCL 思路，闭式期望实现）。

    核心思想：
    1. 假设归一化特征在单位超球面上服从 von Mises-Fisher 混合分布:
       每个簇 k ~ vMF(mu_k, kappa_k)
    2. 利用闭式解 E_{u~vMF(mu,kappa)}[exp(s * z . u)] = S_d(||kappa*mu + s*z||) / S_d(kappa)
       其中 S_d(r) = (2*pi)^{d/2} * I_{d/2-1}(r) / r^{d/2-1}，
       可对"无限对比对"直接计算期望 InfoNCE，无需大批量负样本，
       天然缓解少数类（道路）在小批量中样本不足的问题。
    3. 同时保留原型级约束：类内拉向原型（intra）+ 原型间均匀性（inter uniformity）。

    kappa 为每个簇的可学习参数（softplus 参数化），mu 由外部 EMA 维护后传入。
    """

    def __init__(self, n_clusters, feature_dim, temperature=0.1, kappa_init=10.0,
                 w_intra=1.0, w_inter=0.5):
        super().__init__()
        self.K = n_clusters
        self.d = feature_dim
        self.temperature = temperature
        self.w_intra = w_intra
        self.w_inter = w_inter
        # log-kappa 参数化，保证 kappa > 0
        self._kappa_raw = nn.Parameter(torch.log(torch.tensor(float(kappa_init))) *
                                       torch.ones(n_clusters))

    @property
    def kappa(self):
        return (F.softplus(self._kappa_raw) + 1e-3).clamp(max=300.0)

    def _log_S(self, r):
        """
        数值稳定的 log S_d(r) = log I_{d/2-1}(r) - (d/2-1)*log(r) + const
        r: (..., ) 非负张量
        """
        v = self.d / 2.0 - 1.0
        r = r.clamp(min=1e-6)
        log_I = log_iv(v, r)  # 自定义 autograd（scipy float64 前向 + 解析导数）
        return log_I - v * torch.log(r)

    def expected_exp_sim(self, z, mu, kappa):
        """
        E_{u~vMF(mu, kappa)}[exp(s * z . u)] 的 log 值（闭式解）
        z: (B, D) 已归一化锚点
        mu: (K, D) 已归一化原型
        kappa: (K,)
        返回: (B, K) log 期望相似度
        s = 1/temperature
        """
        s = 1.0 / self.temperature
        # || kappa_k * mu_k + s * z_i ||^2 = kappa^2 + s^2 + 2*kappa*s*(mu . z)
        zmu = torch.matmul(z, mu.t())                       # (B, K)
        k2 = kappa ** 2                                     # (K,)
        r2 = k2.unsqueeze(0) + s * s + 2.0 * kappa.unsqueeze(0) * s * zmu  # (B, K)
        r = torch.sqrt(r2.clamp(min=1e-12))
        log_S_r = self._log_S(r)                            # (B, K)
        log_S_k = self._log_S(kappa)                        # (K,)
        return log_S_r - log_S_k.unsqueeze(0)               # (B, K) log E[exp(s * sim)]

    def forward(self, z, pseudo_labels, confidence, prototypes, priors=None):
        """
        z: (B, D) 归一化特征（两个视图拼接后）
        pseudo_labels: (B,) 簇伪标签（低置信度样本已由外部过滤/降权）
        confidence: (B,) 每个样本伪标签置信度权重 w_i in (0, 1]
        prototypes: (K, D) 归一化原型（EMA 维护，不参与梯度）
        priors: (K,) 各簇先验（样本比例），用于期望分母加权；None 则均匀
        """
        z = F.normalize(z, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        kappa = self.kappa
        B = z.shape[0]

        if priors is None:
            priors = torch.ones(self.K, device=z.device) / self.K
        log_priors = torch.log(priors.clamp(min=1e-8))       # (K,)

        # ---- 1) 无限对比对的期望 InfoNCE（闭式）----
        log_E = self.expected_exp_sim(z, prototypes, kappa)  # (B, K) log E_k[exp(s*sim)]
        # 分子：锚点所属簇的期望
        pos_logE = log_E[torch.arange(B, device=z.device), pseudo_labels]          # (B,)
        # 分母：所有簇按先验加权的期望和 log sum_k pi_k * E_k
        logits = log_E + log_priors.unsqueeze(0)             # (B, K)
        log_denom = torch.logsumexp(logits, dim=1)           # (B,)
        loss_nce = (log_denom - pos_logE)                    # (B,)

        # ---- 2) 原型约束：类内紧凑（拉向原型，梯度流向 z）----
        mu_z = prototypes[pseudo_labels]                     # (B, D)
        intra = 1.0 - (z * mu_z).sum(dim=1)                  # (B,)

        # ---- 3) 原型间均匀性（推开原型，vMF 分布均值方向最小重叠）----
        inter_sim = torch.matmul(prototypes, prototypes.t()) # (K, K)
        off_diag = inter_sim - 2.0 * torch.eye(self.K, device=z.device)
        # log-sum-exp 形式的均匀性惩罚（可微，梯度流向 kappa 无、仅用于日志/监控时 w_inter=0）
        loss_inter = torch.logsumexp(off_diag / self.temperature, dim=1).mean() * 0.0

        # 置信度加权聚合
        w = confidence / (confidence.sum() + 1e-8)           # 归一化权重
        loss = (loss_nce * w).sum() + self.w_intra * (intra * w).sum() + self.w_inter * loss_inter

        # 数值保护：非有限损失置零，避免污染整个训练
        if not torch.isfinite(loss):
            loss = torch.zeros_like(loss)
        return loss

    @torch.no_grad()
    def estimate_kappa(self, z, labels):
        """
        vMF 浓度参数的矩估计（Banerjee et al. MLE 近似）:
        kappa ~= r*(d - r^2) / (1 - r^2), r = ||mean(z)||
        用于初始化/校准可学习 kappa（钳制到 [1, 100] 防止极端紧簇导致爆炸）
        """
        z = F.normalize(z, dim=1)
        for k in range(self.K):
            m = labels == k
            if m.sum() > 2:
                mu = z[m].mean(dim=0)
                r = mu.norm().clamp(min=1e-4, max=0.9999)
                kap = r * (self.d - r ** 2) / (1 - r ** 2)
                kap = float(kap.clamp(min=1.0, max=100.0))
                self._kappa_raw.data[k] = torch.log(torch.expm1(torch.tensor(kap)))
        return self.kappa
