import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer
from .ssm import CausalSSMEncoder


class TC(nn.Module):
    """
    v2: 双分支时间对比模块
      - Transformer 分支（全局上下文建模，保留原 TS-TCC 设计）
      - 时序分支：use_ssm=True 时为因果 S4D（线性复杂度长程依赖），否则为原始 GRU（基线）
    """

    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.hidden_dim = configs.TC.hidden_dim
        self.device = device
        self.activation_func = nn.GELU()
        self.use_ssm = getattr(configs.TC, 'use_ssm', True)

        # --- 双分支编码器 ---
        # 1. Transformer 分支（全局建模）
        self.transformer = Seq_Transformer(
            patch_size=self.num_channels,
            dim=self.hidden_dim,
            depth=4,
            heads=4,
            mlp_dim=64
        )

        # 2. 时序分支
        if self.use_ssm:
            self.ssm_encoder = CausalSSMEncoder(
                d_model=self.num_channels,
                d_state=getattr(configs.TC, 'ssm_state', 64),
                hidden_dim=self.hidden_dim,
                n_layers=getattr(configs.TC, 'ssm_layers', 2),
                dropout=getattr(configs.TC, 'ssm_dropout', 0.1),
            )
        else:
            # 原始 GRU 分支（基线 Run_17 结构，2 层单向）
            self.gru = nn.GRU(
                input_size=self.num_channels,
                hidden_size=self.hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=False
            )

        # --- 特征融合层 ---
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # 合并双分支特征
            nn.Tanh(),
            nn.LayerNorm(self.hidden_dim)
        )

        # --- 预测未来时间步的线性层 ---
        self.Wk = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.num_channels)
            for _ in range(self.timestep)
        ])

        # --- 对比学习损失函数 ---
        self.lsoftmax = nn.LogSoftmax(dim=1)

        # --- 投影头 ---
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),                    # 轨迹最佳
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_channels // 2)
        )

    def encode(self, features_aug1):
        """
        编码单视图特征 -> 上下文向量 c_t
        features_aug1: (B, C, T) 卷积特征
        """
        z_aug1 = features_aug1.transpose(1, 2)  # [B, T, C]
        seq_len = z_aug1.shape[1]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).item()

        # 1. Transformer 分支（全局）
        c_transformer = self.transformer(z_aug1)  # [B, hidden]

        # 2. 时序分支
        if self.use_ssm:
            ssm_seq = self.ssm_encoder(z_aug1)          # [B, T, hidden]
            c_seq = ssm_seq[:, t_samples, :]            # [B, hidden]
        else:
            forward_seq = z_aug1[:, :t_samples + 1, :]
            _, h_n = self.gru(forward_seq)
            c_seq = h_n[-1]                             # [B, hidden]

        # --- 特征融合 ---
        c_combined = torch.cat([c_transformer, c_seq], dim=1)  # [B, 2*hidden]
        c_t = self.fusion(c_combined)                          # [B, hidden]
        return c_t, t_samples

    def forward(self, features_aug1, features_aug2):
        # 输入维度转换: (batch, channels, seq_len) -> (batch, seq_len, channels)
        z_aug1 = features_aug1.transpose(1, 2)  # [B, T, C]
        z_aug2 = features_aug2.transpose(1, 2)
        batch_size, seq_len, _ = z_aug1.shape

        # 随机采样时间点（对比学习正样本）
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).item()

        # --- 双分支编码 ---
        # 1. Transformer 分支（全局）
        c_transformer = self.transformer(z_aug1)  # [B, hidden]

        # 2. 时序分支：取 t_samples 时刻（严格因果，只看过去）
        if self.use_ssm:
            ssm_seq = self.ssm_encoder(z_aug1)          # [B, T, hidden]
            c_seq = ssm_seq[:, t_samples, :]            # [B, hidden]
        else:
            forward_seq = z_aug1[:, :t_samples + 1, :]
            _, h_n = self.gru(forward_seq)
            c_seq = h_n[-1]                             # [B, hidden]

        # --- 特征融合 ---
        c_combined = torch.cat([c_transformer, c_seq], dim=1)  # [B, 2*hidden]
        c_t = self.fusion(c_combined)                          # [B, hidden]

        # --- 对比学习损失计算 ---
        # 从aug2中提取未来timestep个正样本
        encode_samples = torch.empty((self.timestep, batch_size, self.num_channels)).float().to(self.device)
        pred = torch.empty((self.timestep, batch_size, self.num_channels)).float().to(self.device)
        for i in range(self.timestep):
            encode_samples[i] = z_aug2[:, t_samples + i + 1, :]  # [B, C]
            pred[i] = self.Wk[i](c_t)  # 线性预测 [B, C]

        # 批量计算对比损失（NCE）
        total = torch.bmm(encode_samples, pred.transpose(1, 2))  # [timestep, B, B]
        nce = torch.sum(torch.diagonal(self.lsoftmax(total), dim1=1, dim2=2))
        nce /= -1. * batch_size * self.timestep

        # 投影特征（用于下游任务与 vMF 概率对比）
        features = self.projection_head(c_t)  # [B, C//2]

        return nce, features
