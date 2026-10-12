# models/ssm.py
# 纯 PyTorch 实现的 S4D（对角状态空间模型，S4D-Legs 风格初始化）
# 参考: "On the Parameterization and Initialization of Diagonal State Space Models" (Gu et al., 2022)
# 及 MS4 (2026): S4D 在时序分类上精度与效率均优于 Mamba，且实现简单、Windows 兼容
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class S4DKernel(nn.Module):
    """
    对角 SSM 卷积核: k[n] = 2 * Re( c * lambda^(n-1) )
    lambda = exp(log_a) * exp(i * theta), log_a < 0 保证衰减（稳定）
    通过 FFT 实现长卷积的并行计算
    """

    def __init__(self, d_state=64, d_model=128):
        super().__init__()
        self.H = d_state
        # 可学习的实部（对数参数化，保证为负 => 稳定）
        # S4D-Legs 初始化: a = -n/H, n=1..H => 记忆尺度从 ~1 步到 ~H 步
        log_a = torch.log(torch.arange(1, d_state + 1).float() / d_state)
        self.log_a = nn.Parameter(log_a)  # (H,)

        # 虚部: 均匀分布在 [pi/2H, pi] 内（S4D-Legs 风格）
        theta = torch.arange(1, d_state + 1) * math.pi / d_state
        self.theta = nn.Parameter(theta)  # (H,)

        # 复数系数 c 的实部与虚部（可学习）
        self.c_re = nn.Parameter(torch.randn(d_state) / (d_state ** 0.5))
        self.c_im = nn.Parameter(torch.randn(d_state) / (d_state ** 0.5))

        # 输入投影: d_model -> H（复数通道的实输入）
        self.in_proj = nn.Linear(d_model, d_state)

    def forward(self, x, L):
        """
        x: (B, d_model, T) 输入序列
        L: 卷积核长度（= 序列长度）
        返回: (B, H, T) 状态响应（实部）
        """
        B, C, T = x.shape
        # 参数
        a = -torch.exp(self.log_a)          # (H,) 负实部
        theta = self.theta                   # (H,)
        lam = torch.complex(a * torch.cos(theta), a * torch.sin(theta))  # (H,)

        # 输入投影 (B, T, d_model) -> (B, T, H)
        u = self.in_proj(x.transpose(1, 2))  # (B, T, H)

        # 构造核 k[n] = 2*Re(c * lam^n), n = 0..L-1
        n = torch.arange(L, device=x.device, dtype=torch.float32)  # (L,)
        # lam^n = exp(n * (log|lam| + i*arg)) ；|lam| = exp(log_a)
        log_mag = self.log_a                    # (H,)
        # (L, H) 复指数
        pow_re = torch.exp(n[:, None] * log_mag[None, :]) * torch.cos(n[:, None] * theta[None, :])
        pow_im = torch.exp(n[:, None] * log_mag[None, :]) * torch.sin(n[:, None] * theta[None, :])
        # k[n] = 2 * Re( (c_re + i c_im) * (pow_re + i pow_im) )
        k_re = 2.0 * (self.c_re[None, :] * pow_re - self.c_im[None, :] * pow_im)  # (L, H)

        # FFT 卷积（因果）
        K = L + T - 1
        fft_len = 1 << (K - 1).bit_length()
        kf = torch.fft.rfft(k_re, n=fft_len, dim=0)                       # (fft_len/2+1, H)
        uf = torch.fft.rfft(u, n=fft_len, dim=1)                          # u: (B, T, H)
        yf = uf * kf.unsqueeze(0)                                         # (B, F, H)
        y = torch.fft.irfft(yf, n=fft_len, dim=1)[:, :T, :]               # (B, T, H) 因果
        return y.transpose(1, 2)                                          # (B, H, T)


class S4DBlock(nn.Module):
    """
    单层 S4D 块: 前置 LN + 线性投影 + S4D 卷积 + GLU 门控 + 残差
    输入/输出: (B, T, d_model)
    """

    def __init__(self, d_model=128, d_state=64, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S4DKernel(d_state=d_state, d_model=d_model)
        self.out_proj = nn.Linear(d_state, d_model * 2)  # GLU
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm(x)
        y = self.ssm(x.transpose(1, 2), L=x.shape[1])   # (B, H, T)
        y = y.transpose(1, 2)                            # (B, T, H)
        gate = self.out_proj(y)                          # (B, T, 2D)
        out, g = gate.chunk(2, dim=-1)
        out = out * torch.sigmoid(g)
        return residual + self.drop(out)


class CausalSSMEncoder(nn.Module):
    """
    因果 S4D 编码器（用于时间对比任务：只依赖过去信息）
    输入: (B, T, D) -> 输出: (B, T, hidden) 每个时间步的隐状态
    """

    def __init__(self, d_model=128, d_state=64, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            S4DBlock(d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: (B, T, D)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)  # (B, T, hidden)
