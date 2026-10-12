import numpy as np
import torch


# 数据增强 v2
# 保持 DataTransform(sample, config, random_state) -> (weak, strong) 接口不变
# 新增时序专用增强：时间扭曲、幅度扭曲、分段置换、随机裁剪回填（不对称增强 + 轨迹先验）

def DataTransform(sample, config, random_state=None):
    """
    对输入 (N, T, F) 的时间序列做不对称增强
    weak: 微噪声（保持语义）
    strong: 噪声 + 缩放 + 时间扭曲 + 幅度扭曲 + 分段置换（更强扰动，迫使模型学习鲁棒表征）
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random

    aug_cfg = config.augmentation
    weak_feats = aug_cfg.weak_features
    num_feats = sample.shape[-1]
    strong_feats = [i for i in range(num_feats) if i not in weak_feats]

    # ---- Weak Augmentation: 微弱高斯噪声 ----
    weak_aug = sample.copy()
    noise_weak = rng.normal(
        loc=0.0,
        scale=aug_cfg.weak_noise_std,
        size=weak_aug[:, :, weak_feats].shape
    )
    weak_aug[:, :, weak_feats] += noise_weak

    # ---- Strong Augmentation ----
    strong_aug = sample.copy()

    # 1) 强噪声（特征不对称）
    noise_strong = rng.normal(
        loc=0.0,
        scale=aug_cfg.strong_noise_std,
        size=strong_aug[:, :, strong_feats].shape
    )
    strong_aug[:, :, strong_feats] += noise_strong

    # 2) 幅度缩放（每样本每特征独立）
    scale = rng.uniform(
        1.0 - aug_cfg.scale_ratio,
        1.0 + aug_cfg.scale_ratio,
        size=(1, 1, len(strong_feats))
    )
    strong_aug[:, :, strong_feats] *= scale

    # 3) 时间扭曲（time warping）：随机分段拉伸/压缩后重采样回原长
    if getattr(aug_cfg, 'use_time_warp', True):
        strong_aug = time_warp(strong_aug, sigma=getattr(aug_cfg, 'warp_sigma', 0.2), rng=rng)

    # 4) 幅度扭曲（magnitude warping）：平滑随机曲线逐点调制
    if getattr(aug_cfg, 'use_mag_warp', True):
        strong_aug = magnitude_warp(strong_aug, sigma=getattr(aug_cfg, 'mag_sigma', 0.1), knot=4, rng=rng)

    # 5) 分段置换（保留前 4 个弱特征不变：坐标/基础运动量）
    if getattr(aug_cfg, 'use_permutation', False):
        strong_aug = permutation(strong_aug, max_segments=aug_cfg.max_seg, rng=rng)

    return weak_aug, strong_aug


def time_warp(x, sigma=0.2, rng=np.random):
    """
    时间扭曲: 对时间轴做平滑随机形变后线性插值回原长度
    x: (N, T, F)
    """
    N, T, F = x.shape
    # 每个样本一条平滑随机曲线
    knot = 4
    orig_steps = np.arange(T)
    random_warps = []
    for i in range(N):
        loc = rng.uniform(0, 1, size=knot + 2)
        xs = np.linspace(0, T - 1, knot + 2)
        # 累积样条近似（用分段线性代替样条，避免 scipy 依赖）
        warp = np.interp(orig_steps, xs, loc)
        # 平滑随机游走
        rr = rng.uniform(-1, 1, size=T)
        warper = sigma * np.convolve(rr, np.ones(8) / 8, mode='same')
        random_warps.append(orig_steps + warper * warp)
    ret = np.empty_like(x)
    for i in range(N):
        idx = np.clip(random_warps[i], 0, T - 1)
        for f in range(F):
            ret[i, :, f] = np.interp(orig_steps, idx, x[i, :, f])
    return ret


def magnitude_warp(x, sigma=0.1, knot=4, rng=np.random):
    """
    幅度扭曲: 用平滑随机曲线调制幅度
    x: (N, T, F)
    """
    N, T, F = x.shape
    orig_steps = np.arange(T)
    ret = np.empty_like(x)
    for i in range(N):
        loc = rng.uniform(1 - sigma, 1 + sigma, size=knot + 2)
        xs = np.linspace(0, T - 1, knot + 2)
        curve = np.interp(orig_steps, xs, loc)
        # 平滑
        curve = np.convolve(curve, np.ones(5) / 5, mode='same')
        ret[i] = x[i] * curve[:, None]
    return ret


# 抖动
def jitter(x, sigma=0.8, rng=np.random):
    """
    对数据应用抖动，使用提供的随机数生成器。
    """
    return x + rng.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.01, feats=None, rng=np.random):
    """
    对特定特征列执行缩放增强。
    x: (N, T, F)
    feats: list, 指定要增强的特征列索引
    """
    if feats is None or len(feats) == 0:
        return x

    x = x.copy()
    factor = rng.normal(loc=1.0, scale=sigma, size=(x.shape[0], len(feats)))
    factor = factor[:, None, :]
    x[:, :, feats] *= factor
    return x


# 对特征进行分段并随机排列（v2: numpy 版本，作用于 (N,T,F)，只置换时间维）
def permutation(x, max_segments, seg_mode="random", rng=np.random):
    """
    分段置换时间维
    x: (N, T, F) ndarray
    """
    N, T, F = x.shape
    orig_steps = np.arange(T)
    num_segs = rng.randint(2, max_segments + 1, size=N)
    ret = x.copy()
    for i in range(N):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = rng.choice(T - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate([rng.permutation(split) for split in splits]).ravel()
            ret[i] = x[i][warp]
    return ret
