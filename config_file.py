"""配置各种参数"""


class Config(object):
    def __init__(self):
        # 模型配置
        self.input_channels = 28
        self.kernel_size = 9
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 2
        self.dropout = 0.5
        self.features_len = 80  # 时间步长

        # 训练配置
        self.num_epoch = 50

        # 优化器参数
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4  # Run_17 基线用 3e-4（1e-3 是后来提速改动，消融证明导致特征坍缩）

        # 数据清洗配置
        self.moving_average_num = 20
        self.scatter_distance_threshold = 1  # 单位：m

        # 时间窗口配置
        self.window_size = 80
        self.step = 5

        # 数据集配置
        self.drop_last = True
        self.batch_size = 1000  # 实验室 4080 16GB：Run_17 原生 batch（本地 4060 8GB 跑不动才降 600~850）
        self.use_amp = False    # 实验室用 fp32 完全复刻 Run_17（AMP 仅本地 8GB 显存妥协用）

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        self.PrototypeLoss = PrototypeLoss()

        # ===== v2 新增 =====
        # vMF 概率对比
        self.VMF = VMF_configs()
        # 掩码重建预训练
        self.Mask = Mask_configs()
        # 硬负样本 NTXent
        self.HardNeg = HardNeg_configs()
        # 损失权重
        self.lambda_temporal = 1.0     # 时间对比 NCE
        self.lambda_instance = 0.7     # 实例级 NTXent（硬负样本版）
        self.lambda_vmf = 0.0          # vMF 概率对比（Run_4 消融：关闭，先复现基线配方）
        self.lambda_recon = 0.5        # 掩码重建
        # 线性探测
        self.linear_class_weight = 'none'  # 'inverse_freq' | 'sqrt_inv' | 'none'（Run_17 基线无加权）
        self.linear_class_power = None    # 通用幂次 w=inv_freq^alpha（None=关闭；0=无权重, 0.5=sqrt, 1=inverse），优先于 linear_class_weight
        self.select_by = 'macro_f1'       # 最优模型选择标准: 'macro_f1' | 'bal_acc'（验证集平衡准确率）
        # 聚类调度
        self.cluster_interval = 5   # 对齐基线 Run_17 的 K=5
        self.warmup_epochs = 10     # 预热 epoch 数
        self.use_stratified_sampler = False  # PSS 分层采样（Run_17 基线未使用）


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.01
        self.jitter_ratio = 0.01
        self.max_seg = 5

        self.weak_features = [0, 1, 2, 3]
        self.weak_noise_std = 0.001
        self.strong_noise_std = 0.01
        self.scale_ratio = 0.01

        # v2: 时序专用增强
        # 注意：time_warp/mag_warp 会破坏轨迹的速度动力学语义（道路/田间的核心判别信号），
        # 消融确认开启后特征坍缩（Run_1 z-proj 聚类纯度仅 0.53），故默认关闭
        self.use_time_warp = False
        self.warp_sigma = 0.2
        self.use_mag_warp = False
        self.mag_sigma = 0.1
        self.use_permutation = False  # 默认关闭：轨迹时间顺序语义强，置换可能破坏语义


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 128
        self.timesteps = 48  # 时间步长的60%左右

        # v2: S4D 分支
        self.use_ssm = False     # True: S4D 分支；False: 原始 GRU 分支（Run_7 对照：分离编码器变量）
        self.ssm_state = 64      # SSM 状态维数
        self.ssm_layers = 2      # S4D 层数
        self.ssm_dropout = 0.1


class PrototypeLoss(object):
    def __init__(self):
        self.lambda_prototype = 0.0  # 原型损失权重（Run_4 对齐基线 Run_17：仅 CPSE，无原型损失）
        self.distance_type = 'cosine'  # 'softmax'（旧论文 Eq.4）| 'cosine' | 'mse'
        self.temperature = 0.2   # softmax 版温度 τ（与 CPSE 温度一致）


class VMF_configs(object):
    def __init__(self):
        self.temperature = 0.1   # 期望 InfoNCE 温度
        self.kappa_init = 10.0   # 初始浓度参数
        self.w_intra = 1.0       # 类内紧凑权重
        self.conf_tau = 0.1      # 伪标签置信度 softmax 温度
        self.conf_threshold = 0.5  # 置信度门槛：低于此值的样本不参与 vMF 损失
        self.uniform_prior = True  # 分母使用均匀簇先验（类均衡化）


class Mask_configs(object):
    def __init__(self):
        self.ratio = 0.25        # 掩码比例
        self.span = 8            # 连续掩码跨度（时间步）
        self.prob = 0.0          # 应用掩码的批次概率（Run_4 消融：关闭。注意掩码会破坏 aug2 的时间对比正样本目标）
        self.ema_momentum = 0.99 # EMA 教师动量


class HardNeg_configs(object):
    def __init__(self):
        self.top_k = 0           # <=0 使用标准 NTXent；>0 启用硬负样本挖掘（2 类不平衡下易放大假负样本，默认关闭）
