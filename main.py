import argparse
import torch
import os
import numpy as np
from datetime import datetime
from config_file import Config
from utils import _logger, set_requires_grad
from dataloader.dataloader_notype import data_generator
from trainer import Trainer, model_evaluate # 假设 Trainer 函数会被修改
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.base_model import base_Model

torch.serialization.add_safe_globals([Config])

# --- 辅助函数：获取类别分布 ---
def get_class_distribution(data_path):
    # 直接加载训练集文件
    train_data = torch.load(os.path.join(data_path, "train.pt"), weights_only=True)

    # v2: 使用真实标签计算类别分布（而非 cluster_tag）
    all_labels = train_data["labels"]

    # 转换为numpy数组（如果是张量）
    if isinstance(all_labels, torch.Tensor):
        all_labels = all_labels.numpy() if all_labels.is_cpu else all_labels.cpu().numpy()

    # 统计类别分布
    class_counts = np.bincount(all_labels)  # array([n_class0, n_class1])
    # v2: 逆频率类权重（应对不平衡），归一化到均值为 1
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.mean()

    return class_counts, class_weights

start_time = datetime.now()

parser = argparse.ArgumentParser()

home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp_1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='Run_1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='train_linear', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--logs_save_dir', default='logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
# --- 新增参数：继续训练功能 ---
parser.add_argument('--resume_training', action='store_true',
                    help='Set to true to resume training from a checkpoint.')
parser.add_argument('--resume_path', default=None, type=str,
                    help='Path to the checkpoint to resume training from. '
                         'If --resume_training is true and this is None, it will look for "ckp_last.pt" '
                         'in the current experiment\'s saved_models directory.')
# -----------------------------
parser.add_argument('--epochs', default=None, type=int,
                    help='Override num_epoch in config (for quick smoke tests).')
parser.add_argument('--warmup', default=None, type=int,
                    help='Override warmup_epochs.')
parser.add_argument('--cluster_interval', default=None, type=int,
                    help='Override cluster interval K.')
parser.add_argument('--use_ssm', default=None, type=int,
                    help='Override TC temporal branch: 1=S4D, 0=GRU (baseline).')
parser.add_argument('--lambda_vmf', default=None, type=float,
                    help='Override vMF contrastive loss weight (0=off).')
parser.add_argument('--mask_prob', default=None, type=float,
                    help='Override mask augmentation probability (0=off).')
parser.add_argument('--linear_class_weight', default=None, type=str,
                    help="Override linear probe CE weighting: 'inverse_freq' or 'none'.")
parser.add_argument('--linear_class_power', default=None, type=float,
                    help='CE weight = inv_freq^alpha (0=none, 0.5=sqrt, 1=inverse). Overrides --linear_class_weight.')
parser.add_argument('--select_by', default=None, type=str,
                    help="Best-model selection criterion: 'macro_f1' (default) or 'bal_acc'.")
parser.add_argument('--train_fraction', default=None, type=float,
                    help='Label-efficiency: stratified fraction of training set (e.g. 0.05).')
parser.add_argument('--pretrained_from', default=None, type=str,
                    help='Override pretrained checkpoint path for fine_tune/train_linear.')
parser.add_argument('--train_batch_size', default=None, type=int,
                    help='Override train loader batch size (for small subsets).')
parser.add_argument('--data_path', default='datasets', type=str,
                    help='Dataset directory containing train/val/test.pt (e.g. datasets_js).')
parser.add_argument('--lambda_cluster_scale', default=None, type=float,
                    help='Scale factor for CPSE cluster loss (0 = pure TS-TCC baseline).')
parser.add_argument('--lambda_proto', default=None, type=float,
                    help='Override PGA prototype alignment loss weight (0=off).')
parser.add_argument('--proto_type', default=None, type=str,
                    help="PGA distance type: 'softmax' (paper Eq.4) or 'cosine'.")
parser.add_argument('--use_pss', default=None, type=int,
                    help='1 = PSS stratified sampler after clustering (0 = random).')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

configs = Config()
if args.epochs is not None:
    configs.num_epoch = args.epochs
if args.warmup is not None:
    configs.warmup_epochs = args.warmup
if args.cluster_interval is not None:
    configs.cluster_interval = args.cluster_interval
if args.use_ssm is not None:
    configs.TC.use_ssm = bool(args.use_ssm)
if args.lambda_vmf is not None:
    configs.lambda_vmf = args.lambda_vmf
if args.mask_prob is not None:
    configs.Mask.prob = args.mask_prob
if args.linear_class_weight is not None:
    configs.linear_class_weight = args.linear_class_weight
if args.linear_class_power is not None:
    configs.linear_class_power = args.linear_class_power
if args.select_by is not None:
    configs.select_by = args.select_by
if args.lambda_cluster_scale is not None:
    configs.lambda_cluster_scale = args.lambda_cluster_scale
if args.lambda_proto is not None:
    configs.PrototypeLoss.lambda_prototype = args.lambda_proto
if args.proto_type is not None:
    configs.PrototypeLoss.distance_type = args.proto_type
if args.use_pss is not None:
    configs.use_stratified_sampler = bool(args.use_pss)

SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

experiment_log_dir = os.path.join(logs_save_dir, experiment_description,
                                  run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

counter = 0
src_counter = 0

log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# 加载数据
data_path = args.data_path
class_counts, class_weights = get_class_distribution(data_path)
logger.debug(f"类别权重（道路点：田间点）{class_weights}")
data_loaders = data_generator(data_path, configs, training_mode, SEED)
train_ds = data_loaders['train_dataset']
train_dl = data_loaders['train']
valid_dl = data_loaders['valid']
test_dl = data_loaders['test']
cluster_dl = data_loaders['cluster']
# file_count = data_loaders['file_count']

# --- 标签效率实验：分层子采样训练集（val/test 不变，两臂共用同一子集划分） ---
if args.train_fraction is not None and 0 < args.train_fraction < 1.0:
    from torch.utils.data import Subset, DataLoader
    labels_np = train_ds.labels
    if isinstance(labels_np, torch.Tensor):
        labels_np = labels_np.numpy()
    labels_np = np.asarray(labels_np)
    rng = np.random.RandomState(SEED)
    keep = []
    for c in np.unique(labels_np):
        idx_c = np.where(labels_np == c)[0]
        n_keep = max(1, int(round(len(idx_c) * args.train_fraction)))
        keep.extend(rng.choice(idx_c, size=n_keep, replace=False).tolist())
    keep = sorted(keep)
    logger.debug(f"Label-efficiency subset: {len(keep)}/{len(labels_np)} "
                 f"(fraction={args.train_fraction}, seed={SEED})")
    sub_bs = args.train_batch_size if args.train_batch_size else configs.batch_size
    train_dl = DataLoader(Subset(train_ds, keep), batch_size=sub_bs, shuffle=True,
                          num_workers=0, drop_last=False)
elif args.train_batch_size is not None:
    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True,
                          num_workers=0, drop_last=False)

logger.debug("Data loaded ...")

# 模型初始化
base_model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

# --- 预训练模型加载逻辑  ---
if training_mode == "fine_tune":
    if args.pretrained_from:
        ckpt_path = args.pretrained_from
    else:
        ckpt_path = os.path.join(logs_save_dir, experiment_description, run_description,
                                 f"self_supervised_seed_{SEED}", "saved_models", "ckp_last.pt")
    chkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    pretrained_dict = chkpoint["base_model_state_dict"]
    model_dict = base_model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    base_model.load_state_dict(model_dict)

if training_mode == "train_linear" or "tl" in training_mode:
    if args.pretrained_from:
        ckpt_path = args.pretrained_from
    else:
        ckpt_path = os.path.join(logs_save_dir, experiment_description, run_description,
                                 f"self_supervised_seed_{SEED}", "saved_models", "ckp_last.pt")
    chkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    pretrained_dict = chkpoint["base_model_state_dict"]
    model_dict = base_model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    base_model.load_state_dict(model_dict)
    # 预训练的权重冻结起来，把logits层删除，重新训练
    set_requires_grad(base_model, pretrained_dict, requires_grad=False)

if training_mode == "random_init":
    model_dict = base_model.state_dict()

    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(base_model, model_dict, requires_grad=False)

# 初始化优化器
model_optimizer = torch.optim.Adam(base_model.parameters(), lr=configs.lr,
                                   betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr,
                                            betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# --- 继续训练逻辑 ---
start_epoch = 1 # 默认从第1个epoch开始训练
scheduler_loaded_state = None
if args.resume_training:
    resume_checkpoint_path = args.resume_path
    if resume_checkpoint_path is None:
        # 如果没有指定路径，尝试从默认保存目录加载
        resume_checkpoint_path = os.path.join(experiment_log_dir, "saved_models", "ckp_last.pt")

    if os.path.exists(resume_checkpoint_path):
        logger.debug(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)

        # 加载模型状态
        base_model.load_state_dict(checkpoint['base_model_state_dict'])
        temporal_contr_model.load_state_dict(checkpoint['temporal_contr_model_state_dict'])

        # 加载优化器状态
        model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
        temporal_contr_optimizer.load_state_dict(checkpoint['temporal_contr_optimizer_state_dict'])

        scheduler_loaded_state = checkpoint.get('scheduler_state_dict', None) # 获取调度器状态

        # 设置起始 epoch
        start_epoch = checkpoint['epoch'] + 1 # 从上次的下一个epoch开始

        logger.debug(f"Resumed from epoch {checkpoint['epoch']}, continuing from epoch {start_epoch}")

        # 释放检查点副本（避免整个训练期间多占 ~1.5GB 显存引发 WDDM 分页颠簸）
        del checkpoint
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        logger.warning(f"Resume training requested, but checkpoint not found at {resume_checkpoint_path}. Starting from scratch.")
        # 如果找不到检查点，则仍然从头开始
        start_epoch = 1
# ---------------------

if training_mode == "self_supervised":
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description))

# 调用 Trainer 函数，传入 start_epoch
# Trainer(base_model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl,
#         file_count, device, logger, configs, experiment_log_dir, training_mode, class_weights,
#         start_epoch=start_epoch, scheduler_state_dict=scheduler_loaded_state)
Trainer(base_model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, cluster_dl, train_ds,
        device, logger, configs, experiment_log_dir, training_mode, class_weights,
        start_epoch=start_epoch, scheduler_state_dict=scheduler_loaded_state)

if training_mode != "self_supervised":
    outs = model_evaluate(base_model, temporal_contr_model, test_dl, device, training_mode, configs, class_weights)
    total_loss, total_acc, macro_f1, pred_labels, true_labels = outs
    logger.debug(f"Test Macro-F1 : {macro_f1:.4f}")
    _calc_metrics(pred_labels, true_labels, experiment_log_dir)

logger.debug(f"Training time is : {datetime.now()-start_time}")
