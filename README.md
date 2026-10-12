# Masked Probabilistic Contrastive Pre-training for Agricultural Machinery Trajectory Classification

**English | [中文](#中文说明)**

This repository contains the official source code for the paper *Masked Probabilistic Contrastive Pre-training for Class-Imbalanced Agricultural Machinery Trajectory Field–Road Classification*.

## Overview

We pre-train a TS-TCC encoder to classify GNSS trajectory windows of agricultural machinery into **in-field operation** and **on-road transfer**, without using any labels or any external trajectory corpus during pre-training. The framework adds three complementary objectives on top of the backbone:

1. **Span-masked reconstruction with an EMA teacher** — random temporal spans of the strongly augmented view are masked, and the encoder must reconstruct the EMA teacher's representation at the masked positions, forcing it to infer local kinematics at field–road transitions from context.
2. **Closed-form vMF probabilistic contrastive loss** — normalized embeddings are modeled as a per-cluster von Mises–Fisher mixture. The expected InfoNCE denominator under this mixture has a closed form, so the loss integrates infinitely many virtual negatives without large batches. A uniform cluster prior, learnable per-cluster concentrations, and confidence weighting counteract class imbalance without labels.
3. **Cluster-level positive sample expansion (CPSE)** — anchors whose pseudo-labels pass a high-quality filter recognize same-cluster high-quality members as positives, adding a sample-level intra-cluster pull.

See the overall framework figure in [docs/overall.pdf](docs/overall.pdf).

## Repository Structure

```
├── main.py                  # Entry point: self-supervised pre-training / linear probing / fine-tuning
├── trainer.py               # Training loops for all modes
├── config_file.py           # All hyper-parameters
├── classification.py        # Linear-probe / fine-tune classifier heads
├── clustering.py            # Periodic K-Means re-clustering, pseudo-labels, confidences
├── ensemble_eval.py         # Logit-averaging ensembles (greedy / fixed)
├── paper_metrics.py         # Per-class Precision/Recall/F1, mF1, accuracy from saved predictions
├── eval_cm.py               # Confusion-matrix evaluation utilities
├── predict.py               # Trajectory-level prediction and visualization
├── models/                  # TS-TCC encoder (TC head, attention, losses incl. vMF/CPSE/masked reconstruction)
├── dataloader/              # Weak/strong augmentation views and loaders
├── data_preprocessing/      # Feature extraction, cleaning, file-level splitting, scaling
├── baselines/               # Wrapper scripts for TS2Vec / FCACC baselines
└── docs/                    # Framework figure
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0 (developed and tested with 2.5.1, single RTX 4080)
- scikit-learn, numpy, pandas, matplotlib

```bash
pip install torch scikit-learn numpy pandas matplotlib
```

## Data Preparation

The Wheat-HQ benchmark (150 labelled wheat-harvester trajectories, 741,031 points) is public. The preprocessing pipeline in `data_preprocessing/` turns raw trajectories into windowed tensors (`T = 80`, stride 5, 28 kinematic features):

| Script | Purpose |
|---|---|
| `cal_feature.py` | Compute kinematic features (speed, acceleration, heading, turning rate + window statistics) |
| `clean_data.py` | Remove stop points, drift points, and outliers |
| `split_data_file.py` | File-level (trajectory-disjoint) train/val/test split |
| `scaler_data.py` | Min–max scaling to `[-1, 1]`, fitted on training files only |

After preprocessing, place `train.pt` / `val.pt` / `test.pt` in a data directory (default `datasets/`).

## Usage

All modes share `main.py`. Key arguments: `--training_mode` (`self_supervised` | `train_linear` | `fine_tune`), `--data_path`, `--seed`, `--lambda_vmf`, `--mask_prob`, `--linear_class_power`, `--select_by`.

**1. Self-supervised pre-training (full R19 recipe):**

```bash
python main.py --experiment_description Exp1 --run_description R19 \
  --training_mode self_supervised --seed 42 --data_path datasets \
  --lambda_vmf 0.4 --mask_prob 0.5
```

**2. Linear probing (frozen encoder):**

```bash
python main.py --experiment_description Exp1 --run_description R19_lin \
  --training_mode train_linear --seed 42 --data_path datasets \
  --pretrained_from logs/Exp1/R19/self_supervised_seed_42/saved_models/ckp_last.pt
```

**3. Class-balanced fine-tuning:**

```bash
python main.py --experiment_description Exp1 --run_description R19_ft \
  --training_mode fine_tune --seed 42 --data_path datasets \
  --pretrained_from logs/Exp1/R19/self_supervised_seed_42/saved_models/ckp_last.pt \
  --linear_class_power 0.25 --select_by bal_acc
```

**4. Ensembling and metrics:**

```bash
python ensemble_eval.py --runs Run_A Run_B Run_C --fixed Run_A Run_B Run_C
python paper_metrics.py --singles Run_A Run_B Run_C
```

**Baselines.** `baselines/run_ts2vec.py` and `baselines/run_fcacc.py` are the wrapper scripts used in the paper; they expect the official `ts2vec.py` / `fcacc.py` model files (available from the respective authors) to be placed alongside them.

## Main Results (Wheat-HQ, file-level protocol)

| Method | Protocol | Acc | mF1 | Road F1 | Field F1 |
|---|---|---|---|---|---|
| TS-TCC (backbone) | linear | 87.67 ± 0.40 | 73.82 | 54.78 | 92.86 |
| Ours (R19) | linear | **89.92 ± 0.19** | **79.25** | **64.37** | **94.13** |
| Ours (ensemble-18) | fine-tune | **91.74** | **83.89** | **72.63** | **95.14** |

See the paper for the full $2^3$ factor ablation, label-efficiency study, and cross-region validation (Wheat-LQ).

## Citation

If you find this code useful, please cite our paper (citation entry will be added upon acceptance).

---

# 中文说明

本仓库为论文《面向类别不平衡农业机械轨迹田间–道路分类的掩码概率对比预训练》的官方源码。

## 概述

我们在不使用任何标签、不依赖任何外部轨迹语料的条件下，对 TS-TCC 编码器进行自监督预训练，用于农机 GNSS 轨迹窗口的**田间作业**与**道路转移**二分类。框架在骨干之上引入三个互补目标：

1. **带 EMA 教师的跨度掩码重建**：对强增强视图施加随机时间跨度掩码，编码器须在掩码位置重建 EMA 教师的表征，迫使其从上下文推断田间–道路过渡处的局部运动学特征；
2. **闭式 vMF 概率对比损失**：将归一化嵌入建模为逐簇 von Mises–Fisher 混合，InfoNCE 分母在该混合下的期望具有闭式解，无需大 batch 即可积分无穷多虚拟负样本；均匀簇先验、可学习浓度与置信度加权在无标签条件下抵消类别不平衡；
3. **簇级正样本扩展（CPSE）**：伪标签通过高质量过滤的锚点将同簇高质量成员识别为正样本，提供样本级簇内凝聚拉力。

## 环境依赖

- Python >= 3.10
- PyTorch >= 2.0（开发环境 2.5.1，单卡 RTX 4080）
- scikit-learn、numpy、pandas、matplotlib

```bash
pip install torch scikit-learn numpy pandas matplotlib
```

## 数据准备

Wheat-HQ 基准（150 条有标注小麦收割机轨迹、741,031 点）为公开数据集。`data_preprocessing/` 中的流水线将原始轨迹转换为窗口张量（`T = 80`、步长 5、28 维运动学特征）：特征计算（`cal_feature.py`）→ 数据清洗（`clean_data.py`）→ 文件级划分（`split_data_file.py`）→ 归一化（`scaler_data.py`，仅在训练文件上拟合）。预处理完成后，将 `train.pt` / `val.pt` / `test.pt` 放入数据目录（默认 `datasets/`）。

## 使用方法

所有模式共用 `main.py`，核心参数：`--training_mode`（`self_supervised` / `train_linear` / `fine_tune`）、`--data_path`、`--seed`、`--lambda_vmf`、`--mask_prob`、`--linear_class_power`、`--select_by`。

**1. 自监督预训练（完整 R19 配方）：**

```bash
python main.py --experiment_description Exp1 --run_description R19 \
  --training_mode self_supervised --seed 42 --data_path datasets \
  --lambda_vmf 0.4 --mask_prob 0.5
```

**2. 线性探测（冻结编码器）：**

```bash
python main.py --experiment_description Exp1 --run_description R19_lin \
  --training_mode train_linear --seed 42 --data_path datasets \
  --pretrained_from logs/Exp1/R19/self_supervised_seed_42/saved_models/ckp_last.pt
```

**3. 类平衡微调：**

```bash
python main.py --experiment_description Exp1 --run_description R19_ft \
  --training_mode fine_tune --seed 42 --data_path datasets \
  --pretrained_from logs/Exp1/R19/self_supervised_seed_42/saved_models/ckp_last.pt \
  --linear_class_power 0.25 --select_by bal_acc
```

**4. 集成与指标：**

```bash
python ensemble_eval.py --runs Run_A Run_B Run_C --fixed Run_A Run_B Run_C
python paper_metrics.py --singles Run_A Run_B Run_C
```

**基线。** `baselines/run_ts2vec.py` 与 `baselines/run_fcacc.py` 为论文所用封装脚本，需将官方 `ts2vec.py` / `fcacc.py` 模型文件放置在同目录下运行。

## 主要结果（Wheat-HQ、文件级协议）

| 方法 | 协议 | Acc | mF1 | 道路 F1 | 田 F1 |
|---|---|---|---|---|---|
| TS-TCC（骨干） | 线性 | 87.67 ± 0.40 | 73.82 | 54.78 | 92.86 |
| 本文（R19） | 线性 | **89.92 ± 0.19** | **79.25** | **64.37** | **94.13** |
| 本文（集成-18） | 微调 | **91.74** | **83.89** | **72.63** | **95.14** |

完整 $2^3$ 因子消融、标签效率研究与跨区域验证（Wheat-LQ）见论文正文。

## 引用

如本代码对您有帮助，请引用我们的论文（录用后补充引用条目）。
