# dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import os
import numpy as np
import random
from typing import Iterator, List
from .augmentations import DataTransform


# ===============================
# Dataset：按类别存储1111111111111
# ===============================
class UnlabeledWindowDataset(Dataset):
    def __init__(self, data_path, config, training_mode, seed):
        super().__init__()
        self.training_mode = training_mode
        self.config = config
        self.seed = seed

        data = torch.load(data_path, weights_only=False)

        self.samples = data["samples"]          # [N, T, C]
        self.labels = data.get("labels", None)  # [N] or None
        self.cluster_tag = data.get("cluster_tag", None)  # [N] or None
        self.center_idx = data["center_idx"]    # [N]
        self.file_id = data["file_id"]           # [N]

        if training_mode == "self_supervised":
            self._init_augmentations()

    def _init_augmentations(self):
        self.aug1, self.aug2 = DataTransform(
            self.samples.numpy(),
            self.config,
            random_state=self.seed
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]

        # ---- self-supervised ----
        if self.training_mode == "self_supervised":
            label = -1 if self.labels is None else self.labels[idx]
            cluster = -1 if self.cluster_tag is None else self.cluster_tag[idx]

            return (
                x,                      # data
                label,                  # dummy label
                cluster,                # dummy cluster_tag
                self.aug1[idx],         # aug1
                self.aug2[idx],         # aug2
                self.center_idx[idx],
                self.file_id[idx],
                idx
            )

        # ---- supervised / linear probe ----
        else:
            return (
                x,                      # data
                self.labels[idx],       # true label
                self.cluster_tag[idx] if self.cluster_tag is not None else -1,
                x,                      # aug1 (占位)
                x,                      # aug2 (占位)
                self.center_idx[idx],
                self.file_id[idx],
                idx
            )


# ===============================
# Stratified Sampler
# ===============================
class StratifiedSampler(Sampler):
    """
    基于 cluster_tag 的动态分层采样
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # ========= warmup阶段 =========
        if self.dataset.cluster_tag is None:
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)
            return iter(indices)

        # ========= 分层采样 =========
        cluster_tags = np.array(self.dataset.cluster_tag)
        unique_clusters = np.unique(cluster_tags)
        print(f"unique_clusters: {unique_clusters}")

        cluster_indices = {}
        for c in unique_clusters:
            idxs = np.where(cluster_tags == c)[0].tolist()
            random.shuffle(idxs)
            cluster_indices[c] = idxs
        final_indices = []
        # 每个簇取多少
        samples_per_cluster = max(
            1,
            self.batch_size // len(unique_clusters)
        )
        pointers = {
            c: 0 for c in unique_clusters
        }
        while True:
            batch = []
            for c in unique_clusters:
                idxs = cluster_indices[c]
                # ========= oversampling =========
                if pointers[c] + samples_per_cluster >= len(idxs):
                    random.shuffle(idxs)
                    pointers[c] = 0
                selected = idxs[
                    pointers[c]:
                    pointers[c] + samples_per_cluster
                ]
                batch.extend(selected)
                pointers[c] += samples_per_cluster

            if len(batch) == 0:
                break
            random.shuffle(batch)
            final_indices.extend(batch)
            # 防止无限循环
            if len(final_indices) >= len(self.dataset):
                break
        return iter(final_indices[:len(self.dataset)])
    def __len__(self):
        return len(self.dataset)


# ===============================
# DataLoader 生成器
# ===============================
def data_generator(data_path, configs, training_mode, seed):

    train_ds = UnlabeledWindowDataset(
        os.path.join(data_path, "train.pt"),
        configs,
        training_mode,
        seed,
    )

    val_ds = UnlabeledWindowDataset(
        os.path.join(data_path, "val.pt"),
        configs,
        training_mode,
        seed,
    )

    test_ds = UnlabeledWindowDataset(
        os.path.join(data_path, "test.pt"),
        configs,
        training_mode,
        seed,
    )

    # train_sampler = StratifiedSampler(
    # train_ds,
    # configs.batch_size
    # )

    train_loader = DataLoader(
        train_ds,
        batch_size=configs.batch_size,
        # sampler=train_sampler,
        shuffle=True,   # Run_17 原版行为（探针v2 bit-exact 验证；False 会导致 batch 内负样本过相似，掉 ~3 个点）
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=0,
    )

    cluster_loader = DataLoader(
        train_ds,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return {
        "train_dataset": train_ds,
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
        "cluster": cluster_loader,
    }
