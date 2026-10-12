"""论文口径指标一次性计算：
1) 单模型 per-class Precision/Recall/F1 + Macro-F1（混淆矩阵口径，从落盘 labels 计算）
2) 集成（指定成员 ckpt 的 test logits 平均）同口径指标
3) 预训练表征聚类质量：h 空间（encoder 特征）与 z 空间（TC 投影）K-Means NMI/ARI（测试集）
   —— 用于对齐论文 Table 1/2 的 NMI/ARI 列，并验证论文 34.08/55.72 的来源口径
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_file import Config
from models.base_model import base_Model
from models.TC import TC
from dataloader.dataloader_notype import data_generator

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "Exp_SATA_V3")


def cm_metrics(t, p):
    n = int(max(t.max(), p.max())) + 1
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (t.astype(int), p.astype(int)), 1)
    acc = np.trace(cm) / cm.sum()
    rows = []
    for c in range(n):
        tp = cm[c, c]
        prec = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rec = tp / cm[c, :].sum() if cm[c, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append((prec, rec, f1))
    macro_f1 = float(np.mean([r[2] for r in rows]))
    macro_p = float(np.mean([r[0] for r in rows]))
    macro_r = float(np.mean([r[1] for r in rows]))
    return cm, acc, macro_p, macro_r, macro_f1, rows


def fmt(name, cm, acc, mp, mr, mf1, rows):
    s = "%-26s Acc=%.2f P=%.2f R=%.2f mF1=%.2f" % (name, acc * 100, mp * 100, mr * 100, mf1 * 100)
    for i, (p, r, f1) in enumerate(rows):
        s += " | C%d: P=%.2f R=%.2f F1=%.2f" % (i, p * 100, r * 100, f1 * 100)
    print(s)


def find_ckpt(run, ckpt="ckp_best.pt"):
    for d in sorted(glob.glob(os.path.join(ROOT, run, "*_seed_*", "saved_models"))):
        p = os.path.join(d, ckpt)
        if os.path.exists(p):
            return p
    return None


def collect_logits(model, loader, device):
    ls, ys = [], []
    model.eval()
    with torch.no_grad():
        for data, labels, *_ in loader:
            predictions, _ = model(data.float().to(device))
            ls.append(predictions.float().cpu().numpy())
            ys.append(labels.numpy())
    return np.concatenate(ls), np.concatenate(ys)


def cluster_eval(ckpt_path, device, tag):
    """聚类质量评估，对齐 cluster_validation.py 的论文口径：
    z = TC(normalize(base(x)), 同视图)，K-Means(k=2, random_state=42, n_init=10)
    同时报告 h 空间（encoder 特征时间维平均池化）作对照
    """
    configs = Config()
    base = base_Model(configs).to(device)
    tc = TC(configs, device).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # strict=False：原版 ckpt（Exp_21）没有 v2 新增的 recon_proj 键
    base.load_state_dict(ckpt["base_model_state_dict"], strict=False)
    tc.load_state_dict(ckpt["temporal_contr_model_state_dict"], strict=False)
    base.eval()
    tc.eval()
    loaders = data_generator("datasets", configs, "fine_tune", 42)
    torch.manual_seed(0)  # TC 内部随机采样 t_samples，固定种子保证可复现
    H, Z, Y = [], [], []
    with torch.no_grad():
        for data, labels, *_ in loaders["test"]:
            x = data.float().to(device)
            _, f = base(x)                       # (B, C, T) 卷积时序特征
            fn = F.normalize(f, dim=1)
            _, z = tc(fn, fn)                    # cluster_validation.py 口径：同视图
            H.append(f.mean(dim=2).cpu().numpy())  # h: 时间维平均池化 (B, C)
            Z.append(z.cpu().numpy())              # z: TC 投影 (B, C//2)
            Y.append(labels.numpy())
    H = np.concatenate(H)
    Z = np.concatenate(Z)
    Y = np.concatenate(Y)
    for space, F_mat in [("z(TC,同视图)", Z), ("h(时间池化)", H)]:
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(F_mat)
        nmi = normalized_mutual_info_score(Y, km.labels_)
        ari = adjusted_rand_score(Y, km.labels_)
        print("%-26s %s KMeans(k=2): NMI=%.4f ARI=%.4f" % (tag, space, nmi, ari))
    del base, tc, ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles", nargs="*", default=[])
    ap.add_argument("--ensemble", nargs="*", default=[])
    ap.add_argument("--cluster", nargs="*", default=[],
                    help="run 名（Exp_SATA_V3 下）或 ckpt 完整路径")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = Config()

    if args.singles:
        print("== 单模型（混淆矩阵口径 per-class P/R/F1；C0=道路, C1=田间）==")
        for run in args.singles:
            found = False
            for d in sorted(glob.glob(os.path.join(ROOT, run, "*_seed_*", "labels"))):
                t = np.load(os.path.join(d, "true_labels.npy"))
                p = np.load(os.path.join(d, "predicted_labels.npy"))
                cm, acc, mp, mr, mf1, rows = cm_metrics(t, p)
                fmt(run + "/" + os.path.basename(os.path.dirname(d)), cm, acc, mp, mr, mf1, rows)
                found = True
            if not found:
                print("%-26s no labels" % run)

    if args.ensemble:
        print("\n== 集成（test logits 平均；C0=道路, C1=田间）==")
        loaders = data_generator("datasets", configs, "fine_tune", 42)
        logits_list, ys = [], None
        members = []
        for run in args.ensemble:
            ck = find_ckpt(run)
            if not ck:
                print("%s: no ckpt, skip" % run)
                continue
            model = base_Model(configs).to(device)
            ckpt = torch.load(ck, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["base_model_state_dict"])
            tl, ys = collect_logits(model, loaders["test"], device)
            logits_list.append(tl)
            members.append(run)
            del model, ckpt
            if device.type == "cuda":
                torch.cuda.empty_cache()
        avg = np.stack(logits_list).mean(axis=0)
        pred = avg.argmax(1)
        cm, acc, mp, mr, mf1, rows = cm_metrics(ys, pred)
        fmt("ENS(" + "+".join(members) + ")", cm, acc, mp, mr, mf1, rows)

    if args.cluster:
        print("\n== 预训练表征聚类质量（测试集 K-Means, k=2）==")
        for item in args.cluster:
            if item.endswith(".pt"):
                ck = item
            else:
                ck = os.path.join(ROOT, item, "self_supervised_seed_42",
                                  "saved_models", "ckp_last.pt")
            if not os.path.exists(ck):
                print("%-26s ckpt 不存在: %s" % (item, ck))
                continue
            try:
                cluster_eval(ck, device, item)
            except Exception as e:
                print("%-26s cluster error: %s" % (item, e))


if __name__ == "__main__":
    main()
