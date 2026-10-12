"""计算实验目录下 labels/*.npy 的混淆矩阵口径指标：acc / 道路召回 / 田间召回
用法: python eval_cm.py Run_19FTF Run_19FTS ...
  或: python eval_cm.py --all  (扫描 logs/Exp_SATA_V3 下所有含 labels 的目录)
"""
import os
import sys
import glob
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "Exp_SATA_V3")


def eval_one(labels_dir):
    t = np.load(os.path.join(labels_dir, "true_labels.npy"))
    p = np.load(os.path.join(labels_dir, "predicted_labels.npy"))
    n_cls = int(max(t.max(), p.max())) + 1
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    np.add.at(cm, (t.astype(int), p.astype(int)), 1)
    acc = np.trace(cm) / cm.sum()
    recalls = [cm[i, i] / cm[i].sum() for i in range(n_cls)]
    return cm, acc, recalls


def fmt(run, labels_dir):
    cm, acc, recalls = eval_one(labels_dir)
    rec = " ".join("r%d=%.4f" % (i, r) for i, r in enumerate(recalls))
    return "%-14s acc=%.4f  %s  cm=%s" % (run, acc, rec, cm.tolist())


def find_labels_dir(run_dir):
    # 自动扫描任意 seed 的模式目录（fine_tune_seed_0 等）
    import glob as _g
    for d in sorted(_g.glob(os.path.join(run_dir, "*_seed_*", "labels"))):
        return d
    for sub in ["fine_tune_seed_42", "train_linear_seed_42", "self_supervised_seed_42", ""]:
        d = os.path.join(run_dir, sub, "labels") if sub else os.path.join(run_dir, "labels")
        if os.path.isdir(d):
            return d
    return None


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--all":
        for run_dir in sorted(glob.glob(os.path.join(ROOT, "Run_*"))):
            d = find_labels_dir(run_dir)
            if d:
                print(fmt(os.path.basename(run_dir), d))
    else:
        for run in args:
            run_dir = os.path.join(ROOT, run)
            d = find_labels_dir(run_dir)
            if d:
                print(fmt(run, d))
            else:
                print("%-14s no labels dir" % run)


if __name__ == "__main__":
    main()
