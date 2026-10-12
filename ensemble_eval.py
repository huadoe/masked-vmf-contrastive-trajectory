"""模型集成评估：多个微调 checkpoint 的 logits 平均。
在验证集上贪心选择集成组合（目标：val bal_acc 最大化），
测试集上报告最终指标。合法协议：组合选择只用 val，不碰 test。

用法:
  python ensemble_eval.py --runs Run_31 Run_32 Run_41 Run_51 ...   # 贪心搜索
  python ensemble_eval.py --runs Run_43 --fixed Run_43 Run_49 Run_50  # 指定组合
"""
import os
import sys
import argparse
import itertools
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_file import Config
from models.base_model import base_Model
from dataloader.dataloader_notype import data_generator

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "Exp_SATA_V3")


def find_ckpt(run):
    """找 run 目录下的微调 best ckpt（任意 seed）"""
    import glob
    for d in sorted(glob.glob(os.path.join(ROOT, run, "*_seed_*", "saved_models"))):
        p = os.path.join(d, "ckp_best.pt")
        if os.path.exists(p):
            return p
    return None


def collect_logits(model, loader, device):
    logits_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for data, labels, *_ in loader:
            data = data.float().to(device)
            predictions, features = model(data)
            logits_list.append(predictions.float().cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(logits_list), np.concatenate(labels_list)


def metrics(logits, labels):
    pred = logits.argmax(1)
    acc = (pred == labels).mean()
    r0 = (pred[labels == 0] == 0).mean()
    r1 = (pred[labels == 1] == 1).mean()
    return acc, r0, r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--fixed', nargs='*', default=None, help='固定组合（跳过贪心）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_k', type=int, default=5)
    ap.add_argument('--greedy_target', default='bal_acc', choices=['bal_acc', 'min_recall', 'macro_f1'])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs = Config()
    loaders = data_generator('datasets', configs, 'fine_tune', args.seed)
    val_logits_list, test_logits_list = [], []
    valid_runs = []
    for run in args.runs:
        ck = find_ckpt(run)
        if ck is None:
            print(f"{run}: no ckpt, skip")
            continue
        model = base_Model(configs).to(device)
        ckpt = torch.load(ck, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['base_model_state_dict'])
        vl, vy = collect_logits(model, loaders['valid'], device)
        tl, ty = collect_logits(model, loaders['test'], device)
        val_logits_list.append(vl)
        test_logits_list.append(tl)
        valid_runs.append(run)
        va, vr0, vr1 = metrics(vl, vy)
        ta, tr0, tr1 = metrics(tl, ty)
        print(f"{run:12s} VAL acc={va:.4f} road={vr0:.4f} field={vr1:.4f} | TEST acc={ta:.4f} road={tr0:.4f} field={tr1:.4f}")
        del model, ckpt
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    val_logits = np.stack(val_logits_list)   # (M, N, 2)
    test_logits = np.stack(test_logits_list)
    # 验证集标签（各模型相同 split）
    val_labels = vy
    test_labels = ty
    M = len(valid_runs)

    def ensemble_val(idx):
        avg = val_logits[list(idx)].mean(axis=0)
        return metrics(avg, val_labels)

    def ensemble_test(idx):
        avg = test_logits[list(idx)].mean(axis=0)
        return metrics(avg, test_labels)

    def ensemble(idx):
        return ensemble_val(idx), ensemble_test(idx)

    def score(idx):
        va, vr0, vr1 = ensemble_val(idx)
        if args.greedy_target == 'bal_acc':
            return (vr0 + vr1) / 2
        if args.greedy_target == 'min_recall':
            return min(vr0, vr1)
        # macro f1 on val
        pred = val_logits[list(idx)].mean(axis=0).argmax(1)
        from sklearn.metrics import f1_score
        return f1_score(val_labels, pred, average='macro')

    if args.fixed:
        combo = [r for r in args.fixed if r in valid_runs]
        idx = [valid_runs.index(r) for r in combo]
        va, vr0, vr1 = ensemble_val(idx)
        ta, tr0, tr1 = ensemble_test(idx)
        print(f"\n[FIXED {'+'.join(combo)}]")
        print(f"VAL  acc={va:.4f} road={vr0:.4f} field={vr1:.4f}")
        print(f"TEST acc={ta:.4f} road={tr0:.4f} field={tr1:.4f}")
        return

    # 贪心前向选择
    print(f"\n--- greedy selection (target={args.greedy_target}) ---")
    selected = []
    best_score = -1
    while len(selected) < min(args.max_k, M):
        cand_best, cand_score = None, -1
        for i in range(M):
            if i in selected:
                continue
            s = score(selected + [i])
            if s > cand_score:
                cand_score, cand_best = s, i
        if cand_best is None or cand_score <= best_score + 1e-6:
            break
        selected.append(cand_best)
        best_score = cand_score
        va, vr0, vr1 = ensemble_val(selected)
        ta, tr0, tr1 = ensemble_test(selected)
        combo = "+".join(valid_runs[i] for i in selected)
        print(f"k={len(selected)} [{combo}] val_score={best_score:.4f}")
        print(f"    VAL acc={va:.4f} road={vr0:.4f} field={vr1:.4f} | TEST acc={ta:.4f} road={tr0:.4f} field={tr1:.4f}")

    print("\n=== FINAL (greedy on val, report on test) ===")
    va, vr0, vr1 = ensemble_val(selected)
    ta, tr0, tr1 = ensemble_test(selected)
    print(f"ensemble: {'+'.join(valid_runs[i] for i in selected)}")
    print(f"TEST acc={ta:.4f} road={tr0:.4f} field={tr1:.4f}")
    ok = ta >= 0.9223 and tr0 >= 0.6906 and tr1 >= 0.97
    print(">>> ALL TARGETS MET <<<" if ok else ">>> not all met <<<")


if __name__ == '__main__':
    main()
