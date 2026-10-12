"""TS2Vec 基线（官方代码）跑我们的数据集：训练(无标签) -> 线性探测 + KMeans 聚类
用法: python run_ts2vec.py --data ../../trajectory_classification_linux/datasets_clean --gpu 0 --epochs 50
放在 baselines/ts2vec/ 目录下运行
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ts2vec import TS2Vec
from utils import init_dl_program


def cm_metrics(t, p):
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (t.astype(int), p.astype(int)), 1)
    acc = np.trace(cm) / cm.sum()
    rows = []
    for c in range(2):
        tp = cm[c, c]
        prec = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rec = tp / cm[c, :].sum() if cm[c, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append((prec, rec, f1))
    return cm, acc, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--repr-dims', type=int, default=320)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    init_dl_program(args.gpu, seed=args.seed, max_threads=8)

    def load(name):
        d = torch.load(os.path.join(args.data, '%s.pt' % name), weights_only=True)
        x = d['samples'].numpy().astype(np.float32)   # (N, T, C)
        y = d['labels'].numpy().astype(np.int64)
        return x, y

    train_x, train_y = load('train')
    valid_x, valid_y = load('val')
    test_x, test_y = load('test')
    print('train', train_x.shape, 'val', valid_x.shape, 'test', test_x.shape, flush=True)

    model = TS2Vec(
        input_dims=train_x.shape[2],
        output_dims=args.repr_dims,
        batch_size=args.batch_size,
        lr=args.lr,
        device='cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu',
    )
    loss_log = model.fit(train_x, n_epochs=args.epochs, verbose=True)

    tr = model.encode(train_x, encoding_window='full_series')
    vr = model.encode(valid_x, encoding_window='full_series')
    te = model.encode(test_x, encoding_window='full_series')

    # 线性探测（逻辑回归，与 ts2vec 官方 fit_lr 一致），val 选 C
    from sklearn.metrics import f1_score
    best = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        trs = sc.fit_transform(tr)
        vrs = sc.transform(vr)
        tes = sc.transform(te)
        clf = LogisticRegression(max_iter=1000, C=C).fit(trs, train_y)
        vf1 = f1_score(valid_y, clf.predict(vrs), average='macro')
        if best is None or vf1 > best[0]:
            best = (vf1, clf, tes)
    pred = best[1].predict(best[2])
    cm, acc, rows = cm_metrics(test_y, pred)
    mf1 = f1_score(test_y, pred, average='macro')
    print('TS2Vec linear: acc=%.4f mF1=%.4f' % (acc, mf1))
    print('  road  P=%.4f R=%.4f F1=%.4f' % rows[0])
    print('  field P=%.4f R=%.4f F1=%.4f' % rows[1])
    print('  cm=', cm.tolist())

    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(te)
    print('TS2Vec kmeans: NMI=%.4f ARI=%.4f' % (
        normalized_mutual_info_score(test_y, km.labels_),
        adjusted_rand_score(test_y, km.labels_)))
    model.save(args.out or 'ts2vec_ours.pkl')
    print('TS2VEC BASELINE DONE')


if __name__ == '__main__':
    main()
