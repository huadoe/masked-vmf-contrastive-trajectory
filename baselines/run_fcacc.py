"""FCACC 基线（官方代码）跑我们的数据集：
官方协议 = transductive（train+test 合并训练聚类），测试指标在 test 子集上报
用法: python run_fcacc.py --data ../../trajectory_classification_linux/datasets_clean --gpu 0
放在 baselines/FCACC/ 目录下运行
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fcacc import FCACCModel


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


@torch.no_grad()
def encode_split(model, x, bs=256):
    reps = []
    model.encoder.eval()
    for i in range(0, len(x), bs):
        xb = torch.tensor(x[i:i + bs], dtype=torch.float32).to(model.device)
        r = model.encode_with_pooling(xb)
        reps.append(r.detach().cpu().numpy())
    return np.concatenate(reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--pretraining-epoch', type=int, default=50)
    ap.add_argument('--maxiter', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--name', default='ours_clean')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def load(name):
        d = torch.load(os.path.join(args.data, '%s.pt' % name), weights_only=True)
        return d['samples'].numpy().astype(np.float32), d['labels'].numpy().astype(np.int64)

    train_x, train_y = load('train')
    test_x, test_y = load('test')
    n_train = len(train_x)

    # 官方协议：train+test 合并（transductive clustering）
    comb_x = np.concatenate([train_x, test_x], axis=0)
    comb_y = np.concatenate([train_y, test_y], axis=0)
    ds = TensorDataset(torch.tensor(comb_x), torch.tensor(comb_y),
                       torch.arange(len(comb_x)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    config = {
        'batch_size': args.batch_size,
        'output_dims': 64,
        'lr': 0.001,
        'pretraining_epoch': args.pretraining_epoch,
        'MaxIter': args.maxiter,
        'w_c': 0.2,
        'm': 1.5,
        'hard_w': 0.2,
    }
    config.update({
        'data_loader': loader,
        'dataset_size': len(comb_x),
        'timesteps_len': comb_x.shape[1],
        'n_cluster': 2,
        'dataset_name': args.name,
        'input_dims': comb_x.shape[2],
        'device': 'cuda',
    })
    print('FCACC config:', {k: v for k, v in config.items() if k != 'data_loader'}, flush=True)

    model = FCACCModel(**config)
    model.Pretraining()
    model.Finetuning()

    # test 子集编码 -> 线性探测（train 表征训练 LR）+ KMeans
    tr = encode_split(model, train_x)
    te = encode_split(model, test_x)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    sc = StandardScaler()
    trs = sc.fit_transform(tr)
    tes = sc.transform(te)
    clf = LogisticRegression(max_iter=1000).fit(trs, train_y)
    pred = clf.predict(tes)
    cm, acc, rows = cm_metrics(test_y, pred)
    print('FCACC linear(test): acc=%.4f mF1=%.4f' % (acc, f1_score(test_y, pred, average='macro')))
    print('  road  P=%.4f R=%.4f F1=%.4f' % rows[0])
    print('  field P=%.4f R=%.4f F1=%.4f' % rows[1])
    print('  cm=', cm.tolist())

    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(te)
    print('FCACC kmeans(test): NMI=%.4f ARI=%.4f' % (
        normalized_mutual_info_score(test_y, km.labels_),
        adjusted_rand_score(test_y, km.labels_)))
    print('FCACC BASELINE DONE')


if __name__ == '__main__':
    main()
