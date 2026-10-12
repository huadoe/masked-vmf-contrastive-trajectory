import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss_function import (
    NTXentLoss, cluster_contrastive_loss, prototype_alignment_loss,
    HardNegativeNTXentLoss, vMFProbContrastiveLoss,
)
from clustering import ClusterMemory
from dataloader.dataloader_notype import StratifiedSampler
from torch.utils.data import DataLoader

sys.path.append("..")
K = 10  # 每K个epoch聚类一次数据，增加聚类间隔以稳定伪标签
warmup_epochs = 10  # 预热epoch数


def _apply_schedule_config(config):
    """允许从 config / CLI 覆盖聚类间隔与预热轮数"""
    global K, warmup_epochs
    K = getattr(config, 'cluster_interval', K)
    warmup_epochs = getattr(config, 'warmup_epochs', warmup_epochs)


def random_time_mask(x, mask_ratio=0.25, span=8, span_jitter=True):
    """
    v2: 随机连续时间段掩码（TimeMAE 式 span masking）
    x: (B, T, C)
    返回: 掩码后的 x, mask (B, T) bool（True = 被掩码）
    """
    B, T, C = x.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    n_spans = max(1, int(round(T * mask_ratio / span)))
    for _ in range(n_spans):
        s = torch.randint(0, max(1, T - span + 1), (1,)).item()
        e = min(T, s + span)
        mask[:, s:e] = True
    # 保证至少掩码 ratio*0.5 的点，最多不超过 0.6
    frac = mask.float().mean(dim=1, keepdim=True)
    mask = mask & (frac < 0.6)
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


@torch.no_grad()
def ema_update(teacher, student, momentum=0.99):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(momentum).add_(sp.detach(), alpha=1 - momentum)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


# def Trainer(base_model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl,
#             file_count, device, logger, config, experiment_log_dir, training_mode, class_weights,
#             start_epoch=1, # 起始 epoch
#             scheduler_state_dict=None): # 调度器状态字典
def Trainer(base_model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, cluster_dl, train_ds,
            device, logger, config, experiment_log_dir, training_mode, class_weights,
            start_epoch=1,  # 起始 epoch
            scheduler_state_dict=None):  # 调度器状态字典
    # 确保 class_weights 是在 CPU 上转换为 Tensor，再移到 device
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    _apply_schedule_config(config)
    logger.debug(f"Training started .... (cluster_interval={K}, warmup={warmup_epochs})")

    # 线性探测的类权重：'inverse_freq' 全量逆频率 / 'sqrt_inv' 平方根折中 / 'none' 无加权
    # linear_class_power: 通用幂次 w = inv_freq^alpha（0=无权重, 0.5=sqrt, 1=inverse），优先于 lw_mode
    lw_mode = getattr(config, 'linear_class_weight', 'none')
    lw_power = getattr(config, 'linear_class_power', None)
    if training_mode != "self_supervised" and (lw_power is not None or lw_mode != 'none'):
        w = class_weights.clone()
        if lw_power is not None:
            w = w ** float(lw_power)
            w = w / w.mean()
            logger.debug(f"CE weight = inv_freq^{lw_power}: {w.tolist()}")
        elif lw_mode == 'sqrt_inv':
            w = torch.sqrt(w)
            w = w / w.mean()
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min', patience=10)

    # --- 新增：加载调度器状态 ---
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
        logger.debug("Loaded LR scheduler state.")
    # ---------------------------

    # 确保保存模型的目录存在
    save_models_dir = os.path.join(experiment_log_dir, "saved_models")
    os.makedirs(save_models_dir, exist_ok=True)

    # 加载聚类
    cluster_memory = ClusterMemory(
        n_clusters=config.num_classes,
        feature_dim=int(config.final_out_channels / 2),  # temporal_contr_model 的输出维度
        device=device
    )

    # ===== v2 初始化 =====
    # EMA 教师（掩码重建目标；掩码关闭时不创建，节省显存）
    teacher_model = None
    if training_mode == "self_supervised" and config.Mask.prob > 0 and config.lambda_recon > 0:
        teacher_model = copy.deepcopy(base_model).to(device)
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.eval()

    # vMF 概率对比损失（kappa 可学习，挂到 base_model 优化器之后单独管理；权重为 0 时不创建）
    vmf_criterion = None
    vmf_optimizer = None
    if training_mode == "self_supervised" and config.lambda_vmf > 0:
        vmf_criterion = vMFProbContrastiveLoss(
            n_clusters=config.num_classes,
            feature_dim=int(config.final_out_channels / 2),
            temperature=config.VMF.temperature,
            kappa_init=config.VMF.kappa_init,
            w_intra=config.VMF.w_intra,
        ).to(device)
        vmf_optimizer = torch.optim.Adam(vmf_criterion.parameters(), lr=1e-2)

    # 实例对比损失：top_k<=0 时用标准 NTXent（硬负样本挖掘在 2 类不平衡下会放大假负样本）
    if config.HardNeg.top_k and config.HardNeg.top_k > 0:
        hn_criterion = HardNegativeNTXentLoss(
            device, config.batch_size, config.Context_Cont.temperature,
            use_cosine_similarity=True, top_k=config.HardNeg.top_k
        )
    else:
        hn_criterion = NTXentLoss(device, config.batch_size,
                                  config.Context_Cont.temperature,
                                  config.Context_Cont.use_cosine_similarity)

    # 线性探测最优模型跟踪
    best_val_score = -1.0
    best_path = os.path.join(save_models_dir, 'ckp_best.pt')

    # AMP 混合精度（自监督前向；线性探测不开）
    use_amp = bool(getattr(config, 'use_amp', False)) and device.type == 'cuda' and training_mode == "self_supervised"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        logger.debug("AMP autocast + GradScaler enabled (self_supervised)")

    # 修改 epoch 循环，从 start_epoch 开始
    for epoch in range(start_epoch, config.num_epoch + 1):
        # 聚类流程（lambda_cluster_scale=0 时为纯 TS-TCC 基线）
        cluster_scale = getattr(config, 'lambda_cluster_scale', 1.0)
        lambda_cluster = 0 if epoch <= warmup_epochs else 0.4 * (epoch / config.num_epoch) * cluster_scale  # 聚类损失权重（Run_17 基线系数 0.4）
        if (epoch % K == 0 or epoch == 1) and training_mode == "self_supervised":  # Run_17：epoch 1 先聚一次给初值
            all_features = []
            base_model.eval()
            with torch.no_grad():
                for data, labels, cluster_tags, aug1, aug2, center_idx, file_id, idx in cluster_dl:
                    aug1 = aug1.float().to(device)
                    aug2 = aug2.float().to(device)
                    predictions1, features1 = base_model(aug1)
                    predictions2, features2 = base_model(aug2)

                    # 标准化特征向量
                    features1 = F.normalize(features1, dim=1)
                    features2 = F.normalize(features2, dim=1)

                    temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
                    temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)
                    z = (temp_cont_lstm_feat1 + temp_cont_lstm_feat2) / 2

                    all_features.append(z)

            all_features = torch.cat(all_features, dim=0)
            print("开始聚类")
            cluster_memory.run_clustering(all_features)
            cluster_memory.compute_high_quality(all_features, threshold=0.5)  # Run_17 基线固定阈值 0.5
            cluster_memory.compute_prototypes(all_features)
            # v2: 伪标签置信度 + vMF kappa 矩估计校准（vMF 启用时）
            if vmf_criterion is not None:
                cluster_memory.compute_confidences(all_features, tau=config.VMF.conf_tau)
                vmf_criterion.estimate_kappa(all_features, cluster_memory.labels)
                print("聚类完成",
                      "| 簇比例:",
                      torch.bincount(cluster_memory.labels.cpu()).tolist(),
                      "| 平均置信度: %.4f" % cluster_memory.confidences.mean().item(),
                      "| kappa:", [round(k, 2) for k in vmf_criterion.kappa.tolist()])
            else:
                print("聚类完成 | 簇比例:",
                      torch.bincount(cluster_memory.labels.cpu()).tolist())

            train_ds.cluster_tag = (
                cluster_memory.labels
                .cpu()
                .numpy()
            )
            # PSS 分层采样（可选；Run_17 基线使用普通顺序加载器）
            if getattr(config, 'use_stratified_sampler', False):
                train_sampler = StratifiedSampler(
                    train_ds,
                    config.batch_size
                )
                train_loader = DataLoader(
                    train_ds,
                    batch_size=config.batch_size,
                    sampler=train_sampler,
                    shuffle=False,
                    num_workers=0,
                    drop_last=True,
                )
                train_dl = train_loader

        # 训练和验证
        if training_mode == "self_supervised":
            valid_loss, valid_acc = 0.0, 0.0
            train_loss, train_acc, loss_temporal, loss_inst, loss_vmf, loss_rec, loss_clu = model_train(
                base_model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                criterion, train_dl, config, device, training_mode,
                class_weights, cluster_memory, lambda_cluster, epoch,
                teacher_model=teacher_model, vmf_criterion=vmf_criterion,
                vmf_optimizer=vmf_optimizer, hn_criterion=hn_criterion,
                scaler=scaler, use_amp=use_amp)
            logger.debug(f'\nEpoch : {epoch}\n'
                         f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                         f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}\n'
                         f'Temporal NCE   : {loss_temporal:.4f}\t | \tInstance NTXent : {loss_inst:.4f}\n'
                         f'Cluster CPSE   : {loss_clu:.4f}\t | \tvMF Contrast    : {loss_vmf:.4f}\n'
                         f'Mask Recon     : {loss_rec:.4f}')
        else:
            train_loss, train_acc = model_train(base_model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode,
                                            class_weights, cluster_memory, lambda_cluster, epoch,
                                            scaler=scaler, use_amp=False)
            valid_loss, valid_acc, val_f1, v_preds, v_trgs = model_evaluate(
                base_model,
                temporal_contr_model,
                valid_dl,
                device,
                training_mode,
                config,
                class_weights
            )
            scheduler.step(valid_loss)
            # v2: 验证集每类召回（诊断类平衡用）
            v_preds = np.array(v_preds)
            v_trgs = np.array(v_trgs)
            val_r0 = float((v_preds[v_trgs == 0] == 0).mean()) if (v_trgs == 0).any() else 0.0
            val_r1 = float((v_preds[v_trgs == 1] == 1).mean()) if (v_trgs == 1).any() else 0.0
            val_bal_acc = (val_r0 + val_r1) / 2
            logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}\t | \tValid MacroF1 : {val_f1:2.4f}\n'
                     f'Valid Recall   : road={val_r0:.4f}\t | \tfield={val_r1:.4f}\t | \tBalAcc : {val_bal_acc:.4f}')

            # v2: 模型选择标准：select_by='macro_f1'（默认）或 'bal_acc'（验证集平衡准确率）
            select_by = getattr(config, 'select_by', 'macro_f1')
            sel_score = val_bal_acc if select_by == 'bal_acc' else val_f1
            if sel_score > best_val_score:
                best_val_score = sel_score
                torch.save({'epoch': epoch,
                            'base_model_state_dict': base_model.state_dict(),
                            'temporal_contr_model_state_dict': temporal_contr_model.state_dict()},
                           best_path)
                logger.debug(f"Best model saved at epoch {epoch} ({select_by}={sel_score:.4f})")

        try:
            current_lrs = scheduler.get_last_lr()
        except Exception:
            current_lrs = [g['lr'] for g in model_optimizer.param_groups]
        print(f"Epoch {epoch}, Current Learning Rates: {current_lrs}")

        # --- 保存检查点（每个 epoch 覆盖 ckp_last.pt）---
        chkpoint = {
            'epoch': epoch,
            'base_model_state_dict': base_model.state_dict(),
            'temporal_contr_model_state_dict': temporal_contr_model.state_dict(),
            'model_optimizer_state_dict': model_optimizer.state_dict(),
            'temporal_contr_optimizer_state_dict': temp_cont_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        if teacher_model is not None:
            chkpoint['teacher_state_dict'] = teacher_model.state_dict()
        if vmf_criterion is not None:
            chkpoint['vmf_state_dict'] = vmf_criterion.state_dict()
        torch.save(chkpoint, os.path.join(save_models_dir, f'ckp_last.pt'))
        logger.debug(f"Checkpoint saved for epoch {epoch}")

    # v2: 线性/微调模式下加载最优模型用于测试
    select_by = getattr(config, 'select_by', 'macro_f1')
    if training_mode != "self_supervised" and os.path.exists(best_path):
        best = torch.load(best_path, map_location=device, weights_only=False)
        base_model.load_state_dict(best['base_model_state_dict'])
        logger.debug(f"Loaded best model (epoch {best['epoch']}, val {select_by}={best_val_score:.4f}) for test evaluation")

    # 最终模型评估
    if training_mode != "self_supervised":  # 自监督模型不需要测试
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _, _ = model_evaluate(base_model, temporal_contr_model, test_dl,
                                                      device, training_mode, config, class_weights)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")


class WeightedNTXentLoss(nn.Module):
    def __init__(self, device, batch_size, temperature, use_cosine_sim, class_weights):
        super().__init__()
        self.ntxent = NTXentLoss(device, batch_size, temperature, use_cosine_sim)
        self.class_weights = class_weights  # 传入的类别权重

    def forward(self, zis, zjs, pseudo_labels):
        # 计算原始对比损失
        loss = self.ntxent(zis, zjs)

        # 根据伪标签类别加权
        weighted_loss = loss * self.class_weights[pseudo_labels].mean()
        return weighted_loss


# def model_train(base_model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl,
#                 train_count, config, device, training_mode, class_weights):
def model_train(base_model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl,
                config, device, training_mode, class_weights, cluster_memory, lambda_cluster, epoch,
                teacher_model=None, vmf_criterion=None, vmf_optimizer=None, hn_criterion=None,
                scaler=None, use_amp=False):
    total_loss = []
    total_acc = []
    base_model.train()
    temporal_contr_model.train()
    if teacher_model is not None:
        teacher_model.eval()

    running = {'temporal': 0.0, 'inst': 0.0, 'vmf': 0.0, 'recon': 0.0, 'cluster': 0.0, 'n': 0}

    for batch_idx, (data, labels, cluster_tags, aug1, aug2, center_idx, file_id, idx) in enumerate(train_dl):
        data, labels = data.float().to(device), labels.long().to(device)
        cluster_tags = cluster_tags.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # 优化器
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()
        if vmf_optimizer is not None:
            vmf_optimizer.zero_grad()

        if training_mode == "self_supervised":
            # ===== v2: 随机时间段掩码（作用于强增强视图）=====
            use_mask = (np.random.rand() < config.Mask.prob)
            if use_mask:
                aug2_masked, mask_t = random_time_mask(aug2, config.Mask.ratio, config.Mask.span)
            else:
                aug2_masked, mask_t = aug2, None

            with torch.autocast('cuda', enabled=use_amp):
                predictions1, features1 = base_model(aug1)
                predictions2, features2 = base_model(aug2_masked)

                # 标准化特征向量
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

                zis = temp_cont_lstm_feat1  # (B, D)
                zjs = temp_cont_lstm_feat2

                # ===== v2: 掩码重建损失（EMA 教师目标，表征级回归）=====
                loss_recon = torch.tensor(0.0, device=device)
                if use_mask and teacher_model is not None:
                    with torch.no_grad():
                        _, teacher_feat = teacher_model(aug2)          # (B, C, T) 无掩码目标
                    teacher_feat = F.normalize(teacher_feat, dim=1)    # 与学生侧相同的归一化
                    student_feat = base_model.recon_proj(features2)    # (B, C, T)，features2 已归一化
                    mask_feat = mask_t.unsqueeze(1).expand_as(student_feat)  # (B, C, T)
                    loss_recon = F.mse_loss(student_feat[mask_feat], teacher_feat[mask_feat])

                # ===== v2: 硬负样本实例对比 =====
                loss_inst = hn_criterion(zis, zjs)

                # ===== CPSE: 聚类对比损失 + 原型对齐（基线配方，warmup 后启用）=====
                loss_cluster = torch.tensor(0.0, device=device)
                loss_prototype = torch.tensor(0.0, device=device)
                if epoch >= warmup_epochs and cluster_memory.labels is not None and lambda_cluster > 0:
                    z = torch.cat([zis, zjs], dim=0)   # (2B, D)
                    z = F.normalize(z, dim=1)
                    clabels = cluster_memory.labels[idx]   # (B,)
                    clabels = torch.cat([clabels, clabels], dim=0)
                    hq_mask = cluster_memory.high_quality_mask[idx]
                    hq_mask = torch.cat([hq_mask, hq_mask], dim=0)
                    label_matrix = clabels.unsqueeze(1) == clabels.unsqueeze(0)
                    quality_matrix = hq_mask.unsqueeze(1) & hq_mask.unsqueeze(0)

                    positive_mask = label_matrix & quality_matrix
                    positive_mask.fill_diagonal_(False)
                    loss_cluster = cluster_contrastive_loss(z, positive_mask)

                    if cluster_memory.prototypes is not None:
                        loss_prototype = prototype_alignment_loss(
                            z, clabels, cluster_memory.prototypes,
                            distance_type=config.PrototypeLoss.distance_type,
                            temperature=config.PrototypeLoss.temperature
                        )

                # ===== v2: vMF 概率对比（warmup 后，基于聚类伪标签 + 置信度过滤加权）=====
                loss_vmf = torch.tensor(0.0, device=device)
                if epoch >= warmup_epochs and vmf_criterion is not None and cluster_memory.labels is not None:
                    z = torch.cat([zis, zjs], dim=0)
                    z = F.normalize(z, dim=1)
                    plabels = cluster_memory.labels[idx]
                    plabels = torch.cat([plabels, plabels], dim=0)
                    conf = cluster_memory.confidences[idx]
                    conf = torch.cat([conf, conf], dim=0)
                    # 置信度门槛过滤：低于阈值的样本不参与（类比旧版 CPSE 的高质量样本过滤）
                    conf = torch.where(conf > config.VMF.conf_threshold, conf, torch.zeros_like(conf))
                    if conf.sum() > 1e-6:
                        priors = None
                        if config.VMF.uniform_prior:
                            priors = torch.ones(config.num_classes, device=device) / config.num_classes
                        loss_vmf = vmf_criterion(z, plabels, conf, cluster_memory.prototypes, priors=priors)

                # ===== 总损失 =====
                loss = (temp_cont_loss1 + temp_cont_loss2) * config.lambda_temporal \
                    + loss_inst * config.lambda_instance \
                    + loss_cluster * lambda_cluster \
                    + loss_prototype * config.PrototypeLoss.lambda_prototype \
                    + loss_vmf * config.lambda_vmf \
                    + loss_recon * config.lambda_recon

            running['temporal'] += float(temp_cont_loss1 + temp_cont_loss2)
            running['inst'] += float(loss_inst)
            running['vmf'] += float(loss_vmf)
            running['recon'] += float(loss_recon)
            running['cluster'] += float(loss_cluster)
            running['n'] += 1

        else:
            # 监督 / 线性探测模式
            output = base_model(data)
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        # 数值保护：非有限损失跳过本 batch（不回传、不更新）
        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at batch {batch_idx}, skipped")
            model_optimizer.zero_grad()
            temp_cont_optimizer.zero_grad()
            if vmf_optimizer is not None:
                vmf_optimizer.zero_grad()
            continue
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(model_optimizer)
            scaler.step(temp_cont_optimizer)
            if vmf_optimizer is not None:
                scaler.step(vmf_optimizer)
            scaler.update()
        else:
            loss.backward()
            model_optimizer.step()
            temp_cont_optimizer.step()
            if vmf_optimizer is not None:
                vmf_optimizer.step()

        # EMA 教师更新
        if teacher_model is not None and training_mode == "self_supervised":
            ema_update(teacher_model, base_model, momentum=config.Mask.ema_momentum)

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        n = max(1, running['n'])
        total_acc = 0
        return (total_loss, total_acc,
                running['temporal'] / n, running['inst'] / n,
                running['vmf'] / n, running['recon'] / n,
                running['cluster'] / n)
    else:
        total_acc = torch.tensor(total_acc).mean()
        return total_loss, total_acc


# def model_evaluate(base_model, temporal_contr_model, dataloader, file_count, device, training_mode, config, class_weights):
def model_evaluate(base_model, temporal_contr_model, dataloader, device, training_mode, config, class_weights):
    base_model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []  # 自监督模式下为 0

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, cluster_tags, aug1, aug2, center_idx, file_id, idx in dataloader:
            data, labels = data.float().to(device), labels.long().to(device)
            cluster_tags = cluster_tags.long().to(device)
            aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

            if training_mode == "self_supervised":
                # 自监督阶段不进行验证
                return torch.tensor(0.0), 0, 0.0, [], []

            else:  # 监督模式或微调模式
                output = base_model(data)
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        # v2: 计算 Macro-F1（模型选择依据）
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(trgs, outs, average='macro')
    else:
        total_loss = torch.tensor(total_loss).mean()
        total_acc = 0
        return total_loss, total_acc, 0.0, [], []

    return total_loss, total_acc, macro_f1, outs, trgs
