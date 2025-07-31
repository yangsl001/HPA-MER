# src/trainers/hpa_mer_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm
import numpy as np
import json

# 导入我们自己的模块
from src.models.model_main import get_model
from src.utils.metrics import calculate_regression_metrics, calculate_classification_metrics
from src.utils.logger import setup_logger


class HPA_MER_Trainer:
    """
    一个为HPA-MER模型设计的、功能完备的训练器。
    """

    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

        # 1. 初始化日志和模型保存路径
        run_timestamp = str(int(time.time()))
        self.run_dir = os.path.join(config.RUNS_DIR, f"{config.EXPERIMENT_NAME}_{run_timestamp}")
        log_dir = os.path.join(self.run_dir, "logs")
        self.model_save_dir = os.path.join(self.run_dir, "models")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger = setup_logger(log_dir, f"{config.EXPERIMENT_NAME}")

        self.logger.info(f"--- 实验 '{config.EXPERIMENT_NAME}' 开始 ---")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"所有输出将保存在: {self.run_dir}")

        # 2. 初始化模型、优化器和损失函数
        self.model = get_model(config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.reg_criterion = nn.L1Loss()
        self.cls_criterion = nn.CrossEntropyLoss()

        self.logger.info(
            f"模型 '{config.MODEL_NAME}' 已初始化，可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    # --- 损失计算辅助函数 ---
    def _compute_disentanglement_loss(self, output_dict):
        if self.config.LAMBDA_DISENTANGLE <= 0:
            return torch.tensor(0.0).to(self.device)

        features = output_dict['disentanglement_features']

        def _info_nce_loss(f1, f2, temp=0.07):
            # F.normalize(input, p=2, dim=1, eps=1e-8)
            # p=2 表示L2范数，dim=1表示按行进行归一化
            f1_norm = F.normalize(f1, p=2, dim=1, eps=1e-8)
            f2_norm = F.normalize(f2, p=2, dim=1, eps=1e-8)
            logits = torch.mm(f1_norm, f2_norm.t()) / temp
            return F.cross_entropy(logits, torch.arange(len(f1), device=self.device))

        ortho_loss = sum(F.cosine_similarity(f_exp, f_imp, dim=1).pow(2).mean() for f_exp, f_imp in features.values())
        mi_loss = sum(_info_nce_loss(f_exp, f_imp) for f_exp, f_imp in features.values())

        return self.config.LAMBDA_DISENTANGLE * ((ortho_loss / 3.0) + (mi_loss / 3.0))

    def _compute_cma_loss(self, output_dict, labels):
        if not self.config.USE_CMA:
            return torch.tensor(0.0).to(self.device)

        intermediate_preds = output_dict['intermediate_predictions']
        lam, index = output_dict['cma_info']

        reg_labels = labels['regression_labels']
        mixed_a = lam * reg_labels['label_A'] + (1 - lam) * reg_labels['label_A'][index]
        mixed_v = lam * reg_labels['label_V'] + (1 - lam) * reg_labels['label_V'][index]

        loss_cma = self.reg_criterion(intermediate_preds['acoustic'], mixed_a) + \
                   self.reg_criterion(intermediate_preds['visual'], mixed_v)

        return self.config.LAMBDA_CMA * (loss_cma / 2.0)

    def _compute_task_loss(self, output_dict, labels):
        total_loss = torch.tensor(0.0).to(self.device)

        if self.config.USE_REGRESSION:
            preds = output_dict['regression']
            targets = torch.stack([labels['regression_labels'][k] for k in self.config.REGRESSION_TARGETS], dim=1)

            for i, key in enumerate(self.config.REGRESSION_TARGETS):
                weight_key = 'main' if key == 'label' else key.split('_')[1].lower()
                weight = self.config.REGRESSION_LOSS_WEIGHTS.get(weight_key, 0.0)
                if not self.config.USE_MODALITY_SPECIFIC_LOSS and key != 'label':
                    weight = 0.0
                total_loss += weight * self.reg_criterion(preds[:, i], targets[:, i])

        if self.config.USE_CLASSIFICATION:
            total_loss += self.config.CLASSIFICATION_LOSS_WEIGHT * self.cls_criterion(output_dict['classification'],
                                                                                      labels['classification_label'])

        return total_loss

    # --- 主训练和评估循环 ---
    def do_train(self, train_loader, valid_loader, test_loader):
        self.logger.info("======== 开始训练循环 ========")
        best_valid_mae = float('inf')
        best_epoch = -1

        for epoch in range(self.config.NUM_EPOCHS):
            self.model.train()
            epoch_losses = {'total': 0, 'task': 0, 'disentangle': 0, 'cma': 0}

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            for batch in progress_bar:
                # 数据准备
                text = batch['text_features'].to(self.device)
                audio = batch['acoustic_features'].to(self.device)
                vision = batch['visual_features'].to(self.device)
                labels = {
                    'regression_labels': {k: v.to(self.device) for k, v in batch['regression_labels'].items()},
                    'classification_label': batch['classification_label'].to(self.device)
                }

                self.optimizer.zero_grad()
                output_dict = self.model(text, audio, vision, use_cma=self.config.USE_CMA)

                task_loss = self._compute_task_loss(output_dict, labels)
                dis_loss = self._compute_disentanglement_loss(output_dict)
                cma_loss = self._compute_cma_loss(output_dict, labels)

                loss = task_loss + dis_loss + cma_loss
                loss.backward()
                if self.config.GRAD_CLIP_VALUE:
                    nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP_VALUE)
                self.optimizer.step()

                epoch_losses['total'] += loss.item()
                epoch_losses['task'] += task_loss.item()
                epoch_losses['disentangle'] += dis_loss.item()
                epoch_losses['cma'] += cma_loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.3f}", task=f"{task_loss.item():.3f}",
                                         dis=f"{dis_loss.item():.3f}", cma=f"{cma_loss.item():.3f}")

            avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
            self.logger.info(
                f"Epoch {epoch + 1} Train Loss - Total: {avg_losses['total']:.4f} | Task: {avg_losses['task']:.4f} | Dis: {avg_losses['disentangle']:.4f} | CMA: {avg_losses['cma']:.4f}")
            self.writer.add_scalars('Loss/train', avg_losses, epoch)

            valid_metrics = self.do_eval(valid_loader, epoch, "Valid")
            if self.config.USE_REGRESSION and 'label' in valid_metrics.get('regression', {}):
                current_mae = valid_metrics['regression']['label']['MAE']
                if current_mae < best_valid_mae:
                    best_valid_mae = current_mae
                    best_epoch = epoch + 1
                    torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, 'best_model.pth'))
                    self.logger.info(f"==> 新的最佳模型已保存 (Epoch {best_epoch}, Val MAE: {best_valid_mae:.4f})")

        self.logger.info(f"======== 训练结束, 最佳模型发现在 Epoch {best_epoch} ========")
        self.writer.close()

        best_model_path = os.path.join(self.model_save_dir, 'best_model.pth')
        self.logger.info(f"\n======== 在测试集上进行最终评估 ========")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            self.logger.info(f"已加载最佳模型: {best_model_path}")
        else:
            self.logger.warning("未找到最佳模型，测试将使用最后一个epoch的模型。")

        test_metrics = self.do_eval(test_loader, phase="Test")
        with open(os.path.join(self.log_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(test_metrics, f, indent=4, ensure_ascii=False)
        self.logger.info(f"测试结果已保存到: {os.path.join(self.log_dir, 'test_results.json')}")

    # src/trainers/hpa_mer_trainer.py -> do_eval 方法 (最终修正版)

    def do_eval(self, data_loader, epoch=None, phase="Valid"):
        self.model.eval()

        # 初始化用于收集所有预测和标签的容器
        all_outputs = {
            'regression': [],
            'classification': []
        }
        all_labels = {
            'regression_labels': {key: [] for key in self.config.REGRESSION_TARGETS},
            'classification_label': []
        }

        # 在no_grad模式下进行评估，节省显存和计算
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1} [{phase}]" if epoch is not None else f"[{phase}]"):
                # 准备数据并移动到设备
                text = batch['text_features'].to(self.device)
                audio = batch['acoustic_features'].to(self.device)
                vision = batch['visual_features'].to(self.device)

                # 模型前向传播 (在eval模式下，use_cma应为False)
                outputs = self.model(text, audio, vision, use_cma=False)

                # 收集所有预测结果 (转移到CPU)
                if self.config.USE_REGRESSION:
                    all_outputs['regression'].append(outputs['regression'].cpu())
                if self.config.USE_CLASSIFICATION:
                    all_outputs['classification'].append(outputs['classification'].cpu())

                # 收集所有真实标签 (转移到CPU)
                if self.config.USE_REGRESSION:
                    for key in self.config.REGRESSION_TARGETS:
                        all_labels['regression_labels'][key].append(batch['regression_labels'][key].cpu())
                if self.config.USE_CLASSIFICATION:
                    all_labels['classification_label'].append(batch['classification_label'].cpu())

        # --- 数据后处理与指标计算 ---
        final_metrics = {'regression': {}, 'classification': {}}
        self.logger.info(f"--- {phase} Phase Results (Epoch: {epoch + 1 if epoch is not None else 'N/A'}) ---")

        # 1. 处理并计算回归指标
        if self.config.USE_REGRESSION:
            # 将list of tensors合并成一个大的numpy数组
            all_reg_preds = torch.cat(all_outputs['regression']).numpy()

            for i, task_name in enumerate(self.config.REGRESSION_TARGETS):
                # 合并对应任务的真实标签
                task_true_labels = torch.cat(all_labels['regression_labels'][task_name]).numpy()
                # 提取对应任务的预测结果
                task_preds = all_reg_preds[:, i]

                # 计算回归指标
                metrics = calculate_regression_metrics(task_true_labels, task_preds)
                final_metrics['regression'][task_name] = metrics
                self.logger.info(
                    f"  [Reg] Task: {task_name:<10} | MAE={metrics['MAE']:.4f}, Corr={metrics['Correlation']:.4f}")

                # 写入TensorBoard
                if self.writer and epoch is not None:
                    self.writer.add_scalars(f"Metrics_{phase}_Regression/{task_name}", metrics, epoch)

        # 2. 处理并计算分类指标
        if self.config.USE_CLASSIFICATION:
            # --- 核心逻辑修正 ---
            # a. 真正的分类任务指标 (基于分类头的logits和'annotation'标签)
            cls_logits = torch.cat(all_outputs['classification'])
            cls_pred_ids = cls_logits.argmax(dim=1).numpy()
            cls_true_ids = torch.cat(all_labels['classification_label']).numpy()

            cls_metrics = calculate_classification_metrics(cls_true_ids, cls_pred_ids)
            final_metrics['classification']['annotation_based'] = cls_metrics
            self.logger.info(
                f"  [Cls] Task: {'annotation':<10} | Acc2={cls_metrics['Accuracy_Binary']:.4f}, F1_Bin={cls_metrics['F1_Score_Binary']:.4f}")

            if self.writer and epoch is not None:
                self.writer.add_scalars(f"Metrics_{phase}_Classification/Annotation-Based", cls_metrics, epoch)

            # b. 衍生的分类任务指标 (基于主回归任务'label'的预测值和真实值)
            # 只有在回归任务被启用时才能计算
            if self.config.USE_REGRESSION and 'label' in self.config.REGRESSION_TARGETS:
                main_reg_true = torch.cat(all_labels['regression_labels']['label']).numpy()
                main_reg_pred = all_reg_preds[:, self.config.REGRESSION_TARGETS.index('label')]

                derived_cls_metrics = calculate_classification_metrics(main_reg_true, main_reg_pred,
                                                                       is_regression_values=True)
                final_metrics['classification']['derived_from_label'] = derived_cls_metrics
                self.logger.info(
                    f"  [Cls] Task (derived): {'from_label':<10} | Acc2={derived_cls_metrics['Accuracy_Binary']:.4f}, F1_Bin={derived_cls_metrics['F1_Score_Binary']:.4f}")

                if self.writer and epoch is not None:
                    self.writer.add_scalars(f"Metrics_{phase}_Classification/Derived-from-Label", derived_cls_metrics,
                                            epoch)

        return final_metrics