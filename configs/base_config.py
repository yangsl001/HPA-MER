# configs/base_config.py
import torch
import os


class BaseConfig:
    # --- 基础路径设置 ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, "data")
    RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

    # --- 通用训练参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_VALUE = 1.0
    SEED = 42

    # --- 任务开关与定义 ---
    # 回归任务
    USE_REGRESSION = True
    # 分类任务
    USE_CLASSIFICATION = True
    # 模态特定损失 (辅助任务)
    USE_MODALITY_SPECIFIC_LOSS = False
    # CMA (Mixup) 数据增强
    USE_CMA = True

    # --- 标签与类别定义 ---
    # 回归任务的目标列名
    REGRESSION_TARGETS = ["label", "label_T", "label_A", "label_V"]
    # 分类任务的目标列名
    CLASSIFICATION_TARGET = "annotation"
    # 分类任务的类别
    CLASS_LABELS = ['Negative', 'Neutral', 'Positive']
    NUM_CLASSES = len(CLASS_LABELS)

    # 控制如何处理'Neutral'类别
    # True:  Neutral -> Positive (1)
    # False: 忽略Neutral样本
    MAP_NEUTRAL_TO_POSITIVE = True
# Base configurations shared by all experiments
