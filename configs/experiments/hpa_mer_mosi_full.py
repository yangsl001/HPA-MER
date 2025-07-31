# configs/experiments/hpa_mer_mosi_full.py (修正并补全所有参数)
from ..base_config import BaseConfig
import os


class Config(BaseConfig):
    # --- 实验特定信息 ---
    EXPERIMENT_NAME = "hpa_mer_mosi_full"
    DATASET_NAME = "mosi"
    DATA_DIR = os.path.join(BaseConfig.DATA_ROOT_DIR, DATASET_NAME)
    DATA_TYPE = "unaligned_50.pkl"

    # --- 数据集特定维度 ---
    TEXT_EMBED_DIM = 768
    ACOUSTIC_FEAT_DIM = 5
    VISUAL_FEAT_DIM = 20

    # --- 模型选择 ---
    MODEL_NAME = "hpa_mer"

    # ===================================================================
    # HPA-MER 模型所有超参数 (核心补充部分)
    # ===================================================================
    D_MODEL = 128

    # MFE (多模态特征编码器) 参数
    MFE_A_HIDDEN_DIM = 64
    MFE_V_HIDDEN_DIM = 64
    MFE_A_LAYERS = 1
    MFE_V_LAYERS = 1
    MFE_DROPOUT = 0.2

    # UEICD (单模态显隐线索解耦) 参数
    UEICP_ATTN_HEADS = 4
    UEICP_EXP_LAYERS = 2
    UEICP_IMP_LAYERS = 2
    UEICP_EXP_FFN_DIM = 256
    UEICP_IMP_FFN_DIM = 256
    UEICP_DROPOUT = 0.15

    # CACE (跨模态自适应互补增强) 参数
    CACE_EXP_FUSION_HEADS = 4
    CACE_DROPOUT = 0.15

    # 输出头参数
    CLASSIFIER_HIDDEN_DIM = 64
    # ===================================================================

    # --- 损失函数权重 ---
    LAMBDA_DISENTANGLE = 0.3
    LAMBDA_CMA = 0.8
    REGRESSION_LOSS_WEIGHTS = {"main": 1.0, "text": 0.2, "acoustic": 0.2, "visual": 0.2}
    CLASSIFICATION_LOSS_WEIGHT = 0.5