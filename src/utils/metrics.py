# Functions for calculating all evaluation metrics
# src/utils/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def calculate_regression_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))

    # 处理y_pred或y_true标准差为0的特殊情况
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        corr = 0.0
    else:
        corr = np.corrcoef(y_pred, y_true)[0, 1]

    return {"MAE": mae, "Correlation": corr}


def calculate_classification_metrics(y_true, y_pred, is_regression_values=False):
    """
    计算二分类任务的核心指标。

    Args:
        y_true (np.array): 真实标签的一维数组。
        y_pred (np.array): 预测结果的一维数组。
        is_regression_values (bool): 指示输入是否为回归值。
                                    - 如果为True, 函数会先将y_true和y_pred转为0/1标签。
                                    - 如果为False, 函数假设输入已经是0/1标签。
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    if is_regression_values:
        # 从回归值转换为二分类标签
        y_true_binary = (y_true >= 0).astype(int)
        y_pred_binary = (y_pred >= 0).astype(int)
    else:
        # 输入已经是类别ID
        y_true_binary = y_true
        y_pred_binary = y_pred

    acc2 = accuracy_score(y_true_binary, y_pred_binary)
    f1_binary = f1_score(y_true_binary, y_pred_binary, average='weighted')

    return {"Accuracy_Binary": acc2, "F1_Score_Binary": f1_binary}