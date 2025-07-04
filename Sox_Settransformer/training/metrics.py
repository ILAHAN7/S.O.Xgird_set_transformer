"""
[파일 목적]
- 모델 예측과 실제 label을 비교하는 다양한 평가 지표 함수 제공
- 회귀/분류 모두 지원: MAE, MSE, grid accuracy(회귀), accuracy(분류)
- evaluate_model.py, trainer.py 등에서 import하여 재사용

[주요 함수]
- mean_absolute_error(y_true, y_pred): MAE (회귀)
- mean_squared_error(y_true, y_pred): MSE (회귀)
- grid_accuracy(y_true, y_pred): (xId, yId) 완전일치 비율 (회귀)
- accuracy(y_true, y_pred): 분류 정확도 (분류)
"""

import numpy as np
import torch

def mean_absolute_error(y_true, y_pred):
    """MAE (numpy or torch.Tensor)"""
    if isinstance(y_true, torch.Tensor):
        return torch.mean(torch.abs(y_true - y_pred)).item()
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    """MSE (numpy or torch.Tensor)"""
    if isinstance(y_true, torch.Tensor):
        return torch.mean((y_true - y_pred) ** 2).item()
    return np.mean((y_true - y_pred) ** 2)

def grid_accuracy(y_true, y_pred):
    """(xId, yId) 완전일치 비율 (numpy or torch.Tensor)"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    return np.mean(np.all(np.round(y_true) == np.round(y_pred), axis=1))

def accuracy(y_true, y_pred):
    """분류 정확도: y_pred는 logits 또는 class index, y_true는 class index"""
    if isinstance(y_pred, torch.Tensor):
        if y_pred.ndim > 1:
            y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    return np.mean(y_true == y_pred) 