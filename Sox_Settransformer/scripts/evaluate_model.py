"""
[파일 목적]
- 예측 결과(predictions.csv)와 실제 label(HDF5/csv 등)을 비교하여 평가 지표를 산출하고, 평가 리포트를 저장하는 스크립트
- config['model']['task']에 따라 회귀/분류 평가 지표 분기(MAE/MSE/grid_accuracy vs accuracy)

[실행 예시]
python scripts/evaluate_model.py --config configs/wifi_hash.yaml --predictions experiments/exp001/outputs/predictions.csv --labels experiments/exp001/encoded/labels.h5 --output experiments/exp001/outputs/

[주요 흐름]
- 예측/정답 로딩 → config 로딩 → 평가 지표 분기 계산 → 평가 결과 저장

[의존성]
- utils/io, numpy, training/metrics 등
"""

import argparse
import os
import numpy as np
import json
from utils.io import load_from_h5
from utils.config import load_config
from training.metrics import mean_absolute_error, mean_squared_error, grid_accuracy, accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config yaml 경로')
    parser.add_argument('--predictions', type=str, required=True, help='예측 결과 CSV 경로')
    parser.add_argument('--labels', type=str, required=True, help='정답 label HDF5 경로')
    parser.add_argument('--output', type=str, required=True, help='평가 결과 저장 디렉토리')
    args = parser.parse_args()

    config = load_config(args.config)
    task = config['model'].get('task', 'regression')

    os.makedirs(args.output, exist_ok=True)
    preds = np.loadtxt(args.predictions, delimiter=',', skiprows=1)
    labels = load_from_h5(args.labels, dataset_name='labels')

    report = {'num_samples': int(labels.shape[0])}
    if task == 'classification':
        acc = accuracy(labels, preds)
        report['accuracy'] = float(acc)
    else:
        mae = mean_absolute_error(labels, preds)
        mse = mean_squared_error(labels, preds)
        grid_acc = grid_accuracy(labels, preds)
        report['mae'] = float(mae)
        report['mse'] = float(mse)
        report['grid_accuracy'] = float(grid_acc)

    with open(os.path.join(args.output, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print("Evaluation report:", report)

if __name__ == '__main__':
    main() 