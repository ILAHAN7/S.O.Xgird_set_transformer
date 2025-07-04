"""
[파일 목적]
- 저장된 모델 체크포인트와 테스트 feature(HDF5)를 불러와 추론을 실행하고, 예측 결과를 저장하는 스크립트

[실행 예시]
python scripts/run_inference.py --config configs/wifi_multi_field_example.yaml --model experiments/exp001/checkpoints/best_model.pt --test_data experiments/exp001/encoded/features.h5 --output experiments/exp001/outputs/

[주요 흐름]
- config 로딩 → feature 로딩 → 모델 초기화 → 모델 로딩 → 추론 실행 → 예측 결과 저장

[의존성]
- utils/config, utils/io, models 등
"""

import argparse
import os
import numpy as np
import torch
from utils.config import load_config
from utils.io import load_from_h5
from models.set_transformer import SetTransformer
from data.encoders import get_encoder_by_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config yaml 경로')
    parser.add_argument('--model', type=str, required=True, help='모델 체크포인트(.pt) 경로')
    parser.add_argument('--test_data', type=str, required=True, help='테스트 feature HDF5 경로')
    parser.add_argument('--output', type=str, required=True, help='예측 결과 저장 디렉토리')
    parser.add_argument('--batch_size', type=int, default=128, help='추론 배치 크기')
    args = parser.parse_args()

    # Config 로딩
    config = load_config(args.config)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 파일 존재 확인
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data file not found: {args.test_data}")
    
    # Feature 데이터 로딩
    print(f"Loading features from {args.test_data}")
    features = load_from_h5(args.test_data, dataset_name='features')
    print(f"Loaded features shape: {features.shape}")
    
    # 인코더에서 input_dim 계산
    encoder = get_encoder_by_config(config['data']['encoder'])
    input_dim = encoder.get_total_feature_dim()
    
    # 인코더 정보 출력
    encoder_info = encoder.get_encoder_info()
    print(f"Inference - Encoder: {encoder_info['type']}")
    print(f"Inference - MAC encoding: {encoder_info['mac_encoding']}")
    print(f"Inference - Total feature dim: {input_dim}")
    
    # 모델 초기화 (체크포인트와 동일한 구조)
    model_params = config['model'].get('params', {})
    model = SetTransformer(
        input_dim=input_dim,
        output_dim=config['model']['output_dim'],
        task=config['model']['task'],
        **model_params
    )
    
    # 모델 가중치 로딩
    print(f"Loading model from {args.model}")
    try:
        model.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    model.eval()
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # 배치 단위 추론
    batch_size = args.batch_size
    predictions = []
    
    print(f"Starting inference with batch_size={batch_size}")
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_features = features[i:i+batch_size]
            inputs = torch.tensor(batch_features, dtype=torch.float32).to(device)
            
            # 추론 실행
            batch_preds = model(inputs)
            predictions.append(batch_preds.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_features)}/{len(features)} samples")
    
    # 예측 결과 병합
    predictions = np.concatenate(predictions, axis=0)
    print(f"Final predictions shape: {predictions.shape}")
    
    # 예측 결과 저장
    output_file = os.path.join(args.output, 'predictions.csv')
    if config['model']['task'] == 'regression':
        # 회귀: xId, yId
        np.savetxt(output_file, predictions, delimiter=',', 
                  header='xId,yId', comments='')
    else:
        # 분류: class_index
        np.savetxt(output_file, predictions, delimiter=',', 
                  header='class_index', comments='')
    
    print(f"Predictions saved to {output_file}")
    
    # 예측 통계 출력
    if config['model']['task'] == 'regression':
        print(f"Prediction range - xId: [{predictions[:, 0].min():.2f}, {predictions[:, 0].max():.2f}]")
        print(f"Prediction range - yId: [{predictions[:, 1].min():.2f}, {predictions[:, 1].max():.2f}]")
    else:
        print(f"Predicted classes: {np.unique(predictions)}")

if __name__ == '__main__':
    main() 