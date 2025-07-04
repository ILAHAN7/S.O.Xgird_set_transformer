"""
[파일 목적]
- 인코딩된 feature/label(HDF5)을 로딩하여 모델 학습을 수행하는 전체 파이프라인 스크립트

[실행 예시]
python scripts/train_model.py --config configs/wifi_hash.yaml --experiment exp001_wifi_hash --encoded_data experiments/exp001/encoded/

[주요 흐름]
- config 로딩 → feature/label HDF5 로딩 → 모델/트레이너 초기화 → 학습 loop → 체크포인트/로그 저장

[의존성]
- utils/config, utils/io, models, training 등
"""

import argparse
import os
import numpy as np
import torch
from utils.config import load_config
from utils.io import load_from_h5
from models.set_transformer import SetTransformer
from training.trainer import UniversalTrainer
from torch.utils.data import DataLoader
from data.hdf5_dataset import HDF5Dataset
from data.encoders import get_encoder_by_config

def _normalize_model_params(params):
    """기존 config 호환성을 위한 파라미터 정규화"""
    normalized = params.copy()
    param_mapping = {
        'dim_hidden': 'hidden_dim',
        'num_enc_layers': 'num_enc_layers',
        'num_dec_layers': 'num_dec_layers'
    }
    for old_name, new_name in param_mapping.items():
        if old_name in normalized:
            normalized[new_name] = normalized.pop(old_name)
    defaults = {
        'hidden_dim': 128,
        'num_heads': 4,
        'num_inds': 32,
        'num_enc_layers': 2,
        'num_dec_layers': 2,
        'dropout': 0.0,
        'ln': True
    }
    for key, default_value in defaults.items():
        if key not in normalized:
            normalized[key] = default_value
    return normalized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config yaml 경로')
    parser.add_argument('--experiment', type=str, required=True, help='실험명')
    parser.add_argument('--encoded_data', type=str, required=True, help='인코딩 데이터 디렉토리')
    args = parser.parse_args()

    # Config 로딩 및 검증
    config = load_config(args.config)
    
    # 실험 디렉토리 생성
    exp_dir = os.path.join('experiments', args.experiment)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)

    # 데이터 로딩
    features_path = os.path.join(args.encoded_data, 'features.h5')
    labels_path = os.path.join(args.encoded_data, 'labels.h5')
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    dataset = HDF5Dataset(features_path, labels_path, feature_name='features', label_name='labels')
    
    # 인코더에서 직접 feature dimension 가져오기
    encoder = get_encoder_by_config(config['data']['encoder'])
    input_dim = encoder.get_total_feature_dim()
    
    # 인코더 정보 출력 (디버깅용)
    encoder_info = encoder.get_encoder_info()
    print(f"Encoder: {encoder_info['type']}")
    print(f"MAC encoding: {encoder_info['mac_encoding']}")
    print(f"Total feature dim: {input_dim}")
    print(f"Feature dims: {encoder_info['feature_dims']}")
    
    # 샘플 데이터로 검증
    sample_feature, sample_label = dataset[0]
    print(f"Sample feature shape: {sample_feature.shape}")
    print(f"Sample label shape: {sample_label.shape}")
    
    # DataLoader 설정
    batch_size = config['training'].get('batch_size', 64)
    shuffle = config['training'].get('shuffle', True)
    num_workers = config['training'].get('num_workers', 0)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )

    # 모델 초기화
    model_params = config['model'].get('params', {})
    model_params = _normalize_model_params(model_params)  # 호환성 보장
    if 'input_dim' in model_params:
        del model_params['input_dim']
    model = SetTransformer(
        input_dim=input_dim,
        output_dim=config['model']['output_dim'],
        task=config['model']['task'],
        **model_params
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 트레이너 초기화
    trainer = UniversalTrainer(model, config)

    # 학습 실행
    print(f"Starting training for {config['training']['epochs']} epochs...")
    trainer.train(dataloader, exp_dir=exp_dir)
    print("Training completed!")

if __name__ == '__main__':
    main() 