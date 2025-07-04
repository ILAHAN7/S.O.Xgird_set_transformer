"""
[파일 목적]
- HDF5 파일 로딩/저장, 체크포인트 관리 등 I/O 관련 유틸리티 함수들

[주요 기능]
- HDF5 데이터 로딩/저장
- 체크포인트 저장/로딩
- 메모리 효율적인 chunked loading
- 데이터 검증 및 통계
"""

import h5py
import numpy as np
import torch
import os
from typing import Optional, Tuple, Dict, Any, Generator
import logging

logger = logging.getLogger(__name__)

def load_from_h5(file_path: str, dataset_name: str = 'data') -> np.ndarray:
    """
    HDF5 파일에서 데이터셋을 로딩
    
    Args:
        file_path: HDF5 파일 경로
        dataset_name: 로딩할 데이터셋 이름
    
    Returns:
        로딩된 numpy 배열
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_name not in f:
                raise KeyError(f"Dataset '{dataset_name}' not found in {file_path}")
            data = f[dataset_name][:]
            logger.info(f"Loaded {data.shape} from {file_path}")
            return data
    except Exception as e:
        logger.error(f"Failed to load from {file_path}: {e}")
        raise

def save_to_h5(data: np.ndarray, file_path: str, dataset_name: str = 'data', 
               compression: str = 'gzip', compression_opts: int = 9) -> None:
    """
    numpy 배열을 HDF5 파일로 저장
    
    Args:
        data: 저장할 numpy 배열
        file_path: 저장할 HDF5 파일 경로
        dataset_name: 저장할 데이터셋 이름
        compression: 압축 방식 ('gzip', 'lzf', None)
        compression_opts: 압축 옵션 (gzip의 경우 0-9)
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with h5py.File(file_path, 'w') as f:
            f.create_dataset(dataset_name, data=data, 
                           compression=compression, 
                           compression_opts=compression_opts)
        logger.info(f"Saved {data.shape} to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save to {file_path}: {e}")
        raise

def load_chunked_h5(file_path: str, dataset_name: str = 'data', 
                   chunk_size: int = 1000) -> Generator[np.ndarray, None, None]:
    """
    HDF5 파일을 chunk 단위로 메모리 효율적으로 로딩
    
    Args:
        file_path: HDF5 파일 경로
        dataset_name: 로딩할 데이터셋 이름
        chunk_size: 각 chunk의 크기
    
    Yields:
        chunk 단위의 numpy 배열
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_name not in f:
                raise KeyError(f"Dataset '{dataset_name}' not found in {file_path}")
            
            dataset = f[dataset_name]
            total_size = len(dataset)
            
            for start_idx in range(0, total_size, chunk_size):
                end_idx = min(start_idx + chunk_size, total_size)
                chunk = dataset[start_idx:end_idx]
                yield chunk
                
    except Exception as e:
        logger.error(f"Failed to load chunks from {file_path}: {e}")
        raise

def get_h5_info(file_path: str, dataset_name: str = 'data') -> Dict[str, Any]:
    """
    HDF5 파일의 정보를 반환
    
    Args:
        file_path: HDF5 파일 경로
        dataset_name: 데이터셋 이름
    
    Returns:
        파일 정보 딕셔너리
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_name not in f:
                raise KeyError(f"Dataset '{dataset_name}' not found in {file_path}")
            
            dataset = f[dataset_name]
            info = {
                'shape': dataset.shape,
                'dtype': str(dataset.dtype),
                'size_mb': dataset.nbytes / (1024 * 1024),
                'chunks': dataset.chunks,
                'compression': dataset.compression
            }
            return info
    except Exception as e:
        logger.error(f"Failed to get info from {file_path}: {e}")
        raise

def validate_h5_data(file_path: str, dataset_name: str = 'data', 
                    expected_shape: Optional[Tuple] = None,
                    expected_dtype: Optional[np.dtype] = None) -> bool:
    """
    HDF5 데이터의 유효성을 검증
    
    Args:
        file_path: HDF5 파일 경로
        dataset_name: 데이터셋 이름
        expected_shape: 예상되는 shape (None이면 검증 안함)
        expected_dtype: 예상되는 dtype (None이면 검증 안함)
    
    Returns:
        유효성 검증 결과
    """
    try:
        info = get_h5_info(file_path, dataset_name)
        
        if expected_shape and info['shape'] != expected_shape:
            logger.warning(f"Shape mismatch: expected {expected_shape}, got {info['shape']}")
            return False
        
        if expected_dtype and str(info['dtype']) != str(expected_dtype):
            logger.warning(f"Dtype mismatch: expected {expected_dtype}, got {info['dtype']}")
            return False
        
        # NaN/Inf 체크 (첫 번째 chunk만)
        with h5py.File(file_path, 'r') as f:
            dataset = f[dataset_name]
            sample = dataset[0:min(1000, len(dataset))]
            if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
                logger.warning("Found NaN or Inf values in data")
                return False
        
        logger.info(f"Data validation passed for {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed for {file_path}: {e}")
        return False

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: str = 'cpu') -> Dict[str, Any]:
    """
    모델 체크포인트를 로딩
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        model: 로딩할 모델
        optimizer: 로딩할 옵티마이저 (선택사항)
        device: 디바이스
    
    Returns:
        체크포인트 정보 딕셔너리
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, checkpoint_path: str,
                   additional_info: Optional[Dict[str, Any]] = None) -> None:
    """
    모델 체크포인트를 저장
    
    Args:
        model: 저장할 모델
        optimizer: 저장할 옵티마이저
        epoch: 현재 epoch
        loss: 현재 loss
        checkpoint_path: 저장할 체크포인트 경로
        additional_info: 추가 정보 (선택사항)
    """
    try:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
        raise 