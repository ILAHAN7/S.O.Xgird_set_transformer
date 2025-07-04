"""
[파일 목적]
- HDF5 파일에서 feature/label을 로딩하는 PyTorch Dataset 클래스
- 메모리 효율적인 chunked loading 지원
- 스레드 안전성 보장

[주요 기능]
- HDF5 파일에서 feature/label 동시 로딩
- 스레드 안전한 파일 접근
- 메모리 효율적인 데이터 로딩
- 데이터 검증 및 통계
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import threading
import logging
from typing import Optional, Tuple, Dict, Any
from utils.io import get_h5_info, validate_h5_data

logger = logging.getLogger(__name__)

class HDF5Dataset(Dataset):
    """
    HDF5 파일에서 feature/label을 로딩하는 PyTorch Dataset
    
    스레드 안전성을 위해 각 스레드에서 별도의 파일 핸들을 사용
    """
    
    def __init__(self, features_path: str, labels_path: str, 
                 feature_name: str = 'features', label_name: str = 'labels',
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None):
        """
        Args:
            features_path: feature HDF5 파일 경로
            labels_path: label HDF5 파일 경로
            feature_name: feature 데이터셋 이름
            label_name: label 데이터셋 이름
            transform: feature 변환 함수
            target_transform: label 변환 함수
        """
        self.features_path = features_path
        self.labels_path = labels_path
        self.feature_name = feature_name
        self.label_name = label_name
        self.transform = transform
        self.target_transform = target_transform
        
        # 파일 정보 검증
        self._validate_files()
        
        # 데이터 크기 확인
        self._get_data_info()
        
        # 스레드별 파일 핸들 관리를 위한 로컬 스토리지
        self._local = threading.local()
        
        logger.info(f"HDF5Dataset initialized: {self.features_path}, {self.labels_path}")
        logger.info(f"Dataset size: {len(self)}")
    
    def _validate_files(self):
        """파일 존재 여부와 데이터 유효성 검증"""
        if not validate_h5_data(self.features_path, self.feature_name):
            raise ValueError(f"Invalid feature file: {self.features_path}")
        if not validate_h5_data(self.labels_path, self.label_name):
            raise ValueError(f"Invalid label file: {self.labels_path}")
    
    def _get_data_info(self):
        """데이터 크기와 정보 확인"""
        feature_info = get_h5_info(self.features_path, self.feature_name)
        label_info = get_h5_info(self.labels_path, self.label_name)
        
        if feature_info['shape'][0] != label_info['shape'][0]:
            raise ValueError(f"Feature and label sizes don't match: "
                           f"{feature_info['shape'][0]} vs {label_info['shape'][0]}")
        
        self._size = feature_info['shape'][0]
        self._feature_shape = feature_info['shape'][1:]
        self._label_shape = label_info['shape'][1:]
        
        logger.info(f"Feature shape: {self._feature_shape}")
        logger.info(f"Label shape: {self._label_shape}")
    
    def _get_file_handles(self):
        """스레드별 파일 핸들 반환 (스레드 안전)"""
        if not hasattr(self._local, 'feature_file'):
            self._local.feature_file = h5py.File(self.features_path, 'r')
            self._local.label_file = h5py.File(self.labels_path, 'r')
        
        return self._local.feature_file, self._local.label_file
    
    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인덱스에 해당하는 feature/label 반환
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            (feature, label) 튜플
        """
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Index {idx} out of range [0, {self._size})")
        
        try:
            # 스레드별 파일 핸들 사용
            feature_file, label_file = self._get_file_handles()
            
            # 데이터 로딩
            feature = feature_file[self.feature_name][idx]
            label = label_file[self.label_name][idx]
            
            # numpy 배열로 변환
            feature = np.asarray(feature, dtype=np.float32)
            label = np.asarray(label, dtype=np.float32)
            
            # 변환 함수 적용
            if self.transform:
                feature = self.transform(feature)
            if self.target_transform:
                label = self.target_transform(label)
            
            # torch tensor로 변환
            feature = torch.tensor(feature, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            
            return feature, label
            
        except Exception as e:
            logger.error(f"Error loading data at index {idx}: {e}")
            raise
    
    def get_feature_shape(self) -> Tuple[int, ...]:
        """feature shape 반환"""
        return self._feature_shape
    
    def get_label_shape(self) -> Tuple[int, ...]:
        """label shape 반환"""
        return self._label_shape
    
    def get_data_info(self) -> Dict[str, Any]:
        """데이터셋 정보 반환"""
        return {
            'size': self._size,
            'feature_shape': self._feature_shape,
            'label_shape': self._label_shape,
            'feature_path': self.features_path,
            'label_path': self.labels_path
        }
    
    def __del__(self):
        """소멸자: 파일 핸들 정리"""
        if hasattr(self._local, 'feature_file'):
            try:
                self._local.feature_file.close()
            except:
                pass
        if hasattr(self._local, 'label_file'):
            try:
                self._local.label_file.close()
            except:
                pass

class ChunkedHDF5Dataset(Dataset):
    """
    메모리 효율적인 chunked HDF5 데이터셋
    대용량 데이터를 chunk 단위로 로딩하여 메모리 사용량을 줄임
    LRU 캐시 기반으로 메모리 효율성 및 스레드 안전성 강화
    """
    def __init__(self, features_path: str, labels_path: str,
                 feature_name: str = 'features', label_name: str = 'labels',
                 chunk_size: int = 1000, max_chunks_in_memory: int = 2,
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None):
        self.features_path = features_path
        self.labels_path = labels_path
        self.feature_name = feature_name
        self.label_name = label_name
        self.chunk_size = chunk_size
        self.max_chunks_in_memory = max_chunks_in_memory
        self.transform = transform
        self.target_transform = target_transform
        self._validate_files()
        self._get_data_info()
        self._local = threading.local()
        self._lock = threading.RLock()
        self.chunk_cache = {}
        self.access_order = []
        logger.info(f"ChunkedHDF5Dataset initialized with chunk_size={chunk_size}, max_chunks_in_memory={max_chunks_in_memory}")

    def _load_chunk_from_file(self, chunk_idx: int):
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self._size)
        feature_file, label_file = self._get_file_handles()
        features = feature_file[self.feature_name][start_idx:end_idx]
        labels = label_file[self.label_name][start_idx:end_idx]
        return (features, labels)

    def _load_chunk_with_lru_cache(self, chunk_idx: int):
        with self._lock:
            if chunk_idx in self.chunk_cache:
                self.access_order.remove(chunk_idx)
                self.access_order.append(chunk_idx)
                return self.chunk_cache[chunk_idx]
            chunk_data = self._load_chunk_from_file(chunk_idx)
            if len(self.chunk_cache) >= self.max_chunks_in_memory:
                oldest_chunk = self.access_order.pop(0)
                del self.chunk_cache[oldest_chunk]
            self.chunk_cache[chunk_idx] = chunk_data
            self.access_order.append(chunk_idx)
            return chunk_data

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Index {idx} out of range")
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        features, labels = self._load_chunk_with_lru_cache(chunk_idx)
        feature = features[local_idx]
        label = labels[local_idx]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return feature, label

    def __del__(self):
        if hasattr(self._local, 'feature_file'):
            try:
                self._local.feature_file.close()
            except:
                pass
        if hasattr(self._local, 'label_file'):
            try:
                self._local.label_file.close()
            except:
                pass 