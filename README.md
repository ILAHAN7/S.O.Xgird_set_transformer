# Sox_Settransformer

**무선 신호 기반 위치 추정(SetTransformer 기반) 대용량 실험/재현성/확장성 중심 파이프라인**  
**WiFi-based Location Estimation Pipeline with SetTransformer for Large-scale Experiments, Reproducibility, and Scalability**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Core Modules](#core-modules)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Advanced Features & Extensibility](#advanced-features--extensibility)
- [Reference](#reference)

---

## Project Overview

### 목적 / Purpose

- WiFi/LTE 등 무선 신호로부터 위치(그리드셀) 예측  
  Predict location (grid cell) from wireless signals like WiFi/LTE
- 대용량 데이터, 병렬 인코딩, HDF5 기반 효율적 실험  
  Efficient large-scale experiments with parallel encoding and HDF5-based pipeline
- 완전한 config 기반, 실험별 reproducibility, 확장성  
  Complete config-driven approach with experiment reproducibility and scalability

### 주요 특징 / Key Features

- **대용량 처리:** HDF5 + 병렬 인코딩으로 대용량 데이터 처리  
  Large-scale Processing: HDF5 + parallel encoding for massive data handling
- **재현성:** Config 기반 실험으로 완전한 재현성 보장  
  Reproducibility: Config-driven experiments ensuring complete reproducibility
- **확장성:** 모듈화된 구조로 새로운 인코더/모델 추가 용이  
  Scalability: Modular structure for easy addition of new encoders/models
- **견고성:** 에러 처리 및 로깅 시스템  
  Robustness: Error handling and logging system

---

## Architecture

### 디렉토리 구조 / Directory Structure

```
Sox_Settransformer/
├── configs/                # 실험/모델/데이터 설정 YAML
│   └── wifi_multi_field_example.yaml
├── data/                   # 데이터 로딩/인코딩/그리드셀 변환/HDF5Dataset
│   ├── base.py
│   ├── db_reader.py
│   ├── latlon_to_gridcell.py
│   ├── hdf5_dataset.py
│   └── encoders/
│       ├── __init__.py
│       └── wifi_multi_field.py
├── models/
│   ├── base.py
│   ├── set_transformer.py
│   └── __init__.py
├── training/
│   ├── trainer.py
│   ├── metrics.py
│   └── __init__.py
├── utils/
│   ├── config.py
│   ├── io.py
│   └── __init__.py
├── scripts/
│   ├── encode_data.py
│   ├── merge_hdf5_parts.py
│   ├── train_model.py
│   ├── run_inference.py
│   └── evaluate_model.py
├── experiments/
├── requirements.txt
└── README.md
```

---

### 데이터 플로우 / Data Flow

1. **Database → Chunk Loading**
2. **Parallel Encoding → HDF5 Part Files**
3. **Part Merging → Unified HDF5**
4. **HDF5Dataset + DataLoader → Training**
5. **Model Checkpoints → Inference/Evaluation**

---

## Core Modules

### 1. WiFi Multi-Field Encoder

```python
class WiFiMultiFieldEncoder(BaseEncoder):
    def __init__(self, max_mac_count=32, max_ipcikey_count=8, max_icellid_count=4):
        ...

    def _mac_to_bytes(self, mac):
        # "00:40:5a:af:bc:4a" → [0, 64, 90, 175, 188, 74]
        ...

    def _ipcikey_to_vec(self, key):
        # '158_1350_2_0' → [158,1350,2,0]
        ...

    def encode(self, batch_data):
        ...
```

- **MAC 주소 인코딩:** 16진수 문자열을 6바이트 정수로 변환  
  MAC Address Encoding: Convert hex string to 6-byte integers
- **RSSI 정규화:** (wrssi+100)/70으로 [-100, -30] 범위를 [0, 1]로 변환  
  RSSI Normalization: Transform [-100, -30] range to [0, 1]
- **ipcikey 분해:** '_' 구분자로 4차원 벡터로 분해  
  ipcikey Decomposition: Split by '_' into 4D vector
- **icellid 처리:** 정수값 그대로 사용  
  icellid Processing: Use integer values directly
- **패딩 처리:** 최대 개수에 맞춰 0으로 패딩  
  Padding: Zero-padding to match maximum counts

---

### 2. HDF5 Dataset

```python
class HDF5Dataset(Dataset):
    def __init__(self, features_path, labels_path, feature_name='features', label_name='labels'):
        ...
    def _ensure_open(self):
        ...
    def __getitem__(self, idx):
        ...
```

- **Persistent Handle:** 파일 핸들을 한 번만 열고 재사용  
  Open file handles once and reuse
- **Threading Lock:** 멀티스레딩 환경에서 안전한 파일 접근  
  Safe file access in multi-threading environment
- **On-demand Loading:** 필요할 때만 데이터 로딩  
  Load data only when needed
- **PyTorch 호환:** DataLoader와 완전 호환  
  PyTorch Compatible: Fully compatible with DataLoader

---

### 3. SetTransformer Model

```python
class SetTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=4, num_inds=32, 
                 num_enc_layers=2, num_dec_layers=2, dropout=0.0, 
                 output_dim=2, task='regression'):
        ...
    def forward(self, X):
        ...
```

- **인코더:** ISAB 블록들로 구성 (집합 요소 간 상호작용 모델링)  
  Encoder: ISAB blocks for modeling interactions between set elements
- **디코더:** PMA + SAB 블록들 + MLP (고정 크기 출력 생성)  
  Decoder: PMA + SAB blocks + MLP for fixed-size output
- **Task 분기:** 회귀(2D 좌표) vs 분류(클래스)  
  Task Branching: Regression (2D coordinates) vs Classification (classes)
- **순열 불변성:** 집합의 순서에 무관한 처리  
  Permutation Invariance: Order-independent set processing

---

### 4. Universal Trainer

```python
class UniversalTrainer:
    def __init__(self, model, config):
        ...
    def train(self, dataloader, exp_dir):
        ...
```

- **Task별 Loss:** 회귀(MSE) vs 분류(CrossEntropy) 자동 선택  
  Task-specific Loss: Regression (MSE) or Classification (CrossEntropy)
- **Early Stopping:** 검증 손실 기반 조기 종료  
  Early Stopping: Based on validation loss
- **Checkpoint Management:** best/last/epoch별 모델 저장  
  Model saving for best/last/epoch
- **Reproducibility:** 시드 설정  
  Reproducibility guarantee through seed setting
- **Device 관리:** GPU/CPU 자동 감지  
  Automatic GPU/CPU detection and allocation

---

## Usage Guide

### 1. 데이터 인코딩 / Data Encoding

```bash
# 1. 데이터 인코딩 (병렬 처리)
python scripts/encode_data.py --config configs/wifi_multi_field_example.yaml --output experiments/exp001/encoded/
python scripts/{인코더 스크립트명/endcoder python file name} --config configs/{사용할 설정파일명/the name of config file to use} --output {결과 출력 경로/directory to output}

# 2. 파트 파일 병합
python scripts/merge_hdf5_parts.py --input_dir experiments/exp001/encoded/ --output_dir experiments/exp001/merged/
python scripts/merge_hdf5_parts.py --input_dir {입력 파일 경로/directory to pull input} --output_dir {결과 출력 경로/directory to output}
- **Step 1:** DB에서 청크 단위로 데이터를 읽어 병렬 인코딩하여 part-*.h5 파일로 저장 

  Read data from DB in chunks, parallel encode, and save as part-*.h5 files
- **Step 2:** part 파일들을 하나의 features.h5, labels.h5로 병합  
  Merge part files into single features.h5 and labels.h5

---

### 2. 모델 학습 / Model Training

```bash
python scripts/train_model.py --config configs/wifi_multi_field_example.yaml --experiment exp001 --encoded_data experiments/exp001/merged/
```
- **Large-scale batch training** with HDF5Dataset + DataLoader  
- **UniversalTrainer** directly uses DataLoader  
- **Automatic checkpoint saving and early stopping**

---

### 3. 추론 및 평가 / Inference and Evaluation

```bash
# 추론 실행
python scripts/run_inference.py --model experiments/exp001/checkpoints/best_model.pt --test_data experiments/exp001/merged/features.h5 --output experiments/exp001/outputs/

# 모델 평가
python scripts/evaluate_model.py --config configs/wifi_multi_field_example.yaml --predictions experiments/exp001/outputs/predictions.csv --labels experiments/exp001/merged/labels.h5 --output experiments/exp001/reports/
```
- **Inference:** Perform predictions on test data using saved model  
- **Evaluation:** Calculate performance metrics by comparing predictions with actual labels

---

## Configuration

### Example (YAML)

```yaml
data:
  db:
    host: localhost
    port: 3306
    user: user
    password: pw
    database: wifi
    table: signals
    id_col: id
    chunk_size: 100000
  encoder:
    type: WiFiMultiFieldEncoder
    params:
      max_mac_count: 32
      max_ipcikey_count: 8
      max_icellid_count: 4
  gridcell:
    level: 5
  x_max: 1000

model:
  name: set_transformer
  task: regression
  input_dim: 7
  output_dim: 2
  params:
    hidden_dim: 256
    num_heads: 4
    num_inds: 32
    num_enc_layers: 2
    num_dec_layers: 2
    dropout: 0.0

training:
  batch_size: 128
  epochs: 50
  optimizer: adam
  lr: 0.001
  weight_decay: 0.0
  seed: 42
  early_stopping: 10
  save_all_epochs: true
  save_every: 10

experiment:
  name: wifi_multi_field_exp001
  output_dir: experiments/wifi_multi_field_exp001/
```

### Configuration Item Descriptions

- **db:** Database connection info
- **encoder:** Data encoder type and parameters
- **gridcell:** Grid cell resolution
- **x_max:** Number of classes (for classification)
- **model:** Model type, task, and hyperparameters
- **training:** Training batch size, epochs, learning rate, etc.

### 주요 설정 옵션 설명 / Key Option Explanations

- `hidden_dim`: Hidden dimension (e.g., 128, 256)
- `num_heads`: Number of attention heads (recommend power of 2)
- `num_inds`: Number of induced points for ISAB
- `num_enc_layers`/`num_dec_layers`: Number of encoder/decoder layers
- `dropout`: Dropout rate (0.0~0.3)
- `max_mac_count`, `max_ipcikey_count`, `max_icellid_count`: Maximum counts for each feature
- `chunk_size`, `max_chunks_in_memory`: Chunk size and LRU cache size (for large datasets)
- `batch_size`, `epochs`, `optimizer`, `lr`, `early_stopping`: Standard training options

---

## Advanced Features & Extensibility

- **LayerNorm/Dropout**: Enable with `ln: true`, `dropout: 0.1` in model params
- **ChunkedHDF5Dataset LRU cache**:  
  Set in data config for large datasets
- **Add new encoders/models**:  
  Implement new classes in `data/encoders/` or `models/`, register in factory/init, and update config

### Experiment Reproducibility & Extensibility

- All experiment settings are managed via config files for full reproducibility.
- Different encoder/model/data combinations can be tested by just changing the config.
- Results, checkpoints, logs are saved in per-experiment directories.

---

## Reference

Based on the official PyTorch implementation of:  
[Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks](https://github.com/juho-lee/set_transformer)

---
