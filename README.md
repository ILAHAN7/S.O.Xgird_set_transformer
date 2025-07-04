Sox_Settransformer
무선 신호 기반 위치 추정(SetTransformer 기반) 대용량 실험/재현성/확장성 중심 파이프라인

WiFi-based Location Estimation Pipeline with SetTransformer for Large-scale Experiments, Reproducibility, and Scalability

목차 / Table of Contents
프로젝트 개요 / Project Overview
아키텍처 / Architecture
핵심 모듈 / Core Modules
실행 가이드 / Usage Guide
설정 파일 / Configuration
프로젝트 개요 / Project Overview
목적 / Purpose
WiFi/LTE 등 무선 신호로부터 위치(그리드셀) 예측
대용량 데이터, 병렬 인코딩, HDF5 기반 효율적 실험
완전한 config 기반, 실험별 reproducibility, 확장성
Purpose:

Predict location (grid cell) from wireless signals like WiFi/LTE
Efficient large-scale experiments with parallel encoding and HDF5-based pipeline
Complete config-driven approach with experiment reproducibility and scalability
주요 특징 / Key Features
대용량 처리: HDF5 + 병렬 인코딩으로 대용량 데이터 처리
재현성: Config 기반 실험으로 완전한 재현성 보장
확장성: 모듈화된 구조로 새로운 인코더/모델 추가 용이
견고성: 에러 처리 및 로깅 시스템
Key Features:

Large-scale Processing: HDF5 + parallel encoding for massive data handling
Reproducibility: Config-driven experiments ensuring complete reproducibility
Scalability: Modular structure for easy addition of new encoders/models
Robustness: Error handling and logging system
아키텍처 / Architecture
디렉토리 구조 / Directory Structure
Sox_Settransformer/
├── configs/                # 실험/모델/데이터 설정 YAML
│   └── wifi_multi_field_example.yaml
├── data/                   # 데이터 로딩/인코딩/그리드셀 변환/HDF5Dataset
│   ├── base.py            # BaseEncoder 추상 클래스
│   ├── db_reader.py       # MariaDB 데이터 로더
│   ├── latlon_to_gridcell.py  # 위경도→그리드셀 변환
│   ├── hdf5_dataset.py    # 대용량 HDF5 데이터셋
│   └── encoders/
│       ├── __init__.py    # 인코더 팩토리
│       └── wifi_multi_field.py  # WiFi 다중 필드 인코더
├── models/                 # SetTransformer 및 관련 모델
│   ├── base.py            # BaseModel 추상 클래스
│   ├── set_transformer.py # SetTransformer 구현체
│   └── __init__.py
├── training/               # Trainer, metrics, utils
│   ├── trainer.py         # UniversalTrainer
│   ├── metrics.py         # 평가 지표 함수들
│   └── __init__.py
├── utils/                  # config/io/logging 등 유틸
│   ├── config.py          # YAML 설정 로더
│   ├── io.py              # HDF5 입출력 함수
│   └── __init__.py
├── scripts/                # 실행 스크립트
│   ├── encode_data.py     # DB→인코딩→HDF5 (병렬/part 지원)
│   ├── merge_hdf5_parts.py # part-*.h5 병합
│   ├── train_model.py     # HDF5 기반 학습
│   ├── run_inference.py   # 추론/예측
│   └── evaluate_model.py  # 평가/리포트
├── experiments/            # 실험별 결과/체크포인트/로그
├── requirements.txt        # 의존성 패키지
└── README.md
데이터 플로우 / Data Flow
1. Database → Chunk Loading
   ↓
2. Parallel Encoding → HDF5 Part Files
   ↓
3. Part Merging → Unified HDF5
   ↓
4. HDF5Dataset + DataLoader → Training
   ↓
5. Model Checkpoints → Inference/Evaluation
핵심 모듈 / Core Modules
1. WiFi 다중 필드 인코더 / WiFi Multi-Field Encoder
class WiFiMultiFieldEncoder(BaseEncoder):
    def __init__(self, max_mac_count=32, max_ipcikey_count=8, max_icellid_count=4):
        self.max_mac_count = max_mac_count
        self.max_ipcikey_count = max_ipcikey_count
        self.max_icellid_count = max_icellid_count

    def _mac_to_bytes(self, mac):
        # "00:40:5a:af:bc:4a" → [0, 64, 90, 175, 188, 74]
        return [int(b, 16) for b in mac.split(':')]

    def _ipcikey_to_vec(self, key):
        # '158_1350_2_0' → [158,1350,2,0]
        try:
            return [int(x) for x in key.split('_')]
        except Exception:
            return [0,0,0,0]

    def encode(self, batch_data):
        # WiFi MAC: "00:40:5a:af:bc:4a" → [0,64,90,175,188,74] (6D)
        # WiFi RSSI: (wrssi+100)/70 normalization (1D)
        # ipcikey: "158_1350_2_0" → [158,1350,2,0] (4D)
        # icellid: integer value (1D)
        
        # Output: (batch, max_mac+max_ipci+max_icellid, 7/4/1)
기능 / Functionality:

MAC 주소 인코딩: 16진수 문자열을 6바이트 정수로 변환
RSSI 정규화: (wrssi+100)/70으로 [-100, -30] 범위를 [0, 1]로 변환
ipcikey 분해: '_' 구분자로 4차원 벡터로 분해
icellid 처리: 정수값 그대로 사용
패딩 처리: 최대 개수에 맞춰 0으로 패딩
Features:

MAC Address Encoding: Convert hex string to 6-byte integers
RSSI Normalization: Transform [-100, -30] range to [0, 1] via (wrssi+100)/70
ipcikey Decomposition: Split by '_' into 4D vector
icellid Processing: Use integer values directly
Padding: Zero-padding to match maximum counts
2. HDF5 데이터셋 / HDF5 Dataset
class HDF5Dataset(Dataset):
    def __init__(self, features_path, labels_path, feature_name='features', label_name='labels'):
        self.features_path = features_path
        self.labels_path = labels_path
        self.feature_name = feature_name
        self.label_name = label_name
        # 파일 열어서 shape만 파악
        with h5py.File(self.features_path, 'r') as f:
            self.length = f[self.feature_name].shape[0]
        self._features_h5 = None
        self._labels_h5 = None
        self._lock = threading.Lock()

    def _ensure_open(self):
        # 각 worker마다 파일 핸들을 한 번만 열고 재사용
        if self._features_h5 is None or self._labels_h5 is None:
            with self._lock:
                if self._features_h5 is None:
                    self._features_h5 = h5py.File(self.features_path, 'r')
                if self._labels_h5 is None:
                    self._labels_h5 = h5py.File(self.labels_path, 'r')

    def __getitem__(self, idx):
        self._ensure_open()
        x = self._features_h5[self.feature_name][idx]
        y = self._labels_h5[self.label_name][idx]
        return x, y
기능 / Functionality:

Persistent Handle: 파일 핸들을 한 번만 열고 재사용
Threading Lock: 멀티스레딩 환경에서 안전한 파일 접근
On-demand Loading: 필요할 때만 데이터 로딩
PyTorch 호환: DataLoader와 완전 호환
Features:

Persistent Handle: Open file handles once and reuse
Threading Lock: Safe file access in multi-threading environment
On-demand Loading: Load data only when needed
PyTorch Compatible: Fully compatible with DataLoader
3. SetTransformer 모델 / SetTransformer Model
class SetTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_heads=4, num_inds=32, 
                 num_enc_layers=2, num_dec_layers=2, dropout=0.0, 
                 output_dim=2, task='regression'):
        super().__init__()
        # Encoder: 여러 SAB/ISAB
        enc_blocks = []
        enc_blocks.append(ISAB(input_dim, hidden_dim, num_heads, num_inds, dropout))
        for _ in range(num_enc_layers - 1):
            enc_blocks.append(ISAB(hidden_dim, hidden_dim, num_heads, num_inds, dropout))
        self.encoder = nn.Sequential(*enc_blocks)
        
        # Decoder: PMA + (SAB × num_dec_layers) + MLP
        dec_blocks = [PMA(hidden_dim, num_heads, 1, dropout)]
        for _ in range(num_dec_layers):
            dec_blocks.append(SAB(hidden_dim, hidden_dim, num_heads, dropout))
        self.decoder = nn.Sequential(*dec_blocks)
        
        self.task = task
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, X):
        # X: (batch, set_size, feature_dim)
        H = self.encoder(X)
        H = self.decoder(H)
        H = H.squeeze(1)  # (batch, hidden_dim)
        out = self.mlp(H)  # (batch, output_dim)
        if self.task == 'classification':
            return out  # (batch, num_classes)
        else:
            return out  # (batch, 2)
아키텍처 특징 / Architectural Features:

인코더: ISAB 블록들로 구성 (집합 요소 간 상호작용 모델링)
디코더: PMA + SAB 블록들 + MLP (고정 크기 출력 생성)
Task 분기: 회귀(2D 좌표) vs 분류(클래스) 자동 분기
순열 불변성: 집합의 순서에 무관한 처리
Architecture Features:

Encoder: ISAB blocks for modeling interactions between set elements
Decoder: PMA + SAB blocks + MLP for fixed-size output generation
Task Branching: Automatic branching for regression (2D coordinates) vs classification (classes)
Permutation Invariance: Order-independent set processing
4. UniversalTrainer / Universal Trainer
class UniversalTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.lr = config['training']['lr']
        self.task = config['model'].get('task', 'regression')
        
        if self.task == 'classification':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        seed = config['training'].get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, dataloader, exp_dir):
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            n_samples = 0
            
            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
            
            epoch_loss /= n_samples
            print(f"Epoch {epoch}/{self.epochs} - Loss: {epoch_loss:.4f}")
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
                torch.save(self.model.state_dict(), os.path.join(exp_dir, 'checkpoints', 'best_model.pt'))
            else:
                patience += 1
            
            if patience >= self.early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Best loss: {best_loss:.4f}")
기능 / Functionality:

Task별 Loss: 회귀(MSE) vs 분류(CrossEntropy) 자동 선택
Early Stopping: 검증 손실 기반 조기 종료
Checkpoint Management: best/last/epoch별 모델 저장
Reproducibility: 시드 설정으로 재현성 보장
Device 관리: GPU/CPU 자동 감지 및 할당
Features:

Task-specific Loss: Automatic selection between regression (MSE) and classification (CrossEntropy)
Early Stopping: Early termination based on validation loss
Checkpoint Management: Model saving for best/last/epoch
Reproducibility: Reproducibility guarantee through seed setting
Device Management: Automatic GPU/CPU detection and allocation
실행 가이드 / Usage Guide
1. 데이터 인코딩 / Data Encoding
# 1. 데이터 인코딩 (병렬 처리)
python scripts/encode_data.py --config configs/wifi_multi_field_example.yaml --output experiments/exp001/encoded/

# 2. 파트 파일 병합
python scripts/merge_hdf5_parts.py --input_dir experiments/exp001/encoded/ --output_dir experiments/exp001/merged/
설명 / Description:

1단계: DB에서 청크 단위로 데이터를 읽어 병렬 인코딩하여 part-*.h5 파일로 저장
2단계: part 파일들을 하나의 features.h5, labels.h5로 병합
Explanation:

Step 1: Read data from DB in chunks, parallel encode, and save as part-*.h5 files
Step 2: Merge part files into single features.h5 and labels.h5
2. 모델 학습 / Model Training
# 모델 학습
python scripts/train_model.py --config configs/wifi_multi_field_example.yaml --experiment exp001 --encoded_data experiments/exp001/merged/
설명 / Description:

HDF5Dataset + DataLoader로 대용량 데이터 배치 단위 학습
UniversalTrainer가 DataLoader 직접 사용
체크포인트 자동 저장 및 조기 종료
Explanation:

Large-scale data batch training with HDF5Dataset + DataLoader
UniversalTrainer directly uses DataLoader
Automatic checkpoint saving and early stopping
3. 추론 및 평가 / Inference and Evaluation
# 추론 실행
python scripts/run_inference.py --model experiments/exp001/checkpoints/best_model.pt --test_data experiments/exp001/merged/features.h5 --output experiments/exp001/outputs/

# 모델 평가
python scripts/evaluate_model.py --config configs/wifi_multi_field_example.yaml --predictions experiments/exp001/outputs/predictions.csv --labels experiments/exp001/merged/labels.h5 --output experiments/exp001/outputs/
설명 / Description:

추론: 저장된 모델로 테스트 데이터에 대한 예측 수행
평가: 예측 결과와 실제 라벨을 비교하여 성능 지표 계산
Explanation:

Inference: Perform predictions on test data using saved model
Evaluation: Calculate performance metrics by comparing predictions with actual labels
설정 파일 / Configuration
설정 예시 / Configuration Example
data:
  db:
    host: localhost         # DB 주소
    port: 3306              # DB 포트
    user: user              # DB 계정
    password: pw            # DB 비밀번호
    database: wifi          # DB명
    table: signals          # 테이블명
    id_col: id              # 프라이머리 키 컬럼명, 그 각 행을 구별 해줄 고유 숫자 번호가 부여된 열이름름
    chunk_size: 100000      # chunk 크기 - DB가 너무 크면 chunk로 나누어 로드하는데 이때 chunk가 담을 데이터 크기기
  encoder:
    type: WiFiMultiFieldEncoder
    params:
      max_mac_count: 32          # WiFi MAC+RSSI 최대 개수
      max_ipcikey_count: 8       # ipcikey 최대 개수
      max_icellid_count: 4       # icellid 최대 개수
  gridcell:
    level: 5   # 25m 셀
  x_max: 1000                    # 분류 task에서 클래스 개수 계산용

model:
  name: set_transformer
  task: regression               # regression/classification
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
설정 항목 설명 / Configuration Item Descriptions:

데이터 설정 / Data Configuration
db: 데이터베이스 연결 정보
encoder: 데이터 인코더 타입 및 파라미터
gridcell: 그리드셀 해상도 설정
x_max: 분류 작업에서 클래스 수 계산용
모델 설정 / Model Configuration
name: 사용할 모델 타입
task: 회귀 또는 분류 작업
params: 모델 하이퍼파라미터
학습 설정 / Training Configuration
batch_size: 배치 크기
epochs: 학습 에포크 수
lr: 학습률
early_stopping: 조기 종료 patience
seed: 재현성을 위한 랜덤 시드
설정 옵션별 상세 설명 / Configuration Option Details
주요 설정 옵션 설명
hidden_dim

SetTransformer의 히든 차원
예시: 128, 256
주의: 값이 클수록 성능은 좋아지나 메모리 사용량이 증가합니다.
num_heads

Multihead Attention의 head 수
예시: 4, 8
주의: 2의 배수 사용 권장
num_inds

ISAB의 induced point 수
예시: 16, 32
주의: 집합 크기와 비슷하게 설정
num_enc_layers / num_dec_layers

인코더(ISAB)/디코더(SAB) 레이어 수
예시: 2
주의: 너무 많으면 과적합 위험
dropout

Dropout 비율(과적합 방지)
예시: 0.0 ~ 0.3
주의: 너무 높으면 underfitting 발생
ln

LayerNorm 사용 여부
예시: true/false
주의: 안정성↑, 속도 약간 감소
max_mac_count / max_ipcikey_count / max_icellid_count

WiFi/IPC/CellID 최대 개수
예시: 1032, 58, 3~4
주의: 데이터 특성에 맞게 조정
mac_encoding

MAC 인코딩 방식
예시: bytes, hash, int64
주의: 인코더에 따라 다름
chunk_size / max_chunks_in_memory

DB/HDF5 chunk 크기, LRU 캐시 크기
예시: 10000, 2~4
주의: 메모리 상황에 맞게, 너무 작으면 I/O 증가
batch_size / epochs

학습 배치 크기, epoch 수
예시: 64, 128 / 50, 100
주의: GPU 메모리 상황에 맞게
optimizer / lr / early_stopping

Optimizer 종류, learning rate, early stopping patience
예시: Adam, 0.001, 10
save_frequency / save_best_only

체크포인트 저장 주기, best 모델만 저장 여부
예시: 10, true/false
실험 목적별 config 예시
과적합 방지 실험
model:
  params:
    dropout: 0.2
    ln: true
대용량 데이터 실험
data:
  chunk_size: 10000
  max_chunks_in_memory: 3
다양한 인코딩 방식 실험
data:
  encoder:
    type: WiFiMultiFieldEncoder
    params:
      mac_encoding: hash
모델 구조 실험
model:
  params:
    hidden_dim: 256
    num_enc_layers: 3
    num_dec_layers: 2
    num_heads: 8
부가 기능 및 확장성 / Advanced Features & Extensibility
LayerNorm/Dropout 사용법
활성화 방법:
model:
  params:
    ln: true
    dropout: 0.1
효과: 학습 안정성 향상, 과적합 방지
주의: 작은 데이터셋은 dropout을 높이고, 큰 데이터셋은 낮게 설정하는 것이 일반적입니다.
ChunkedHDF5Dataset LRU 캐시 사용법
활성화 방법:
data:
  chunk_size: 10000
  max_chunks_in_memory: 3
효과: 대용량 데이터셋도 메모리 초과 없이 처리 가능
주의: 캐시 크기가 너무 작으면 I/O가 증가할 수 있습니다. 실험 환경에 맞게 조정하세요.
새로운 인코더/모델 추가 방법
인코더 추가:
data/encoders/에 새로운 인코더 클래스 구현 (BaseEncoder 상속)
data/encoders/__init__.py의 팩토리 함수에 등록
config에서 type만 변경하여 사용
모델 추가:
models/에 새로운 모델 클래스 구현 (BaseModel 또는 nn.Module 상속)
config에서 type만 변경하여 사용
실험 reproducibility/확장성
모든 실험 설정은 config 파일로 관리하여 완전한 재현성 보장
다양한 인코더/모델/데이터 조합을 config만 바꿔서 손쉽게 실험 가능
실험 결과, 체크포인트, 로그 등은 실험별 디렉토리에 자동 저장
기타 옵션 및 주의사항
config 파라미터명은 자동 정규화되어 기존 실험 config와의 호환성이 유지됩니다.
input_dim은 인코더에서 자동 계산되므로 config에서 중복 입력하지 않아도 됩니다.
멀티스레드/멀티프로세스 환경에서는 파일 핸들, 캐시 관리에 주의하세요.
실험 reproducibility를 위해 seed, 체크포인트, 로그를 철저히 관리하세요.
set_transformer
Official PyTorch implementation of the paper Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks .




----------------------------------------------------------
Referance from
https://github.com/juho-lee/set_transformer
_---------------------------------------------------------
