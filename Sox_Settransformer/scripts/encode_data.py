"""
[파일 목적]
- DB에서 chunk 단위로 데이터를 읽어 인코딩 및 그리드셀 변환(label) 후 HDF5로 저장하는 전체 파이프라인 스크립트
- config['model']['task']가 'regression'이면 label=[xId, yId], 'classification'이면 label=class_index(xId + yId * x_max)로 저장

[실행 예시]
python scripts/encode_data.py --config configs/wifi_hash.yaml --output experiments/exp001/encoded/

[주요 흐름]
- config 로딩 → DB chunk iterator → 인코더(feature) → 그리드셀 변환(label) → HDF5 append 저장
- label 생성 시 회귀/분류 분기 처리

[의존성]
- utils/config, data/db_reader, data/encoders, data/latlon_to_gridcell, utils/io 등
"""

import argparse
import os
import numpy as np
import multiprocessing
from utils.config import load_config
from data.db_reader import load_data_chunks_from_db
from data.encoders import get_encoder_by_config
from data.latlon_to_gridcell import latlon_to_gridcell
from utils.io import append_to_h5
import traceback

def encode_and_save_chunk(args):
    chunk, chunk_idx, config, encoder, task, x_max, output_dir = args
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'encode_chunks.log')
    try:
        features = encoder.encode(chunk)
        labels = []
        for sample in chunk:
            lon = float(sample['longitude'])
            lat = float(sample['latitude'])
            level = config['data']['gridcell']['level']
            xId, yId = latlon_to_gridcell(lon, lat, level=level)
            if task == 'classification':
                label = xId + yId * x_max
                labels.append(label)
            else:
                labels.append([xId, yId])
        labels = np.array(labels, dtype=np.int32)
        feature_h5 = os.path.join(output_dir, f'part-{chunk_idx:03d}_features.h5')
        label_h5 = os.path.join(output_dir, f'part-{chunk_idx:03d}_labels.h5')
        append_to_h5(feature_h5, features, dataset_name='features')
        append_to_h5(label_h5, labels, dataset_name='labels')
        msg = f"SUCCESS chunk {chunk_idx}: {features.shape[0]} samples\n"
        print(msg.strip())
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(msg)
        return (chunk_idx, True, None)
    except Exception as e:
        err_msg = f"FAIL chunk {chunk_idx}: {repr(e)}\n{traceback.format_exc()}\n"
        print(err_msg.strip())
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(err_msg)
        return (chunk_idx, False, str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config yaml 경로')
    parser.add_argument('--output', type=str, required=True, help='HDF5 저장 디렉토리')
    parser.add_argument('--chunk_size', type=int, default=10000, help='DB chunk 크기')
    parser.add_argument('--num_workers', type=int, default=None, help='병렬 인코딩 프로세스 수 (기본: CPU 코어 수)')
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(args.output, exist_ok=True)

    encoder = get_encoder_by_config(config['data']['encoder'])
    db_config = config['data']['db']
    chunk_size = args.chunk_size
    num_workers = args.num_workers or multiprocessing.cpu_count()

    task = config['model'].get('task', 'regression')
    x_max = config['data'].get('x_max', 1000)

    # chunk list 생성
    chunks = []
    for i, chunk in enumerate(load_data_chunks_from_db(db_config, chunk_size=chunk_size)):
        chunks.append((chunk, i, config, encoder, task, x_max, args.output))

    print(f"Encoding {len(chunks)} chunks with {num_workers} workers...")
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(encode_and_save_chunk, chunks)
    n_success = sum(1 for r in results if r[1])
    n_fail = sum(1 for r in results if not r[1])
    print(f"All chunks processed. Success: {n_success}, Fail: {n_fail}")
    if n_fail > 0:
        print("Failed chunks:")
        for r in results:
            if not r[1]:
                print(f"  chunk {r[0]}: {r[2]}")
    print("(part 파일 병합은 별도 merge_hdf5_parts.py에서 수행)")

if __name__ == '__main__':
    main() 