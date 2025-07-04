import os
import argparse
import h5py
import numpy as np
from glob import glob
import traceback

def merge_h5_parts(input_dir, output_dir, prefix='part-', feature_name='features', label_name='labels'):
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'merge_parts.log')
    # part 파일 리스트 정렬
    feature_files = sorted(glob(os.path.join(input_dir, f'{prefix}*_features.h5')))
    label_files = sorted(glob(os.path.join(input_dir, f'{prefix}*_labels.h5')))
    if len(feature_files) != len(label_files):
        msg = f"feature/label part 파일 개수 불일치: {len(feature_files)} vs {len(label_files)}\n"
        print(msg.strip())
        with open(log_path, 'a', encoding='utf-8') as logf:
            logf.write(msg)
        return
    print(f"Merging {len(feature_files)} part files...")
    with open(log_path, 'a', encoding='utf-8') as logf:
        logf.write(f"Merging {len(feature_files)} part files...\n")
    # feature/label shape 파악
    shapes = []
    label_shapes = []
    for f in feature_files:
        with h5py.File(f, 'r') as h5f:
            shapes.append(h5f[feature_name].shape)
    for f in label_files:
        with h5py.File(f, 'r') as h5f:
            label_shapes.append(h5f[label_name].shape)
    total_samples = sum(s[0] for s in shapes)
    feature_shape = (total_samples,) + shapes[0][1:]
    label_total = sum(s[0] for s in label_shapes)
    label_shape = (label_total,) + label_shapes[0][1:]
    # 병합 저장
    feature_out = os.path.join(output_dir, 'features.h5')
    label_out = os.path.join(output_dir, 'labels.h5')
    with h5py.File(feature_out, 'w') as fout, h5py.File(label_out, 'w') as lout:
        fset = fout.create_dataset(feature_name, shape=feature_shape, dtype='float32')
        lset = lout.create_dataset(label_name, shape=label_shape, dtype='int32')
        f_idx = 0
        l_idx = 0
        n_success = 0
        n_fail = 0
        for i, (ff, lf) in enumerate(zip(feature_files, label_files)):
            try:
                with h5py.File(ff, 'r') as fpart, h5py.File(lf, 'r') as lpart:
                    fdata = fpart[feature_name][:]
                    ldata = lpart[label_name][:]
                    # shape 체크
                    if fdata.shape[0] != ldata.shape[0]:
                        msg = f"FAIL part {i}: feature/label 샘플 수 불일치 {fdata.shape[0]} vs {ldata.shape[0]}\n"
                        print(msg.strip())
                        with open(log_path, 'a', encoding='utf-8') as logf:
                            logf.write(msg)
                        n_fail += 1
                        continue
                    fset[f_idx:f_idx+fdata.shape[0]] = fdata
                    lset[l_idx:l_idx+ldata.shape[0]] = ldata
                    f_idx += fdata.shape[0]
                    l_idx += ldata.shape[0]
                msg = f"SUCCESS part {i}: {fdata.shape[0]} samples\n"
                print(msg.strip())
                with open(log_path, 'a', encoding='utf-8') as logf:
                    logf.write(msg)
                n_success += 1
            except Exception as e:
                err_msg = f"FAIL part {i}: {repr(e)}\n{traceback.format_exc()}\n"
                print(err_msg.strip())
                with open(log_path, 'a', encoding='utf-8') as logf:
                    logf.write(err_msg)
                n_fail += 1
    summary = f"Done. Total parts: {len(feature_files)}, Success: {n_success}, Fail: {n_fail}\n"
    print(summary.strip())
    with open(log_path, 'a', encoding='utf-8') as logf:
        logf.write(summary)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='part 파일들이 있는 디렉토리')
    parser.add_argument('--output_dir', type=str, required=True, help='병합된 파일 저장 디렉토리')
    parser.add_argument('--prefix', type=str, default='part-', help='part 파일 prefix')
    args = parser.parse_args()
    merge_h5_parts(args.input_dir, args.output_dir, prefix=args.prefix)

if __name__ == '__main__':
    main() 