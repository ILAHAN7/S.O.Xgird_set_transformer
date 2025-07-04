"""
WiFiMultiFieldEncoder
====================

WiFi MAC 주소(6바이트 정수), wrssi(정규화), ipcikey(4D 정수), icellid(정수)를 모두 인코딩하여 feature vector로 반환하는 인코더.

[사용 예시]
------------
config 예시:
  encoder:
    type: WiFiMultiFieldEncoder
    params:
      max_mac_count: 32
      max_ipcikey_count: 8
      max_icellid_count: 4

from data.encoders import get_encoder_by_config
encoder = get_encoder_by_config(config['data']['encoder'])
features = encoder.encode(batch_data)

[입력]
------
- wmac: MAC 주소 문자열 리스트 (쉼표구분)
- wrssi: wrssi(float) 리스트 (MAC 순서와 일치)
- ipcikey: 복합키 문자열 리스트 (쉼표구분, 각 원소는 '_'로 분해)
- icellid: 정수형 셀ID 리스트 (쉼표구분)

[출력]
-------
- feature vector (np.ndarray): shape = (batch, max_mac_count+max_ipcikey_count+max_icellid_count, 7/4/1)
  (WiFi: [mac_byte0..5, wrssi_norm], ipcikey: [p0,p1,p2,p3], icellid: [cellid])

[파라미터 설명]
---------------
- max_mac_count: WiFi MAC+RSSI 최대 개수 (패딩)
- max_ipcikey_count: ipcikey 최대 개수 (패딩)
- max_icellid_count: icellid 최대 개수 (패딩)

[인코딩 방식]
-------------
- wmac: "00:40:5a:af:bc:4a" → [0,64,90,175,188,74] (6D)
- wrssi: (wrssi+100)/70 (정규화, 1D)
- ipcikey: "158_1350_2_0" → [158,1350,2,0] (4D)
- icellid: 정수 그대로 (1D)

[반환 shape]
------------
- (batch, max_mac_count+max_ipcikey_count+max_icellid_count, feature_dim)
  (feature_dim: 7 for WiFi, 4 for ipcikey, 1 for icellid)

[주의]
------
- 각 필드가 없을 경우 0으로 패딩됨
- 모델 입력 시 feature flatten/concat 필요
"""

import numpy as np
import re
from data.base import BaseEncoder

class WiFiMultiFieldEncoder(BaseEncoder):
    def __init__(self, max_mac_count=32, max_ipcikey_count=8, max_icellid_count=4):
        self.max_mac_count = max_mac_count
        self.max_ipcikey_count = max_ipcikey_count
        self.max_icellid_count = max_icellid_count
        
        # MAC 주소 검증을 위한 정규표현식
        self.mac_pattern = re.compile(r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$')

    def _validate_mac(self, mac):
        """MAC 주소 형식 검증"""
        if not mac or not isinstance(mac, str):
            return False
        return bool(self.mac_pattern.match(mac.strip()))

    def _mac_to_bytes(self, mac):
        """MAC 주소를 바이트 배열로 변환 (검증 포함)"""
        try:
            if not self._validate_mac(mac):
                return [0, 0, 0, 0, 0, 0]
            # "00:40:5a:af:bc:4a" → [0, 64, 90, 175, 188, 74]
            return [int(b, 16) for b in mac.strip().split(':')]
        except Exception:
            return [0, 0, 0, 0, 0, 0]

    def _validate_rssi(self, rssi):
        """RSSI 값 범위 검증"""
        try:
            rssi_val = float(rssi)
            return -120 <= rssi_val <= 0  # 일반적인 WiFi RSSI 범위
        except (ValueError, TypeError):
            return False

    def _normalize_rssi(self, rssi):
        """RSSI 정규화 (검증 포함)"""
        try:
            rssi_val = float(rssi)
            if not self._validate_rssi(rssi_val):
                rssi_val = -100.0  # 기본값
            # (rssi + 100) / 70 → [0, 1] 범위로 정규화
            return max(0.0, min(1.0, (rssi_val + 100) / 70))
        except Exception:
            return 0.0

    def _ipcikey_to_vec(self, key):
        """ipcikey를 벡터로 변환 (검증 포함)"""
        try:
            if not key or not isinstance(key, str):
                return [0, 0, 0, 0]
            # '158_1350_2_0' → [158,1350,2,0]
            parts = key.strip().split('_')
            if len(parts) != 4:
                return [0, 0, 0, 0]
            return [int(x) for x in parts]
        except Exception:
            return [0, 0, 0, 0]

    def _validate_icellid(self, icellid):
        """icellid 값 검증"""
        try:
            val = int(icellid)
            return 0 <= val <= 999999  # 합리적인 범위
        except (ValueError, TypeError):
            return False

    def _parse_string_list(self, data_str, field_name):
        """쉼표로 구분된 문자열을 리스트로 파싱 (검증 포함)"""
        if not data_str or not isinstance(data_str, str):
            return []
        
        try:
            # 빈 문자열, 공백만 있는 경우 처리
            parts = [part.strip() for part in data_str.split(',')]
            return [part for part in parts if part]  # 빈 문자열 제거
        except Exception:
            print(f"Warning: Failed to parse {field_name}: {data_str}")
            return []

    def encode(self, batch_data):
        """
        Args:
            batch_data: List[dict], 각 dict에 'wmac', 'wrssi', 'ipcikey', 'icellid' key (쉼표구분 문자열)
        Returns:
            np.ndarray: (batch, max_mac_count+max_ipcikey_count+max_icellid_count, 7/4/1)
        """
        if not batch_data:
            raise ValueError("Empty batch_data provided")
        
        features = []
        for sample_idx, sample in enumerate(batch_data):
            try:
                # WiFi MAC 및 RSSI 처리
                macs = self._parse_string_list(sample.get('wmac', ''), 'wmac')
                wrssis = self._parse_string_list(sample.get('wrssi', ''), 'wrssi')
                
                # MAC과 RSSI 개수 일치 확인
                if len(macs) != len(wrssis):
                    print(f"Warning: MAC count ({len(macs)}) != RSSI count ({len(wrssis)}) in sample {sample_idx}")
                    # 더 작은 개수로 맞춤
                    min_count = min(len(macs), len(wrssis))
                    macs = macs[:min_count]
                    wrssis = wrssis[:min_count]
                
                wifi_feats = []
                for i in range(self.max_mac_count):
                    if i < len(macs):
                        mac_bytes = self._mac_to_bytes(macs[i])
                        wrssi_norm = self._normalize_rssi(wrssis[i] if i < len(wrssis) else -100.0)
                        feat = np.array(mac_bytes + [wrssi_norm], dtype=np.float32)
                    else:
                        feat = np.zeros(7, dtype=np.float32)
                    wifi_feats.append(feat)
                
                # ipcikey 처리
                ipcikeys = self._parse_string_list(sample.get('ipcikey', ''), 'ipcikey')
                ipci_feats = []
                for i in range(self.max_ipcikey_count):
                    if i < len(ipcikeys):
                        vec = self._ipcikey_to_vec(ipcikeys[i])
                        feat = np.array(vec[:4], dtype=np.float32)
                    else:
                        feat = np.zeros(4, dtype=np.float32)
                    ipci_feats.append(feat)
                
                # icellid 처리
                icellids = self._parse_string_list(sample.get('icellid', ''), 'icellid')
                icellid_feats = []
                for i in range(self.max_icellid_count):
                    if i < len(icellids):
                        try:
                            val = float(icellids[i]) if self._validate_icellid(icellids[i]) else 0.0
                        except Exception:
                            val = 0.0
                        feat = np.array([val], dtype=np.float32)
                    else:
                        feat = np.zeros(1, dtype=np.float32)
                    icellid_feats.append(feat)
                
                # 모든 feature 연결
                sample_feat = np.concatenate([
                    np.stack(wifi_feats), 
                    np.stack(ipci_feats), 
                    np.stack(icellid_feats)
                ], axis=0)
                features.append(sample_feat)
                
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                # 에러 발생 시 기본값으로 채움
                default_feat = np.zeros(
                    self.max_mac_count * 7 + self.max_ipcikey_count * 4 + self.max_icellid_count * 1,
                    dtype=np.float32
                ).reshape(-1, 1)
                features.append(default_feat)
        
        return np.array(features, dtype=np.float32)

    def get_feature_dim(self):
        """각 필드의 feature dimension 반환"""
        return (7, 4, 1)  # WiFi, ipcikey, icellid
    
    def get_total_feature_dim(self):
        """전체 feature dimension 반환"""
        return self.max_mac_count * 7 + self.max_ipcikey_count * 4 + self.max_icellid_count * 1
    
    def get_encoder_info(self):
        """WiFi 인코더 정보 반환"""
        info = super().get_encoder_info()
        info.update({
            'max_mac_count': self.max_mac_count,
            'max_ipcikey_count': self.max_ipcikey_count,
            'max_icellid_count': self.max_icellid_count,
            'mac_encoding': 'bytes'  # 현재 인코딩 방식
        })
        return info 