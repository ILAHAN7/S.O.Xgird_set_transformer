"""
[파일 목적]
- 위도/경도 좌표를 고정된 기준점/셀 크기/수식으로 그리드셀 id(xId, yId)로 변환하는 함수 제공
- 셀 크기(level)는 실험 목적에 따라 조절, 나머지 상수/수식은 항상 고정

[고정 상수/수식]
- ORG_MIN_X = 124.54117 (기준 경도)
- ORG_MIN_Y = 32.928463 (기준 위도)
- OFFSET_5M_X = 0.0000555 (경도 5m 단위)
- OFFSET_5M_Y = 0.0000460 (위도 5m 단위)
- level: 셀 크기 배수 (5=25m, 10=50m 등)
- xId = int((lon - ORG_MIN_X) / (OFFSET_5M_X * level)) + 1
- yId = int((lat - ORG_MIN_Y) / (OFFSET_5M_Y * level)) + 1

[config 예시]
data:
  gridcell:
    level: 5   # 25m 셀

[사용 예시]
from data.latlon_to_gridcell import latlon_to_gridcell
xId, yId = latlon_to_gridcell(126.998153, 37.566629, level=5)

[주의]
- ORG_MIN_X, ORG_MIN_Y, OFFSET_5M_X, OFFSET_5M_Y, 수식은 항상 고정
- level만 실험 목적에 따라 조절
- config에서 cell_size, grid_level 등은 level로 변환해서 사용하거나 level만 직접 지정

[의존성]
- scripts/encode_data.py 등에서 label 생성에 사용
"""

ORG_MIN_X = 124.54117
ORG_MIN_Y = 32.928463
OFFSET_5M_X = 0.0000555
OFFSET_5M_Y = 0.0000460

def latlon_to_gridcell(lon, lat, level=5):
    """
    위경도 → 그리드셀 id(xId, yId) 변환 (고정 수식)
    Args:
        lon (float): 경도
        lat (float): 위도
        level (int): 셀 크기 배수 (5=25m, 10=50m 등)
    Returns:
        (int, int): (xId, yId)
    """
    xId = int((lon - ORG_MIN_X) / (OFFSET_5M_X * level)) + 1
    yId = int((lat - ORG_MIN_Y) / (OFFSET_5M_Y * level)) + 1
    return xId, yId 