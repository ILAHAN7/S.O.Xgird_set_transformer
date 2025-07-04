"""
[파일 목적]
- MariaDB(MySQL 호환)에서 직접 데이터를 쿼리하여 List[dict]로 반환하는 데이터 로더
- 대용량 데이터 처리를 위해 chunk iterator(load_data_chunks_from_db) 지원
- 범위/limit/id_col 기반 쿼리로 메모리 효율적 데이터 로딩 가능

[주요 함수]
- load_data_from_db(config, start_id=None, end_id=None, limit=None): 범위/limit 쿼리 지원
- load_data_chunks_from_db(config, chunk_size=10000, min_id=None, max_id=None): chunk iterator

[의존성]
- scripts/encode_data.py에서 사용
- configs/에서 DB 접속 정보 및 id_col, 쿼리문 등 읽어옴
"""

import pymysql

def load_data_from_db(config, start_id=None, end_id=None, limit=None):
    """
    MariaDB에서 범위/limit 단위로 데이터를 쿼리하여 List[dict]로 반환
    Args:
        config (dict): DB 접속 정보 및 쿼리/id_col
        start_id (int, optional): 시작 id
        end_id (int, optional): 끝 id
        limit (int, optional): 최대 row 수
    Returns:
        List[dict]: 각 row가 dict인 리스트
    """
    base_query = config['query'].strip()
    id_col = config.get('id_col')
    where_clauses = []
    if id_col:
        if start_id is not None and end_id is not None:
            where_clauses.append(f"{id_col} BETWEEN {start_id} AND {end_id}")
        elif start_id is not None:
            where_clauses.append(f"{id_col} >= {start_id}")
        elif end_id is not None:
            where_clauses.append(f"{id_col} <= {end_id}")
    if where_clauses:
        if 'where' in base_query.lower():
            base_query += ' AND ' + ' AND '.join(where_clauses)
        else:
            base_query += ' WHERE ' + ' AND '.join(where_clauses)
    if limit is not None:
        base_query += f' LIMIT {limit}'
    conn = pymysql.connect(
        host=config['host'],
        port=int(config['port']),
        user=config['user'],
        password=config['password'],
        database=config['database'],
        charset='utf8'
    )
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(base_query)
            data = cursor.fetchall()
    finally:
        conn.close()
    return data

def load_data_chunks_from_db(config, chunk_size=10000, min_id=None, max_id=None):
    """
    MariaDB에서 chunk 단위로 데이터를 반복적으로 읽어오는 제너레이터
    Args:
        config (dict): DB 접속 정보 및 쿼리/id_col
        chunk_size (int): 한 번에 읽을 row 수
        min_id (int, optional): 시작 id (없으면 0)
        max_id (int, optional): 끝 id (없으면 무한)
    Yields:
        List[dict]: chunk_size만큼의 row 리스트
    """
    id_col = config.get('id_col')
    if not id_col:
        raise ValueError('config에 id_col(고유 정수 열) 지정 필요')
    # min_id, max_id 자동 추정 (없으면 DB에서 쿼리)
    conn = pymysql.connect(
        host=config['host'],
        port=int(config['port']),
        user=config['user'],
        password=config['password'],
        database=config['database'],
        charset='utf8'
    )
    try:
        with conn.cursor() as cursor:
            if min_id is None:
                cursor.execute(f"SELECT MIN({id_col}) FROM {config['table']}")
                min_id = cursor.fetchone()[0] or 0
            if max_id is None:
                cursor.execute(f"SELECT MAX({id_col}) FROM {config['table']}")
                max_id = cursor.fetchone()[0]
    finally:
        conn.close()
    for start in range(min_id, max_id + 1, chunk_size):
        end = min(start + chunk_size - 1, max_id)
        chunk = load_data_from_db(config, start_id=start, end_id=end)
        if not chunk:
            break
        yield chunk

# DB 연결 및 쿼리용 스켈레톤 