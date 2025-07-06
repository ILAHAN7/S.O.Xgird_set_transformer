import yaml
import os
from typing import Dict, Any, List

def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Config 설정을 검증하고 오류/경고 메시지 리스트를 반환합니다.
    Args:
        config (dict): 검증할 config
    Returns:
        List[str]: 오류/경고 메시지 리스트 (빈 리스트면 검증 통과)
    """
    errors = []
    warnings = []
    
    # 필수 섹션 검증
    required_sections = ['data', 'model', 'training', 'output', 'experiment']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    if errors:  # 필수 섹션이 없으면 더 이상 검증하지 않음
        return errors
    
    # Data 섹션 검증
    data = config['data']
    if 'db' not in data:
        errors.append("Missing 'db' configuration in data section")
    else:
        db = data['db']
        required_db_fields = ['host', 'port', 'user', 'password', 'database', 'table', 'id_col']
        for field in required_db_fields:
            if field not in db:
                errors.append(f"Missing required DB field: {field}")
    
    if 'encoder' not in data:
        errors.append("Missing 'encoder' configuration in data section")
    else:
        encoder = data['encoder']
        if 'type' not in encoder:
            errors.append("Missing encoder type")
        if 'params' not in encoder:
            warnings.append("Missing encoder params (using defaults)")
    
    # Model 섹션 검증
    model = config['model']
    required_model_fields = ['name', 'task', 'output_dim']
    for field in required_model_fields:
        if field not in model:
            errors.append(f"Missing required model field: {field}")
    
    if 'task' in model and model['task'] not in ['regression', 'classification']:
        errors.append("Model task must be 'regression' or 'classification'")
    
    if 'params' in model:
        params = model['params']
        if 'hidden_dim' in params and params['hidden_dim'] <= 0:
            errors.append("Model hidden_dim must be positive")
        if 'num_heads' in params and params['num_heads'] <= 0:
            errors.append("Model num_heads must be positive")
        if 'dropout' in params and not (0 <= params['dropout'] <= 1):
            errors.append("Model dropout must be between 0 and 1")
    
    # Training 섹션 검증
    training = config['training']
    required_training_fields = ['batch_size', 'epochs', 'lr']
    for field in required_training_fields:
        if field not in training:
            errors.append(f"Missing required training field: {field}")
    
    if 'batch_size' in training and training['batch_size'] <= 0:
        errors.append("Training batch_size must be positive")
    if 'epochs' in training and training['epochs'] <= 0:
        errors.append("Training epochs must be positive")
    if 'lr' in training and training['lr'] <= 0:
        errors.append("Training learning rate must be positive")
    if 'weight_decay' in training and training['weight_decay'] < 0:
        errors.append("Training weight_decay must be non-negative")
    
    # Output 섹션 검증
    output = config['output']
    if 'save_every' in output and output['save_every'] <= 0:
        errors.append("Output save_every must be positive")
    
    # Experiment 섹션 검증
    experiment = config['experiment']
    if 'name' not in experiment:
        errors.append("Missing experiment name")
    if 'output_dir' not in experiment:
        errors.append("Missing experiment output_dir")
    
    # 경고 메시지도 포함하여 반환
    return errors + warnings

def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML config 파일을 로딩하고 검증하여 dict로 반환합니다.
    Args:
        config_path (str): config yaml 파일 경로
    Returns:
        dict: 검증된 config 내용
    Raises:
        FileNotFoundError: 파일이 없을 때
        yaml.YAMLError: YAML 파싱 에러
        ValueError: Config 검증 실패
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"YAML parsing error: {e}")
    
    # Config 검증
    validation_messages = validate_config(config)
    
    if validation_messages:
        print("Config validation messages:")
        for msg in validation_messages:
            print(f"  - {msg}")
        
        # 오류가 있으면 예외 발생
        errors = [msg for msg in validation_messages if not msg.startswith("Warning")]
        if errors:
            raise ValueError(f"Config validation failed with {len(errors)} errors")
    
    return config

def create_default_config() -> Dict[str, Any]:
    """
    기본 config 템플릿을 생성합니다.
    Returns:
        dict: 기본 config 구조
    """
    return {
        'data': {
            'db': {
                'host': 'localhost',
                'port': 3306,
                'user': 'user',
                'password': 'password',
                'database': 'wifi',
                'table': 'signals',
                'id_col': 'id',
                'chunk_size': 100000
            },
            'encoder': {
                'type': 'WiFiMultiFieldEncoder',
                'params': {
                    'max_mac_count': 32,
                    'max_ipcikey_count': 8,
                    'max_icellid_count': 4
                }
            },
            'gridcell': {
                'level': 5
            },
            'x_max': 1000
        },
        'model': {
            'name': 'set_transformer',
            'task': 'regression',
            # 'input_dim': 7,  # (자동 추론됨, 명시 불필요)
            'output_dim': 2,
            'params': {
                'hidden_dim': 256,
                'num_heads': 4,
                'num_inds': 32,
                'num_enc_layers': 2,
                'num_dec_layers': 2,
                'dropout': 0.1
            }
        },
        'training': {
            'batch_size': 128,
            'epochs': 50,
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0001,
            'seed': 42,
            'early_stopping': True,
            'patience': 5,
            'save_all_epochs': True
        },
        'output': {
            'save_every': 10,
            'save_best_only': False,
            'output_format': ['csv', 'json']
        },
        'experiment': {
            'name': 'default_experiment',
            'output_dir': 'experiments/default_experiment/'
        }
    } 