from .wifi_multi_field import WiFiMultiFieldEncoder

def get_encoder_by_config(encoder_config):
    """
    config dict에서 type/params를 읽어 인코더 객체 반환
    Args:
        encoder_config (dict): {'type': 'WiFiMultiFieldEncoder', 'params': {...}}
    Returns:
        BaseEncoder 하위 인코더 객체
    """
    encoder_type = encoder_config['type']
    params = encoder_config.get('params', {})
    if encoder_type == 'WiFiMultiFieldEncoder':
        return WiFiMultiFieldEncoder(**params)
    else:
        raise ValueError(f'Unknown encoder type: {encoder_type}') 