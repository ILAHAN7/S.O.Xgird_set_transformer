class BaseEncoder:
    def encode(self, raw_data):
        """원시 데이터를 인코딩하여 feature vector로 반환"""
        raise NotImplementedError
    
    def get_feature_dim(self):
        """각 필드의 feature dimension 반환 (기존 메서드)"""
        raise NotImplementedError
    
    def get_total_feature_dim(self):
        """전체 feature dimension 반환 (새로 추가)"""
        raise NotImplementedError
    
    def get_encoder_info(self):
        """인코더 정보 반환 (디버깅/로깅용)"""
        return {
            'type': self.__class__.__name__,
            'total_feature_dim': self.get_total_feature_dim(),
            'feature_dims': self.get_feature_dim()
        } 