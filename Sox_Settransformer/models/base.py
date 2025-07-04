class BaseModel:
    def forward(self, features):
        raise NotImplementedError
    def get_config(self):
        raise NotImplementedError 