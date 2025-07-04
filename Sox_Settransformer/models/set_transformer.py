"""
[파일 목적]
- SetTransformer 논문/원본 구현의 핵심 구조(SAB, ISAB, PMA, SetTransformer 전체)를 PyTorch 기반 클래스로 이식
- config['model']['task']가 'regression'이면 output_dim=2, 'classification'이면 output_dim=num_classes로 분기
- config 기반 파라미터, 입력/출력 shape, forward 메서드, 저장/불러오기 등 리팩토링 구조에 맞게 구현

[주요 클래스]
- MAB: Multihead Attention Block (원본 논문 기반)
- SAB: Self-Attention Block
- ISAB: Induced Set Attention Block
- PMA: Pooling by Multihead Attention
- SetTransformer: 전체 모델(encoder + decoder)

[입력/출력]
- 입력: (batch, set_size, feature_dim)
- 출력: (batch, 2) 또는 (batch, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MAB(nn.Module):
    """
    Multihead Attention Block - 원본 SetTransformer 논문 구현
    Q, K, V를 사용한 attention 메커니즘
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, dropout=0.0, ln=True):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.dropout = dropout
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        else:
            self.ln0 = None
            self.ln1 = None
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        A = self.attn_dropout(A)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        if self.ln0 is not None:
            O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.out_dropout(O)
        if self.ln1 is not None:
            O = self.ln1(O)
        return O

class SAB(nn.Module):
    """
    Self-Attention Block - 자기 자신과의 attention
    """
    def __init__(self, dim_in, dim_out, num_heads, dropout=0.0, ln=True):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, dropout, ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    """
    Induced Set Attention Block - 원본 논문 구현
    Step 1: Induced points와 input 간의 attention
    Step 2: Input과 induced representation 간의 attention
    """
    def __init__(self, dim_in, dim_out, num_heads, num_inds, dropout=0.0, ln=True):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, dropout, ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, dropout, ln)

    def forward(self, X):
        H = self.mab0(self.I.expand(X.size(0), -1, -1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    """
    Pooling by Multihead Attention - 집합을 고정 크기로 압축
    """
    def __init__(self, dim, num_heads, num_seeds, dropout=0.0, ln=True):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, dropout, ln)

    def forward(self, X):
        return self.mab(self.S.expand(X.size(0), -1, -1), X)

class SetTransformer(nn.Module):
    """
    SetTransformer - WiFi 신호 기반 위치추정에 최적화된 구조
    """
    def __init__(self, input_dim, hidden_dim=256, num_heads=4, num_inds=32, 
                 num_enc_layers=2, num_dec_layers=2, dropout=0.0, ln=True,
                 output_dim=2, task='regression'):
        super().__init__()
        
        # Encoder: 여러 ISAB로 구성
        enc_blocks = []
        enc_blocks.append(ISAB(input_dim, hidden_dim, num_heads, num_inds, dropout, ln))
        for _ in range(num_enc_layers - 1):
            enc_blocks.append(ISAB(hidden_dim, hidden_dim, num_heads, num_inds, dropout, ln))
        self.encoder = nn.Sequential(*enc_blocks)
        
        # Decoder: PMA + SAB + MLP
        dec_blocks = [PMA(hidden_dim, num_heads, 1, dropout, ln)]
        for _ in range(num_dec_layers):
            dec_blocks.append(SAB(hidden_dim, hidden_dim, num_heads, dropout, ln))
        self.decoder = nn.Sequential(*dec_blocks)
        
        # 출력 레이어
        self.task = task
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, X):
        # X: (batch, set_size, feature_dim)
        H = self.encoder(X)  # (batch, set_size, hidden_dim)
        H = self.decoder(H)  # (batch, 1, hidden_dim)
        H = H.squeeze(1)     # (batch, hidden_dim)
        out = self.mlp(H)    # (batch, output_dim)
        
        if self.task == 'classification':
            return out  # (batch, num_classes)
        else:
            return out  # (batch, 2) - 위치 좌표 