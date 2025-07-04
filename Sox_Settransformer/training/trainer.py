"""
[파일 목적]
- UniversalTrainer: 모델 학습/평가/체크포인트 저장을 담당하는 범용 트레이너 클래스
- config['model']['task']와 config['training']['loss']에 따라 회귀/분류 분기 지원
- PyTorch 기반 optimizer, loss, epoch/batch loop, validation, checkpoint, early stopping, reproducibility 등 지원

[주요 함수]
- train(features, labels, exp_dir): 학습 loop, 체크포인트/로그 저장
- evaluate(features, labels): 평가(손실, MAE/accuracy 등)

[입력/출력]
- 회귀: features (N, set_size, feature_dim), labels (N, 2)
- 분류: features (N, set_size, feature_dim), labels (N,) (int, class index)
"""

import os
import torch
import numpy as np
import torch.nn.functional as F

class UniversalTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training 설정
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.lr = config['training']['lr']
        self.weight_decay = config['training'].get('weight_decay', 0.0)
        self.early_stopping = config['training'].get('early_stopping', 20)
        
        # Output 설정 (수정된 부분)
        self.save_every = config['output'].get('save_every', 10)
        self.save_best_only = config['output'].get('save_best_only', False)
        
        # Model 설정
        self.task = config['model'].get('task', 'regression')
        self.loss_type = config['training'].get('loss', 'mse')
        
        # Loss function 설정
        if self.task == 'classification' or self.loss_type == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()
        
        # Optimizer 설정
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # 재현성을 위한 시드 설정
        seed = config['training'].get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def train(self, dataloader, exp_dir):
        best_loss = float('inf')
        patience = 0
        save_all_epochs = self.config['training'].get('save_all_epochs', True)
        
        print(f"Training on device: {self.device}")
        print(f"Total epochs: {self.epochs}")
        print(f"Save every: {self.save_every} epochs")
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            n_samples = 0
            
            for batch_idx, (xb, yb) in enumerate(dataloader):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                
                self.optimizer.zero_grad()
                preds = self.model(xb)
                
                if self.task == 'classification':
                    loss = self.criterion(preds, yb)
                else:
                    loss = self.criterion(preds, yb)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * xb.size(0)
                n_samples += xb.size(0)
                
                # 진행상황 출력 (매 100 배치마다)
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            epoch_loss /= n_samples
            print(f"Epoch {epoch}/{self.epochs} - Loss: {epoch_loss:.4f}")
            
            # Early stopping 체크
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
                # Best 모델 저장
                torch.save(self.model.state_dict(), 
                          os.path.join(exp_dir, 'checkpoints', 'best_model.pt'))
            else:
                patience += 1
            
            # Epoch별 저장 (옵션)
            if save_all_epochs:
                torch.save(self.model.state_dict(), 
                          os.path.join(exp_dir, 'checkpoints', f'epoch_{epoch}.pt'))
            
            # 주기적 체크포인트 저장
            if epoch % self.save_every == 0:
                torch.save(self.model.state_dict(), 
                          os.path.join(exp_dir, 'checkpoints', f'epoch_{epoch}_saveevery.pt'))
            
            # 마지막 epoch 저장
            if epoch == self.epochs:
                torch.save(self.model.state_dict(), 
                          os.path.join(exp_dir, 'checkpoints', 'last_model.pt'))
            
            # Early stopping
            if patience >= self.early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"Training completed. Best loss: {best_loss:.4f}")

    def evaluate(self, features, labels):
        self.model.eval()
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        if self.task == 'classification':
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        else:
            labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(features)
            
            if self.task == 'classification':
                loss = self.criterion(preds, labels).item()
                pred_class = torch.argmax(preds, dim=1)
                acc = (pred_class == labels).float().mean().item()
                print(f"Eval Loss: {loss:.4f}, Accuracy: {acc:.4f}")
                return loss, acc
            else:
                loss = self.criterion(preds, labels).item()
                mae = torch.mean(torch.abs(preds - labels)).item()
                print(f"Eval Loss: {loss:.4f}, MAE: {mae:.4f}")
                return loss, mae 