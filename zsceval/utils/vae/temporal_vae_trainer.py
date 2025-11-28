"""
Temporal VAE Trainer

이 모듈은 Temporal VAE를 학습하기 위한 유틸리티를 제공합니다.

Features:
    - 시계열 데이터 수집 및 버퍼 관리
    - VAE 학습 루프
    - 학습 모니터링 및 체크포인트 저장
    - multi_overcooked 환경과의 통합

Author: Julie
Date: 2025-10-22
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import pickle
import json
from tqdm import tqdm
import wandb

from .temporal_vae import (
    TemporalVAE, 
    TemporalVAEConfig, 
    vae_loss,
    EncodingScheme,
    VAEMode
)


class TemporalObservationBuffer:
    """
    시계열 관찰을 저장하고 관리하는 버퍼
    
    에피소드 동안 관찰을 수집하고 VAE 학습을 위한 
    시계열 시퀀스를 생성합니다.
    """
    
    def __init__(self, 
                 k_timesteps: int = 5,
                 max_episodes: int = 1000,
                 encoding_scheme: EncodingScheme = EncodingScheme.OAI_LOSSLESS):
        """
        Args:
            k_timesteps: 시퀀스에 포함할 이전 타임스텝 수
            max_episodes: 버퍼에 저장할 최대 에피소드 수
            encoding_scheme: 사용할 인코딩 스킴
        """
        self.k = k_timesteps
        self.max_episodes = max_episodes
        self.encoding_scheme = encoding_scheme
        
        self.episodes: deque = deque(maxlen=max_episodes)
        self.current_episode: List = []
    
    def add_observation(self, obs: Union[np.ndarray, Dict]):
        """
        현재 에피소드에 관찰 추가
        
        Args:
            obs: 관찰 데이터 (dict 또는 numpy array)
        """
        # Dict 형태의 관찰에서 적절한 키 추출
        if isinstance(obs, dict):
            if self.encoding_scheme == EncodingScheme.OAI_FEATS:
                obs_data = obs.get('agent_obs', obs.get('obs', obs))
            else:
                obs_data = obs.get('visual_obs', obs.get('obs', obs))
        else:
            obs_data = obs
        
        self.current_episode.append(obs_data)
    
    def finish_episode(self):
        """현재 에피소드를 버퍼에 저장하고 초기화"""
        if len(self.current_episode) > self.k:  # 최소 k+1개의 관찰 필요
            self.episodes.append(np.array(self.current_episode))
        self.current_episode = []
    
    def get_sequences(self, mode: VAEMode = VAEMode.RECONSTRUCTION) -> Tuple[np.ndarray, np.ndarray]:
        """
        버퍼에서 학습용 시계열 시퀀스 생성
        
        Args:
            mode: VAE 모드 (reconstruction 또는 predictive)
        
        Returns:
            inputs: (N, k+1, *obs_shape) - 입력 시퀀스
            targets: (N, k+1, *obs_shape) - 타겟 시퀀스
        """
        inputs = []
        targets = []
        
        for episode in self.episodes:
            episode_length = len(episode)
            
            # 각 에피소드에서 가능한 모든 시퀀스 추출
            for t in range(self.k, episode_length):
                # 입력: t-k부터 t까지
                input_seq = episode[t-self.k:t+1]
                
                if mode == VAEMode.RECONSTRUCTION:
                    # Reconstruction: 동일한 시퀀스를 재구성
                    target_seq = input_seq
                elif mode == VAEMode.PREDICTIVE:
                    # Predictive: 한 스텝 앞 예측
                    if t + 1 < episode_length:
                        target_seq = episode[t-self.k+1:t+2]
                    else:
                        continue  # 마지막 타임스텝은 스킵
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                inputs.append(input_seq)
                targets.append(target_seq)
        
        if len(inputs) == 0:
            return np.array([]), np.array([])
        
        return np.array(inputs), np.array(targets)
    
    def __len__(self):
        """버퍼에 저장된 에피소드 수"""
        return len(self.episodes)
    
    def save(self, filepath: Union[str, Path]):
        """버퍼를 파일로 저장"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'k': self.k,
            'max_episodes': self.max_episodes,
            'encoding_scheme': self.encoding_scheme.value,
            'episodes': list(self.episodes),
            'current_episode': self.current_episode
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TemporalObservationBuffer':
        """파일에서 버퍼 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        buffer = cls(
            k_timesteps=data['k'],
            max_episodes=data['max_episodes'],
            encoding_scheme=EncodingScheme(data['encoding_scheme'])
        )
        buffer.episodes = deque(data['episodes'], maxlen=data['max_episodes'])
        buffer.current_episode = data['current_episode']
        
        return buffer


class TemporalVAEDataset(Dataset):
    """
    VAE 학습을 위한 PyTorch Dataset
    
    실제 데이터는 (W, H, C) 형태이지만, VAE는 (C, H, W) 형태를 기대하므로
    자동으로 변환합니다.
    """
    
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Args:
            inputs: Input sequences (N, k+1, W, H, C) - 실제 데이터 형태
            targets: Target sequences (N, k+1, W, H, C) - 실제 데이터 형태
        """
        # numpy array를 torch tensor로 변환
        inputs_tensor = torch.from_numpy(inputs).float()
        targets_tensor = torch.from_numpy(targets).float()
        
        # (W, H, C) -> (C, H, W) 변환 (3D observation인 경우)
        # inputs shape: (N, k+1, W, H, C)
        if inputs_tensor.ndim == 5:
            # (N, k+1, W, H, C) -> (N, k+1, C, H, W)
            inputs_tensor = inputs_tensor.permute(0, 1, 4, 3, 2)
            targets_tensor = targets_tensor.permute(0, 1, 4, 3, 2)
        
        self.inputs = inputs_tensor
        self.targets = targets_tensor
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class TemporalVAETrainer:
    """
    Temporal VAE를 학습하는 Trainer
    
    Features:
        - 학습 루프 및 최적화
        - 학습 모니터링
        - 체크포인트 저장/로드
        - Tensorboard/Wandb 통합 (optional)
    """
    
    def __init__(self,
                 model: TemporalVAE,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: Optional[Union[str, Path]] = None,
                 use_bce: bool = False,
                 use_wandb: bool = False,
                 wandb_project: str = 'vae-training',
                 wandb_run_name: Optional[str] = None,
                 beta_warmup_steps: int = 5000,
                 beta_annealing_steps: int = 10000):
        """
        Args:
            model: TemporalVAE 모델
            learning_rate: 학습률
            device: 학습에 사용할 디바이스
            checkpoint_dir: 체크포인트 저장 디렉토리
            use_bce: Binary Cross Entropy 사용 여부 (raw_image의 경우 True 권장)
            use_wandb: Wandb 사용 여부
            wandb_project: Wandb 프로젝트 이름
            wandb_run_name: Wandb run 이름
            beta_warmup_steps: Beta warm-up 스텝 수 (이 스텝까지 beta=0으로 고정)
            beta_annealing_steps: Beta annealing 스텝 수 (warm-up 이후 이 스텝에 걸쳐 beta를 0에서 목표값까지 증가)
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.use_bce = use_bce
        self.use_wandb = use_wandb
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb 초기화
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        'learning_rate': learning_rate,
                        'hidden_dim': model.config.hidden_dim,
                        'k_timesteps': model.config.k_timesteps,
                        'mode': model.config.mode.value,
                        'encoding_scheme': model.config.encoding_scheme.value,
                        'input_shape': model.config.input_shape,
                        'beta': model.config.beta,
                        'use_bce': use_bce,
                        'beta_warmup_steps': beta_warmup_steps,
                        'beta_annealing_steps': beta_annealing_steps
                    }
                )
                # Model architecture 로깅
                wandb.watch(self.model, log='all', log_freq=100)
                print("Wandb initialized successfully")
            except ImportError:
                print("Warning: wandb not installed. Install with: pip install wandb")
                self.use_wandb = False
        
        self.training_history = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': []
        }
        self.epoch = 0
        self.global_step = 0
        self.beta_warmup_steps = beta_warmup_steps
        self.beta_annealing_steps = beta_annealing_steps
    
    def train_epoch(self, 
                    dataloader: DataLoader,
                    beta: float = 1.0) -> Dict[str, float]:
        """
        한 에폭 학습
        
        Args:
            dataloader: 학습 데이터 로더
            beta: KL divergence 가중치
        
        Returns:
            평균 loss 딕셔너리
        """
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0
        }
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {self.epoch}")):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Beta warm-up and annealing
            if self.global_step < self.beta_warmup_steps:
                # Warm-up: 첫 5,000 스텝 동안 beta = 0으로 고정
                current_beta = 0.0
            elif self.global_step < self.beta_warmup_steps + self.beta_annealing_steps:
                # Annealing: 5,000 스텝 이후부터 10,000 스텝에 걸쳐 beta를 0에서 목표값까지 증가
                annealing_progress = (self.global_step - self.beta_warmup_steps) / self.beta_annealing_steps
                current_beta = beta * annealing_progress
            else:
                # Annealing 완료 후: 목표 beta 값 사용
                current_beta = beta
            
            # Forward pass
            x_recon, mu, logvar = self.model(inputs)
            
            # Compute loss (annealed beta 사용)
            losses = vae_loss(x_recon, targets, mu, logvar, beta=current_beta, use_bce=self.use_bce)
            
            self.global_step += 1
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 실시간 Wandb 로깅 (매 10 배치마다)
            if self.use_wandb and batch_idx % 10 == 0:
                import wandb
                wandb.log({
                    'train/total_loss': losses['total_loss'].item(),
                    'train/recon_loss': losses['recon_loss'].item(),
                    'train/kl_loss': losses['kl_loss'].item(),
                    'train/beta': current_beta,  # 실시간 beta 값
                    'epoch': self.epoch,
                    'batch': batch_idx,
                    'global_step': self.global_step
                })
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Update history
        for key in epoch_losses:
            self.training_history[key].append(epoch_losses[key])
        
        # Wandb 로깅 (에폭별 평균 - beta는 실시간 로깅에서만)
        if self.use_wandb:
            import wandb
            wandb.log({
                'train/total_loss': epoch_losses['total_loss'],
                'train/recon_loss': epoch_losses['recon_loss'],
                'train/kl_loss': epoch_losses['kl_loss'],
                'epoch': self.epoch
            })
        
        self.epoch += 1
        return epoch_losses
    
    @torch.no_grad()
    def evaluate(self, 
                 dataloader: DataLoader,
                 beta: float = 1.0) -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            dataloader: 평가 데이터 로더
            beta: KL divergence 가중치
        
        Returns:
            평균 loss 딕셔너리
        """
        self.model.eval()
        eval_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0
        }
        num_batches = 0
        
        # Validation progress bar 추가
        from tqdm import tqdm
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            x_recon, mu, logvar = self.model(inputs)
            
            # Compute loss
            losses = vae_loss(x_recon, targets, mu, logvar, beta=beta, use_bce=self.use_bce)
            
            # Accumulate losses
            for key in eval_losses:
                eval_losses[key] += losses[key].item()
            num_batches += 1
        
        # Average losses
        for key in eval_losses:
            eval_losses[key] /= num_batches
        
        return eval_losses
    
    def train(self,
              buffer: TemporalObservationBuffer,
              num_epochs: int = 100,
              batch_size: int = 32,
              beta: float = 1.0,
              val_split: float = 0.1,
              save_interval: int = 10,
              early_stopping_patience: Optional[int] = None) -> Dict[str, List[float]]:
        """
        전체 학습 프로세스
        
        Args:
            buffer: 관찰 데이터 버퍼
            num_epochs: 학습 에폭 수
            batch_size: 배치 크기
            beta: KL divergence 가중치 (β-VAE)
            val_split: 검증 데이터 비율
            save_interval: 체크포인트 저장 간격
            early_stopping_patience: Early stopping patience (None이면 비활성화)
        
        Returns:
            학습 히스토리
        """
        # 데이터 준비
        print("Preparing data...")
        inputs, targets = buffer.get_sequences(mode=self.model.config.mode)
        
        if len(inputs) == 0:
            raise ValueError("No sequences available in buffer!")
        
        print(f"Total sequences: {len(inputs)}")
        
        # Train/Val split
        dataset = TemporalVAEDataset(inputs, targets)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training for {num_epochs} epochs...")
        print(f"Train size: {train_size}, Val size: {val_size}")
        
        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, beta=beta)
            
            # Validate
            val_losses = self.evaluate(val_loader, beta=beta)
            
            # Logging
            print(f"Epoch {self.epoch}/{num_epochs}")
            print(f"  Train - Total: {train_losses['total_loss']:.4f}, "
                  f"Recon: {train_losses['recon_loss']:.4f}, "
                  f"KL: {train_losses['kl_loss']:.4f}")
            print(f"  Val   - Total: {val_losses['total_loss']:.4f}, "
                  f"Recon: {val_losses['recon_loss']:.4f}, "
                  f"KL: {val_losses['kl_loss']:.4f}")
            
            # Wandb 검증 로깅
            if self.use_wandb:
                import wandb
                wandb.log({
                    'eval/total_loss': val_losses['total_loss'],
                    'eval/recon_loss': val_losses['recon_loss'],
                    'eval/kl_loss': val_losses['kl_loss'],
                    'epoch': self.epoch
                })
            
            # Save checkpoint
            if self.checkpoint_dir and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{self.epoch}.pt")
            
            # Early stopping
            if early_stopping_patience is not None:
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    patience_counter = 0
                    if self.checkpoint_dir:
                        self.save_checkpoint("best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {self.epoch}")
                        break
        
        # Save final model
        if self.checkpoint_dir:
            self.save_checkpoint("final_model.pt")
        
        return self.training_history
    
    def train_with_dataset(self, dataset, num_epochs: int = 100, batch_size: int = 32, 
                          beta: float = 1.0, val_split: float = 0.2, 
                          save_interval: int = 10, early_stopping_patience: int = None):
        """
        Lazy dataset을 사용한 VAE 학습
        
        Args:
            dataset: LazyTemporalVAEDataset
            num_epochs: 학습 에폭 수
            batch_size: 배치 크기
            beta: KL divergence 가중치 (β-VAE)
            val_split: 검증 데이터 비율
            save_interval: 체크포인트 저장 간격
            early_stopping_patience: Early stopping patience (None이면 비활성화)
        
        Returns:
            학습 히스토리
        """
        print("Preparing lazy dataset...")
        
        # Train/Val split
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Lazy loading과 함께 사용 시 메모리 중복 방지
            pin_memory=False  # 메모리 절약
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Lazy loading과 함께 사용 시 메모리 중복 방지
            pin_memory=False  # 메모리 절약
        )
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training for {num_epochs} epochs...")
        print(f"Train samples: {train_size}, Val samples: {val_size}")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'reconstruction_loss': [],
            'kl_loss': []
        }
        
        from time import time
        start_time = time()
        
        for epoch in range(num_epochs):
            epoch_start = time()
            
            # Training
            print(f"\nEpoch {epoch+1}/{num_epochs} - Training...")
            train_loss = self.train_epoch(train_loader, beta)
            
            # Validation
            print(f"Epoch {epoch+1}/{num_epochs} - Validation...")
            val_loss = self.evaluate(val_loader, beta)
            
            # History update
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # 시간 계산
            epoch_time = time() - epoch_start
            elapsed_time = time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = num_epochs - epoch - 1
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            # Progress (매 에폭마다 출력)
            if eta_hours >= 1:
                eta_str = f"{eta_hours:.1f}h"
            else:
                eta_str = f"{eta_minutes:.1f}m"
            
            print(f"Epoch {epoch:3d}/{num_epochs}: "
                  f"Train={train_loss['total_loss']:.4f} "
                  f"Val={val_loss['total_loss']:.4f} "
                  f"Time={epoch_time:.1f}s "
                  f"ETA={eta_str}")
            
            # Checkpoint saving
            if save_interval and (epoch + 1) % save_interval == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            current_val_loss = val_loss['total_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                # Save best model
                best_model_path = self.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(best_model_path)
                print(f"✓ New best model saved (val_loss={best_val_loss:.4f}): {best_model_path}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save model every epoch
            epoch_model_path = self.checkpoint_dir / f"model_epoch_{epoch+1}.pt"
            self.save_checkpoint(epoch_model_path)
            print(f"✓ Epoch model saved: {epoch_model_path}")
            
            # Early stopping
            if early_stopping_patience:
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        print("Training completed!")
        return history
    
    def save_checkpoint(self, filename: str):
        """
        체크포인트 저장
        
        Args:
            filename: 저장할 파일명
        """
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is not set!")
        
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config.__dict__,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: Union[str, Path]):
        """
        체크포인트 로드
        
        Args:
            filepath: 로드할 체크포인트 파일
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Resuming from epoch {self.epoch}")


# 편의 함수들
def collect_episodes_from_env(
    env,
    buffer: TemporalObservationBuffer,
    num_episodes: int = 100,
    agents = None,
    encoding_fn = None
):
    """
    환경에서 에피소드를 수집하여 버퍼에 저장
    
    Args:
        env: Multi-agent 환경
        buffer: TemporalObservationBuffer
        num_episodes: 수집할 에피소드 수
        agents: 사용할 에이전트들 (None이면 랜덤 액션)
                - None: 랜덤 액션 사용
                - List[Agent]: 각 에이전트의 policy 사용
                - 지원 policy: RandomPolicy, RuleBasedPolicy, TrainedRLPolicy 등
        encoding_fn: 관찰 인코딩 함수 (None이면 환경의 기본 관찰 사용)
    
    Examples:
        # 1. 랜덤 액션으로 데이터 수집
        collect_episodes_from_env(env, buffer, num_episodes=500, agents=None)
        
        # 2. Rule-based policy로 데이터 수집
        from pymarlzooplus.envs.oai_agents.policies.rule_based import RuleBasedAgent
        agents = [RuleBasedAgent(env, i) for i in range(env.n_agents)]
        collect_episodes_from_env(env, buffer, num_episodes=500, agents=agents)
        
        # 3. 학습된 RL policy로 데이터 수집
        trained_agent = load_trained_agent('path/to/model.pt')
        agents = [trained_agent] * env.n_agents
        collect_episodes_from_env(env, buffer, num_episodes=500, agents=agents)
    """
    print(f"Collecting {num_episodes} episodes...")
    
    for ep in tqdm(range(num_episodes)):
        reset_result = env.reset()
        # env.reset()이 (obs, info) 튜플을 반환하는 경우 처리
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        done = False
        step_count = 0
        
        while not done:
            step_count += 1
            # 에이전트가 없으면 랜덤 액션
            if agents is None:
                n_actions = env.action_space[0].n if hasattr(env, 'action_space') else 6
                num_agents = env.num_agents if hasattr(env, 'num_agents') else 2
                actions = [np.random.randint(0, n_actions) for _ in range(num_agents)]
            else:
                # 각 에이전트가 액션 선택
                actions = []
                for i, agent in enumerate(agents):
                    if hasattr(agent, 'get_actions'):
                        # info에서 all_agent_obs 추출하여 전달
                        if isinstance(info, dict) and 'all_agent_obs' in info:
                            all_obs = info['all_agent_obs']
                            action = agent.get_actions(all_obs, explore=True, agent_idx=i)
                        else:
                            # obs를 직접 전달
                            if isinstance(obs, (tuple, list)) and len(obs) > i:
                                agent_obs = obs[i]
                            else:
                                agent_obs = obs
                            action = agent.get_actions(agent_obs, explore=True, agent_idx=i)
                    else:
                        n_actions = env.action_space[0].n if hasattr(env, 'action_space') else 6
                        action = np.random.randint(0, n_actions)
                    actions.append(action)
            
            # 관찰 인코딩 (필요한 경우)
            if encoding_fn is not None:
                encoded_obs = encoding_fn(obs)
            else:
                encoded_obs = obs
            
            # 각 에이전트의 관찰을 버퍼에 추가 (첫 번째 에이전트 관찰 사용)
            if isinstance(encoded_obs, (list, tuple)) and len(encoded_obs) > 0:
                buffer.add_observation(encoded_obs[0])
            else:
                buffer.add_observation(encoded_obs)
            
            # 스텝 진행
            step_result = env.step(actions)
            # Overcooked 환경은 (obs, share_obs, reward, done, info, available_actions) 반환
            if len(step_result) == 6:
                obs, share_obs, rewards, done, info, available_actions = step_result
            elif len(step_result) == 5:
                obs, share_obs, rewards, done, info = step_result
            elif len(step_result) == 4:
                obs, share_obs, rewards, done = step_result
                info = {}
            else:
                raise ValueError(f"Unexpected step_result length: {len(step_result)}")
            
            # done이 리스트인 경우 (여러 에이전트)
            if isinstance(done, (list, tuple)):
                done = any(done)
        
        # 에피소드 종료
        buffer.finish_episode()
    
    print(f"Collection complete. Buffer has {len(buffer)} episodes.")
