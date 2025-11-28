"""
Temporal VAE Integration

VAE를 강화학습 에이전트와 통합하는 유틸리티
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

from .temporal_vae import TemporalVAE, TemporalVAEConfig, EncodingScheme, VAEMode


class VAEEnhancedEncoder:
    """VAE latent representation을 제공하는 래퍼"""
    
    def __init__(self, vae_model: TemporalVAE, k_timesteps: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.vae = vae_model.to(device)
        self.vae.eval()
        self.k = k_timesteps
        self.device = device
        self.observation_history = []
    
    def reset(self):
        """에피소드 시작 시 히스토리 초기화"""
        self.observation_history = []
    
    def encode(self, obs: Union[np.ndarray, Dict]) -> np.ndarray:
        """관찰을 VAE latent representation으로 인코딩"""
        if isinstance(obs, dict):
            obs_data = obs.get('visual_obs', obs.get('agent_obs', obs.get('obs', obs)))
        else:
            obs_data = obs
        
        self.observation_history.append(obs_data)
        
        # k+1개가 모일 때까지는 첫 관찰 복제
        if len(self.observation_history) < self.k + 1:
            padded_history = [self.observation_history[0]] * (self.k + 1 - len(self.observation_history))
            padded_history.extend(self.observation_history)
            sequence = np.array(padded_history)
        else:
            sequence = np.array(self.observation_history[-(self.k + 1):])
        
        with torch.no_grad():
            sequence_tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(self.device)
            latent_repr = self.vae.get_latent_representation(sequence_tensor)
            return latent_repr.cpu().numpy().squeeze()
    
    def encode_batch(self, sequences: np.ndarray) -> np.ndarray:
        """배치 관찰 시퀀스를 인코딩"""
        with torch.no_grad():
            sequences_tensor = torch.from_numpy(sequences).float().to(self.device)
            latent_reprs = self.vae.get_latent_representation(sequences_tensor)
            return latent_reprs.cpu().numpy()


class VAEIntegratedObservationEncoder:
    """기존 observation encoder와 VAE를 통합"""
    
    def __init__(self, base_encoding_fn, vae_encoder: Optional[VAEEnhancedEncoder] = None,
                 use_vae_only: bool = False):
        self.base_encoding_fn = base_encoding_fn
        self.vae_encoder = vae_encoder
        self.use_vae_only = use_vae_only
    
    def reset(self):
        """에피소드 시작 시 VAE 히스토리 초기화"""
        if self.vae_encoder:
            self.vae_encoder.reset()
    
    def encode(self, raw_state, mdp, player_idx: int, **kwargs) -> Dict:
        """통합 인코딩"""
        base_obs = self.base_encoding_fn(mdp, raw_state, **kwargs)
        
        if self.vae_encoder is None:
            return base_obs
        
        if isinstance(base_obs, dict):
            obs_for_vae = base_obs.get('visual_obs', base_obs.get('agent_obs', base_obs))
        else:
            obs_for_vae = base_obs
        
        vae_latent = self.vae_encoder.encode(obs_for_vae)
        
        if self.use_vae_only:
            return {'vae_latent': vae_latent}
        
        if isinstance(base_obs, dict):
            result = base_obs.copy()
            result['vae_latent'] = vae_latent
        else:
            result = {'base_obs': base_obs, 'vae_latent': vae_latent}
        
        return result


def save_vae_model(model: TemporalVAE, filepath: Union[str, Path],
                   additional_info: Optional[Dict] = None):
    """VAE 모델 저장"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': model.config.__dict__,
        'additional_info': additional_info or {}
    }
    
    torch.save(save_dict, filepath)
    print(f"VAE model saved: {filepath}")


def load_vae_model(filepath: Union[str, Path],
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                   ) -> Tuple[TemporalVAE, Dict]:
    """VAE 모델 로드"""
    checkpoint = torch.load(filepath, map_location=device)
    
    config_dict = checkpoint['config']
    config_dict['mode'] = VAEMode(config_dict['mode'])
    config_dict['encoding_scheme'] = EncodingScheme(config_dict['encoding_scheme'])
    config = TemporalVAEConfig(**config_dict)
    
    model = TemporalVAE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    additional_info = checkpoint.get('additional_info', {})
    
    print(f"VAE model loaded: {filepath}")
    return model, additional_info


def save_encoder_only(model: TemporalVAE, filepath: Union[str, Path],
                      additional_info: Optional[Dict] = None):
    """
    Encoder만 저장 (차원 압축 목적)
    
    실제 RL agent에서는 encoder만 사용하므로 encoder만 저장하여 메모리 절약
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'encoder_state_dict': model.encoder.state_dict(),
        'config': model.config.__dict__,
        'additional_info': additional_info or {}
    }
    
    torch.save(save_dict, filepath)
    print(f"VAE Encoder saved: {filepath}")


def load_encoder_only(filepath: Union[str, Path],
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Encoder만 로드
    
    Returns:
        encoder: TemporalEncoder 모델
        config: TemporalVAEConfig
        additional_info: 추가 정보
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    config_dict = checkpoint['config']
    config_dict['mode'] = VAEMode(config_dict['mode'])
    config_dict['encoding_scheme'] = EncodingScheme(config_dict['encoding_scheme'])
    config = TemporalVAEConfig(**config_dict)
    
    # Encoder만 생성
    from .temporal_vae import TemporalEncoder
    encoder = TemporalEncoder(config)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder = encoder.to(device)
    encoder.eval()
    
    additional_info = checkpoint.get('additional_info', {})
    
    print(f"VAE Encoder loaded: {filepath}")
    return encoder, config, additional_info
