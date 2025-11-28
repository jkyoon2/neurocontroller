"""
VAE (Variational Autoencoder) 모듈

Temporal VAE를 사용하여 Overcooked 환경의 state encoding을 강화합니다.
"""

# Temporal VAE for State Encoding
from .temporal_vae import (
    TemporalVAE,
    TemporalVAEConfig,
    TemporalEncoder,
    TemporalDecoder,
    EncodingScheme,
    VAEMode,
    vae_loss,
    create_reconstruction_vae,
    create_predictive_vae
)

from .temporal_vae_trainer import (
    TemporalObservationBuffer,
    TemporalVAEDataset,
    TemporalVAETrainer,
    collect_episodes_from_env
)

from .temporal_vae_integration import (
    VAEEnhancedEncoder,
    VAEIntegratedObservationEncoder,
    save_vae_model,
    load_vae_model,
    save_encoder_only,
    load_encoder_only
)

from .lazy_vae_dataset import (
    LazyVAEDataset,
    CachedVAEDataset
)

# API 래퍼들은 gym_environments 디렉토리로 이동됨

# 편의 함수들은 각각의 래퍼 모듈에서 제공됨

# 버전 정보
__version__ = "1.0.0"

# 주요 클래스들을 __all__에 포함
__all__ = [
    # Temporal VAE 클래스들
    'TemporalVAE',
    'TemporalVAEConfig',
    'TemporalEncoder',
    'TemporalDecoder',
    'EncodingScheme',
    'VAEMode',
    'vae_loss',
    'create_reconstruction_vae',
    'create_predictive_vae',
    
    # Temporal VAE 트레이닝
    'TemporalObservationBuffer',
    'TemporalVAEDataset',
    'TemporalVAETrainer',
    'collect_episodes_from_env',
    
    # Temporal VAE 통합
    'VAEEnhancedEncoder',
    'VAEIntegratedObservationEncoder',
    'save_vae_model',
    'load_vae_model',
    'save_encoder_only',
    'load_encoder_only',
    
    # Lazy Dataset
    'LazyVAEDataset',
    'CachedVAEDataset',
]
