"""
Temporal VAE for State Encoding Enhancement

이 모듈은 multi_overcooked 환경에서 state의 인코딩 방식을 강화하기 위한 
Variational Autoencoder (VAE)를 제공합니다.

주요 기능:
1. 시계열 관찰 시퀀스 (t-k부터 t까지)를 입력으로 받아 인코딩
2. Reconstruction mode: 동일한 입력 재구성
3. Predictive mode: 다음 타임스텝 예측 (t-k+1부터 t+1까지)
4. 다양한 인코딩 스킴 지원 (OAI_feats, OAI_lossless, OAI_egocentric, raw image)
5. 설정 가능한 hidden dimension

Author: Julie
Date: 2025-10-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
from enum import Enum


class EncodingScheme(Enum):
    """지원되는 인코딩 스킴"""
    OAI_FEATS = "OAI_feats"  # Feature-based encoding (1D vector)
    OAI_LOSSLESS = "OAI_lossless"  # Lossless visual encoding (3D tensor)
    OAI_EGOCENTRIC = "OAI_egocentric"  # Egocentric visual encoding (3D tensor)
    RAW_IMAGE = "raw_image"  # Raw image representation
    OAI_RAW_IMAGE = "OAI_raw_image"  # Alias for raw_image (compatibility)


class VAEMode(Enum):
    """VAE 동작 모드"""
    RECONSTRUCTION = "reconstruction"  # t-k부터 t까지 재구성
    PREDICTIVE = "predictive"  # t-k+1부터 t+1까지 예측


@dataclass
class TemporalVAEConfig:
    """
    Temporal VAE 설정 클래스
    
    Parameters:
        k_timesteps: 포함할 이전 타임스텝 수 (k)
        mode: VAE 동작 모드 (reconstruction 또는 predictive)
        encoding_scheme: 인코딩 스킴 (OAI_feats, OAI_lossless, etc.)
        hidden_dim: VAE의 latent space dimension
        encoder_hidden_dims: Encoder의 hidden layer dimensions
        decoder_hidden_dims: Decoder의 hidden layer dimensions
        input_shape: 입력 데이터의 shape (encoding_scheme에 따라 다름)
        beta: KL divergence term의 가중치 (β-VAE)
        use_conv: 시각적 인코딩에 CNN 사용 여부
    """
    k_timesteps: int = 5
    mode: VAEMode = VAEMode.RECONSTRUCTION
    encoding_scheme: EncodingScheme = EncodingScheme.OAI_LOSSLESS
    hidden_dim: int = 128
    encoder_hidden_dims: List[int] = None
    decoder_hidden_dims: List[int] = None
    input_shape: Tuple = None  # Will be set based on encoding_scheme
    beta: float = 1.0
    use_conv: bool = True
    
    def __post_init__(self):
        """기본값 설정"""
        if self.encoder_hidden_dims is None:
            self.encoder_hidden_dims = [256, 512, 256]
        if self.decoder_hidden_dims is None:
            self.decoder_hidden_dims = [256, 512, 256]
        
        # encoding_scheme에 따른 기본 input_shape 설정
        if self.input_shape is None:
            if self.encoding_scheme == EncodingScheme.OAI_FEATS:
                self.input_shape = (96,)  # Feature vector
            elif self.encoding_scheme == EncodingScheme.OAI_LOSSLESS:
                self.input_shape = (25, 5, 13)  # (channels, height, width) - 실제 데이터 크기
            elif self.encoding_scheme == EncodingScheme.OAI_EGOCENTRIC:
                self.input_shape = (25, 5, 13)  # (channels, height, width) - 실제 데이터 크기
            elif self.encoding_scheme in [EncodingScheme.RAW_IMAGE, EncodingScheme.OAI_RAW_IMAGE]:
                self.input_shape = (3, 64, 64)  # RGB image


class TemporalEncoder(nn.Module):
    """
    시계열 관찰을 latent representation으로 인코딩하는 Encoder
    
    입력: (batch, k+1, *input_shape)
    출력: (batch, hidden_dim) - 평균과 분산
    """
    
    def __init__(self, config: TemporalVAEConfig):
        super().__init__()
        self.config = config
        self.k = config.k_timesteps
        
        # 인코딩 타입에 따라 다른 아키텍처 사용
        if config.encoding_scheme == EncodingScheme.OAI_FEATS:
            self._build_feature_encoder()
        else:
            if config.use_conv:
                self._build_conv_encoder()
            else:
                self._build_mlp_encoder()
    
    def _build_feature_encoder(self):
        """Feature-based encoding을 위한 MLP Encoder"""
        input_dim = np.prod(self.config.input_shape) * (self.k + 1)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.config.encoder_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder_net = nn.Sequential(*layers)
        
        # Latent space: 평균과 로그 분산
        self.fc_mu = nn.Linear(prev_dim, self.config.hidden_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.config.hidden_dim)
    
    def _build_conv_encoder(self):
        """시각적 인코딩을 위한 CNN Encoder"""
        c, h, w = self.config.input_shape
        # 시계열 데이터를 채널 방향으로 concat: (k+1)*c channels
        input_channels = c * (self.k + 1)
        
        # 입력 크기에 따라 다른 CNN 구조 사용
        if h <= 10 and w <= 10:
            # 작은 그리드 (6x5 등): 공간 해상도 유지에 집중
            self.conv_layers = nn.Sequential(
                # 1x1 conv로 채널 수 먼저 줄이기
                nn.Conv2d(input_channels, 64, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                
                # Spatial feature extraction (stride=1로 해상도 유지)
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                
                # Global average pooling으로 공간 차원 제거
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        else:
            # 큰 이미지 (64x64 등): 다운샘플링 적극 사용
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
                
                nn.AdaptiveAvgPool2d(1),  # Global average pooling
                nn.Flatten()
            )
        
        # MLP layers
        mlp_layers = []
        prev_dim = 256
        
        for hidden_dim in self.config.encoder_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder_net = nn.Sequential(*mlp_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, self.config.hidden_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.config.hidden_dim)
    
    def _build_mlp_encoder(self):
        """Flatten된 입력을 사용하는 MLP Encoder"""
        input_dim = np.prod(self.config.input_shape) * (self.k + 1)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.config.encoder_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder_net = nn.Sequential(*layers)
        
        # Latent space
        self.fc_mu = nn.Linear(prev_dim, self.config.hidden_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.config.hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, k+1, *input_shape)
        
        Returns:
            mu: Mean of latent distribution (batch, hidden_dim)
            logvar: Log variance of latent distribution (batch, hidden_dim)
        """
        batch_size = x.shape[0]
        
        if self.config.encoding_scheme == EncodingScheme.OAI_FEATS:
            # Flatten temporal and feature dimensions
            x = x.reshape(batch_size, -1)
            h = self.encoder_net(x)
        else:
            if self.config.use_conv:
                # Reshape: (batch, k+1, c, h, w) -> (batch, (k+1)*c, h, w)
                x = x.reshape(batch_size, -1, x.shape[-2], x.shape[-1])
                h = self.conv_layers(x)
                h = self.encoder_net(h)
            else:
                # Flatten all dimensions
                x = x.reshape(batch_size, -1)
                h = self.encoder_net(x)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class TemporalDecoder(nn.Module):
    """
    Latent representation에서 시계열 관찰을 재구성하는 Decoder
    
    입력: (batch, hidden_dim)
    출력: (batch, k+1, *input_shape) - reconstruction mode
          (batch, k+1, *input_shape) - predictive mode (shifted)
    """
    
    def __init__(self, config: TemporalVAEConfig):
        super().__init__()
        self.config = config
        self.k = config.k_timesteps
        
        # 인코딩 타입에 따라 다른 아키텍처 사용
        if config.encoding_scheme == EncodingScheme.OAI_FEATS:
            self._build_feature_decoder()
        else:
            if config.use_conv:
                self._build_conv_decoder()
            else:
                self._build_mlp_decoder()
    
    def _build_feature_decoder(self):
        """Feature-based encoding을 위한 MLP Decoder"""
        output_dim = np.prod(self.config.input_shape) * (self.k + 1)
        
        layers = []
        prev_dim = self.config.hidden_dim
        
        for hidden_dim in reversed(self.config.decoder_hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder_net = nn.Sequential(*layers)
    
    def _build_conv_decoder(self):
        """시각적 인코딩을 위한 Deconvolution Decoder"""
        c, h, w = self.config.input_shape
        output_channels = c * (self.k + 1)
        
        # 입력 크기에 따라 다른 구조 사용
        if h <= 10 and w <= 10:
            # 작은 그리드 (5x13 등): 공간 해상도 유지에 집중
            # latent vector를 feature map으로 reshape할 크기 계산
            self.latent_h = 2  # 작은 크기
            self.latent_w = 2
            self.latent_c = self.config.hidden_dim // (self.latent_h * self.latent_w)
            
            # hidden_dim이 나누어떨어지지 않는 경우 처리
            if self.config.hidden_dim % (self.latent_h * self.latent_w) != 0:
                self.latent_c += 1
                self.actual_latent_size = self.latent_c * self.latent_h * self.latent_w
            else:
                self.actual_latent_size = self.config.hidden_dim
            
            # 작은 크기에 맞는 Deconvolution layers
            self.deconv_layers = nn.Sequential(
                # 2x2 → 4x4
                nn.ConvTranspose2d(self.latent_c, 128, kernel_size=3, stride=2, padding=0),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                
                # 4x4 → 원하는 크기로 조정
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                
                # 최종 채널 수 맞추기
                nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
                
                # Adaptive pooling으로 정확한 크기 맞추기
                nn.AdaptiveAvgPool2d((h, w))
            )
        else:
            # 큰 이미지 (64x64 등): 기존 구조 사용
            # latent vector를 feature map으로 reshape할 크기 계산
            self.latent_h = 4  # 고정 spatial 크기
            self.latent_w = 4
            self.latent_c = self.config.hidden_dim // (self.latent_h * self.latent_w)
            
            # hidden_dim이 나누어떨어지지 않는 경우 처리
            if self.config.hidden_dim % (self.latent_h * self.latent_w) != 0:
                # 나머지를 처리하기 위해 추가 채널 사용
                self.latent_c += 1
                # 실제 사용할 크기 계산
                self.actual_latent_size = self.latent_c * self.latent_h * self.latent_w
            else:
                self.actual_latent_size = self.config.hidden_dim
            
            # Deconvolution layers
            self.deconv_layers = nn.Sequential(
                # 4x4 → 8x8
                nn.ConvTranspose2d(self.latent_c, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                
                # 8x8 → 16x16
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                
                # 최종 채널 수 맞추기
                nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
                
                # Adaptive pooling으로 정확한 크기 맞추기
                nn.AdaptiveAvgPool2d((h, w))
            )
        
        # Raw image 및 OAI_lossless의 경우 [0, 1] 범위를 위한 Sigmoid
        # OAI_lossless도 정규화된 데이터를 사용하므로 sigmoid 필요
        # 단, BCEWithLogitsLoss 사용 시에는 sigmoid를 제거해야 함
        self.use_sigmoid = (self.config.encoding_scheme in [
            EncodingScheme.RAW_IMAGE, 
            EncodingScheme.OAI_LOSSLESS
        ])
        
        # BCEWithLogitsLoss 사용 여부 (외부에서 설정 가능)
        self.use_bce_with_logits = False
    
    def _build_mlp_decoder(self):
        """Flatten된 출력을 생성하는 MLP Decoder"""
        output_dim = np.prod(self.config.input_shape) * (self.k + 1)
        
        layers = []
        prev_dim = self.config.hidden_dim
        
        for hidden_dim in reversed(self.config.decoder_hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder_net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            z: Latent representation (batch, hidden_dim)
        
        Returns:
            reconstruction: Reconstructed observations (batch, k+1, *input_shape)
        """
        batch_size = z.shape[0]
        
        if self.config.encoding_scheme == EncodingScheme.OAI_FEATS:
            x_recon = self.decoder_net(z)
            # Reshape to (batch, k+1, feature_dim)
            x_recon = x_recon.reshape(batch_size, self.k + 1, -1)
        else:
            if self.config.use_conv:
                # latent vector를 feature map으로 reshape
                # hidden_dim이 나누어떨어지지 않는 경우 패딩 추가
                if self.actual_latent_size > self.config.hidden_dim:
                    # 패딩으로 크기 맞추기
                    padding_size = self.actual_latent_size - self.config.hidden_dim
                    z_padded = torch.cat([z, torch.zeros(batch_size, padding_size, device=z.device)], dim=1)
                else:
                    z_padded = z
                
                z_reshaped = z_padded.view(batch_size, self.latent_c, self.latent_h, self.latent_w)
                
                # Deconvolution으로 upsampling
                x_recon = self.deconv_layers(z_reshaped)
                
                # BCEWithLogitsLoss 사용 시: sigmoid 제거 (raw logit 반환)
                # 일반 BCE 사용 시: sigmoid 적용하여 [0, 1] 범위 보장
                if self.use_bce_with_logits:
                    # sigmoid 제거 - raw logit 반환
                    pass
                elif self.use_sigmoid:
                    x_recon = torch.sigmoid(x_recon)
                else:
                    # Sigmoid를 사용하지 않는 경우 Min-Max 정규화
                    min_val = x_recon.min()
                    max_val = x_recon.max()
                    if max_val > min_val:
                        x_recon = (x_recon - min_val) / (max_val - min_val)
                
                # Reshape: (batch, (k+1)*c, h, w) -> (batch, k+1, c, h, w)
                c, h, w = self.config.input_shape
                x_recon = x_recon.reshape(batch_size, self.k + 1, c, h, w)
            else:
                x_recon = self.decoder_net(z)
                # Reshape to (batch, k+1, *input_shape)
                x_recon = x_recon.reshape(batch_size, self.k + 1, *self.config.input_shape)
                
                # BCEWithLogitsLoss 사용 시: sigmoid 제거 (raw logit 반환)
                # 일반 BCE 사용 시: sigmoid 적용하여 [0, 1] 범위 보장
                if self.use_bce_with_logits:
                    # sigmoid 제거 - raw logit 반환
                    pass
                elif self.use_sigmoid:
                    x_recon = torch.sigmoid(x_recon)
        
        return x_recon


class TemporalVAE(nn.Module):
    """
    시계열 관찰을 위한 Variational Autoencoder
    
    이 모듈은 multi_overcooked 환경의 state encoding을 강화합니다.
    
    Features:
        - 시계열 입력 (t-k부터 t까지)
        - Reconstruction 또는 Predictive mode
        - 다양한 인코딩 스킴 지원
        - β-VAE로 disentangled representation 학습 가능
    """
    
    def __init__(self, config: TemporalVAEConfig):
        super().__init__()
        self.config = config
        
        self.encoder = TemporalEncoder(config)
        self.decoder = TemporalDecoder(config)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input sequence (batch, k+1, *input_shape)
        
        Returns:
            x_recon: Reconstructed sequence
            mu: Latent mean
            logvar: Latent log variance
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation (inference only)
        
        Args:
            x: Input sequence (batch, k+1, *input_shape)
        
        Returns:
            z: Latent representation (batch, hidden_dim)
        """
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to observations
        
        Args:
            z: Latent representation (batch, hidden_dim)
        
        Returns:
            x_recon: Reconstructed observations (batch, k+1, *input_shape)
        """
        return self.decoder(z)
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        강화학습 에이전트에서 사용할 latent representation 추출
        
        Args:
            x: Input sequence (batch, k+1, *input_shape)
        
        Returns:
            z: Latent representation (batch, hidden_dim)
        """
        with torch.no_grad():
            mu, _ = self.encoder(x)
            return mu  # Use mean instead of sampling for deterministic inference


def vae_loss(x_recon: torch.Tensor, 
             x_target: torch.Tensor,
             mu: torch.Tensor,
             logvar: torch.Tensor,
             beta: float = 1.0,
             use_bce: bool = False,
             pos_weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    VAE loss 계산
    
    Args:
        x_recon: Reconstructed observations (logits if use_bce_with_logits, otherwise [0,1] range)
        x_target: Target observations ([0, 1] 범위)
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term (β-VAE)
        use_bce: Binary Cross Entropy 사용 여부 (이미지의 경우 True 권장)
        pos_weight: Channel-wise positive weights for BCEWithLogitsLoss (None이면 사용 안 함)
    
    Returns:
        Dictionary containing total loss and components
    """
    # Reconstruction loss
    if use_bce:
        if pos_weight is not None:
            # BCEWithLogitsLoss 사용 (pos_weight 적용)
            # x_recon은 logits (sigmoid 통과 전), x_target은 [0, 1] 범위
            # pos_weight shape: (C,) - 채널별 가중치
            # x_recon shape: (batch, k+1, C, H, W) 또는 (batch, k+1, *obs_shape)
            # pos_weight를 올바른 shape으로 확장해야 함
            if x_recon.ndim == 5 and pos_weight.ndim == 1:
                # (batch, k+1, C, H, W) 형태
                # pos_weight를 (1, 1, C, 1, 1)로 확장
                pos_weight_expanded = pos_weight.view(1, 1, -1, 1, 1)
            elif x_recon.ndim == 3:
                # Feature-based: pos_weight 사용 불가
                pos_weight_expanded = None
            else:
                pos_weight_expanded = pos_weight
            
            if pos_weight_expanded is not None:
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_expanded, reduction='sum')
            else:
                criterion = nn.BCEWithLogitsLoss(reduction='sum')
            recon_loss = criterion(x_recon, x_target)
        else:
            # 기존 방식: Binary Cross Entropy (sigmoid가 이미 적용된 경우)
            # x_recon과 x_target 모두 [0, 1] 범위여야 함
            recon_loss = F.binary_cross_entropy(x_recon, x_target, reduction='sum')
    else:
        # Mean Squared Error (기본)
        recon_loss = F.mse_loss(x_recon, x_target, reduction='sum')
    
    # KL divergence loss
    # KL(q(z|x) || p(z)) where p(z) = N(0, I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }


# 편의 함수들
def create_reconstruction_vae(
    k_timesteps: int = 5,
    encoding_scheme: str = "OAI_lossless",
    hidden_dim: int = 128,
    input_shape: Optional[Tuple] = None,
    **kwargs
) -> TemporalVAE:
    """
    Reconstruction mode VAE 생성
    
    Args:
        k_timesteps: 포함할 이전 타임스텝 수
        encoding_scheme: 인코딩 스킴
        hidden_dim: Latent dimension
        input_shape: 입력 shape (None이면 자동 설정)
        **kwargs: 추가 설정
    
    Returns:
        TemporalVAE 모델
    """
    config = TemporalVAEConfig(
        k_timesteps=k_timesteps,
        mode=VAEMode.RECONSTRUCTION,
        encoding_scheme=EncodingScheme(encoding_scheme),
        hidden_dim=hidden_dim,
        input_shape=input_shape,
        **kwargs
    )
    return TemporalVAE(config)


def create_predictive_vae(
    k_timesteps: int = 5,
    encoding_scheme: str = "OAI_lossless",
    hidden_dim: int = 128,
    input_shape: Optional[Tuple] = None,
    **kwargs
) -> TemporalVAE:
    """
    Predictive mode VAE 생성
    
    Args:
        k_timesteps: 포함할 이전 타임스텝 수
        encoding_scheme: 인코딩 스킴
        hidden_dim: Latent dimension
        input_shape: 입력 shape (None이면 자동 설정)
        **kwargs: 추가 설정
    
    Returns:
        TemporalVAE 모델
    """
    config = TemporalVAEConfig(
        k_timesteps=k_timesteps,
        mode=VAEMode.PREDICTIVE,
        encoding_scheme=EncodingScheme(encoding_scheme),
        hidden_dim=hidden_dim,
        input_shape=input_shape,
        **kwargs
    )
    return TemporalVAE(config)

