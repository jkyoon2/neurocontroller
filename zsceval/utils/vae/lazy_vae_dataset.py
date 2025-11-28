"""
Lazy Loading VAE Dataset for Memory-Efficient Training
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
from .temporal_vae import VAEMode


class LazyVAEDataset(Dataset):
    """
    메모리 효율적인 Lazy Loading VAE Dataset
    
    파일 레벨 LRU 캐싱을 사용하여 매번 파일을 로드하지 않습니다.
    """
    
    def __init__(self, buffer_files: List[str], k_timesteps: int = 5, 
                 mode: VAEMode = VAEMode.RECONSTRUCTION,
                 encoding: str = 'OAI_lossless',
                 max_cached_files: int = 3):
        """
        Args:
            buffer_files: 버퍼 파일 경로 리스트
            k_timesteps: 시퀀스 길이
            mode: VAE 모드
            encoding: 인코딩 타입 (raw_image, OAI_raw_image, OAI_lossless 등)
            max_cached_files: 메모리에 캐싱할 최대 파일 수
        """
        self.buffer_files = [Path(f) for f in buffer_files]
        self.k = k_timesteps
        self.mode = mode
        self.encoding = encoding
        self.max_cached_files = max_cached_files
        
        # 파일 레벨 LRU 캐시
        self.file_cache: OrderedDict[int, Dict] = OrderedDict()
        
        # 각 파일의 시퀀스 정보 미리 계산 (메타데이터만 로드)
        self.file_sequence_counts = []
        self.file_sequence_offsets = [0]
        self.file_episode_counts = []
        
        print(f"Scanning {len(self.buffer_files)} buffer files...")
        for i, file_path in enumerate(self.buffer_files):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # 이 파일에서 생성 가능한 시퀀스 수 계산
            num_sequences = 0
            num_episodes = len(data['episodes'])
            
            for episode in data['episodes']:
                episode_length = len(episode)
                if mode == VAEMode.RECONSTRUCTION:
                    num_sequences += max(0, episode_length - k_timesteps)
                else:  # PREDICTIVE
                    num_sequences += max(0, episode_length - k_timesteps - 1)
            
            self.file_sequence_counts.append(num_sequences)
            self.file_episode_counts.append(num_episodes)
            self.file_sequence_offsets.append(
                self.file_sequence_offsets[-1] + num_sequences
            )
            
            if i % 5 == 0:
                print(f"  Scanned {i+1}/{len(self.buffer_files)} files...")
        
        self.total_sequences = self.file_sequence_offsets[-1]
        print(f"✓ Total sequences across {len(self.buffer_files)} files: {self.total_sequences:,}")
        print(f"✓ Total episodes: {sum(self.file_episode_counts):,}")
        print(f"✓ File-level caching enabled (max {max_cached_files} files in memory)")
    
    def _load_file(self, file_idx: int) -> Dict:
        """파일을 로드하거나 캐시에서 가져옴 (LRU)"""
        # 캐시에 있으면 반환 (LRU 업데이트)
        if file_idx in self.file_cache:
            self.file_cache.move_to_end(file_idx)
            return self.file_cache[file_idx]
        
        # 파일 로드
        with open(self.buffer_files[file_idx], 'rb') as f:
            data = pickle.load(f)
        
        # 캐시에 추가
        self.file_cache[file_idx] = data
        self.file_cache.move_to_end(file_idx)
        
        # 캐시 크기 제한 (LRU eviction)
        if len(self.file_cache) > self.max_cached_files:
            oldest_file_idx = next(iter(self.file_cache))
            del self.file_cache[oldest_file_idx]
        
        return data
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        인덱스에 해당하는 시퀀스를 로드
        
        Returns:
            input_seq: (k+1, *obs_shape)
            target_seq: (k+1, *obs_shape)
        """
        # 어느 파일의 몇 번째 시퀀스인지 찾기
        file_idx = 0
        for i, offset in enumerate(self.file_sequence_offsets[1:]):
            if idx < offset:
                file_idx = i
                break
        
        local_idx = idx - self.file_sequence_offsets[file_idx]
        
        # 파일 로드 (캐시 사용)
        data = self._load_file(file_idx)
        
        # 시퀀스 추출
        current_idx = 0
        for episode in data['episodes']:
            episode_length = len(episode)
            max_seqs = max(0, episode_length - self.k - (1 if self.mode == VAEMode.PREDICTIVE else 0))
            
            if current_idx + max_seqs > local_idx:
                # 이 에피소드에서 시퀀스 추출
                t = local_idx - current_idx + self.k
                
                input_seq = episode[t - self.k:t + 1]
                
                if self.mode == VAEMode.RECONSTRUCTION:
                    target_seq = input_seq
                else:  # PREDICTIVE
                    target_seq = episode[t - self.k + 1:t + 2]
                
                # numpy array를 torch tensor로 변환
                input_tensor = torch.from_numpy(input_seq).float()
                target_tensor = torch.from_numpy(target_seq).float()
                
                # 인코딩 타입에 따라 다른 처리
                # Min-Max 정규화 함수 (0~1 범위로)
                def normalize_to_01(tensor):
                    """Min-Max 정규화: [0, 1] 범위로 변환"""
                    min_val = tensor.min()
                    max_val = tensor.max()
                    if max_val > min_val:  # 0으로 나누기 방지
                        return (tensor - min_val) / (max_val - min_val)
                    else:
                        return tensor  # 모든 값이 같으면 그대로
                
                if self.encoding in ['raw_image', 'OAI_raw_image']:
                    # Raw image: Min-Max 정규화 [0, 1] (BCE loss를 위해)
                    input_tensor = normalize_to_01(input_tensor)
                    target_tensor = normalize_to_01(target_tensor)
                elif self.encoding in ['OAI_lossless']:
                    # Lossless: (W, H, C) -> (C, H, W) 변환 (VAE는 (C, H, W) 형태 기대)
                    # input_seq shape: (k+1, W, H, C)
                    if input_tensor.ndim == 4:  # (k+1, W, H, C)
                        # (k+1, W, H, C) -> (k+1, C, H, W)
                        input_tensor = input_tensor.permute(0, 3, 2, 1)
                        target_tensor = target_tensor.permute(0, 3, 2, 1)
                    
                    # Min-Max 정규화 추가 (0~1 범위로, BCE loss를 위해)
                    # 전체 배치의 최대값으로 정규화 (채널별이 아닌 전체)
                    input_tensor = normalize_to_01(input_tensor)
                    target_tensor = normalize_to_01(target_tensor)
                # 다른 인코딩 타입은 그대로 사용
                
                return (input_tensor, target_tensor)
            
            current_idx += max_seqs
        
        raise IndexError(f"Index {idx} out of range")
    
    def get_sample_shape(self) -> Tuple:
        """샘플의 shape 반환 (input_shape 확인용)"""
        if len(self) == 0:
            raise ValueError("Dataset is empty!")
        sample_input, _ = self[0]
        return sample_input.shape  # (k+1, C, H, W) or (k+1, *obs_shape)


class CachedVAEDataset(Dataset):
    """
    일부 캐싱을 사용하는 VAE Dataset (중간 방식)
    
    자주 사용되는 데이터는 캐싱하여 성능 향상
    """
    
    def __init__(self, buffer_files: List[str], k_timesteps: int = 5,
                 mode: VAEMode = VAEMode.RECONSTRUCTION,
                 cache_size: int = 1000):
        """
        Args:
            cache_size: 캐시할 시퀀스 수
        """
        self.lazy_dataset = LazyVAEDataset(buffer_files, k_timesteps, mode)
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}
    
    def __len__(self):
        return len(self.lazy_dataset)
    
    def __getitem__(self, idx: int):
        # 캐시에 있으면 반환
        if idx in self.cache:
            self.access_count[idx] = self.access_count.get(idx, 0) + 1
            return self.cache[idx]
        
        # 캐시에 없으면 로드
        item = self.lazy_dataset[idx]
        
        # 캐시가 꽉 찼으면 가장 덜 사용된 항목 제거
        if len(self.cache) >= self.cache_size:
            least_used = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_used]
            del self.access_count[least_used]
        
        # 캐시에 추가
        self.cache[idx] = item
        self.access_count[idx] = 1
        
        return item

