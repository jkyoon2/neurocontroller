"""
VAE 데이터 수집 스크립트

랜덤 policy와 학습된 체크포인트들을 사용하여 VAE 학습용 데이터를 수집합니다.

Usage:
    # 랜덤 policy만
    python -m zsceval.utils.vae.collect_vae_data --layout forced_coordination
    
    # 학습된 체크포인트 자동 로드 (MAPPO)
    python -m zsceval.utils.vae.collect_vae_data \
        --layout forced_coordination \
        --checkpoint-dir results/models/single_layout_forced_coordination \
        --algorithm mappo
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
from types import SimpleNamespace

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args
from zsceval.envs.env_wrappers import ShareDummyVecEnv
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from zsceval.runner.separated.base_runner import make_trainer_policy_cls
from zsceval.utils.train_util import setup_seed
from .temporal_vae_trainer import TemporalObservationBuffer
from .temporal_vae import EncodingScheme


def _t2n(x):
    """Tensor to numpy"""
    return x.detach().cpu().numpy()


class RandomPolicy:
    """랜덤 액션 policy"""
    def __init__(self, n_actions):
        self.n_actions = n_actions
    
    def get_actions(self, obs, explore=True):
        return np.random.randint(0, self.n_actions)


def find_checkpoints(checkpoint_dir):
    """디렉토리에서 모든 체크포인트 찾기 (서브디렉토리 포함)"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    
    # 1. 직접 .pt, .pth, .th 파일 찾기
    for ext in ['*.pt', '*.pth', '*.th']:
        checkpoints.extend(checkpoint_dir.glob(ext))
    
    # 2. 서브디렉토리의 agent.th 파일 찾기 (MAPPO 형식)
    for subdir in checkpoint_dir.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():  # 숫자 이름의 디렉토리
            agent_file = subdir / 'agent.th'
            if agent_file.exists():
                checkpoints.append(agent_file)
    
    checkpoints.sort()  # 파일명 순 정렬
    
    return checkpoints


def collect_episodes_with_trainers(
    eval_envs,
    buffers: list,  # [수정] 단일 buffer -> buffers 리스트
    trainers,
    num_episodes: int,
    all_args,
    encoding_fn=None
):
    """
    Trainer를 사용하여 에피소드 수집 (post-hoc-eval.ipynb와 동일한 방식)
    
    Args:
        eval_envs: ShareDummyVecEnv 환경 (post-hoc-eval.ipynb와 동일)
        buffers: List[TemporalObservationBuffer] - 각 에이전트별 버퍼 리스트
        trainers: List[Trainer] - 각 에이전트별 Trainer 리스트
        num_episodes: 수집할 에피소드 수
        all_args: 환경 설정 args
        encoding_fn: 관찰 인코딩 함수
    """
    num_agents = len(trainers)
    
    print(f"Collecting {num_episodes} episodes with trained policies...")
    
    for ep in tqdm(range(num_episodes)):
        # 환경 리셋 (post-hoc-eval.ipynb와 동일)
        eval_obs_batch, eval_info_list = eval_envs.reset()
        
        # 관찰 추출 (post-hoc-eval.ipynb와 동일)
        eval_obs = np.array([info['all_agent_obs'] for info in eval_info_list])
        eval_available_actions = np.array([info['available_actions'] for info in eval_info_list])
        
        # RNN states 초기화 (post-hoc-eval.ipynb와 동일)
        eval_rnn_states = np.zeros(
            (
                all_args.n_eval_rollout_threads,
                num_agents,
                all_args.recurrent_N,
                all_args.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((all_args.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
        
        done = False
        step_count = 0
        
        while not done:
            step_count += 1
            eval_actions = []
            
            # 각 에이전트별로 액션 선택 (post-hoc-eval.ipynb와 동일)
            for agent_id in range(num_agents):
                trainers[agent_id].prep_rollout()
                eval_action, eval_rnn_state = trainers[agent_id].policy.act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id],
                    deterministic=False,  # Exploration (데이터 수집용)
                )
                
                eval_action = _t2n(eval_action)
                eval_actions.append(eval_action)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
            
            # Action 변환 (post-hoc-eval.ipynb와 동일)
            eval_actions = np.stack(eval_actions).transpose(1, 0, 2)
            
            # [수정] 관찰 인코딩 및 각 에이전트 버퍼에 추가
            for agent_id in range(num_agents):
                # 각 에이전트의 현재 관찰 가져오기
                curr_obs = eval_obs[0, agent_id]
                
                if encoding_fn is not None:
                    encoded_obs = encoding_fn(curr_obs)
                else:
                    encoded_obs = curr_obs
                
                # 해당 에이전트의 전용 버퍼에 추가
                buffers[agent_id].add_observation(encoded_obs)
            
            # 환경 스텝 (post-hoc-eval.ipynb와 동일)
            (
                _eval_obs_batch_single_agent,
                _,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = eval_envs.step(eval_actions)
            
            # 관찰 업데이트 (post-hoc-eval.ipynb와 동일)
            eval_obs = np.array([info['all_agent_obs'] for info in eval_infos])
            
            # RNN states 및 masks 업데이트 (post-hoc-eval.ipynb와 동일)
            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), all_args.recurrent_N, all_args.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((all_args.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            # 에피소드 종료 확인
            if np.all(eval_dones):
                done = True
        
        # [수정] 모든 버퍼의 에피소드 종료 처리
        for buffer in buffers:
            buffer.finish_episode()
    
    # 총 수집된 에피소드 수 출력 (모든 버퍼 합산)
    total_eps = sum(len(b) for b in buffers)
    print(f"Collection complete. Total accumulated episodes across all agents: {total_eps}")


def collect_episodes_random(
    eval_envs,
    buffers: list,  # [수정] 단일 buffer -> buffers 리스트
    num_episodes: int,
    all_args,
    encoding_fn=None
):
    """랜덤 액션으로 에피소드 수집 (post-hoc-eval.ipynb와 동일한 환경 상호작용 방식)"""
    num_agents = all_args.num_agents
    n_actions = eval_envs.action_space[0].n
    
    print(f"Collecting {num_episodes} episodes with random actions...")
    
    for ep in tqdm(range(num_episodes)):
        # 환경 리셋 (post-hoc-eval.ipynb와 동일)
        eval_obs_batch, eval_info_list = eval_envs.reset()
        
        # 관찰 추출 (post-hoc-eval.ipynb와 동일)
        eval_obs = np.array([info['all_agent_obs'] for info in eval_info_list])
        
        done = False
        
        while not done:
            # 랜덤 액션 (ShareDummyVecEnv 형식: (n_envs, num_agents, 1))
            eval_actions = np.random.randint(0, n_actions, size=(all_args.n_eval_rollout_threads, num_agents, 1))
            
            # [수정] 모든 에이전트 순회하며 데이터 수집
            for agent_id in range(num_agents):
                curr_obs = eval_obs[0, agent_id]
                
                if encoding_fn is not None:
                    encoded_obs = encoding_fn(curr_obs)
                else:
                    encoded_obs = curr_obs
                
                buffers[agent_id].add_observation(encoded_obs)
            
            # 환경 스텝 (post-hoc-eval.ipynb와 동일)
            (
                _eval_obs_batch_single_agent,
                _,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = eval_envs.step(eval_actions)
            
            # 관찰 업데이트 (post-hoc-eval.ipynb와 동일)
            eval_obs = np.array([info['all_agent_obs'] for info in eval_infos])
            
            # 에피소드 종료 확인
            if np.all(eval_dones):
                done = True
        
        # [수정] 모든 버퍼 종료
        for buffer in buffers:
            buffer.finish_episode()
    
    # 총 수집된 에피소드 수 출력 (모든 버퍼 합산)
    total_eps = sum(len(b) for b in buffers)
    print(f"Collection complete. Total accumulated episodes across all agents: {total_eps}")


def main():
    parser = argparse.ArgumentParser(description='Collect VAE training data')
    
    # 환경 설정
    parser.add_argument('--layout', type=str, default='3_chefs_smartfactory')
    parser.add_argument('--num-agents', type=int, default=3)
    parser.add_argument('--encoding', type=str, default='OAI_lossless')
    
    # 데이터 수집 설정
    parser.add_argument('--episodes-random', type=int, default=200,
                       help='랜덤 policy로 수집할 에피소드 수')
    parser.add_argument('--episodes-per-checkpoint', type=int, default=10,
                       help='각 체크포인트당 수집할 에피소드 수')
    parser.add_argument('--k-timesteps', type=int, default=4)
    parser.add_argument('--image-size', type=int, nargs=2, default=[16, 16],
                       help='raw_image 인코딩 시 이미지 크기 (기본: 16x16)')
    parser.add_argument('--obs-width', type=int, default=None,
                       help='패딩 후 observation 가로 길이 (예: 16)')
    parser.add_argument('--obs-height', type=int, default=None,
                       help='패딩 후 observation 세로 길이 (예: 16)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='중간 저장 간격 (체크포인트 개수)')
    parser.add_argument('--clear-buffer-after-save', action='store_true',
                       help='중간 저장 후 버퍼 비우기 (메모리 절약)')
    
    # 체크포인트 설정
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='체크포인트 디렉토리 (예: results/models/single_layout_forced_coordination)')
    parser.add_argument('--checkpoint-dirs', type=str, nargs='+', default=None,
                       help='여러 체크포인트 디렉토리 리스트 (예: dir1 dir2 dir3)')
    parser.add_argument('--checkpoint-files', type=str, nargs='+', default=None,
                       help='특정 체크포인트 파일 경로 리스트 (예: agent0.pt agent1.pt)')
    parser.add_argument('--checkpoint-step', type=int, default=None,
                       help='특정 스텝의 체크포인트만 선택 (checkpoint-dir과 함께 사용)')
    parser.add_argument('--algorithm', type=str, default='mappo',
                       choices=['mappo', 'mappo_ns', 'rmappo'],
                       help='RL 알고리즘')
    
    # 저장 설정
    parser.add_argument('--save-path', type=str, default='./vae_data/buffer_raw_image.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VAE Data Collection")
    print("=" * 80)
    print(f"Layout: {args.layout}")
    print(f"Encoding: {args.encoding}")
    if args.encoding in ['raw_image', 'OAI_raw_image']:
        print(f"Image size: {args.image_size[0]}x{args.image_size[1]}")
    if args.obs_width is not None and args.obs_height is not None:
        print(f"Observation padding: {args.obs_width}x{args.obs_height}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Random episodes: {args.episodes_random}")
    if args.checkpoint_dir:
        print(f"Checkpoint dir: {args.checkpoint_dir}")
        print(f"Episodes per checkpoint: {args.episodes_per_checkpoint}")
    if args.checkpoint_dirs:
        print(f"Checkpoint dirs: {len(args.checkpoint_dirs)} directories")
        print(f"Episodes per checkpoint: {args.episodes_per_checkpoint}")
    print(f"Save interval: every {args.save_interval} checkpoints")
    print(f"Clear buffer after save: {args.clear_buffer_after_save}")
    print("=" * 80)
    
    # Config 설정 (post-hoc-eval.ipynb 방식)
    print("\nCreating environment...")
    parser_config = get_config()
    parser_config = get_overcooked_args(parser_config)
    
    # 필수 인자 설정 (post-hoc-eval.ipynb와 동일)
    args_list = [
        "--algorithm_name", args.algorithm,
        "--env_name", "Overcooked",
        "--layout_name", args.layout,
        "--experiment_name", "vae_data_collection",
        "--seed", str(args.seed),
        "--overcooked_version", "new",
        "--num_agents", str(args.num_agents),
        "--n_eval_rollout_threads", "1",
        "--episode_length", "400",
        "--use_recurrent_policy",  # Boolean 플래그는 값 없이
        "--cnn_layers_params", "32,3,1,1 64,3,1,1 32,3,1,1",  # train_sp.sh에서 사용된 설정
        "--random_index",  # train_sp.sh에서 사용됨
        "--recurrent_N", "1",  # RNN layers
        # hidden_size는 명시되지 않았으므로 기본값 64 사용
    ]
    
    all_args = parser_config.parse_args(args_list)
    
    # 추가 설정 (post-hoc-eval.ipynb와 동일)
    all_args.use_eval = False
    all_args.eval_stochastic = False
    all_args.use_wandb = False
    all_args.use_render = False
    all_args.n_render_rollout_threads = 1
    all_args.cuda = (args.device == 'cuda' and torch.cuda.is_available())
    all_args.use_phi = False
    all_args.old_dynamics = False
    all_args.use_agent_policy_id = False
    # use_centralized_V는 train_sp.sh에 없으므로 기본값 True 사용
    # 하지만 parse_args에서 action="store_false"이므로 플래그가 없으면 True가 됨
    # 명시적으로 True로 설정하여 확실하게 함
    all_args.use_centralized_V = True
    all_args.use_available_actions = True
    
    # agent_policy_names 처리: rMAPPO 정책을 직접 로드하므로 scripted agent 불필요
    # parse_args()에서 default=None으로 설정되지만, 환경 코드가 len(None)을 호출하려고 해서 에러 발생
    # 환경 코드는 agent_policy_names 길이가 num_agents와 일치해야 하므로, num_agents 길이만큼 None으로 채움
    if not hasattr(all_args, 'agent_policy_names') or all_args.agent_policy_names is None:
        all_args.agent_policy_names = [None] * all_args.num_agents  # num_agents 길이만큼 None으로 채움 (scripted agent 사용 안 함)
    
    # 설정 확인 출력 (post-hoc-eval.ipynb와 동일)
    print(f"\nPolicy configuration (from train_sp.sh):")
    print(f"  - hidden_size: {all_args.hidden_size} (default)")
    print(f"  - cnn_layers_params: {all_args.cnn_layers_params}")
    print(f"  - use_recurrent_policy: {all_args.use_recurrent_policy}")
    print(f"  - use_centralized_V: {all_args.use_centralized_V} (should be True for critic to use share_obs)")
    print(f"  - random_index: {all_args.random_index}")
    
    # Seed 설정 (post-hoc-eval.ipynb와 동일)
    setup_seed(args.seed)
    
    # Device 설정 (post-hoc-eval.ipynb와 동일)
    device = torch.device("cuda" if all_args.cuda and torch.cuda.is_available() else "cpu")
    all_args.device = device
    print(f"Using device: {device}")
    
    # Run directory 설정 (post-hoc-eval.ipynb와 동일)
    run_dir = Path(f"./vae_data_collection/{args.layout}/seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    all_args.run_dir = run_dir
    print(f"Results will be saved to: {run_dir}")
    
    # 환경 생성 (post-hoc-eval.ipynb와 동일한 방식 - ShareDummyVecEnv 사용)
    def make_eval_env(all_args, run_dir):
        def get_env_fn(rank):
            def init_env():
                env = Overcooked_new(all_args, run_dir, rank=rank, evaluation=False)
                env.seed(all_args.seed * 50000 + rank * 10000)
                return env
            return init_env
        
        return ShareDummyVecEnv([get_env_fn(0)])
    
    eval_envs = make_eval_env(all_args, run_dir)
    num_agents = all_args.num_agents
    
    print(f"✓ Environment created: {args.layout} with {num_agents} agents")
    print(f"\nEnvironment observation spaces:")
    print(f"  - use_centralized_V: {all_args.use_centralized_V}")
    for agent_id in range(num_agents):
        print(f"  - Agent {agent_id}:")
        print(f"    - obs_space shape: {eval_envs.observation_space[agent_id].shape}")
        print(f"    - share_obs_space shape: {eval_envs.share_observation_space[agent_id].shape}")
    print(f"  - Action space: {eval_envs.action_space[0]}")
    print(f"  - Number of agents: {num_agents}")
    
    # [수정] 버퍼 생성: 에이전트 수만큼 리스트로 생성
    # EncodingScheme enum에 맞게 변환 (OAI_raw_image -> raw_image)
    encoding_for_buffer = args.encoding.lower()
    if encoding_for_buffer in ['oai_raw_image', 'raw_image']:
        encoding_for_buffer = 'raw_image'
    elif encoding_for_buffer in ['oai_lossless', 'lossless']:
        encoding_for_buffer = 'OAI_lossless'
    elif encoding_for_buffer in ['oai_feats', 'feats']:
        encoding_for_buffer = 'OAI_feats'
    elif encoding_for_buffer in ['oai_egocentric', 'egocentric']:
        encoding_for_buffer = 'OAI_egocentric'
    
    buffers = [
        TemporalObservationBuffer(
            k_timesteps=args.k_timesteps,
            max_episodes=10000,  # 충분히 큰 값
            encoding_scheme=EncodingScheme(encoding_for_buffer)
        ) for _ in range(num_agents)
    ]
    print(f"Created {num_agents} separate buffers (one per agent)")
    
    # Policy/Trainer 생성 (post-hoc-eval.ipynb 방식)
    TrainAlgo, Policy = make_trainer_policy_cls(all_args.algorithm_name, use_single_network=all_args.use_single_network)
    
    # 체크포인트 파일 수집
    checkpoint_pairs = []  # [(agent0_path, agent1_path), ...]
    
    # 특정 체크포인트 파일이 지정된 경우 (agent0, agent1 쌍)
    if args.checkpoint_files:
        if len(args.checkpoint_files) < 2:
            raise ValueError("--checkpoint-files requires at least 2 files (agent0, agent1)")
        
        # agent0, agent1 쌍으로 묶기
        for i in range(0, len(args.checkpoint_files), args.num_agents):
            if i + args.num_agents <= len(args.checkpoint_files):
                pair = tuple(args.checkpoint_files[i:i+args.num_agents])
                checkpoint_pairs.append(pair)
                print(f"  Found checkpoint pair: {pair}")
    
    # 체크포인트 디렉토리에서 찾기
    checkpoint_dirs_to_load = []
    if args.checkpoint_dir:
        checkpoint_dirs_to_load.append(args.checkpoint_dir)
    if args.checkpoint_dirs:
        checkpoint_dirs_to_load.extend(args.checkpoint_dirs)
    
    if checkpoint_dirs_to_load:
        print(f"\nSearching in {len(checkpoint_dirs_to_load)} directories...")
        
        for checkpoint_dir in checkpoint_dirs_to_load:
            checkpoint_dir = Path(checkpoint_dir)
            if not checkpoint_dir.exists():
                print(f"  ⚠ Warning: Directory not found: {checkpoint_dir}")
                continue
            
            # 특정 스텝이 지정된 경우
            if args.checkpoint_step is not None:
                step = args.checkpoint_step
                # agent0, agent1 체크포인트 찾기
                agent_paths = []
                for agent_id in range(args.num_agents):
                    agent_path = checkpoint_dir / f"actor_agent{agent_id}_periodic_{step}.pt"
                    if agent_path.exists():
                        agent_paths.append(str(agent_path))
                    else:
                        print(f"  ⚠ Warning: Checkpoint not found: {agent_path}")
                        break
                
                if len(agent_paths) == args.num_agents:
                    checkpoint_pairs.append(tuple(agent_paths))
                    print(f"  Found checkpoint pair at step {step}: {agent_paths}")
            else:
                # 모든 체크포인트 찾기
                checkpoints = find_checkpoints(checkpoint_dir)
                # agent별로 그룹화
                agent_checkpoints = {i: [] for i in range(args.num_agents)}
                for ckpt in checkpoints:
                    for agent_id in range(args.num_agents):
                        if f"agent{agent_id}" in ckpt.name:
                            agent_checkpoints[agent_id].append(ckpt)
                            break
                
                # 같은 스텝의 체크포인트를 쌍으로 묶기
                # 스텝 번호 추출
                step_to_paths = {}
                for agent_id in range(args.num_agents):
                    for ckpt in agent_checkpoints[agent_id]:
                        # 파일명에서 스텝 번호 추출 (예: actor_agent0_periodic_2020000.pt)
                        import re
                        match = re.search(r'(\d+)\.pt$', ckpt.name)
                        if match:
                            step = int(match.group(1))
                            if step not in step_to_paths:
                                step_to_paths[step] = [None] * args.num_agents
                            step_to_paths[step][agent_id] = str(ckpt)
                
                # 모든 에이전트가 있는 스텝만 추가
                for step, paths in step_to_paths.items():
                    if all(p is not None for p in paths):
                        checkpoint_pairs.append(tuple(paths))
    
    print(f"\nTotal checkpoint pairs to load: {len(checkpoint_pairs)}")
    
    # 패딩 함수: observation을 고정 크기로 패딩
    def pad_observation(obs: np.ndarray, target_w: int, target_h: int):
        """
        obs: (W, H, C) 형태의 observation을 (target_W, target_H, C)로 0 패딩.
        
        현재 환경 설명에 따르면:
          - obs.shape = (13, 5, 25)
          - 13: 가로 (width), 5: 세로 (height), 25: 채널 수
        를 가정하고 구현함.
        """
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if obs.ndim != 3:
            raise ValueError(f"pad_observation expects 3D tensor (W,H,C), got shape {obs.shape}")
        w, h, c = obs.shape  # 가로, 세로, 채널 순서
        if w > target_w or h > target_h:
            raise ValueError(
                f"Cannot pad: obs ({w},{h},{c}) is larger than target ({target_w},{target_h},C). "
                "패딩할 타겟 크기를 더 크게 설정해야 합니다."
            )
        padded = np.zeros((target_w, target_h, c), dtype=obs.dtype)
        padded[:w, :h, :] = obs  # 좌상단 기준으로 그대로 복사
        return padded
    
    # Encoding function: 환경의 observation을 추출
    # ShareDummyVecEnv를 사용하므로 내부 환경에 접근하는 방식이 다름
    def encoding_fn(obs):
        """환경의 observation을 추출하고 첫 번째 에이전트의 obs 반환
        
        post-hoc-eval.ipynb에서 사용하는 방식:
        - eval_obs = np.array([info['all_agent_obs'] for info in eval_info_list])
        - 이것은 이미 lossless state encoding입니다.
        """
        # 1) 먼저 기존 로직대로 'encoded'를 만든다
        if args.encoding in ['lossless', 'OAI_lossless']:
            if isinstance(obs, np.ndarray):
                encoded = obs
            elif isinstance(obs, (list, tuple)):
                encoded = obs[0] if len(obs) > 0 else obs
            else:
                encoded = obs
        elif args.encoding in ['raw_image', 'OAI_raw_image']:
            try:
                # ShareDummyVecEnv에서 내부 환경 접근
                inner_env = eval_envs.envs[0]
                state = inner_env.base_env.state
                
                # lossless encoding을 사용 (raw_image는 나중에 구현)
                encoded = inner_env.base_env.lossless_state_encoding_mdp(state, old_dynamics=inner_env.old_dynamics)
                # 첫 번째 에이전트의 observation 반환
                if isinstance(encoded, (list, tuple)):
                    encoded = encoded[0]
            except Exception as e:
                print(f"Warning: Failed to encode raw_image: {e}")
                import traceback
                traceback.print_exc()
                # fallback: 환경의 기본 관찰 사용
                encoded = obs if isinstance(obs, np.ndarray) else np.array(obs)
        else:
            # 기타 인코딩은 일단 lossless fallback
            if isinstance(obs, np.ndarray):
                encoded = obs
            elif isinstance(obs, (list, tuple)):
                encoded = obs[0] if len(obs) > 0 else obs
            elif isinstance(obs, dict) and 'all_agent_obs' in obs:
                all_obs = obs['all_agent_obs']
                if isinstance(all_obs, np.ndarray) and all_obs.ndim > 1:
                    encoded = all_obs[0]
                else:
                    encoded = all_obs[0] if isinstance(all_obs, (list, tuple)) else all_obs
            else:
                encoded = obs
        
        # 2) 필요하면 numpy로 캐스팅
        if not isinstance(encoded, np.ndarray):
            encoded = np.array(encoded)
        
        # 3) 패딩 적용
        if args.obs_width is not None and args.obs_height is not None:
            encoded = pad_observation(encoded, args.obs_width, args.obs_height)
        
        return encoded
    
    # 데이터 수집
    total_episodes = args.episodes_random + len(checkpoint_pairs) * args.episodes_per_checkpoint
    print(f"\nCollecting {total_episodes} total episodes...")
    print(f"  - Random episodes: {args.episodes_random}")
    print(f"  - Trained policy episodes: {len(checkpoint_pairs) * args.episodes_per_checkpoint}")
    
    checkpoint_count = 0
    
    # 1. 랜덤 policy로 데이터 수집
    if args.episodes_random > 0:
        print(f"\n[1/{(1 if checkpoint_pairs else 0) + (1 if args.episodes_random > 0 else 0)}] Random policy: {args.episodes_random} episodes")
        collect_episodes_random(
            eval_envs=eval_envs,
            buffers=buffers,  # [수정] 리스트 전달
            num_episodes=args.episodes_random,
            all_args=all_args,
            encoding_fn=encoding_fn
        )
    
    # 2. 학습된 체크포인트들로 데이터 수집
    for i, checkpoint_pair in enumerate(checkpoint_pairs):
        checkpoint_count += 1
        print(f"\n[{i+2 if args.episodes_random > 0 else i+1}/{len(checkpoint_pairs) + (1 if args.episodes_random > 0 else 0)}] Checkpoint pair {checkpoint_count}: {args.episodes_per_checkpoint} episodes")
        print(f"  Agent checkpoints: {checkpoint_pair}")
        
        try:
            # 각 에이전트별로 Policy와 Trainer 생성
            policies = []
            trainers = []
            
            for agent_id in range(args.num_agents):
                checkpoint_path = Path(checkpoint_pair[agent_id])
                
                # Share observation space 결정 (post-hoc-eval.ipynb와 동일)
                if all_args.use_centralized_V:
                    share_observation_space = eval_envs.share_observation_space[agent_id]
                else:
                    share_observation_space = eval_envs.observation_space[agent_id]
                
                # Policy 생성 (post-hoc-eval.ipynb와 동일)
                policy = Policy(
                    all_args,
                    eval_envs.observation_space[agent_id],
                    share_observation_space,
                    eval_envs.action_space[agent_id],
                    device=device,
                )
                policies.append(policy)
                
                # Trainer 생성
                trainer = TrainAlgo(all_args, policy, device=device)
                trainers.append(trainer)
                
                # Actor 로드
                if checkpoint_path.exists():
                    actor_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    policy.actor.load_state_dict(actor_state_dict)
                    policy.actor.eval()
                    print(f"  ✓ Agent {agent_id} actor loaded from {checkpoint_path}")
                else:
                    raise FileNotFoundError(f"Actor checkpoint not found: {checkpoint_path}")
                
                # Critic 로드 (선택사항)
                critic_path = checkpoint_path.parent / checkpoint_path.name.replace("actor_", "critic_")
                if critic_path.exists():
                    critic_state_dict = torch.load(critic_path, map_location=device, weights_only=False)
                    policy.critic.load_state_dict(critic_state_dict)
                    policy.critic.eval()
                    print(f"  ✓ Agent {agent_id} critic loaded from {critic_path}")
            
            # 데이터 수집 (post-hoc-eval.ipynb와 동일)
            collect_episodes_with_trainers(
                eval_envs=eval_envs,
                buffers=buffers,  # [수정] 리스트 전달
                trainers=trainers,
                num_episodes=args.episodes_per_checkpoint,
                all_args=all_args,
                encoding_fn=encoding_fn
            )
            
            # 메모리 정리
            del policies, trainers
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 중간 저장 (옵션)
            if args.save_interval > 0 and checkpoint_count % args.save_interval == 0:
                # 병합 후 저장
                main_buffer = buffers[0]
                for i in range(1, num_agents):
                    if hasattr(buffers[i], 'episodes'):
                        main_buffer.episodes.extend(buffers[i].episodes)
                
                intermediate_path = args.save_path.replace('.pkl', f'_ckpt{checkpoint_count}.pkl')
                print(f"\n  [Intermediate Save] Saving to {intermediate_path}...")
                main_buffer.save(intermediate_path)
                print(f"  Saved {len(main_buffer)} episodes (merged from {num_agents} agents)")
                
                # 버퍼 비우기 (선택사항)
                if args.clear_buffer_after_save:
                    print(f"  Clearing buffers to save memory...")
                    for buffer in buffers:
                        buffer.episodes.clear()
                        buffer.current_episode = []
        
        except Exception as e:
            print(f"  ✗ Failed to load checkpoint pair {checkpoint_pair}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 현재 메모리 상태 출력
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_info = process.memory_info()
            print(f"  Memory: {mem_info.rss / 1024**3:.2f} GB")
        else:
            total_eps = sum(len(b) for b in buffers)
            print(f"  Total buffer episodes: {total_eps}")
    
    # [수정] 최종 병합 및 저장
    # 모든 에이전트의 데이터를 하나의 버퍼(첫 번째 버퍼)로 합칩니다.
    print(f"\nMerging buffers from {num_agents} agents...")
    
    main_buffer = buffers[0]
    
    # 나머지 버퍼들의 데이터를 main_buffer로 이동
    for i in range(1, num_agents):
        if hasattr(buffers[i], 'episodes'):
            main_buffer.episodes.extend(buffers[i].episodes)
            print(f"  Agent {i} data merged ({len(buffers[i])} episodes)")
        else:
            print(f"  Warning: Cannot merge buffer for agent {i} (unknown structure)")
    
    print(f"\nSaving merged buffer to {args.save_path}...")
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 합쳐진 main_buffer 저장
    main_buffer.save(args.save_path)
    
    # 통계
    inputs, _ = main_buffer.get_sequences()
    print("\n" + "=" * 80)
    print("Collection Complete!")
    print(f"  Total Merged Episodes: {len(main_buffer)}")
    print(f"  Sequences: {len(inputs)}")
    print(f"  Saved to: {args.save_path}")
    if len(inputs) > 0:
        print(f"  Sequence shape: {inputs[0].shape}")
    print("=" * 80)


if __name__ == '__main__':
    main()

