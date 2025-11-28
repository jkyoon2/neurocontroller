"""
VAE 학습 스크립트

Usage:
    # 새로 데이터 수집 + 학습
    python -m pymarlzooplus.envs.oai_agents.components.train_vae \
        --layout forced_coordination --collect-data
    
    # 기존 버퍼로 학습
    python -m pymarlzooplus.envs.oai_agents.components.train_vae \
        --buffer-path ./vae_data/buffer.pkl
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked


class ArgsNamespace(SimpleNamespace):
    """SimpleNamespace with _get_kwargs() method for compatibility with argparse.Namespace"""
    def _get_kwargs(self):
        """Return dict of all attributes (compatible with argparse.Namespace._get_kwargs())"""
        return vars(self)
from .temporal_vae import TemporalVAE, TemporalVAEConfig, EncodingScheme, VAEMode
from .temporal_vae_trainer import TemporalObservationBuffer, TemporalVAETrainer
from .temporal_vae_integration import save_vae_model, save_encoder_only


def main():
    parser = argparse.ArgumentParser(description='Train VAE')
    
    # 데이터 설정
    parser.add_argument('--collect-data', action='store_true',
                       help='새로 데이터 수집')
    parser.add_argument('--buffer-path', type=str, default=None,
                       help='버퍼 파일 경로 (단일 파일 지정 시 사용)')
    parser.add_argument('--vae-data-dir', type=str, default='/home/juliecandoit98/ZSC-Eval/vae_data',
                       help='VAE 데이터 베이스 디렉토리 (vae_data/)')
    parser.add_argument('--layout', type=str, default=None,
                       help='레이아웃 이름 (지정 시 해당 레이아웃만 처리, None 또는 "all"이면 모든 레이아웃 처리, 기본값: None=모든 레이아웃)')
    parser.add_argument('--seed', type=int, default=None,
                       help='시드 번호 (선택사항, 지정 시 해당 시드만 사용)')
    parser.add_argument('--step', type=int, default=None,
                       help='체크포인트 스텝 (선택사항, 지정 시 해당 스텝만 사용)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='RL 체크포인트 디렉토리 (데이터 수집용)')
    
    # 환경 설정
    parser.add_argument('--num-agents', type=int, default=2)
    parser.add_argument('--encoding', type=str, default='OAI_lossless')
    
    # VAE 설정
    parser.add_argument('--k-timesteps', type=int, default=5)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--mode', type=str, default='reconstruction',
                       choices=['reconstruction', 'predictive'])
    parser.add_argument('--beta', type=float, default=1.0)
    
    # 학습 설정
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    # 저장 설정
    parser.add_argument('--save-path', type=str, default='./vae_models/vae.pt')
    parser.add_argument('--save-encoder-path', type=str, default=None,
                       help='Encoder만 저장할 경로 (None이면 자동 생성)')
    parser.add_argument('--checkpoint-save-dir', type=str, default='./vae_checkpoints')
    parser.add_argument('--train-seed', type=int, default=42,
                       help='학습 시드 (데이터 로드와 무관)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Wandb 설정
    parser.add_argument('--use-wandb', action='store_true',
                       help='Wandb 사용 여부')
    parser.add_argument('--wandb-project', type=str, default='vae-overcooked',
                       help='Wandb 프로젝트 이름')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                       help='Wandb run 이름')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    
    # 버퍼 경로 결정
    import glob
    buffer_files = []
    
    if args.buffer_path:
        # 단일 파일 지정
        if '*' in args.buffer_path or Path(args.buffer_path).is_dir():
            # Glob 패턴 또는 디렉토리
            if Path(args.buffer_path).is_dir():
                pattern = str(Path(args.buffer_path) / "**" / "buffer_*.pkl")
            else:
                pattern = args.buffer_path
            buffer_files = sorted(glob.glob(pattern, recursive=True))
            if len(buffer_files) == 0:
                raise FileNotFoundError(f"No buffer files found with pattern: {pattern}")
            print(f"Found {len(buffer_files)} buffer files with pattern")
        else:
            # 단일 파일
            if not Path(args.buffer_path).exists():
                raise FileNotFoundError(f"Buffer file not found: {args.buffer_path}")
            buffer_files = [args.buffer_path]
    elif args.vae_data_dir:
        # 디렉토리 구조 기반: vae_data/{layout}/seed{seed}/step{step}/buffer_*.pkl
        vae_data_dir = Path(args.vae_data_dir)
        
        # 레이아웃 처리: None이거나 "all"이면 모든 레이아웃 처리
        if args.layout is None or args.layout.lower() == 'all':
            # 모든 레이아웃 디렉토리 찾기
            layout_dirs = sorted([d for d in vae_data_dir.iterdir() if d.is_dir()])
            if len(layout_dirs) == 0:
                raise FileNotFoundError(f"No layout directories found in {vae_data_dir}")
            
            print(f"Processing all layouts: {[d.name for d in layout_dirs]}")
            buffer_files = []
            
            # 각 레이아웃에 대해 버퍼 파일 찾기
            for layout_dir in layout_dirs:
                # 시드 디렉토리 찾기
                seed_dirs = sorted([d for d in layout_dir.iterdir() if d.is_dir() and d.name.startswith('seed')])
                
                # 시드 필터링 (지정된 경우)
                if args.seed is not None:
                    seed_dirs = [d for d in seed_dirs if d.name == f"seed{args.seed}"]
                
                # 각 시드 디렉토리에서 스텝 디렉토리 찾기
                for seed_dir in seed_dirs:
                    step_dirs = sorted([d for d in seed_dir.iterdir() if d.is_dir() and d.name.startswith('step')])
                    
                    # 스텝 필터링 (지정된 경우)
                    if args.step is not None:
                        step_dirs = [d for d in step_dirs if d.name == f"step{args.step}"]
                    
                    # 각 스텝 디렉토리에서 버퍼 파일 찾기
                    for step_dir in step_dirs:
                        buffer_pattern = str(step_dir / "buffer_*.pkl")
                        found_files = glob.glob(buffer_pattern)
                        if found_files:
                            buffer_files.extend(found_files)
            
            if len(buffer_files) == 0:
                raise FileNotFoundError(f"No buffer files found in {vae_data_dir}")
            
            buffer_files = sorted(buffer_files)
            print(f"Found {len(buffer_files)} buffer files across {len(layout_dirs)} layout(s)")
        else:
            # 특정 레이아웃 지정 시: 해당 레이아웃의 모든 시드/스텝 버퍼 자동 찾기
            layout_dir = vae_data_dir / args.layout
            if not layout_dir.exists():
                raise FileNotFoundError(f"Layout directory not found: {layout_dir}")
            
            buffer_files = []
            
            # 시드 디렉토리 찾기
            seed_dirs = sorted([d for d in layout_dir.iterdir() if d.is_dir() and d.name.startswith('seed')])
            if len(seed_dirs) == 0:
                raise FileNotFoundError(f"No seed directories found in {layout_dir}")
            
            # 시드 필터링 (지정된 경우)
            if args.seed is not None:
                seed_dirs = [d for d in seed_dirs if d.name == f"seed{args.seed}"]
                if len(seed_dirs) == 0:
                    raise FileNotFoundError(f"Seed directory not found: seed{args.seed}")
            
            # 각 시드 디렉토리에서 스텝 디렉토리 찾기
            for seed_dir in seed_dirs:
                step_dirs = sorted([d for d in seed_dir.iterdir() if d.is_dir() and d.name.startswith('step')])
                
                # 스텝 필터링 (지정된 경우)
                if args.step is not None:
                    step_dirs = [d for d in step_dirs if d.name == f"step{args.step}"]
                
                # 각 스텝 디렉토리에서 버퍼 파일 찾기
                for step_dir in step_dirs:
                    buffer_pattern = str(step_dir / "buffer_*.pkl")
                    found_files = glob.glob(buffer_pattern)
                    if found_files:
                        buffer_files.extend(found_files)
            
            if len(buffer_files) == 0:
                raise FileNotFoundError(f"No buffer files found in {layout_dir}")
            
            buffer_files = sorted(buffer_files)
            print(f"Found {len(buffer_files)} buffer files across {len(seed_dirs)} seed(s) and multiple steps")
    else:
        raise ValueError("Either --buffer-path or --vae-data-dir must be specified")
    
    print(f"\nBuffer files to load ({len(buffer_files)}):")
    for i, bf in enumerate(buffer_files[:5]):  # 처음 5개만 출력
        print(f"  {i+1}. {bf}")
    if len(buffer_files) > 5:
        print(f"  ... and {len(buffer_files) - 5} more files")
    
    # 데이터 수집 (필요시)
    if args.collect_data:
        if args.layout is None:
            raise ValueError("--layout is required when using --collect-data")
        if args.seed is None:
            raise ValueError("--seed is required when using --collect-data (데이터 수집 시에는 시드 지정 필요)")
        if args.step is None:
            raise ValueError("--step is required when using --collect-data (데이터 수집 시에는 스텝 지정 필요)")
        
        print("Collecting data...")
        from .collect_vae_data import main as collect_main
        import sys
        
        # 저장 경로 자동 생성
        vae_data_dir = Path(args.vae_data_dir)
        save_path = vae_data_dir / args.layout / f"seed{args.seed}" / f"step{args.step}" / f"buffer_{args.layout}_seed{args.seed}_step{args.step}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # collect_vae_data의 args 구성
        collect_args = [
            '--layout', args.layout,
            '--num-agents', str(args.num_agents),
            '--encoding', args.encoding,
            '--k-timesteps', str(args.k_timesteps),
            '--save-path', str(save_path),
            '--seed', str(args.seed)
        ]
        if args.checkpoint_dir:
            collect_args.extend(['--checkpoint-dir', args.checkpoint_dir])
        
        sys.argv[1:] = collect_args
        collect_main()
        
        # 수집된 버퍼 파일을 buffer_files에 추가
        buffer_files = [str(save_path)]
    
    # 버퍼 로드
    print(f"\nLoading buffers...")
    # LazyVAEDataset 사용 가능한 인코딩: raw_image, OAI_raw_image, OAI_lossless
    use_lazy_dataset = args.encoding in ['raw_image', 'OAI_raw_image', 'OAI_lossless']
    
    if use_lazy_dataset:
        # LazyVAEDataset 사용 (메모리 효율) - raw_image와 lossless 모두 지원
        from .lazy_vae_dataset import LazyVAEDataset
        
        print(f"Using LazyVAEDataset for memory efficiency (encoding: {args.encoding})...")
        
        # merge 파일 제외
        chunk_files = [f for f in buffer_files if 'merged' not in Path(f).name]
        
        if len(chunk_files) == 0:
            raise FileNotFoundError(f"No valid buffer files found (all are merged files?)")
        
        print(f"Using {len(chunk_files)} buffer files (merge files excluded)")
        
        dataset = LazyVAEDataset(
            buffer_files=chunk_files,
            k_timesteps=args.k_timesteps,
            mode=VAEMode(args.mode),
            encoding=args.encoding,  # 인코딩 타입 전달
            max_cached_files=3  # 최대 3개 파일만 메모리에 유지
        )
        print(f"✓ Lazy dataset created: {len(dataset):,} sequences")
        buffer = None  # buffer 대신 dataset 사용
    else:
        # 다른 인코딩의 경우: 여러 버퍼를 합쳐서 사용
        print(f"Loading and merging {len(buffer_files)} buffer files...")
        
        # 첫 번째 버퍼 로드
        main_buffer = TemporalObservationBuffer.load(buffer_files[0])
        print(f"  Loaded buffer 1/{len(buffer_files)}: {len(main_buffer)} episodes")
        
        # 나머지 버퍼들을 합치기
        for i, buffer_file in enumerate(buffer_files[1:], start=2):
            temp_buffer = TemporalObservationBuffer.load(buffer_file)
            # 에피소드 합치기
            main_buffer.episodes.extend(temp_buffer.episodes)
            print(f"  Loaded buffer {i}/{len(buffer_files)}: {len(temp_buffer)} episodes (total: {len(main_buffer)} episodes)")
            
            buffer = main_buffer
            dataset = None
            print(f"✓ Merged buffer: {len(buffer)} total episodes")
            
            # 실제 데이터 shape 확인 (디버깅용)
            if len(buffer.episodes) > 0 and len(buffer.episodes[0]) > 0:
                sample_obs = buffer.episodes[0][0]
                print(f"  Sample observation shape: {sample_obs.shape}")
    
    # Layout shape 결정
    # layout 정보 추출 (버퍼 파일 경로에서 또는 args에서)
    # 여러 레이아웃이 섞여 있을 수 있으므로 첫 번째 버퍼의 레이아웃 사용
    if args.layout and args.layout.lower() != 'all':
        layout_name = args.layout
    elif buffer_files:
        # 버퍼 파일 경로에서 레이아웃 추출
        # 예: vae_data/random0_medium/seed5/step3020000/buffer_*.pkl
        path_parts = Path(buffer_files[0]).parts
        if 'vae_data' in path_parts:
            vae_data_idx = path_parts.index('vae_data')
            if vae_data_idx + 1 < len(path_parts):
                layout_name = path_parts[vae_data_idx + 1]
            else:
                layout_name = 'random0_medium'  # 기본값
        else:
            layout_name = 'random0_medium'  # 기본값
    else:
        layout_name = 'random0_medium'  # 기본값
    
    # ArgsNamespace 사용 (_get_kwargs() 메서드 포함)
    env_args = ArgsNamespace()
    env_args.layout_name = layout_name
    env_args.num_agents = args.num_agents
    env_args.episode_length = 400
    env_args.use_render = False
    env_args.use_phi = False
    env_args.use_hsp = False
    env_args.reward_shaping_factor = 1.0
    env_args.initial_reward_shaping_factor = 1.0
    env_args.reward_shaping_horizon = 1e10
    env_args.random_index = False
    env_args.old_dynamics = False
    env_args.random_start_prob = 0.0
    env_args.use_available_actions = True
    env_args.algorithm_name = 'rmappo'
    # agent_policy_names: 환경 코드가 len()을 호출하므로 None이 아닌 리스트로 설정
    env_args.agent_policy_names = [None] * args.num_agents
    env_args.use_agent_policy_id = False
    
    featurize_type = tuple(["ppo"] * args.num_agents)
    env = Overcooked(
        all_args=env_args,
        run_dir=".",
        baselines_reproducible=True,
        featurize_type=featurize_type,
        stuck_time=4,
        rank=0,
        evaluation=False
    )
    layout_shape = env.base_env.mdp.shape if hasattr(env, 'base_env') and hasattr(env.base_env, 'mdp') else (7, 7)
    
    # VAE 설정
    if args.encoding == 'OAI_feats':
        input_shape = (96,)
    elif args.encoding in ['raw_image', 'OAI_raw_image']:
        # Raw image: RGB 이미지
        if dataset is not None:
            # Lazy dataset에서 샘플 shape 확인
            sample_shape = dataset.get_sample_shape()
            # sample_shape: (k+1, W, H, C) 또는 (k+1, C, H, W) -> input_shape 확인 필요
            input_shape = sample_shape[1:]
            print(f"Detected raw_image shape from dataset: {input_shape}")
        else:
            # buffer에서 실제 이미지 크기 확인
            inputs, _ = buffer.get_sequences(mode=VAEMode(args.mode))
            if len(inputs) > 0:
                # inputs shape: (N, k+1, W, H, C) - 실제 데이터는 (W, H, C) 순서
                obs_shape = inputs[0].shape[1:]  # (k+1, W, H, C) -> (W, H, C) 추출
                print(f"Detected raw_image shape from buffer: {obs_shape}")
                # VAE는 (C, H, W) 형태를 기대하므로 변환 필요
                if len(obs_shape) == 3:
                    # (W, H, C) -> (C, H, W) 변환
                    input_shape = (obs_shape[2], obs_shape[1], obs_shape[0])
                    print(f"Converted to VAE input_shape (C, H, W): {input_shape}")
                else:
                    input_shape = obs_shape
            else:
                # fallback: 기본 16x16
                input_shape = (3, 16, 16)
                print(f"Using default raw_image shape: {input_shape}")
    else:
        # OAI_lossless 등: 실제 데이터는 (W, H, C) 형태
        if dataset is not None:
            # Lazy dataset에서 샘플 shape 확인
            sample_shape = dataset.get_sample_shape()
            # sample_shape: (k+1, C, H, W) - LazyVAEDataset에서 이미 변환됨
            input_shape = sample_shape[1:]  # (C, H, W)
            print(f"Detected lossless shape from dataset: {input_shape}")
        elif buffer is not None:
            # 버퍼에서 실제 shape 확인
            inputs, _ = buffer.get_sequences(mode=VAEMode(args.mode))
            if len(inputs) > 0:
                # inputs shape: (N, k+1, W, H, C)
                # 예: (3960, 5, 13, 5, 25) = (N, k+1, W, H, C)
                actual_shape = inputs[0].shape  # (k+1, W, H, C)
                print(f"Actual sequence shape from buffer: {actual_shape}")
                
                if len(actual_shape) == 4:  # (k+1, W, H, C)
                    # 마지막 3차원이 실제 observation shape: (W, H, C)
                    w, h, c = actual_shape[1], actual_shape[2], actual_shape[3]
                    # VAE는 (C, H, W) 형태를 기대하므로 변환
                    # Dataset에서 자동으로 (W, H, C) -> (C, H, W) 변환하므로
                    # input_shape는 (C, H, W)로 설정
                    input_shape = (c, h, w)
                    print(f"Detected observation shape from buffer: (W={w}, H={h}, C={c})")
                    print(f"VAE input_shape (C, H, W): {input_shape} (Dataset will auto-convert)")
                else:
                    # fallback: layout_shape 사용
                    input_shape = (25, layout_shape[0], layout_shape[1])
                    print(f"Using layout-based input_shape: {input_shape}")
            else:
                # fallback: layout_shape 사용
                input_shape = (25, layout_shape[0], layout_shape[1])
                print(f"Using layout-based input_shape: {input_shape}")
        else:
            # fallback: layout_shape 사용
            input_shape = (25, layout_shape[0], layout_shape[1])
            print(f"Using layout-based input_shape: {input_shape}")
    
    # 작은 그리드 데이터 (H, W <= 10)에는 MLP decoder 사용 (더 간단하고 효율적)
    # 큰 이미지에는 CNN decoder 사용
    c, h, w = input_shape if len(input_shape) == 3 else (input_shape[0], 1, 1)
    use_conv = args.encoding in ['raw_image', 'OAI_raw_image'] or (h > 10 or w > 10)
    
    vae_config = TemporalVAEConfig(
        k_timesteps=args.k_timesteps,
        mode=VAEMode(args.mode),
        encoding_scheme=EncodingScheme(args.encoding),
        hidden_dim=args.hidden_dim,
        input_shape=input_shape,
        beta=args.beta,
        use_conv=use_conv  # 작은 그리드는 MLP, 큰 이미지는 CNN
    )
    
    print("\nVAE Config:")
    print(f"  k_timesteps: {vae_config.k_timesteps}")
    print(f"  hidden_dim: {vae_config.hidden_dim}")
    print(f"  mode: {vae_config.mode.value}")
    print(f"  encoding: {vae_config.encoding_scheme.value}")
    print(f"  input_shape: {vae_config.input_shape}")
    print(f"  use_conv: {vae_config.use_conv} ({'CNN Decoder' if vae_config.use_conv else 'MLP Decoder'})")
    
    # VAE 생성
    vae_model = TemporalVAE(vae_config)
    
    # Trainer (raw_image의 경우 BCE 사용)
    use_bce = args.encoding in ['raw_image', 'OAI_raw_image']
    
    # Wandb run 이름 자동 생성
    if args.use_wandb and args.wandb_run_name is None:
        args.wandb_run_name = f"vae_{args.encoding}_{args.mode}_h{args.hidden_dim}_k{args.k_timesteps}"
    
    trainer = TemporalVAETrainer(
        model=vae_model,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_save_dir,
        use_bce=use_bce,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        beta_warmup_steps=5000,
        beta_annealing_steps=10000
    )
    
    if use_bce:
        print("Using Binary Cross Entropy loss for image reconstruction")
    
    # 학습
    print("\nTraining...")
    if dataset is not None:
        # Lazy dataset 사용
        trainer.train_with_dataset(
            dataset=dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            beta=args.beta,
            save_interval=10,
            early_stopping_patience=20
        )
    else:
        # 기존 buffer 방식 사용
        trainer.train(
            buffer=buffer,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            beta=args.beta,
            save_interval=10,
            early_stopping_patience=20
        )
    
    # 모델 저장
    print(f"\nSaving model to {args.save_path}...")
    save_vae_model(
        model=vae_model,
        filepath=args.save_path,
        additional_info={
            'layout': args.layout,
            'encoding': args.encoding,
            'k_timesteps': args.k_timesteps,
            'hidden_dim': args.hidden_dim
        }
    )
    
    # Encoder만 저장 (차원 압축용)
    if args.save_encoder_path is None:
        # 자동 경로 생성: vae.pt -> encoder_vae.pt
        encoder_path = Path(args.save_path).parent / f"encoder_{Path(args.save_path).name}"
    else:
        encoder_path = args.save_encoder_path
    
    print(f"Saving encoder to {encoder_path}...")
    save_encoder_only(
        model=vae_model,
        filepath=encoder_path,
        additional_info={
            'layout': layout_name,
            'encoding': args.encoding,
            'k_timesteps': args.k_timesteps,
            'hidden_dim': args.hidden_dim
        }
    )
    
    print("\nTraining complete!")
    print(f"  Final total loss: {trainer.training_history['total_loss'][-1]:.4f}")
    print(f"  Final recon loss: {trainer.training_history['recon_loss'][-1]:.4f}")
    print(f"  Final KL loss: {trainer.training_history['kl_loss'][-1]:.4f}")
    print(f"  Full model saved: {args.save_path}")
    print(f"  Encoder saved: {encoder_path}")
    
    # Wandb 종료
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()

