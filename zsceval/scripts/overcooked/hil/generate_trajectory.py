"""
Generate Trajectories from Agent Checkpoints

This script loads trained agent checkpoints and generates trajectory JSON files
for human review in the HIL workflow.

Usage:
    python -m zsceval.scripts.overcooked.hil.generate_trajectory \
        --layout_name random3 \
        --checkpoint_agent0 path/to/actor_agent0.pt \
        --checkpoint_agent1 path/to/actor_agent1.pt \
        --num_trajectories 10 \
        --output_dir human_interface/data/trajectories_for_human
        
Or use checkpoint directory (auto-finds latest):
    python -m zsceval.scripts.overcooked.hil.generate_trajectory \
        --layout_name random3 \
        --checkpoint_dir results/Overcooked/random3/rmappo/sp/seed10/models \
        --num_trajectories 10
"""

import argparse
import sys
from pathlib import Path
import torch
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from zsceval.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocDummyBatchVecEnv
from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from zsceval.utils.feedback import TrajectoryGenerator


def find_latest_checkpoint(checkpoint_dir: Path, agent_id: int) -> Path:
    """Find the latest checkpoint for an agent.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        agent_id: Agent ID
        
    Returns:
        Path to latest checkpoint
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Debug: List all files in directory
    all_files = list(checkpoint_dir.glob("*.pt"))
    logger.debug(f"All .pt files in {checkpoint_dir}: {[f.name for f in all_files]}")
    
    # Look for periodic checkpoints first - try multiple patterns
    patterns = [
        f"actor_agent{agent_id}_periodic_*.pt",  # Standard pattern
        f"actor_agent{agent_id}_*.pt",  # Any actor_agent{id} file
    ]
    
    periodic_files = []
    for pattern in patterns:
        found = list(checkpoint_dir.glob(pattern))
        if found:
            # Sort by step number (extract from filename)
            def extract_step(p):
                try:
                    # For "actor_agent0_periodic_10000000.pt", stem is "actor_agent0_periodic_10000000"
                    # Split by '_' and get last part, then convert to int
                    parts = p.stem.split('_')
                    # Find the last numeric part (should be the step number)
                    for part in reversed(parts):
                        if part.isdigit():
                            return int(part)
                    return 0
                except (ValueError, IndexError):
                    return 0
            
            found_sorted = sorted(found, key=extract_step)
            periodic_files = found_sorted
            logger.debug(f"Found {len(found_sorted)} files with pattern '{pattern}': {[f.name for f in found_sorted[:5]]}")
            break
    
    logger.debug(f"Looking for agent {agent_id} checkpoints in {checkpoint_dir}")
    logger.debug(f"Found {len(periodic_files)} periodic files for agent {agent_id}")
    if len(periodic_files) > 0:
        logger.debug(f"  First few: {[f.name for f in periodic_files[:3]]}")
        logger.debug(f"  Last few: {[f.name for f in periodic_files[-3:]]}")
    
    if periodic_files:
        latest = periodic_files[-1]
        logger.info(f"Selected latest checkpoint for agent {agent_id}: {latest.name}")
        return latest
    
    # Fallback to non-periodic - try multiple patterns
    regular_patterns = [
        f"actor_agent{agent_id}.pt",
        f"actor_{agent_id}.pt",
        f"agent{agent_id}_actor.pt",
    ]
    
    for pattern in regular_patterns:
        regular_file = checkpoint_dir / pattern
        if regular_file.exists():
            logger.debug(f"Using non-periodic checkpoint for agent {agent_id}: {regular_file.name}")
            return regular_file
    
    # List available files for better error message
    available_files = [f.name for f in all_files if 'actor' in f.name.lower()]
    agent_files = [f.name for f in all_files if f'agent{agent_id}' in f.name.lower() or f'agent_{agent_id}' in f.name.lower()]
    
    error_msg = (
        f"No checkpoint found for agent {agent_id} in {checkpoint_dir}.\n"
        f"  Patterns searched: {patterns + regular_patterns}\n"
        f"  Available actor files: {available_files[:10]}\n"
        f"  Files containing 'agent{agent_id}': {agent_files[:10]}"
    )
    raise FileNotFoundError(error_msg)


def find_checkpoint_by_step(checkpoint_dir: Path, agent_id: int, step: int) -> Path:
    """Find a specific checkpoint by step number.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        agent_id: Agent ID
        step: Checkpoint step number (e.g., 2020000, 2520000)
        
    Returns:
        Path to checkpoint with specified step
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Look for periodic checkpoint with specific step
    checkpoint_path = checkpoint_dir / f"actor_agent{agent_id}_periodic_{step}.pt"
    
    logger.debug(f"Looking for checkpoint: {checkpoint_path}")
    
    if checkpoint_path.exists():
        logger.debug(f"Found checkpoint for agent {agent_id} at step {step}: {checkpoint_path.name}")
        return checkpoint_path
    
    # List available files for better error message
    all_files = list(checkpoint_dir.glob("*.pt"))
    available_files = [f.name for f in all_files if f'agent{agent_id}' in f.name]
    
    raise FileNotFoundError(
        f"Checkpoint not found for agent {agent_id} at step {step} in {checkpoint_dir}. "
        f"Expected: {checkpoint_path.name}. "
        f"Available files for agent {agent_id}: {available_files}"
    )


def parse_args():
    """Parse command line arguments."""
    parser = get_config()
    parser = get_overcooked_args(parser)
    
    # CRITICAL: Override num_agents default to 2 for Overcooked (2-player game)
    # get_overcooked_args sets default=1, but Overcooked requires 2 agents
    parser.set_defaults(num_agents=2)
    
    # CRITICAL: Override episode_length default to 400 for Overcooked
    # config.py sets default=200, but Overcooked episodes are 400 timesteps
    parser.set_defaults(episode_length=400)
    
    # Checkpoint arguments
    checkpoint_group = parser.add_mutually_exclusive_group(required=False)
    checkpoint_group.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing checkpoints. If not provided, will be auto-constructed from seed and layout_name: "
             "results/Overcooked/{layout_name}/rmappo/sp/seed{seed}/models"
    )
    checkpoint_group.add_argument(
        "--checkpoint_agent0",
        type=str,
        help="Path to agent 0 checkpoint (use with --checkpoint_agent1)"
    )
    
    parser.add_argument(
        "--checkpoint_agent1",
        type=str,
        help="Path to agent 1 checkpoint (use with --checkpoint_agent0)"
    )
    
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=None,
        help="Specific checkpoint step to use (e.g., 2020000, 2520000). "
             "Required when using auto-constructed checkpoint_dir. If not specified with checkpoint_dir, uses latest checkpoint."
    )
    
    # Generation arguments
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=1,
        help="Number of trajectories to generate"
    )
    parser.add_argument(
        "--start_episode_id",
        type=int,
        default=0,
        help="Starting episode ID for filenames"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for trajectory JSON files. "
             "If not provided, defaults to human_interface/data/trajectories_for_human/{layout_name}/"
    )
    
    args = parser.parse_args()
    
    # Ensure num_agents is 2 for Overcooked (critical for multi-agent setup)
    if args.num_agents != 2:
        logger.warning(
            f"num_agents is {args.num_agents}, but Overcooked requires 2 agents. "
            f"Setting num_agents=2."
        )
        args.num_agents = 2
    
    # Add missing attributes for Overcooked environment (same as train_sp.sh)
    if not hasattr(args, 'use_phi'):
        args.use_phi = False
    if not hasattr(args, 'old_dynamics'):
        args.old_dynamics = False
    
    # CRITICAL: Set train_sp.sh default values for policy architecture
    # These MUST match the training configuration exactly for checkpoint loading to work
    # Always set explicitly (don't rely on defaults) to ensure structure matches
    
    # CNN layers (train_sp.sh: --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1")
    args.cnn_layers_params = "32,3,1,1 64,3,1,1 32,3,1,1"
    
    # RNN settings (train_sp.sh: --use_recurrent_policy, --recurrent_N 1 is default)
    args.use_recurrent_policy = True
    args.recurrent_N = 1  # CRITICAL: Must match training config
    
    # Hidden size (default is 64, train_sp.sh doesn't specify)
    args.hidden_size = 64
    
    # Other settings
    args.random_index = True  # train_sp.sh uses --random_index
    
    # share_policy: action="store_false", default=True
    # train_sp.sh uses --share_policy flag, which sets it to False (separated policy)
    args.share_policy = False  # Separated policy (train_sp.sh uses --share_policy flag)
    
    # use_centralized_V: action="store_false", default=True
    # train_sp.sh does NOT use --use_centralized_V flag, so it stays True
    args.use_centralized_V = True  # Default is True, train_sp.sh doesn't override it
    
    # agent_policy_names 처리: rMAPPO 정책을 직접 로드하므로 scripted agent 불필요
    # parse_args()에서 default=None으로 설정되지만, 환경 코드가 len(None)을 호출하려고 해서 에러 발생
    # 따라서 None이 아닌 빈 리스트로 설정하여 환경 코드의 len() 호출을 안전하게 만듦
    if not hasattr(args, 'agent_policy_names') or args.agent_policy_names is None:
        args.agent_policy_names = []  # 빈 리스트로 설정 (scripted agent 사용 안 함)
    
    # Validate checkpoint arguments
    if args.checkpoint_agent0 and not args.checkpoint_agent1:
        parser.error("--checkpoint_agent1 is required when using --checkpoint_agent0")
    if args.checkpoint_agent1 and not args.checkpoint_agent0:
        parser.error("--checkpoint_agent0 is required when using --checkpoint_agent1")
    
    # Auto-construct checkpoint_dir from seed and layout_name if not provided
    if not args.checkpoint_dir and not args.checkpoint_agent0:
        if not hasattr(args, 'seed') or args.seed is None:
            parser.error("Either --checkpoint_dir, --checkpoint_agent0/agent1, or --seed must be provided")
        if not hasattr(args, 'layout_name') or args.layout_name is None:
            parser.error("--layout_name is required when auto-constructing checkpoint_dir")
        # Auto-construct: results/Overcooked/{layout_name}/rmappo/sp/seed{seed}/models
        args.checkpoint_dir = f"results/Overcooked/{args.layout_name}/rmappo/sp/seed{args.seed}/models"
        logger.info(f"Auto-constructed checkpoint_dir: {args.checkpoint_dir}")
    
    # Validate checkpoint_step requires checkpoint_dir (when auto-constructed)
    if args.checkpoint_step is not None and not args.checkpoint_dir:
        parser.error("--checkpoint_step requires --checkpoint_dir")
    
    return args


def main():
    """Main execution function.
    
    IMPORTANT: HIL trajectory generation MUST use single environment because:
    1. Each trajectory is a single episode for human annotation
    2. Buffer is saved per-episode (not aggregated across parallel envs)
    3. Offline training loads one buffer at a time
    """
    args = parse_args()
    
    # CRITICAL: Force n_rollout_threads=1 for HIL
    if not hasattr(args, 'n_rollout_threads'):
        args.n_rollout_threads = 1
    elif args.n_rollout_threads != 1:
        logger.warning(
            f"HIL trajectory generation requires n_rollout_threads=1 "
            f"(got {args.n_rollout_threads}). Forcing to 1."
        )
        args.n_rollout_threads = 1
    
    # Set default output_dir with layout_name if not provided
    if args.output_dir is None:
        args.output_dir = f"human_interface/data/trajectories_for_human/{args.layout_name}"
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("HIL Trajectory Generation")
    print("=" * 80)
    print(f"Layout: {args.layout_name}")
    print(f"Seed: {args.seed}")
    print(f"Num agents: {args.num_agents} (must be 2 for Overcooked)")
    print(f"Num trajectories: {args.num_trajectories}")
    print(f"Output directory: {args.output_dir}")
    print(f"n_rollout_threads: {args.n_rollout_threads} (must be 1 for HIL)")
    print("=" * 80)
    print()
    
    # Validate num_agents
    if args.num_agents != 2:
        print(f"Error: num_agents must be 2 for Overcooked, got {args.num_agents}")
        return 1
    
    # Determine checkpoint paths (dynamically for all agents)
    checkpoint_paths = []
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        
        # Verify directory exists and list contents
        if not checkpoint_dir.exists():
            print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
            return 1
        
        print(f"Checkpoint directory: {checkpoint_dir}")
        all_pt_files = list(checkpoint_dir.glob("*.pt"))
        print(f"  Found {len(all_pt_files)} .pt files in directory")
        if len(all_pt_files) > 0:
            print(f"  Sample files: {[f.name for f in all_pt_files[:5]]}")
        print()
        
        if args.checkpoint_step is not None:
            # Use specific checkpoint step
            print(f"Finding checkpoints at step {args.checkpoint_step} in: {checkpoint_dir}")
            for agent_id in range(args.num_agents):
                try:
                    checkpoint_path = find_checkpoint_by_step(checkpoint_dir, agent_id, args.checkpoint_step)
                    if not checkpoint_path.exists():
                        print(f"  ✗ Agent {agent_id}: Checkpoint file does not exist: {checkpoint_path}")
                        return 1
                    checkpoint_paths.append(checkpoint_path)
                    print(f"  ✓ Agent {agent_id}: {checkpoint_path.name}")
                except FileNotFoundError as e:
                    print(f"  ✗ Agent {agent_id}: {e}")
                    # List available files for this agent
                    agent_files = [f.name for f in all_pt_files if f'agent{agent_id}' in f.name.lower()]
                    if agent_files:
                        print(f"    Available files for agent {agent_id}: {agent_files[:5]}")
                    return 1
        else:
            # Use latest checkpoint
            print(f"Finding latest checkpoints in: {checkpoint_dir}")
            for agent_id in range(args.num_agents):
                try:
                    checkpoint_path = find_latest_checkpoint(checkpoint_dir, agent_id)
                    if not checkpoint_path.exists():
                        print(f"  ✗ Agent {agent_id}: Checkpoint file does not exist: {checkpoint_path}")
                        return 1
                    checkpoint_paths.append(checkpoint_path)
                    print(f"  ✓ Agent {agent_id}: {checkpoint_path.name}")
                except FileNotFoundError as e:
                    print(f"  ✗ Agent {agent_id}: {e}")
                    # List available files for this agent
                    agent_files = [f.name for f in all_pt_files if f'agent{agent_id}' in f.name.lower()]
                    if agent_files:
                        print(f"    Available files for agent {agent_id}: {agent_files[:5]}")
                    return 1
    else:
        # Use explicit checkpoint paths
        if args.checkpoint_agent0 and args.checkpoint_agent1:
            checkpoint_paths = [
                Path(args.checkpoint_agent0),
                Path(args.checkpoint_agent1)
            ]
            # Verify files exist
            for i, ckpt_path in enumerate(checkpoint_paths):
                if not ckpt_path.exists():
                    print(f"Error: Checkpoint file does not exist for agent {i}: {ckpt_path}")
                    return 1
            print(f"✓ Agent 0 checkpoint: {checkpoint_paths[0]}")
            print(f"✓ Agent 1 checkpoint: {checkpoint_paths[1]}")
        else:
            print("Error: Must provide either --checkpoint_dir or both --checkpoint_agent0 and --checkpoint_agent1")
            return 1
    
    # Validate checkpoint paths count matches num_agents
    if len(checkpoint_paths) != args.num_agents:
        print(f"Error: Found {len(checkpoint_paths)} checkpoint(s) but num_agents={args.num_agents}")
        return 1
    
    print()
    
    # Create environment (same as post-hoc-eval.ipynb)
    print("Creating environment...")
    
    def make_eval_env(all_args, run_dir):
        """Create evaluation environment wrapped in VecEnv (same as post-hoc-eval.ipynb)."""
        def get_env_fn(rank):
            def init_env():
                env = Overcooked_new(all_args, run_dir, rank=rank, evaluation=True)
                env.seed(all_args.seed * 50000 + rank * 10000)
                return env
            return init_env
        
        # Use ShareDummyVecEnv for single thread, ShareSubprocDummyBatchVecEnv for multiple
        if all_args.n_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocDummyBatchVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)],
                getattr(all_args, 'dummy_batch_size', 1)
            )
    
    run_dir = Path(".")
    eval_envs = make_eval_env(args, run_dir)
    print("✓ Environment created (wrapped in VecEnv)")
    print()
    
    # Create policies
    print("Initializing policies...")
    policy_list = []
    device = torch.device("cuda" if args.cuda else "cpu")
    
    for agent_id in range(args.num_agents):
        policy = R_MAPPOPolicy(
            args,
            eval_envs.observation_space[agent_id],
            eval_envs.share_observation_space[agent_id],
            eval_envs.action_space[agent_id],
            device=device
        )
        policy_list.append(policy)
        print(f"  Agent {agent_id}: Policy initialized (obs_space: {eval_envs.observation_space[agent_id]}, action_space: {eval_envs.action_space[agent_id]})")
    
    print(f"✓ {len(policy_list)} policies initialized for {args.num_agents} agents")
    print()
    
    # Create trajectory generator
    print("Creating trajectory generator...")
    generator = TrajectoryGenerator(
        env=eval_envs,  # Pass wrapped VecEnv
        trainer_list=policy_list,
        layout_name=args.layout_name,
        trajectory_dir=args.output_dir,
        buffer_dir=args.output_dir.replace('trajectories_for_human', 'buffers_for_training') if 'trajectories_for_human' in args.output_dir else None,
        args=args  # Pass args for buffer creation
    )
    print("✓ Generator created")
    print()
    
    # Load checkpoints
    print("Loading checkpoints...")
    print(f"  Checkpoint paths:")
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"    Agent {i}: {ckpt_path}")
    
    success = generator.load_agent_checkpoints(
        checkpoint_paths=[str(p) for p in checkpoint_paths],
        device=device
    )
    
    if not success:
        print("✗ Failed to load checkpoints")
        return 1
    
    print(f"✓ Successfully loaded checkpoints for {len(checkpoint_paths)} agents")
    print()
    
    # Generate trajectories
    print(f"Generating {args.num_trajectories} trajectory(ies)...")
    print()
    
    # Determine checkpoint_step for JSON naming
    checkpoint_step_for_json = args.checkpoint_step if args.checkpoint_step is not None else "latest"
    
    if args.num_trajectories == 1:
        # Single trajectory - use seed_{checkpoint_step}.json naming
        if args.checkpoint_step is not None:
            json_filename = f"seed{args.seed}_{args.checkpoint_step}.json"
        else:
            # Extract step from checkpoint filename if available
            checkpoint_name = checkpoint_paths[0].stem
            if 'periodic_' in checkpoint_name:
                step = checkpoint_name.split('periodic_')[-1]
                json_filename = f"seed{args.seed}_{step}.json"
            else:
                json_filename = f"seed{args.seed}_latest.json"
        
        json_save_path = Path(args.output_dir) / json_filename
        generator._custom_json_path = str(json_save_path)
        
        result = generator.generate_trajectory_and_save_buffer(
            episode_id=args.start_episode_id,
            episode_length=args.episode_length,
            deterministic=args.deterministic
        )
        
        if result:
            print(f"✓ Trajectory and buffer saved:")
            print(f"  - JSON: {result['json_path']}")
            for i, buf_path in enumerate(result['buffer_paths']):
                print(f"  - Buffer agent{i}: {buf_path}")
        else:
            print("✗ Failed to generate trajectory")
            return 1
    else:
        # Multiple trajectories
        all_results = []
        for i in range(args.num_trajectories):
            episode_id = args.start_episode_id + i
            
            # Determine checkpoint_step for this trajectory
            if args.checkpoint_step is not None:
                # If checkpoint_step is specified, use it for all trajectories
                checkpoint_step = args.checkpoint_step
            else:
                # Extract step from checkpoint filename
                checkpoint_name = checkpoint_paths[0].stem
                if 'periodic_' in checkpoint_name:
                    checkpoint_step = checkpoint_name.split('periodic_')[-1]
                else:
                    checkpoint_step = "latest"
            
            json_filename = f"seed{args.seed}_{checkpoint_step}_{i}.json"
            json_save_path = Path(args.output_dir) / json_filename
            generator._custom_json_path = str(json_save_path)
            
            print(f"Generating trajectory {i+1}/{args.num_trajectories} (episode {episode_id})...")
            
            result = generator.generate_trajectory_and_save_buffer(
                episode_id=episode_id,
                episode_length=args.episode_length,
                deterministic=args.deterministic
            )
            
            if result:
                all_results.append(result)
                print(f"  ✓ Episode {episode_id} complete")
            else:
                print(f"  ✗ Episode {episode_id} failed")
        
        print()
        print(f"✓ Generated {len(all_results)}/{args.num_trajectories} trajectories:")
        for result in all_results:
            json_name = Path(result['json_path']).name if result['json_path'] else 'failed'
            print(f"  - {json_name}")
    
    print()
    print("=" * 80)
    print("✓ Trajectory generation complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Human reviews trajectories and adds feedback")
    print("2. Run train_from_feedback.py to train with feedback")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

