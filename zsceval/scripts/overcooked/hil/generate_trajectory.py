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
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked
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
    # Look for periodic checkpoints first
    periodic_files = sorted(
        checkpoint_dir.glob(f"actor_agent{agent_id}_periodic_*.pt"),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    if periodic_files:
        return periodic_files[-1]
    
    # Fallback to non-periodic
    regular_file = checkpoint_dir / f"actor_agent{agent_id}.pt"
    if regular_file.exists():
        return regular_file
    
    raise FileNotFoundError(f"No checkpoint found for agent {agent_id} in {checkpoint_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = get_config()
    parser = get_overcooked_args(parser)
    
    # Checkpoint arguments
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory containing checkpoints (auto-finds latest)"
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
        default="human_interface/data/trajectories_for_human",
        help="Output directory for trajectory JSON files"
    )
    
    args = parser.parse_args()
    
    # Add missing attributes for Overcooked environment
    if not hasattr(args, 'use_phi'):
        args.use_phi = False
    if not hasattr(args, 'old_dynamics'):
        args.old_dynamics = False
    
    # Validate checkpoint arguments
    if args.checkpoint_agent0 and not args.checkpoint_agent1:
        parser.error("--checkpoint_agent1 is required when using --checkpoint_agent0")
    if args.checkpoint_agent1 and not args.checkpoint_agent0:
        parser.error("--checkpoint_agent0 is required when using --checkpoint_agent1")
    
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
    
    print("=" * 80)
    print("HIL Trajectory Generation")
    print("=" * 80)
    print(f"Layout: {args.layout_name}")
    print(f"Num trajectories: {args.num_trajectories}")
    print(f"Output directory: {args.output_dir}")
    print(f"n_rollout_threads: {args.n_rollout_threads} (must be 1 for HIL)")
    print("=" * 80)
    print()
    
    # Determine checkpoint paths
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        print(f"Finding latest checkpoints in: {checkpoint_dir}")
        checkpoint_paths = [
            find_latest_checkpoint(checkpoint_dir, 0),
            find_latest_checkpoint(checkpoint_dir, 1)
        ]
        print(f"  Agent 0: {checkpoint_paths[0].name}")
        print(f"  Agent 1: {checkpoint_paths[1].name}")
    else:
        checkpoint_paths = [
            Path(args.checkpoint_agent0),
            Path(args.checkpoint_agent1)
        ]
        print(f"Agent 0 checkpoint: {checkpoint_paths[0]}")
        print(f"Agent 1 checkpoint: {checkpoint_paths[1]}")
    
    print()
    
    # Create environment
    print("Creating environment...")
    env = Overcooked(
        all_args=args,
        run_dir=".",
        rank=0,
        evaluation=True
    )
    print("✓ Environment created")
    print()
    
    # Create policies
    print("Initializing policies...")
    policy_list = []
    device = torch.device("cuda" if args.cuda else "cpu")
    
    for agent_id in range(args.num_agents):
        policy = R_MAPPOPolicy(
            args,
            env.observation_space[agent_id],
            env.share_observation_space[agent_id],
            env.action_space[agent_id],
            device=device
        )
        policy_list.append(policy)
    
    print(f"✓ {len(policy_list)} policies initialized")
    print()
    
    # Create trajectory generator
    print("Creating trajectory generator...")
    generator = TrajectoryGenerator(
        env=env,
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
    success = generator.load_agent_checkpoints(
        checkpoint_paths=[str(p) for p in checkpoint_paths],
        device=device
    )
    
    if not success:
        print("✗ Failed to load checkpoints")
        return 1
    
    print()
    
    # Generate trajectories
    print(f"Generating {args.num_trajectories} trajectory(ies)...")
    print()
    
    if args.num_trajectories == 1:
        # Single trajectory
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

