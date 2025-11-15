"""
Offline HIL Runner for Overcooked

This runner implements offline Human-in-the-Loop training:
1. Loads a specific checkpoint
2. Replays trajectories to fill buffer
3. Applies human feedback to the buffer
4. Trains for multiple epochs on the fixed buffer (off-policy style)
5. Saves the updated checkpoint

This enables an asynchronous HIL workflow where humans can provide
feedback independently of the training process.
"""

import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
from loguru import logger

from zsceval.runner.separated.base_runner import Runner
from zsceval.algorithms.r_mappo_hil.hil_trainer import HIL_RMAPPO
from zsceval.utils.hil_rollout_buffer import HILRolloutBuffer
from zsceval.utils.feedback.feedback_loader import FeedbackLoader


def _t2n(x):
    """Convert torch tensor to numpy array."""
    return x.detach().cpu().numpy()


class HilOfflineRunner(Runner):
    """Offline HIL Runner for asynchronous human-in-the-loop training.
    
    Key differences from online runners:
    - Does NOT interact with environment during training
    - Replays trajectories from checkpoints to fill buffer
    - Applies human feedback to buffer
    - Trains for multiple epochs on fixed buffer (like off-policy RL)
    - Designed for checkpoint → trajectory → feedback → train cycle
    """
    
    def __init__(self, config):
        """Initialize Offline HIL Runner.
        
        Args:
            config: Configuration dictionary with all_args, envs, device, etc.
        """
        super().__init__(config)
        
        # Override save_dir if provided in config (for offline HIL, save directly to output_dir)
        if "save_dir" in config and config["save_dir"] is not None:
            self.save_dir = str(config["save_dir"])
        
        # HIL-specific parameters
        # Detect reward type from args
        self.reward_type = self._detect_reward_type()
        
        # Auto-detect penalty magnitude if not provided
        default_penalty = self._get_default_penalty_magnitude(self.reward_type)
        self.hil_penalty_magnitude = getattr(
            self.all_args, 'hil_penalty_magnitude', default_penalty
        )
        self.hil_calibration_window = getattr(
            self.all_args, 'hil_calibration_window', 5
        )
        self.hil_error_buffer_size = getattr(
            self.all_args, 'hil_error_buffer_size', 10000
        )
        self.hil_enable_m2 = getattr(
            self.all_args, 'hil_enable_m2', True
        )
        self.hil_enable_m3 = getattr(
            self.all_args, 'hil_enable_m3', False
        )
        self.hil_constraint_coef = getattr(
            self.all_args, 'hil_constraint_coef', 0.1
        )
        self.hil_constraint_type = getattr(
            self.all_args, 'hil_constraint_type', 'kl'
        )
        self.hil_vae_path = getattr(
            self.all_args, 'hil_vae_path', None
        )
        self.hil_train_epochs = getattr(
            self.all_args, 'hil_train_epochs', 10
        )
        
        # Which agents to apply HIL to (default: all agents)
        self.hil_agent_ids = getattr(
            self.all_args, 'hil_agent_ids', list(range(self.num_agents))
        )
        
        # Replace buffer and trainer for HIL agents
        self._setup_hil_agents()
        
        logger.info(
            f"HilOfflineRunner initialized:\n"
            f"  - Reward Type: {self.reward_type}\n"
            f"  - HIL Agents: {self.hil_agent_ids}\n"
            f"  - Training Epochs: {self.hil_train_epochs}\n"
            f"  - Penalty Magnitude: {self.hil_penalty_magnitude}\n"
            f"  - Calibration Window: ±{self.hil_calibration_window}\n"
            f"  - M2 (Policy Constraint): {self.hil_enable_m2}\n"
            f"  - M3 (State Augmentation): {self.hil_enable_m3}"
        )
    
    def _detect_reward_type(self) -> str:
        """Detect reward type from environment configuration.
        
        Returns:
            'sparse', 'shaped', or 'hsp'
        """
        # Check if HSP is used
        use_hsp = getattr(self.all_args, 'use_hsp', False)
        if use_hsp:
            return 'hsp'
        
        # Check reward shaping factor
        reward_shaping_factor = getattr(self.all_args, 'reward_shaping_factor', 0.0)
        if reward_shaping_factor > 0:
            return 'shaped'
        
        return 'sparse'
    
    def _get_default_penalty_magnitude(self, reward_type: str) -> float:
        """Get default penalty magnitude based on reward type.
        
        For HSP, attempts to estimate appropriate penalty from weight magnitudes.
        HSP rewards can be VERY large (100-600 per step) due to fixed weight scaling.
        
        Args:
            reward_type: 'sparse', 'shaped', or 'hsp'
            
        Returns:
            Recommended penalty magnitude
        """
        if reward_type == 'hsp':
            # HSP: Fixed user-specified weights, rewards can be extremely large!
            # Example: w0 = "...,20,-5,30" with shaped_info values -> rewards of 100-600 per step
            
            try:
                w0 = getattr(self.all_args, 'w0', None)
                
                # Parse w0 if it's a string (e.g., "0,1,2,3,...")
                if w0 is not None and isinstance(w0, str):
                    w0 = np.array([float(x) for x in w0.split(',')])
                
                if w0 is not None and isinstance(w0, (list, np.ndarray)):
                    w0 = np.array(w0)
                    
                    # Calculate statistics of the weight vector
                    max_weight = np.abs(w0).max()
                    avg_weight = np.abs(w0[w0 != 0]).mean() if np.any(w0 != 0) else 1.0
                    
                    # Use max weight as reference (most influential feature)
                    # Penalty should be 2-5x the max weight to be noticeable
                    penalty = -3.0 * max_weight
                    
                    logger.info(
                        f"HSP penalty auto-scaled from weights:\n"
                        f"  - Max weight: {max_weight:.1f}\n"
                        f"  - Avg weight (non-zero): {avg_weight:.2f}\n"
                        f"  - Calculated penalty: {penalty:.1f}"
                    )
                    
                    # Ensure penalty is substantial but not extreme
                    # Clamp between -10 and -100
                    return max(min(penalty, -10.0), -100.0)
            except Exception as e:
                logger.warning(f"Could not estimate HSP penalty from weights: {e}")
            
            # Conservative default for HSP (much larger than shaped)
            # Without weight info, assume moderately large rewards
            return -30.0
        
        elif reward_type == 'shaped':
            # Shaped rewards: moderate density, ~0.5-2.0 per step
            # Penalty of -2.0 is noticeable but not overwhelming
            return -2.0
        
        else:  # sparse
            # Sparse rewards: very rare (only on soup delivery)
            # Need stronger penalty signal to be effective
            return -7.0
    
    def _setup_hil_agents(self):
        """Setup HIL-specific buffer and trainer for specified agents."""
        for agent_id in self.hil_agent_ids:
            if agent_id >= self.num_agents:
                logger.warning(f"HIL agent_id {agent_id} >= num_agents {self.num_agents}, skipping")
                continue
            
            # Create HIL buffer
            share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            
            hil_buffer = HILRolloutBuffer(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
                human_penalty_magnitude=self.hil_penalty_magnitude,
                calibration_window=self.hil_calibration_window,
                error_buffer_size=self.hil_error_buffer_size,
            )
            
            # Create HIL trainer (reusing the policy from parent)
            hil_trainer = HIL_RMAPPO(
                self.all_args,
                self.policy[agent_id],
                device=self.device,
                enable_m2=self.hil_enable_m2,
                enable_m3=self.hil_enable_m3,
                constraint_coef=self.hil_constraint_coef,
                constraint_type=self.hil_constraint_type,
                vae_path=self.hil_vae_path,
            )
            
            # Replace agent's buffer and trainer
            self.buffer[agent_id] = hil_buffer
            self.trainer[agent_id] = hil_trainer
            
            logger.info(
                f"✓ Agent {agent_id} configured for offline HIL:\n"
                f"  - Buffer: HILRolloutBuffer\n"
                f"  - Trainer: HIL_RMAPPO"
            )
    
    def _detect_reward_type(self) -> str:
        """Detect reward type from environment configuration.
        
        Returns:
            'sparse', 'shaped', or 'hsp'
        """
        # Check if HSP is used
        use_hsp = getattr(self.all_args, 'use_hsp', False)
        if use_hsp:
            return 'hsp'
        
        # Check reward shaping factor
        reward_shaping_factor = getattr(self.all_args, 'reward_shaping_factor', 0.0)
        if reward_shaping_factor > 0:
            return 'shaped'
        
        return 'sparse'
    
    def _get_default_penalty_magnitude(self, reward_type: str) -> float:
        """Get default penalty magnitude based on reward type.
        
        Args:
            reward_type: 'sparse', 'shaped', or 'hsp'
            
        Returns:
            Recommended penalty magnitude
        """
        penalty_map = {
            'sparse': -7.0,   # Sparse rewards need stronger penalty signal
            'shaped': -2.0,   # Shaped rewards are denser, moderate penalty
            'hsp': -3.0,      # HSP scale-dependent, conservative default
        }
        
        return penalty_map.get(reward_type.lower(), -2.0)
    
    def load_buffers_from_files(self, buffer_paths: List[str]):
        """Load pre-saved buffers for all HIL agents.
        
        This is the key to true offline HIL training - we load buffers that
        were saved during trajectory generation, avoiding any environment interaction.
        
        Args:
            buffer_paths: List of buffer file paths for each agent
                         (e.g., ["buffer_episode_42_agent0.pt", "buffer_episode_42_agent1.pt"])
        """
        logger.info(f"Loading {len(buffer_paths)} buffer(s) for offline HIL training...")
        
        if len(buffer_paths) != len(self.hil_agent_ids):
            logger.warning(
                f"Buffer paths count ({len(buffer_paths)}) != HIL agents count ({len(self.hil_agent_ids)}). "
                f"Will load buffers for available agents."
            )
        
        for i, agent_id in enumerate(self.hil_agent_ids):
            if i >= len(buffer_paths):
                logger.warning(f"No buffer path provided for agent {agent_id}, skipping")
                continue
            
            buffer_path = buffer_paths[i]
            
            if not isinstance(self.buffer[agent_id], HILRolloutBuffer):
                raise TypeError(f"Agent {agent_id} buffer is not HILRolloutBuffer")
            
            self.buffer[agent_id].load_from_file(buffer_path)
            
            # Verify buffer state
            if self.buffer[agent_id].feedback_applied:
                logger.warning(
                    f"Agent {agent_id} buffer already has feedback applied! "
                    f"This might be intentional if continuing training."
                )
        
        logger.info(f"✓ All buffers loaded successfully")
    
    def apply_feedback_to_buffer(self, feedback_json_path: str):
        """Apply human feedback from JSON file to buffer.
        
        Args:
            feedback_json_path: Path to feedback JSON file
        """
        logger.info(f"Applying feedback from: {feedback_json_path}")
        
        # Load feedback using absolute path (FeedbackLoader defaults to relative paths)
        feedback_path = Path(feedback_json_path)
        if not feedback_path.is_absolute():
            feedback_path = Path.cwd() / feedback_path
        
        # Use FeedbackLoader with absolute path or load directly
        import json
        from zsceval.utils.feedback.feedback_schema import parse_feedback_json
        
        with open(feedback_path, 'r') as f:
            feedback_data = json.load(f)
        feedback = parse_feedback_json(feedback_data)

        # Extract layout name from static_info
        feedback_layout = feedback.static_info.get('layoutName', 'unknown')
        
        # Validate layout (optional - just log warning if mismatch)
        if feedback_layout != self.all_args.layout_name:
            raise ValueError(
                f"Feedback layout ({feedback.layout_name}) doesn't match "
                f"current layout ({self.all_args.layout_name})"
            )
        
        # Apply feedback to all HIL agents
        total_stats = defaultdict(int)
        
        for agent_id in self.hil_agent_ids:
            if not isinstance(self.trainer[agent_id], HIL_RMAPPO):
                continue
            
            stats = self.trainer[agent_id].apply_human_feedback(
                buffer=self.buffer[agent_id],
                realtime_timesteps=feedback.get_realtime_timesteps(),
                calibrated_timesteps=feedback.get_calibrated_timesteps(),
                env_id=0,  # Apply to first rollout thread
                use_calibrated=True,  # Use calibrated by default for offline training
            )
            
            for key, value in stats.items():
                total_stats[key] += value
            
            logger.info(
                f"✓ Feedback applied to Agent {agent_id}:\n"
                f"  - Errors: {stats['num_errors']}\n"
                f"  - Rewards modified: {stats['m1_rewards_modified']}\n"
                f"  - Actions masked: {stats['m2_actions_masked']}\n"
                f"  - States stored: {stats['m3_states_stored']}"
            )
        
        return total_stats
    
    def train_from_buffer(self) -> List[List[Dict]]:
        """Train agents for multiple epochs on the fixed buffer.
        
        This is similar to off-policy training where we reuse the same
        buffer multiple times.
        
        Returns:
            List of training info for each epoch (each epoch has list of agent infos)
        """
        logger.info(f"Training for {self.hil_train_epochs} epochs on fixed buffer...")
        
        all_epoch_infos = []
        
        for epoch in range(self.hil_train_epochs):
            epoch_train_infos = []
            
            for agent_id in range(self.num_agents):
                if not self.trainer_trainable[agent_id]:
                    epoch_train_infos.append({})
                    continue
                
                self.trainer[agent_id].prep_training()
                # Adapt entropy coefficient (use 0 for offline training)
                self.trainer[agent_id].adapt_entropy_coef(0)
                train_info = self.trainer[agent_id].train(self.buffer[agent_id])
                
                # Reset HIL stats
                if agent_id in self.hil_agent_ids and isinstance(self.trainer[agent_id], HIL_RMAPPO):
                    self.trainer[agent_id].reset_hil_stats()
                
                epoch_train_infos.append(train_info)
            
            all_epoch_infos.append(epoch_train_infos)
            
            # Log to wandb/tensorboard
            self.log_train(epoch_train_infos, epoch)
            
            # Log epoch summary
            if epoch % max(1, self.hil_train_epochs // 10) == 0:
                logger.info(f"Epoch {epoch + 1}/{self.hil_train_epochs} complete")
        
        logger.info("✓ Training complete")
        
        return all_epoch_infos
    
    def run_offline_hil(self, buffer_paths: List[str], feedback_json_path: str):
        """Full offline HIL training pipeline.
        
        Steps:
        1. Load pre-saved buffers (NO environment interaction!)
        2. Apply human feedback to buffer
        3. Train for multiple epochs on fixed buffer
        4. Save updated checkpoint
        
        Args:
            buffer_paths: List of paths to saved buffer files for each agent
            feedback_json_path: Path to feedback JSON file
        """
        logger.info("=" * 80)
        logger.info("Starting Offline HIL Training")
        logger.info("=" * 80)
        logger.info(f"Checkpoint loaded from: {self.model_dir}")
        logger.info(f"Buffer files: {len(buffer_paths)} agent(s)")
        for i, path in enumerate(buffer_paths):
            logger.info(f"  - Agent {i}: {Path(path).name}")
        logger.info(f"Feedback file: {Path(feedback_json_path).name}")
        logger.info(f"Training epochs: {self.hil_train_epochs}")
        logger.info("=" * 80)
        print()
        
        start_time = time.time()
        
        # Step 1: Load buffers (TRUE offline - no env interaction!)
        logger.info("Step 1/4: Loading pre-saved buffers...")
        self.load_buffers_from_files(buffer_paths)
        print()
        
        # Step 2: Apply feedback
        logger.info("Step 2/4: Applying human feedback to buffer...")
        feedback_stats = self.apply_feedback_to_buffer(feedback_json_path)
        print()
        
        # Step 3: Train
        logger.info("Step 3/4: Training with feedback...")
        all_epoch_infos = self.train_from_buffer()
        print()
        
        # Step 4: Save
        logger.info("Step 4/4: Saving updated checkpoint...")
        self.save(steps=None, save_critic=True)
        print()
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info("✓ Offline HIL Training Complete")
        logger.info("=" * 80)
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        logger.info(f"Feedback errors processed: {feedback_stats.get('num_errors', 0)}")
        logger.info(f"Checkpoints saved to: {self.save_dir}")
        logger.info("=" * 80)
        
        return all_epoch_infos
    
    def restore(self):
        """Override parent's restore() - we manually load checkpoints instead.
        
        HilOfflineRunner uses periodic checkpoints (e.g., actor_agent0_periodic_10000000.pt)
        which don't follow the standard naming. We load them manually via
        load_checkpoints_from_paths() after __init__().
        """
        logger.debug("Skipping auto-restore (will manually load checkpoints)")
        pass
    
    def load_checkpoints_from_paths(self, checkpoint_paths: List[Path]):
        """Load checkpoints from specific paths (supporting periodic checkpoints).
        
        This should be called AFTER __init__() when self.trainer exists.
        
        Args:
            checkpoint_paths: List of checkpoint file paths for each agent
        """
        logger.info(f"Loading {len(checkpoint_paths)} checkpoint(s)...")
        
        for agent_id in range(min(len(checkpoint_paths), self.num_agents)):
            actor_path = checkpoint_paths[agent_id]
            critic_path = Path(str(actor_path).replace('actor', 'critic'))
            
            try:
                # Load actor
                if actor_path.exists():
                    actor_state_dict = torch.load(actor_path, map_location=self.device, weights_only=False)
                    self.trainer[agent_id].policy.actor.load_state_dict(actor_state_dict)
                    logger.info(f"✓ Agent {agent_id} actor: {actor_path.name}")
                else:
                    raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
                
                # Load critic
                if critic_path.exists():
                    critic_state_dict = torch.load(critic_path, map_location=self.device, weights_only=False)
                    self.trainer[agent_id].policy.critic.load_state_dict(critic_state_dict)
                    logger.info(f"✓ Agent {agent_id} critic: {critic_path.name}")
                else:
                    logger.warning(f"Critic checkpoint not found: {critic_path}")
                    
            except Exception as e:
                logger.error(f"Failed to load checkpoint for agent {agent_id}: {e}")
                raise
        
        logger.info("✓ All checkpoints loaded successfully")
    
    # Override methods that are not used in offline mode
    def run(self):
        """Not used in offline mode. Use run_offline_hil() instead."""
        raise NotImplementedError(
            "HilOfflineRunner does not support run(). "
            "Use run_offline_hil(buffer_paths, feedback_json_path) instead."
        )
    
    def warmup(self):
        """Not used in offline mode (buffers are loaded directly)."""
        raise NotImplementedError(
            "HilOfflineRunner does not need warmup. "
            "Buffers are loaded directly from files."
        )
    
    def collect(self, step):
        """Not used in offline mode (buffers are loaded directly)."""
        raise NotImplementedError(
            "HilOfflineRunner does not collect actions. "
            "Buffers are loaded directly from files."
        )
    
    def insert(self, data):
        """Not used in offline mode (buffers are loaded directly)."""
        raise NotImplementedError(
            "HilOfflineRunner does not insert data. "
            "Buffers are loaded directly from files."
        )

