"""
HIL Rollout Buffer: Extended replay buffer for Human-in-the-Loop training.

This buffer extends SeparatedReplayBuffer to support:
- M1: Reward Shaping (modify rewards based on human feedback)
- M2: Policy Constraint (mark error actions for constraint)
- M3: State Representation Augmentation (store error states for VAE augmentation)
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from zsceval.utils.separated_buffer import SeparatedReplayBuffer


class HILRolloutBuffer(SeparatedReplayBuffer):
    """Extended replay buffer for HIL training.
    
    This buffer maintains the same interface as SeparatedReplayBuffer
    but adds HIL-specific functionality for integrating human feedback.
    """
    
    def __init__(
        self,
        args,
        obs_space,
        share_obs_space,
        act_space,
        # HIL-specific parameters
        human_penalty_magnitude: float = -1.0,
        calibration_window: int = 5,
        error_buffer_size: int = 10000,
    ):
        """Initialize HIL Rollout Buffer.
        
        Args:
            args: Configuration arguments (must contain episode_length, n_rollout_threads, etc.)
            obs_space: Observation space
            share_obs_space: Shared observation space
            act_space: Action space
            human_penalty_magnitude: Penalty value for error timesteps (M1)
            calibration_window: Window size around error timestep to apply penalty
            error_buffer_size: Maximum size of error buffer for M2/M3
        """
        super().__init__(args, obs_space, share_obs_space, act_space)
        
        # HIL-specific configurations
        self.human_penalty_magnitude = human_penalty_magnitude
        self.calibration_window = calibration_window
        self.error_buffer_size = error_buffer_size
        
        # M1: Reward modification tracking
        # Track which rewards have been modified by human feedback
        self.reward_modifications = np.zeros(
            (self.episode_length, self.n_rollout_threads),
            dtype=np.float32
        )
        
        # M2: Error action masks
        # Mark timesteps where errors occurred (for policy constraint)
        self.error_action_masks = np.zeros(
            (self.episode_length, self.n_rollout_threads),
            dtype=np.bool_
        )
        
        # M3: Error buffer for state augmentation
        # Store (obs, share_obs, actions) tuples for error states
        self.error_buffer = deque(maxlen=error_buffer_size)
        
        # Metadata tracking
        self.feedback_applied_count = 0
        self.total_error_timesteps = 0
        self.feedback_applied = False  # Track if human feedback has been applied
        
        logger.info(
            f"HILRolloutBuffer initialized: "
            f"penalty={human_penalty_magnitude}, "
            f"window={calibration_window}, "
            f"error_buffer_size={error_buffer_size}"
        )
    
    def apply_feedback(
        self,
        error_timesteps: List[int],
        env_id: int = 0,
        apply_reward_shaping: bool = True,
        apply_action_masking: bool = True,
        store_error_states: bool = True,
    ) -> Dict[str, int]:
        """Apply human feedback to the buffer.
        
        This method implements M1, M2, and M3 by modifying the buffer data
        based on error timesteps provided by human annotators.
        
        Args:
            error_timesteps: List of timesteps where errors occurred
            env_id: Environment ID (rollout thread) to apply feedback to
            apply_reward_shaping: Enable M1 (reward modification)
            apply_action_masking: Enable M2 (error action masking)
            store_error_states: Enable M3 (store error states for augmentation)
            
        Returns:
            Statistics dictionary with counts of applied feedback
        """
        stats = {
            "rewards_modified": 0,
            "actions_masked": 0,
            "states_stored": 0,
        }
        
        if env_id >= self.n_rollout_threads:
            logger.warning(f"Invalid env_id {env_id} (max: {self.n_rollout_threads})")
            return stats
        
        logger.debug(
            f"Applying feedback to env_id={env_id}: "
            f"{len(error_timesteps)} error timesteps"
        )
        
        for error_t in error_timesteps:
            # Ensure error_t is within valid range
            if error_t < 0 or error_t >= self.episode_length:
                logger.warning(f"Error timestep {error_t} out of range [0, {self.episode_length})")
                continue
            
            # M1: Reward Shaping
            if apply_reward_shaping:
                reward_modified = self._apply_reward_shaping(error_t, env_id)
                stats["rewards_modified"] += reward_modified
            
            # M2: Policy Constraint (mark error action)
            if apply_action_masking:
                action_masked = self._mark_error_action(error_t, env_id)
                stats["actions_masked"] += action_masked
            
            # M3: Store error state for augmentation
            if store_error_states:
                state_stored = self._store_error_state(error_t, env_id)
                stats["states_stored"] += state_stored
        
        self.feedback_applied_count += 1
        self.total_error_timesteps += len(error_timesteps)
        self.feedback_applied = True  # Mark that feedback has been applied
        
        logger.info(
            f"Feedback applied (count={self.feedback_applied_count}): "
            f"rewards_modified={stats['rewards_modified']}, "
            f"actions_masked={stats['actions_masked']}, "
            f"states_stored={stats['states_stored']}"
        )
        
        return stats
    
    def _apply_reward_shaping(self, error_t: int, env_id: int) -> int:
        """M1: Apply reward shaping around error timestep.
        
        Args:
            error_t: Error timestep
            env_id: Environment ID
            
        Returns:
            Number of rewards modified (window size)
        """
        # Apply penalty to window around error timestep
        start_t = max(0, error_t - self.calibration_window)
        end_t = min(self.episode_length, error_t + self.calibration_window + 1)
        
        modified_count = 0
        for t in range(start_t, end_t):
            # Apply penalty (additive)
            self.rewards[t, env_id] += self.human_penalty_magnitude
            
            # Track modification
            self.reward_modifications[t, env_id] += abs(self.human_penalty_magnitude)
            
            modified_count += 1
        
        logger.trace(
            f"M1 Reward shaping: timesteps [{start_t}, {end_t}) "
            f"for env_id={env_id}, penalty={self.human_penalty_magnitude}"
        )
        
        return modified_count
    
    def _mark_error_action(self, error_t: int, env_id: int) -> int:
        """M2: Mark error action for policy constraint.
        
        Args:
            error_t: Error timestep
            env_id: Environment ID
            
        Returns:
            1 if action was marked, 0 otherwise
        """
        # Mark this timestep as containing an error action
        if not self.error_action_masks[error_t, env_id]:
            self.error_action_masks[error_t, env_id] = True
            logger.trace(f"M2 Action masked: timestep={error_t}, env_id={env_id}")
            return 1
        
        return 0
    
    def _store_error_state(self, error_t: int, env_id: int) -> int:
        """M3: Store error state for VAE augmentation.
        
        Args:
            error_t: Error timestep
            env_id: Environment ID
            
        Returns:
            1 if state was stored, 0 otherwise
        """
        try:
            # Extract state-action tuple
            # Note: We store timestep t (before action) not t+1
            error_data = {
                "obs": self.obs[error_t, env_id].copy(),
                "share_obs": self.share_obs[error_t, env_id].copy(),
                "action": self.actions[error_t, env_id].copy(),
                "timestep": error_t,
                "env_id": env_id,
            }
            
            # Add to error buffer (deque will auto-evict oldest if full)
            self.error_buffer.append(error_data)
            
            logger.trace(
                f"M3 State stored: timestep={error_t}, env_id={env_id}, "
                f"buffer_size={len(self.error_buffer)}"
            )
            
            return 1
            
        except Exception as e:
            logger.error(f"Failed to store error state: {e}")
            return 0
    
    def sample_error_states(
        self, 
        batch_size: int,
        return_tensors: bool = True
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Sample error states from the error buffer.
        
        This is used by M3 (State Augmentation) to provide error states
        to the VAE for reconstruction/augmentation.
        
        Args:
            batch_size: Number of error states to sample
            return_tensors: If True, return torch tensors; else numpy arrays
            
        Returns:
            Dictionary with 'obs', 'share_obs', 'actions' batches, or None if buffer too small
        """
        if len(self.error_buffer) < batch_size:
            logger.warning(
                f"Error buffer too small for sampling: "
                f"requested={batch_size}, available={len(self.error_buffer)}"
            )
            return None
        
        # Random sampling without replacement
        indices = np.random.choice(len(self.error_buffer), batch_size, replace=False)
        
        sampled_data = [self.error_buffer[i] for i in indices]
        
        # Stack into batches
        obs_batch = np.stack([d["obs"] for d in sampled_data])
        share_obs_batch = np.stack([d["share_obs"] for d in sampled_data])
        actions_batch = np.stack([d["action"] for d in sampled_data])
        
        if return_tensors:
            obs_batch = torch.from_numpy(obs_batch).float()
            share_obs_batch = torch.from_numpy(share_obs_batch).float()
            actions_batch = torch.from_numpy(actions_batch).float()
        
        return {
            "obs": obs_batch,
            "share_obs": share_obs_batch,
            "actions": actions_batch,
        }
    
    def get_error_action_mask(self, flatten: bool = True) -> np.ndarray:
        """Get error action mask for M2 (Policy Constraint).
        
        Args:
            flatten: If True, flatten to (T*N,); else keep as (T, N)
            
        Returns:
            Error action mask array
        """
        if flatten:
            return self.error_action_masks.reshape(-1)
        return self.error_action_masks
    
    def get_reward_modifications(self, flatten: bool = True) -> np.ndarray:
        """Get reward modification tracking for analysis.
        
        Args:
            flatten: If True, flatten to (T*N,); else keep as (T, N)
            
        Returns:
            Reward modification array
        """
        if flatten:
            return self.reward_modifications.reshape(-1)
        return self.reward_modifications
    
    def reset_hil_tracking(self):
        """Reset HIL-specific tracking arrays.
        
        Call this after updating the policy to prepare for next rollout.
        """
        self.reward_modifications.fill(0)
        self.error_action_masks.fill(False)
        # Note: We don't clear error_buffer as it accumulates across episodes
        
        logger.debug("HIL tracking arrays reset")
    
    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """Feed-forward generator that also yields error action masks.
        
        This extends the base generator to include error_action_masks for M2.
        
        CONNECTION TO POLICYCONSTRAINT:
        1. apply_feedback() marks error timesteps in self.error_action_masks
        2. This generator flattens and batches error_action_masks
        3. HIL_RMAPPO.ppo_update() receives error_masks_batch in sample
        4. PolicyConstraint uses error_masks_batch to compute constraint loss
        """
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"PPO requires the number of processes ({n_rollout_threads}) "
                f"* number of steps ({episode_length}) = {batch_size} "
                f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch})."
            )
            mini_batch_size = batch_size // num_mini_batch
        
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size : (i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        
        # Prepare data (same as base class)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages_flat = advantages.reshape(-1, 1)
        
        # NEW: Prepare error action masks for M2
        # Shape: (episode_length, n_rollout_threads) -> (episode_length * n_rollout_threads, 1)
        error_masks = self.error_action_masks.reshape(-1, 1).astype(np.float32)
        
        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages_flat[indices]
            
            # NEW: Include error masks in sample
            error_masks_batch = error_masks[indices]
            
            yield (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                error_masks_batch,  # NEW: 13th element for M2
            )
    
    def get_error_buffer_stats(self) -> Dict[str, any]:
        """Get statistics about the error buffer.
        
        Returns:
            Dictionary with error buffer statistics
        """
        return {
            "buffer_size": len(self.error_buffer),
            "buffer_capacity": self.error_buffer_size,
            "feedback_applied_count": self.feedback_applied_count,
            "total_error_timesteps": self.total_error_timesteps,
        }
    
    def after_update(self):
        """Override parent method to also reset HIL tracking."""
        super().after_update()
        self.reset_hil_tracking()
    
    def save_to_file(self, save_path: str):
        """Save buffer to file for offline HIL training.
        
        This allows us to:
        1. Generate trajectories once with env interaction
        2. Save buffer with all training data (obs, actions, values, etc.)
        3. Later load buffer and apply feedback without re-running environment
        
        Args:
            save_path: Path to save buffer file (.pt)
        """
        # Determine actual filled length (step pointer)
        filled_length = self.step if hasattr(self, 'step') and self.step > 0 else self.episode_length
        
        buffer_data = {
            # Core RL data
            'obs': self.obs[:filled_length].copy(),
            'share_obs': self.share_obs[:filled_length].copy(),
            'actions': self.actions[:filled_length].copy(),
            'action_log_probs': self.action_log_probs[:filled_length].copy(),
            'value_preds': self.value_preds[:filled_length].copy(),
            'returns': self.returns[:filled_length].copy() if hasattr(self, 'returns') else None,
            'rewards': self.rewards[:filled_length].copy(),
            'masks': self.masks[:filled_length].copy(),
            'bad_masks': self.bad_masks[:filled_length].copy() if hasattr(self, 'bad_masks') else None,
            'active_masks': self.active_masks[:filled_length].copy() if hasattr(self, 'active_masks') else None,
            'rnn_states': self.rnn_states[:filled_length].copy(),
            'rnn_states_critic': self.rnn_states_critic[:filled_length].copy(),
            'available_actions': self.available_actions[:filled_length].copy() if self.available_actions is not None else None,
            
            # HIL-specific data
            'reward_modifications': self.reward_modifications[:filled_length].copy(),
            'error_action_masks': self.error_action_masks[:filled_length].copy(),
            'error_buffer': list(self.error_buffer),  # Convert deque to list
            
            # Metadata
            'metadata': {
                'episode_length': filled_length,
                'n_rollout_threads': self.n_rollout_threads,
                'human_penalty_magnitude': self.human_penalty_magnitude,
                'calibration_window': self.calibration_window,
                'error_buffer_size': self.error_buffer_size,
                'feedback_applied': self.feedback_applied,
                'feedback_applied_count': self.feedback_applied_count,
                'total_error_timesteps': self.total_error_timesteps,
            }
        }
        
        torch.save(buffer_data, save_path)
        logger.info(
            f"✓ Buffer saved to {save_path}:\n"
            f"  - Timesteps: {filled_length}\n"
            f"  - Feedback applied: {self.feedback_applied}\n"
            f"  - Error buffer size: {len(self.error_buffer)}"
        )
    
    def load_from_file(self, load_path: str):
        """Load buffer from file for offline HIL training.
        
        Args:
            load_path: Path to buffer file (.pt)
        """
        buffer_data = torch.load(load_path, map_location='cpu', weights_only=False)
        
        # Extract metadata
        metadata = buffer_data['metadata']
        filled_length = metadata['episode_length']
        
        # Validate buffer dimensions
        if metadata['n_rollout_threads'] != self.n_rollout_threads:
            logger.warning(
                f"Buffer n_rollout_threads mismatch: "
                f"file={metadata['n_rollout_threads']}, "
                f"current={self.n_rollout_threads}"
            )
        
        # Load core RL data
        self.obs[:filled_length] = buffer_data['obs']
        self.share_obs[:filled_length] = buffer_data['share_obs']
        self.actions[:filled_length] = buffer_data['actions']
        self.action_log_probs[:filled_length] = buffer_data['action_log_probs']
        self.value_preds[:filled_length] = buffer_data['value_preds']
        if buffer_data['returns'] is not None:
            self.returns[:filled_length] = buffer_data['returns']
        self.rewards[:filled_length] = buffer_data['rewards']
        self.masks[:filled_length] = buffer_data['masks']
        if buffer_data['bad_masks'] is not None:
            self.bad_masks[:filled_length] = buffer_data['bad_masks']
        if buffer_data['active_masks'] is not None:
            self.active_masks[:filled_length] = buffer_data['active_masks']
        self.rnn_states[:filled_length] = buffer_data['rnn_states']
        self.rnn_states_critic[:filled_length] = buffer_data['rnn_states_critic']
        if buffer_data['available_actions'] is not None:
            self.available_actions[:filled_length] = buffer_data['available_actions']
        
        # Load HIL-specific data
        self.reward_modifications[:filled_length] = buffer_data['reward_modifications']
        self.error_action_masks[:filled_length] = buffer_data['error_action_masks']
        self.error_buffer = deque(buffer_data['error_buffer'], maxlen=self.error_buffer_size)
        
        # Load metadata
        self.feedback_applied = metadata['feedback_applied']
        self.feedback_applied_count = metadata['feedback_applied_count']
        self.total_error_timesteps = metadata['total_error_timesteps']
        
        # Update step pointer
        self.step = filled_length
        
        logger.info(
            f"✓ Buffer loaded from {load_path}:\n"
            f"  - Timesteps: {filled_length}\n"
            f"  - Feedback applied: {self.feedback_applied}\n"
            f"  - Error buffer size: {len(self.error_buffer)}\n"
            f"  - Total error timesteps: {self.total_error_timesteps}"
        )
    
    def __repr__(self) -> str:
        """String representation of HIL buffer state."""
        return (
            f"HILRolloutBuffer("
            f"episode_length={self.episode_length}, "
            f"n_rollout_threads={self.n_rollout_threads}, "
            f"error_buffer_size={len(self.error_buffer)}/{self.error_buffer_size}, "
            f"feedback_applied={self.feedback_applied}, "
            f"feedback_count={self.feedback_applied_count})"
        )

