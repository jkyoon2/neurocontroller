"""
HIL R-MAPPO Trainer: Human-in-the-Loop extension of R-MAPPO.

This trainer integrates human feedback into the R-MAPPO training loop through
three methods:
    M1: Reward shaping
    M2: Policy constraints
    M3: State augmentation (via VAE)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger

from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from zsceval.algorithms.r_mappo.r_mappo import R_MAPPO
from zsceval.utils.util import check, get_gard_norm

from .policy_constraint import PolicyConstraint
from .reward_shaper import RewardShaper
from .state_augmentor import StateAugmentor


class HIL_RMAPPO(R_MAPPO):
    """Human-in-the-Loop R-MAPPO Trainer.
    
    Extends R_MAPPO with human feedback integration capabilities:
    - M1: Reward shaping based on human-identified errors
    - M2: Policy constraints at error points
    - M3: State augmentation using VAE
    """
    
    def __init__(
        self,
        args,
        policy: R_MAPPOPolicy,
        device=torch.device("cpu"),
        # HIL-specific parameters
        enable_m2: bool = True,
        enable_m3: bool = False,  # Disabled by default until VAE is implemented
        # M2 parameters
        constraint_coef: float = 0.1,
        constraint_type: str = "kl",
        # M3 parameters
        vae_path: Optional[str] = None,
        augmentation_factor: int = 2,
    ):
        """Initialize HIL R-MAPPO trainer.
        
        NOTE: M1 (Reward Shaping) is handled by HILRolloutBuffer.apply_feedback(),
        not by this trainer. The buffer already modifies rewards based on human
        feedback with calibration windows.
        
        Args:
            args: Training arguments
            policy: R_MAPPOPolicy instance
            device: Training device
            enable_m2: Enable policy constraints
            enable_m3: Enable state augmentation
            constraint_coef: Coefficient for M2 constraint loss
            constraint_type: Type of constraint ("kl", "l2", "entropy")
            vae_path: Path to pretrained VAE for M3
            augmentation_factor: Number of augmented samples per error state for M3
        """
        # Initialize base R_MAPPO
        super().__init__(args, policy, device)
        
        # HIL configuration
        # M1 is handled by HILRolloutBuffer.apply_feedback()
        self.enable_m1 = True  # Always enabled via buffer
        self.enable_m2 = enable_m2
        self.enable_m3 = enable_m3
        
        # Initialize HIL modules
        # M1: No separate module needed (handled by buffer)
        self.reward_shaper = None
        
        # M2: Policy constraints
        self.policy_constraint = PolicyConstraint(
            constraint_coef=constraint_coef,
            constraint_type=constraint_type
        ) if enable_m2 else None
        
        # M3: State augmentation
        self.state_augmentor = StateAugmentor(
            vae_path=vae_path,
            augmentation_factor=augmentation_factor,
            device=device
        ) if enable_m3 else None
        
        # Load VAE if provided
        if enable_m3 and vae_path:
            self.state_augmentor.load_vae(vae_path)
        
        logger.info(
            f"HIL_RMAPPO initialized: "
            f"M1=enabled(via buffer), M2={enable_m2}, M3={enable_m3}"
        )
    
    def apply_human_feedback(
        self,
        buffer,
        realtime_timesteps: List[int],
        calibrated_timesteps: List[int],
        env_id: int = 0,
        use_calibrated: bool = True,
        apply_m1: bool = True,
        apply_m2: bool = True,
        apply_m3: bool = True
    ) -> Dict[str, any]:
        """Apply human feedback to buffer.
        
        This method delegates to buffer.apply_feedback() which handles:
        - M1: Reward shaping with calibration window
        - M2: Marking error action masks
        - M3: Storing error states in error buffer
        
        Args:
            buffer: HILRolloutBuffer instance
            realtime_timesteps: Real-time error timesteps from human
            calibrated_timesteps: Calibrated error timesteps from human
            env_id: Environment ID to apply feedback to
            use_calibrated: Whether to use calibrated or real-time feedback
            apply_m1: Apply reward shaping (M1)
            apply_m2: Mark error actions for policy constraints (M2)
            apply_m3: Store error states for augmentation (M3)
            
        Returns:
            Dictionary with feedback application statistics
        """
        # Choose which timesteps to use
        error_timesteps = calibrated_timesteps if use_calibrated else realtime_timesteps
        
        # Delegate to buffer's apply_feedback method
        # This handles M1 (reward shaping), M2 (error masking), and M3 (error storage)
        stats = buffer.apply_feedback(
            error_timesteps=error_timesteps,
            env_id=env_id,
            apply_reward_shaping=apply_m1 and self.enable_m1,
            apply_action_masking=apply_m2 and self.enable_m2,
            store_error_states=apply_m3 and self.enable_m3
        )
        
        logger.info(
            f"Human feedback applied to buffer (env_id={env_id}): "
            f"{stats['rewards_modified']} rewards modified, "
            f"{stats['actions_masked']} actions masked, "
            f"{stats['states_stored']} states stored"
        )
        
        return {
            "num_errors": len(error_timesteps),
            "use_calibrated": use_calibrated,
            "m1_rewards_modified": stats["rewards_modified"],
            "m2_actions_masked": stats["actions_masked"],
            "m3_states_stored": stats["states_stored"],
        }
    
    def ppo_update(self, sample, turn_on=True, **kwargs):
        """PPO update with HIL modifications.
        
        Extends base PPO update to include M2 policy constraints.
        
        CONNECTION TO BUFFER:
        - HILRolloutBuffer.feed_forward_generator() yields 13-element tuples
        - Element 13: error_masks_batch (NEW, for M2 policy constraints)
        - This method extracts error_masks_batch and uses it with PolicyConstraint
        
        Args:
            sample: Mini-batch sample from buffer (13 or 14 elements for HIL)
            turn_on: Whether to perform update
            **kwargs: Additional arguments
            
        Returns:
            Tuple of loss components and metrics
        """
        # Unpack sample (HIL buffer yields 13 elements with error_masks_batch)
        if len(sample) == 13:
            # HILRolloutBuffer with error masks
            (
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
            ) = sample
            task_id_batch = None
        elif len(sample) == 14:
            # HILRolloutBuffer with error masks + task_id (for share_policy)
            (
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
                error_masks_batch,  # 13th element
                task_id_batch,  # 14th element
            ) = sample
        else:
            # Standard buffer (12 elements, no error masks, no task_id)
            (
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
            ) = sample
            error_masks_batch = None
            task_id_batch = None
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        
        # === Evaluate actions (same as base R_MAPPO) ===
        values, action_log_probs, dist_entropy, policy_values = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            task_id=task_id_batch,
        )
        
        # === Calculate value loss (same as base) ===
        value_loss = self.cal_value_loss(
            self.value_normalizer,
            values,
            value_preds_batch,
            return_batch,
            active_masks_batch,
        )
        
        # === Calculate policy loss (same as base) ===
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        
        # === M2: Add policy constraint loss ===
        constraint_loss = torch.tensor(0.0, device=self.device)
        if self.enable_m2 and self.policy_constraint and error_masks_batch is not None:
            # Convert error masks to tensor
            error_mask_tensor = check(error_masks_batch).to(**self.tpdv)
            
            # Only compute constraint if there are actual errors in this batch
            if error_mask_tensor.sum() > 0:
                # Get distribution from policy for KL-based constraints
                # Note: For discrete actions, we can use actor's last distribution
                dist = None  # Will be used by PolicyConstraint if needed
                
                constraint_loss = self.policy_constraint.compute_constraint_loss(
                    action_log_probs=action_log_probs,
                    actions_batch=actions_batch,
                    error_mask=error_mask_tensor,
                    dist=dist
                )
                
                logger.trace(
                    f"M2 constraint: {error_mask_tensor.sum().item():.0f} errors "
                    f"in batch, loss={constraint_loss.item():.6f}"
                )
        
        # === Calculate total policy loss ===
        policy_loss = policy_action_loss + constraint_loss
        
        # Entropy loss
        dist_entropy_loss = -dist_entropy.mean()
        
        # === Backpropagation ===
        self.policy.actor_optimizer.zero_grad()
        
        if turn_on:
            (policy_loss - dist_entropy_loss * self.entropy_coef).backward()
        
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        
        self.policy.actor_optimizer.step()
        
        # Value function update
        self.policy.critic_optimizer.zero_grad()
        
        if turn_on:
            (value_loss * self.value_loss_coef).backward()
        
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        
        self.policy.critic_optimizer.step()
        
        # === Calculate metrics ===
        upper_rate = ((ratio > 1.0 + self.clip_param) * active_masks_batch).sum() / active_masks_batch.sum()
        lower_rate = ((ratio < 1.0 - self.clip_param) * active_masks_batch).sum() / active_masks_batch.sum()
        
        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            ratio,
            upper_rate,
            lower_rate,
            self.entropy_coef,
        )
    
    def train(self, buffer, turn_on=True, **kwargs):
        """Train with HIL-enhanced R-MAPPO.
        
        Extends base train method to include M3 state augmentation.
        
        Args:
            buffer: HILRolloutBuffer instance
            turn_on: Whether to perform updates
            **kwargs: Additional arguments
            
        Returns:
            Training info dictionary with loss and metric values
        """
        # M3: Augment error states (if enabled)
        if self.enable_m3 and self.state_augmentor and hasattr(buffer, 'error_buffer'):
            if len(buffer.error_buffer) > 0:
                logger.debug(f"M3: Augmenting {len(buffer.error_buffer)} error states...")
                # TODO: Integrate augmented states into training
                # This will be implemented when VAE is ready
        
        # Call base train method
        return super().train(buffer, turn_on, **kwargs)
    
    def get_hil_stats(self) -> Dict[str, Dict]:
        """Get statistics from HIL modules.
        
        NOTE: M1 statistics are tracked by HILRolloutBuffer, not here.
        
        Returns:
            Dictionary containing stats from M2, M3
        """
        stats = {}
        
        # M1 stats are in the buffer
        stats["m1_reward_shaping"] = {
            "note": "M1 is handled by HILRolloutBuffer.apply_feedback(). "
                    "Check buffer.feedback_applied_count and buffer.total_error_timesteps."
        }
        
        if self.policy_constraint:
            stats["m2_policy_constraint"] = self.policy_constraint.get_stats()
        
        if self.state_augmentor:
            stats["m3_state_augmentation"] = self.state_augmentor.get_stats()
        
        return stats
    
    def reset_hil_stats(self):
        """Reset statistics for all HIL modules."""
        # M1 stats are in the buffer (reset by buffer itself)
        
        if self.policy_constraint:
            self.policy_constraint.reset_stats()
        
        if self.state_augmentor:
            self.state_augmentor.reset_stats()


# Import check utility
from zsceval.algorithms.utils.util import check

