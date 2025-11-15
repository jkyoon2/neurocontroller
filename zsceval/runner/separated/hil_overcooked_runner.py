"""
HIL Overcooked Runner: Human-in-the-Loop training for Overcooked environment.

This runner extends OvercookedRunner to support HIL training:
- Agent 0 uses HILRolloutBuffer and HIL_RMAPPO trainer
- Other agents use standard SeparatedReplayBuffer and R_MAPPO
- Loads and applies human feedback from JSON files
- Tracks HIL-specific metrics
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from loguru import logger

from zsceval.runner.separated.overcooked_runner import OvercookedRunner
from zsceval.utils.hil_rollout_buffer import HILRolloutBuffer
from zsceval.utils.separated_buffer import SeparatedReplayBuffer
from zsceval.algorithms.r_mappo_hil.hil_trainer import HIL_RMAPPO
from zsceval.algorithms.r_mappo.r_mappo import R_MAPPO
from zsceval.utils.feedback.feedback_loader import FeedbackLoader
from zsceval.utils.log_util import eta


def _t2n(x):
    """Convert torch tensor to numpy array."""
    return x.detach().cpu().numpy()


class HilOvercookedRunner(OvercookedRunner):
    """Human-in-the-Loop Overcooked Runner.
    
    Extends OvercookedRunner to integrate human feedback into training:
    - Agent 0 and 1 are trained with HIL (uses HILRolloutBuffer and HIL_RMAPPO)
    - Loads human feedback from JSON files periodically
    - Applies feedback to both agents' buffers
    
    NOTE: For Overcooked, typically we have 2 agents, both receiving HIL feedback
    from the same trajectory JSON (which contains both agents' observations and actions).
    """
    
    def __init__(self, config):
        """Initialize HIL Overcooked Runner.
        
        Args:
            config: Configuration dictionary with all_args, envs, device, etc.
        """
        # Initialize parent (restores checkpoints if model_dir is provided)
        # After super().__init__(), policies are loaded and restored
        super().__init__(config)
        
        # HIL-specific parameters
        self.hil_feedback_dir = getattr(
            self.all_args, 'hil_feedback_dir',
            'human_interface/data/feedback_from_human'
        )
        self.hil_feedback_interval = getattr(
            self.all_args, 'hil_feedback_interval', 10
        )  # Check for new feedback every N episodes
        self.hil_penalty_magnitude = getattr(
            self.all_args, 'hil_penalty_magnitude', -1.0
        )
        self.hil_calibration_window = getattr(
            self.all_args, 'hil_calibration_window', 5
        )
        self.hil_error_buffer_size = getattr(
            self.all_args, 'hil_error_buffer_size', 10000
        )
        self.hil_use_calibrated = getattr(
            self.all_args, 'hil_use_calibrated', True
        )  # Use calibrated feedback by default
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
        
        # Determine which agents use HIL (default: agents 0 and 1 for Overcooked)
        self.hil_agent_ids = getattr(
            self.all_args, 'hil_agent_ids', list(range(self.num_agents))
        )  # Default: all agents use HIL
        
        # Initialize feedback loader
        self.feedback_loader = FeedbackLoader(feedback_dir=self.hil_feedback_dir)
        self.last_feedback_check = 0
        self.feedback_applied_count = 0
        
        # Override buffer and trainer for HIL agents
        # NOTE: This happens AFTER parent's restore(), so checkpoints are already loaded
        self._setup_hil_agents()
        
        logger.info(
            f"HilOvercookedRunner initialized:\n"
            f"  - HIL Agents: {self.hil_agent_ids}\n"
            f"  - Feedback Dir: {self.hil_feedback_dir}\n"
            f"  - Feedback Interval: {self.hil_feedback_interval} episodes\n"
            f"  - Penalty: {self.hil_penalty_magnitude}\n"
            f"  - Calibration Window: ±{self.hil_calibration_window}\n"
            f"  - Use Calibrated: {self.hil_use_calibrated}\n"
            f"  - M2 (Policy Constraint): {self.hil_enable_m2}\n"
            f"  - M3 (State Augmentation): {self.hil_enable_m3}"
        )
    
    def _setup_hil_agents(self):
        """Setup HIL-specific buffer and trainer for specified agents.
        
        NOTE: This is called AFTER parent's __init__(), which means:
        1. self.policy[agent_id] is already created
        2. Checkpoints are already loaded via restore()
        3. We replace buffer and trainer while keeping the loaded policy weights
        """
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
            
            # Create HIL trainer (reusing the already-loaded policy from parent)
            hil_trainer = HIL_RMAPPO(
                self.all_args,
                self.policy[agent_id],  # Policy weights already loaded by restore()
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
                f"✓ Agent {agent_id} configured for HIL training:\n"
                f"  - Buffer: HILRolloutBuffer\n"
                f"  - Trainer: HIL_RMAPPO\n"
                f"  - Policy weights: {'loaded from checkpoint' if self.model_dir else 'randomly initialized'}"
            )
    
    def run(self):
        """Main training loop with HIL feedback integration."""
        self.warmup()
        
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0
        
        for episode in range(episodes):
            # Check for new feedback periodically
            if episode % self.hil_feedback_interval == 0:
                self._check_and_apply_feedback(episode)
            
            # Linear learning rate decay
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
            
            # Rollout
            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                
                # Environment step
                (
                    _obs_batch_single_agent,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                
                # Extract all_agent_obs from info_list
                obs = np.array([info['all_agent_obs'] for info in infos])
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)
                
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                
                # Insert data into buffer
                self.insert(data)
            
            # Compute returns and update networks
            self.compute()
            train_infos = self.train(total_num_steps)
            
            # Add HIL-specific metrics for each HIL agent
            for agent_id in self.hil_agent_ids:
                if agent_id >= len(train_infos):
                    continue
                    
                if isinstance(self.buffer[agent_id], HILRolloutBuffer):
                    hil_stats = self.buffer[agent_id].get_error_buffer_stats()
                    train_infos[agent_id].update({
                        f"hil/error_buffer_size": hil_stats["buffer_size"],
                        f"hil/feedback_applied_count": hil_stats["feedback_applied_count"],
                        f"hil/total_error_timesteps": hil_stats["total_error_timesteps"],
                    })
                    
                    # Add M2, M3 stats if available
                    if isinstance(self.trainer[agent_id], HIL_RMAPPO):
                        trainer_hil_stats = self.trainer[agent_id].get_hil_stats()
                        if "m2_policy_constraint" in trainer_hil_stats:
                            train_infos[agent_id].update({
                                f"hil/m2_{k}": v
                                for k, v in trainer_hil_stats["m2_policy_constraint"].items()
                            })
                        if "m3_state_augmentation" in trainer_hil_stats:
                            train_infos[agent_id].update({
                                f"hil/m3_{k}": v
                                for k, v in trainer_hil_stats["m3_state_augmentation"].items()
                            })
            
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # Save model
            if episode < 50:
                if episode % 2 == 0:
                    self.save(total_num_steps, save_critic=True)
            elif episode < 100:
                if episode % 5 == 0:
                    self.save(total_num_steps, save_critic=True)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save(total_num_steps, save_critic=True)
            
            # Log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                eta_t = eta(start, end, self.num_env_steps, total_num_steps)
                logger.info(
                    "Layout {} Algo {} (HIL) Exp {} Seed {} updates {}/{} episodes, "
                    "total num timesteps {}/{}, FPS {}, ETA {}.".format(
                        self.all_args.layout_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                        eta_t,
                    )
                )
                
                for a in range(self.num_agents):
                    train_infos[a]["average_episode_rewards"] = np.mean(self.buffer[a].rewards) * self.episode_length
                    logger.info(
                        "agent {} average episode rewards is {}".format(a, train_infos[a]["average_episode_rewards"])
                    )
                
                # Log HIL-specific info for all HIL agents
                for agent_id in self.hil_agent_ids:
                    if isinstance(self.buffer[agent_id], HILRolloutBuffer):
                        hil_stats = self.buffer[agent_id].get_error_buffer_stats()
                        logger.info(
                            f"HIL Agent {agent_id}: error_buffer={hil_stats['buffer_size']}, "
                            f"feedback_applied={self.feedback_applied_count}"
                        )
                
                env_infos = defaultdict(list)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)
                
                if self.env_name == "Overcooked":
                    if self.all_args.overcooked_version == "old":
                        from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
                        shaped_info_keys = SHAPED_INFOS
                    else:
                        from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
                        shaped_info_keys = SHAPED_INFOS
                    
                    for info in infos:
                        for a in range(self.num_agents):
                            env_infos[f"ep_sparse_r_by_agent{a}"].append(info["episode"]["ep_sparse_r_by_agent"][a])
                            env_infos[f"ep_shaped_r_by_agent{a}"].append(info["episode"]["ep_shaped_r_by_agent"][a])
                            if "ep_hidden_r_by_agent" in info["episode"]:
                                env_infos[f"ep_hidden_r_by_agent{a}"].append(info["episode"]["ep_hidden_r_by_agent"][a])
                            for i, k in enumerate(shaped_info_keys):
                                env_infos[f"ep_{k}_by_agent{a}"].append(info["episode"]["ep_category_r_by_agent"][a][i])
                        env_infos["ep_sparse_r"].append(info["episode"]["ep_sparse_r"])
                        env_infos["ep_shaped_r"].append(info["episode"]["ep_shaped_r"])
                
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                logger.info(f'average sparse rewards is {np.mean(env_infos["ep_sparse_r"]):.3f}')
            
            # Eval
            if episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                self.eval(total_num_steps)
    
    def _check_and_apply_feedback(self, episode: int):
        """Check for new feedback files and apply them to all HIL agents' buffers.
        
        NOTE: A single trajectory JSON contains observations/actions for BOTH agents.
        We apply the same feedback to both agents' buffers.
        
        Args:
            episode: Current episode number
        """
        logger.debug(f"[Episode {episode}] Checking for new human feedback...")
        
        # Scan feedback directory for JSON files
        feedback_files = self.feedback_loader.scan_feedback_files()
        
        if not feedback_files:
            logger.debug("No feedback files found")
            return
        
        # Apply each feedback file to all HIL agents' buffers
        applied_count = 0
        for feedback_file in feedback_files:
            try:
                # Load feedback
                feedback = self.feedback_loader.load_feedback(feedback_file.name)
                
                # Check if feedback matches current layout
                if feedback.layout_name != self.all_args.layout_name:
                    logger.warning(
                        f"Feedback layout ({feedback.layout_name}) doesn't match "
                        f"current layout ({self.all_args.layout_name}), skipping"
                    )
                    continue
                
                # Choose timesteps based on configuration
                if self.hil_use_calibrated:
                    error_timesteps = feedback.get_calibrated_timesteps()
                    feedback_type = "calibrated"
                else:
                    error_timesteps = feedback.get_realtime_timesteps()
                    feedback_type = "real-time"
                
                if not error_timesteps:
                    logger.warning(f"No {feedback_type} error timesteps in {feedback_file.name}")
                    continue
                
                # Apply feedback to ALL HIL agents' buffers
                # (Same trajectory JSON contains info for both agents)
                for agent_id in self.hil_agent_ids:
                    if not isinstance(self.trainer[agent_id], HIL_RMAPPO):
                        continue
                    
                    stats = self.trainer[agent_id].apply_human_feedback(
                        buffer=self.buffer[agent_id],
                        realtime_timesteps=feedback.get_realtime_timesteps(),
                        calibrated_timesteps=feedback.get_calibrated_timesteps(),
                        env_id=0,  # Always apply to env_id=0 (first rollout thread)
                        use_calibrated=self.hil_use_calibrated,
                    )
                    
                    logger.info(
                        f"✓ Applied feedback to Agent {agent_id} from {feedback_file.name} ({feedback_type}):\n"
                        f"  - Errors: {stats['num_errors']}\n"
                        f"  - Rewards modified: {stats['m1_rewards_modified']}\n"
                        f"  - Actions masked: {stats['m2_actions_masked']}\n"
                        f"  - States stored: {stats['m3_states_stored']}"
                    )
                
                applied_count += 1
                self.feedback_applied_count += 1
                
                # Log to wandb (aggregate stats)
                if self.use_wandb:
                    wandb.log({
                        "hil/feedback_applied": self.feedback_applied_count,
                        "hil/feedback_file": feedback_file.name,
                    }, step=episode)
                
            except Exception as e:
                logger.error(f"Failed to apply feedback from {feedback_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        if applied_count > 0:
            logger.info(f"Applied {applied_count} feedback file(s) at episode {episode}")
        else:
            logger.debug("No new feedback applied")
        
        self.last_feedback_check = episode
    
    def train(self, total_num_steps: int) -> List[Dict]:
        """Train all agents (with HIL for specified agents).
        
        Args:
            total_num_steps: Total number of environment steps so far
            
        Returns:
            List of training info dictionaries for each agent
        """
        train_infos = []
        
        for agent_id in range(self.num_agents):
            if not self.trainer_trainable[agent_id]:
                # Skip non-trainable agents
                train_infos.append({})
                continue
            
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            
            # Reset HIL stats after each training update (for HIL agents)
            if agent_id in self.hil_agent_ids and isinstance(self.trainer[agent_id], HIL_RMAPPO):
                self.trainer[agent_id].reset_hil_stats()
            
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()
        
        return train_infos
    
    def save(self, total_num_steps: int, save_critic: bool = True):
        """Save checkpoints for all agents.
        
        For HIL agents, also save HIL-specific information.
        
        Args:
            total_num_steps: Total number of environment steps
            save_critic: Whether to save critic networks
        """
        # Call parent save method (saves actor/critic checkpoints)
        super().save(total_num_steps, save_critic)
        
        # Save HIL-specific state for each HIL agent
        for agent_id in self.hil_agent_ids:
            if isinstance(self.buffer[agent_id], HILRolloutBuffer):
                hil_state_path = Path(self.save_dir) / f"hil_state_agent{agent_id}_step{total_num_steps}.pt"
                hil_state = {
                    "agent_id": agent_id,
                    "error_buffer_size": len(self.buffer[agent_id].error_buffer),
                    "feedback_applied_count": self.buffer[agent_id].feedback_applied_count,
                    "total_error_timesteps": self.buffer[agent_id].total_error_timesteps,
                    "feedback_applied_count_runner": self.feedback_applied_count,
                }
                torch.save(hil_state, hil_state_path)
                logger.debug(f"Saved HIL state for agent {agent_id} to {hil_state_path}")
    
    def restore(self):
        """Restore checkpoints for all agents.
        
        For HIL agents, also restore HIL-specific information if available.
        
        NOTE: This is called by parent's __init__(), so it happens BEFORE _setup_hil_agents().
        HIL state restoration is informational only (for logging feedback_applied_count).
        The actual error buffer is reset each training session.
        """
        # Call parent restore method (loads actor/critic checkpoints)
        super().restore()
        
        # Try to restore HIL state for each HIL agent
        if self.model_dir is not None:
            for agent_id in self.hil_agent_ids:
                hil_state_files = sorted(Path(self.model_dir).glob(f"hil_state_agent{agent_id}_step*.pt"))
                if hil_state_files:
                    latest_hil_state = hil_state_files[-1]
                    try:
                        hil_state = torch.load(latest_hil_state, map_location='cpu', weights_only=False)
                        self.feedback_applied_count = hil_state.get("feedback_applied_count_runner", 0)
                        logger.info(
                            f"✓ Restored HIL state for agent {agent_id} from {latest_hil_state.name}:\n"
                            f"  - Feedback applied: {self.feedback_applied_count}\n"
                            f"  - Error buffer size (previous session): {hil_state.get('error_buffer_size', 0)}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to restore HIL state for agent {agent_id}: {e}")

