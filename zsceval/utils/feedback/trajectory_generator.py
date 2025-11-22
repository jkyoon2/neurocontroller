"""
Trajectory Generator: Load agent checkpoints and generate trajectories.

This module handles:
1. Loading agent checkpoints from specified paths
2. Running rollout in environment
3. Collecting trajectory data for frontend export
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from loguru import logger

from zsceval.utils.feedback.trajectory_exporter import TrajectoryExporter
from zsceval.utils.hil_rollout_buffer import HILRolloutBuffer
from zsceval.algorithms.r_mappo.r_mappo import R_MAPPO


def _t2n(x):
    """Convert torch tensor to numpy array."""
    return x.detach().cpu().numpy()


class TrajectoryGenerator:
    """Generate trajectories by running agents in environment.
    
    This class loads agent checkpoints and generates trajectories
    that can be exported for human annotation. It also saves the
    full HILRolloutBuffer for offline HIL training.
    """
    
    def __init__(
        self,
        env,
        trainer_list: List[Any],
        layout_name: str,
        trajectory_dir: Optional[str] = None,
        buffer_dir: Optional[str] = None,
        args: Optional[Any] = None
    ):
        """Initialize TrajectoryGenerator.
        
        Args:
            env: Overcooked environment instance
            trainer_list: List of trainers/policies for each agent
            layout_name: Layout name for static info
            trajectory_dir: Directory to save trajectory JSONs (for frontend)
            buffer_dir: Directory to save buffer files (for offline training)
            args: Configuration arguments (needed for buffer creation)
        """
        self.env = env
        self.trainer = trainer_list  # List of trainers, one per agent
        self.num_agents = len(trainer_list)  # Number of agents (should be 2 for Overcooked)
        self.layout_name = layout_name
        self.args = args
        
        # Verify we have the correct number of agents
        if self.num_agents != 2:
            logger.warning(f"Expected 2 agents for Overcooked, got {self.num_agents}")
        
        # Trajectory exporter (for frontend JSONs)
        self.exporter = TrajectoryExporter(
            trajectory_dir=trajectory_dir,
            layout_name=layout_name
        )
        
        # Buffer directory (for offline training)
        if buffer_dir is None:
            self.buffer_dir = Path("human_interface/data/buffers_for_training")
        else:
            self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"TrajectoryGenerator initialized:\n"
            f"  - Agents: {self.num_agents}\n"
            f"  - Layout: {layout_name}\n"
            f"  - Buffer dir: {self.buffer_dir}\n"
            f"  - Trajectory dir: {self.exporter.trajectory_dir}"
        )
    
    def load_agent_checkpoints(
        self,
        checkpoint_paths: List[str],
        device: str = "cpu"
    ) -> bool:
        """Load agent checkpoints from specified paths.
        
        Args:
            checkpoint_paths: List of checkpoint paths for each agent
            device: Device to load checkpoints on
            
        Returns:
            True if all checkpoints loaded successfully
        """
        if len(checkpoint_paths) != self.num_agents:
            logger.error(
                f"Checkpoint paths length ({len(checkpoint_paths)}) "
                f"!= num_agents ({self.num_agents})"
            )
            return False
        
        logger.info(f"Loading {self.num_agents} agent checkpoints and creating trainers...")
        logger.info(f"  Checkpoint paths: {checkpoint_paths}")
        
        # Create new trainer list (wrapping policies with R_MAPPO)
        new_trainers = []
        
        for agent_id in range(self.num_agents):
            ckpt_path = Path(checkpoint_paths[agent_id])
            
            if not ckpt_path.exists():
                logger.error(f"Checkpoint not found for agent {agent_id}: {ckpt_path}")
                return False
            
            try:
                # Get the policy (currently in self.trainer)
                policy = self.trainer[agent_id]
                
                # Load checkpoint weights
                actor_state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
                policy.actor.load_state_dict(actor_state_dict)
                policy.actor.eval()
                
                # Load critic if available
                critic_path = str(ckpt_path).replace('actor', 'critic')
                if Path(critic_path).exists():
                    critic_state_dict = torch.load(critic_path, map_location=device, weights_only=False)
                    policy.critic.load_state_dict(critic_state_dict)
                    policy.critic.eval()
                    logger.info(f"  ✓ Agent {agent_id}: Loaded actor + critic from {ckpt_path.name}")
                else:
                    logger.info(f"  ✓ Agent {agent_id}: Loaded actor from {ckpt_path.name}")
                
                # Wrap policy with R_MAPPO trainer (includes value_normalizer)
                trainer = R_MAPPO(self.args, policy, device=torch.device(device))
                new_trainers.append(trainer)
                
                logger.debug(f"  Agent {agent_id}: Trainer created successfully")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint for agent {agent_id}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Replace policy list with trainer list
        self.trainer = new_trainers
        
        logger.info(f"✓ All {len(new_trainers)} checkpoints loaded with trainers (includes value_normalizer)")
        logger.info(f"  Trainer list length: {len(self.trainer)}, num_agents: {self.num_agents}")
        return True
    
    def generate_trajectory(
        self,
        episode_length: int = 400,
        deterministic: bool = True,
        render: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Generate a single trajectory by running agents in environment.
        
        Args:
            episode_length: Maximum episode length
            deterministic: Use deterministic actions
            render: Render the environment (for debugging)
            
        Returns:
            Tuple of (obs_sequence, actions_sequence, rewards_sequence, info_sequence)
        """
        logger.info(f"Generating trajectory (length={episode_length})...")
        
        # Reset environment
        reset_output = self.env.reset()
        
        logger.debug(f"env.reset() returned {len(reset_output)} items")
        
        if len(reset_output) == 2:
            obs_batch, info_list = reset_output
        else:
            obs_batch, info_list, _ = reset_output
        
        logger.debug(f"obs_batch type: {type(obs_batch)}, shape: {obs_batch.shape if hasattr(obs_batch, 'shape') else 'N/A'}")
        logger.debug(f"info_list type: {type(info_list)}, length: {len(info_list) if isinstance(info_list, list) else 'N/A'}")
        
        # info_list는 dict (단일 env) 또는 list (VecEnv)
        # 단일 env: obs_batch는 list of agent observations
        # obs_batch가 list면 각 agent의 obs를 담고 있음
        if isinstance(obs_batch, list):
            obs = obs_batch  # List of per-agent observations
            logger.debug(f"obs_batch is list, length: {len(obs)}")
        elif isinstance(info_list, dict) and 'all_agent_obs' in info_list:
            # 단일 env, info가 dict
            obs = info_list['all_agent_obs']  # shape: (num_agents, H, W, C)
            logger.debug(f"Using all_agent_obs from info dict, shape: {obs.shape}")
        elif isinstance(info_list, list) and len(info_list) > 0:
            if 'all_agent_obs' in info_list[0]:
                obs = info_list[0]['all_agent_obs']  # shape: (num_agents, H, W, C)
                logger.debug(f"Using all_agent_obs from info_list, shape: {obs.shape}")
            else:
                obs = obs_batch
        else:
            # obs_batch 자체를 사용 (단일 agent observation일 수 있음)
            obs = obs_batch
            logger.debug(f"Using obs_batch directly, shape: {obs.shape}")
        
        # Storage
        obs_sequence = [obs]
        actions_sequence = []
        rewards_sequence = []
        info_sequence = []
        
        # RNN states 초기화
        # R_Actor는 rnn_hidden_size 속성이 없고, args에서 가져와야 함
        # 또는 policy._recurrent_N과 policy.hidden_size 사용
        if hasattr(self.trainer[0], 'policy'):
            policy = self.trainer[0].policy
        else:
            policy = self.trainer[0]
        
        # hidden_size 찾기
        if hasattr(policy, 'hidden_size'):
            rnn_hidden_size = policy.hidden_size
        elif hasattr(policy, '_hidden_size'):
            rnn_hidden_size = policy._hidden_size
        elif hasattr(policy.actor, 'hidden_size'):
            rnn_hidden_size = policy.actor.hidden_size
        else:
            # Fallback: args에서 가져오기
            rnn_hidden_size = 64  # Default
            logger.warning(f"Could not find rnn_hidden_size, using default: {rnn_hidden_size}")
        
        rnn_states = [
            np.zeros((1, 1, rnn_hidden_size))  # (1, 1, hidden_size)
            for _ in range(self.num_agents)
        ]
        
        # Run episode
        for step in range(episode_length):
            # Get actions from each agent
            actions = []
            for agent_id in range(self.num_agents):
                # Get policy object
                if hasattr(self.trainer[agent_id], 'policy'):
                    policy = self.trainer[agent_id].policy
                    if hasattr(self.trainer[agent_id], 'prep_rollout'):
                        self.trainer[agent_id].prep_rollout()
                else:
                    policy = self.trainer[agent_id]
                
                # Get action
                # obs가 list면: obs[agent_id] shape: (H, W, C)
                # obs가 array면: obs[agent_id] shape: (H, W, C)
                # policy.act expects: (1, H, W, C) - batch dimension 필요
                
                if isinstance(obs, list):
                    obs_agent = obs[agent_id][np.newaxis, ...]  # (H,W,C) → (1,H,W,C)
                else:
                    obs_agent = obs[agent_id][np.newaxis, ...]  # (H,W,C) → (1,H,W,C)
                
                if step == 0:  # Log first step only
                    logger.debug(f"  Agent {agent_id} obs_agent shape: {obs_agent.shape}")
                
                try:
                    action, rnn_state = policy.act(
                        obs_agent,
                        rnn_states[agent_id],
                        masks=np.ones((1, 1)),
                        deterministic=deterministic
                    )
                    
                    # Convert to numpy (policy.act returns torch.Tensor)
                    if torch.is_tensor(action):
                        action = action.detach().cpu().numpy()
                    if torch.is_tensor(rnn_state):
                        rnn_state = rnn_state.detach().cpu().numpy()
                    
                except Exception as e:
                    logger.error(f"Error in policy.act() for agent {agent_id}: {e}")
                    logger.error(f"  obs_agent shape: {obs_agent.shape}")
                    logger.error(f"  rnn_states shape: {rnn_states[agent_id].shape}")
                    raise
                
                actions.append(action)
                rnn_states[agent_id] = rnn_state
            
            actions = np.concatenate(actions, axis=0)
            
            # Step environment
            step_output = self.env.step(actions)
            
            # Unpack step output
            # Overcooked returns: (obs_batch, info_list)
            if len(step_output) == 2:
                obs_batch, info_list = step_output
                rewards = None
                dones = None
            else:
                obs_batch, share_obs, rewards, dones, info_list, available_actions = step_output
            
            # Extract all_agent_obs from info_list (separated runner 방식)
            if isinstance(info_list, list) and len(info_list) > 0:
                if 'all_agent_obs' in info_list[0]:
                    obs = info_list[0]['all_agent_obs']  # (num_agents, H, W, C)
                else:
                    obs = obs_batch
                
                # rewards와 dones 추출
                if rewards is None and 'rewards' in info_list[0]:
                    rewards = np.array([info_list[0]['rewards']])
                if dones is None and 'dones' in info_list[0]:
                    dones = np.array([info_list[0]['dones']])
            else:
                obs = obs_batch
            
            # Store
            obs_sequence.append(obs)
            actions_sequence.append(actions)
            rewards_sequence.append(rewards)
            info_sequence.append(info_list)
            
            if render:
                self.env.render()
            
            # Check if done
            if dones[0]:
                logger.info(f"Episode finished at step {step}")
                break
        
        # Convert to arrays
        try:
            obs_array = np.array(obs_sequence)
            actions_array = np.array(actions_sequence)
            rewards_array = np.array(rewards_sequence) if rewards_sequence[0] is not None else np.zeros((len(actions_sequence), 1))
            
            logger.info(
                f"✓ Trajectory generated: {len(actions_sequence)} steps"
            )
            logger.debug(f"  obs_array shape: {obs_array.shape}")
            logger.debug(f"  actions_array shape: {actions_array.shape}")
            logger.debug(f"  rewards_array shape: {rewards_array.shape}")
            
            return obs_array, actions_array, rewards_array, info_sequence
            
        except Exception as e:
            logger.error(f"Failed to convert trajectory to arrays: {e}")
            logger.error(f"  obs_sequence length: {len(obs_sequence)}")
            logger.error(f"  actions_sequence length: {len(actions_sequence)}")
            logger.error(f"  rewards_sequence length: {len(rewards_sequence)}")
            raise
    
    def generate_trajectory_and_save_buffer(
        self,
        episode_id: int,
        episode_length: int = 400,
        deterministic: bool = True
    ) -> Optional[Dict[str, str]]:
        """Generate trajectory, save buffer AND export JSON.
        
        This method follows the same pattern as post-hoc-eval.ipynb:
        - Uses ShareDummyVecEnv or ShareSubprocDummyBatchVecEnv wrapped environment
        - Uses trainer.policy.act() for action selection (not get_actions())
        - Extracts observations from info['all_agent_obs']
        - Matches the exact evaluation workflow
        
        This is the main method for offline HIL workflow:
        1. Creates HILRolloutBuffers for each agent
        2. Runs rollout and fills buffers with full training data
        3. Saves buffers to disk (for later offline training)
        4. Exports JSON to disk (for frontend visualization)
        
        Args:
            episode_id: Episode ID for filenames
            episode_length: Maximum episode length
            deterministic: Use deterministic actions
            
        Returns:
            Dict with 'buffer_paths' (list) and 'json_path' (str), or None if failed
        """
        if self.args is None:
            logger.error("Cannot create buffers without args. Pass args to __init__.")
            return None
        
        # Validate n_rollout_threads (should be 1 for HIL, but we support VecEnv format)
        n_rollout_threads = getattr(self.args, 'n_rollout_threads', 1)
        if n_rollout_threads != 1:
            logger.warning(
                f"HIL trajectory generation typically uses n_rollout_threads=1, "
                f"got {n_rollout_threads}. Using VecEnv format."
            )
        
        try:
            logger.info(f"Generating trajectory {episode_id} with buffer saving...")
            
            # Create HILRolloutBuffers for each agent
            buffers = []
            for agent_id in range(self.num_agents):
                buffer = HILRolloutBuffer(
                    self.args,
                    self.env.observation_space[agent_id],
                    self.env.share_observation_space[agent_id],
                    self.env.action_space[agent_id]
                )
                buffers.append(buffer)
            
            # Initialize environment (VecEnv format: returns (obs_batch, info_list))
            eval_obs_batch, eval_info_list = self.env.reset()
            
            # Extract all_agent_obs from info_list (same as post-hoc-eval.ipynb)
            # info['all_agent_obs'] shape: (num_agents, H, W, C)
            # After np.array([...]): (n_rollout_threads, num_agents, H, W, C)
            eval_obs = np.array([info['all_agent_obs'] for info in eval_info_list])
            # eval_obs shape: (n_rollout_threads, num_agents, H, W, C)
            # For n_rollout_threads=1: (1, num_agents, H, W, C)
            
            # Debug: Verify shape
            logger.debug(f"After reset: eval_info_list length: {len(eval_info_list)}")
            if len(eval_info_list) > 0:
                logger.debug(f"  First info keys: {eval_info_list[0].keys()}")
                if 'all_agent_obs' in eval_info_list[0]:
                    logger.debug(f"  First info['all_agent_obs'].shape: {eval_info_list[0]['all_agent_obs'].shape}")
            
            # Verify eval_obs shape is correct
            if eval_obs.shape[1] != self.num_agents:
                logger.error(
                    f"eval_obs shape mismatch! Expected (n_rollout_threads, {self.num_agents}, H, W, C), "
                    f"got {eval_obs.shape}. This suggests only one agent's observations are being returned."
                )
                # Try to fix: if shape is (1, 1, H, W, C), we might need to check info structure
                if eval_obs.shape[1] == 1 and self.num_agents == 2:
                    logger.warning("Only one agent's observations found. Checking info structure...")
                    # Check if we need to extract differently
                    if len(eval_info_list) > 0 and 'all_agent_obs' in eval_info_list[0]:
                        all_agent_obs = eval_info_list[0]['all_agent_obs']
                        logger.debug(f"  Direct all_agent_obs.shape: {all_agent_obs.shape}")
                        # If it's already (num_agents, H, W, C), we just need to add batch dim
                        if len(all_agent_obs.shape) == 4 and all_agent_obs.shape[0] == self.num_agents:
                            eval_obs = np.expand_dims(all_agent_obs, axis=0)
                            logger.info(f"  Fixed eval_obs.shape to: {eval_obs.shape}")
            
            # Extract available_actions
            eval_available_actions = np.array([info['available_actions'] for info in eval_info_list])
            # eval_available_actions shape: (n_rollout_threads, num_agents, action_dim)
            
            logger.info(f"After reset: eval_obs.shape={eval_obs.shape}, eval_available_actions.shape={eval_available_actions.shape}")
            logger.info(f"  Expected eval_obs shape: (1, {self.num_agents}, H, W, C)")
            
            # Initialize RNN states (same format as post-hoc-eval.ipynb)
            # Shape: (n_rollout_threads, num_agents, recurrent_N, hidden_size)
            eval_rnn_states = np.zeros(
                (n_rollout_threads, self.num_agents, self.args.recurrent_N, self.args.hidden_size),
                dtype=np.float32
            )
            eval_masks = np.ones((n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            # Collect sequences for JSON export
            obs_sequence = [eval_obs[0].copy()]  # Use first rollout thread for JSON
            actions_sequence = []
            rewards_sequence = []
            
            # Rollout
            for step in range(episode_length):
                eval_actions = []
                
                # Collect actions from all agents (same as post-hoc-eval.ipynb)
                for agent_id in range(self.num_agents):
                    trainer = self.trainer[agent_id]
                    trainer.prep_rollout()
                    
                    # Use policy.act() (not get_actions()) - same as post-hoc-eval.ipynb
                    eval_action, eval_rnn_state = trainer.policy.act(
                        eval_obs[:, agent_id],  # Shape: (n_rollout_threads, H, W, C)
                        eval_rnn_states[:, agent_id],  # Shape: (n_rollout_threads, recurrent_N, hidden_size)
                        eval_masks[:, agent_id],  # Shape: (n_rollout_threads, 1)
                        eval_available_actions[:, agent_id],  # Shape: (n_rollout_threads, action_dim)
                        deterministic=deterministic
                    )
                    
                    eval_action = _t2n(eval_action)
                    eval_actions.append(eval_action)
                    eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                    
                    # Debug logging for first step to verify both agents are working
                    if step == 0:
                        logger.debug(f"Agent {agent_id} action shape: {eval_action.shape}, action value: {eval_action}")
                
                # Stack actions and transpose (same as post-hoc-eval.ipynb)
                # eval_actions: list of (n_rollout_threads, action_dim)
                # After stack: (num_agents, n_rollout_threads, action_dim)
                # After transpose: (n_rollout_threads, num_agents, action_dim)
                eval_actions = np.stack(eval_actions).transpose(1, 0, 2)
                
                # Debug logging for first step to verify both agents' actions are included
                if step == 0:
                    logger.info(f"Step {step}: eval_actions shape: {eval_actions.shape}, actions: {eval_actions[0]}")
                    logger.info(f"  Agent 0 action: {eval_actions[0, 0]}, Agent 1 action: {eval_actions[0, 1] if self.num_agents > 1 else 'N/A'}")
                
                # Environment step (VecEnv format)
                (
                    _eval_obs_batch_single_agent,
                    _,
                    eval_rewards,
                    eval_dones,
                    eval_infos,
                    eval_available_actions,
                ) = self.env.step(eval_actions)
                
                # Extract all_agent_obs from info_list (same as post-hoc-eval.ipynb)
                # info['all_agent_obs'] shape: (num_agents, H, W, C)
                # After np.array([...]): (n_rollout_threads, num_agents, H, W, C)
                eval_obs = np.array([info['all_agent_obs'] for info in eval_infos])
                
                # Verify shape is correct
                if eval_obs.shape[1] != self.num_agents:
                    logger.error(
                        f"eval_obs shape mismatch at step {step}! Expected (n_rollout_threads, {self.num_agents}, H, W, C), "
                        f"got {eval_obs.shape}"
                    )
                    # Try to fix if needed
                    if eval_obs.shape[1] == 1 and self.num_agents == 2 and len(eval_infos) > 0:
                        if 'all_agent_obs' in eval_infos[0]:
                            all_agent_obs = eval_infos[0]['all_agent_obs']
                            if len(all_agent_obs.shape) == 4 and all_agent_obs.shape[0] == self.num_agents:
                                eval_obs = np.expand_dims(all_agent_obs, axis=0)
                                logger.warning(f"  Fixed eval_obs.shape at step {step} to: {eval_obs.shape}")
                
                # Store for JSON export (use first rollout thread)
                obs_sequence.append(eval_obs[0].copy())
                actions_sequence.append(eval_actions[0].copy())  # (num_agents, action_dim)
                rewards_sequence.append(eval_rewards[0].copy())  # (num_agents,)
                
                # Update RNN states and masks for done episodes
                eval_rnn_states[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), self.args.recurrent_N, self.args.hidden_size),
                    dtype=np.float32,
                )
                eval_masks = np.ones((n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                
                # Insert into buffers (for each agent, use first rollout thread)
                for agent_id in range(self.num_agents):
                    # Get share_obs (use centralized if available, else use obs)
                    if hasattr(self.args, 'use_centralized_V') and self.args.use_centralized_V:
                        # Extract share_obs from info if available
                        share_obs_agent = np.array([info.get('share_obs', info['all_agent_obs'])[agent_id] for info in eval_infos])
                    else:
                        share_obs_agent = eval_obs[:, agent_id]
                    
                    # Get values using get_actions for buffer (needed for value estimation)
                    trainer = self.trainer[agent_id]
                    with torch.no_grad():
                        value, _, action_log_prob, rnn_state_new, rnn_state_critic_new = trainer.policy.get_actions(
                            share_obs_agent[0:1],  # (1, H, W, C_share)
                            eval_obs[0:1, agent_id],  # (1, H, W, C)
                            eval_rnn_states[0:1, agent_id],  # (1, recurrent_N, hidden_size)
                            eval_rnn_states[0:1, agent_id],  # Use same for critic (will be updated)
                            eval_masks[0:1, agent_id],  # (1, 1)
                            eval_available_actions[0:1, agent_id] if eval_available_actions is not None else None,
                            deterministic=deterministic
                        )
                    
                    value = _t2n(value)
                    action_log_prob = _t2n(action_log_prob)
                    rnn_state_new = _t2n(rnn_state_new)
                    rnn_state_critic_new = _t2n(rnn_state_critic_new)
                    
                    # Insert into buffer
                    buffers[agent_id].insert(
                        share_obs_agent[0:1],  # (1, H, W, C_share)
                        eval_obs[0:1, agent_id],  # (1, H, W, C)
                        rnn_state_new,  # (1, recurrent_N, hidden_size)
                        rnn_state_critic_new,  # (1, recurrent_N, hidden_size)
                        eval_actions[0:1, agent_id:agent_id+1],  # (1, action_dim)
                        action_log_prob,  # (1,)
                        value,  # (1,)
                        eval_rewards[0:1, agent_id:agent_id+1],  # (1,)
                        eval_masks[0:1, agent_id],  # (1, 1)
                        available_actions=eval_available_actions[0:1, agent_id:agent_id+1] if eval_available_actions is not None else None
                    )
                
                # Check if done
                if np.all(eval_dones):
                    logger.debug(f"Episode finished at step {step + 1}")
                    break
            
            # Compute returns for each agent's buffer
            # Use trainer's value_normalizer (same as train_sp.sh)
            for agent_id in range(self.num_agents):
                trainer = self.trainer[agent_id]
                
                # Get share_obs for final value estimation
                if hasattr(self.args, 'use_centralized_V') and self.args.use_centralized_V:
                    share_obs_agent = np.array([info.get('share_obs', info['all_agent_obs'])[agent_id] for info in eval_infos])
                else:
                    share_obs_agent = eval_obs[:, agent_id]
                
                with torch.no_grad():
                    next_value = trainer.policy.get_values(
                        share_obs_agent[0:1],  # (1, H, W, C_share)
                        eval_rnn_states[0:1, agent_id],  # (1, recurrent_N, hidden_size)
                        eval_masks[0:1, agent_id]  # (1, 1)
                    )
                    next_value = _t2n(next_value)
                
                # Use trainer's value_normalizer (same as Runner.compute())
                buffers[agent_id].compute_returns(next_value, trainer.value_normalizer)
            
            # Save buffers
            buffer_paths = []
            for agent_id in range(self.num_agents):
                buffer_path = self.buffer_dir / f"buffer_episode_{episode_id}_agent{agent_id}.pt"
                buffers[agent_id].save_to_file(str(buffer_path))
                buffer_paths.append(str(buffer_path))
            
            # Export JSON (for frontend visualization)
            obs_array = np.array(obs_sequence)
            actions_array = np.array(actions_sequence)
            rewards_array = np.array(rewards_sequence)
            
            # Use custom save_path if provided, otherwise use default naming
            custom_save_path = getattr(self, '_custom_json_path', None)
            if custom_save_path:
                json_save_path = custom_save_path
            else:
                json_save_path = None  # Use default naming
            
            json_path = self.exporter.export_from_raw_data(
                obs_sequence=obs_array[:-1, 0],  # Remove last obs, use agent 0
                actions_sequence=actions_array[:, 0],  # Agent 0 actions
                rewards_sequence=rewards_array[:, 0],  # Agent 0 rewards
                episode_id=episode_id,
                env_info={"layoutName": self.layout_name},
                save_path=json_save_path
            )
            
            logger.info(
                f"✓ Episode {episode_id} complete:\n"
                f"  - Buffer agents 0-{self.num_agents-1}: {self.buffer_dir.name}/\n"
                f"  - JSON: {Path(json_path).name if json_path else 'failed'}"
            )
            
            return {
                'buffer_paths': buffer_paths,
                'json_path': json_path
            }
            
        except Exception as e:
            logger.error(f"Failed to generate trajectory with buffer: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_and_export(
        self,
        episode_id: int,
        episode_length: int = 400,
        deterministic: bool = True,
        custom_save_path: Optional[str] = None
    ) -> Optional[str]:
        """Generate trajectory and export to JSON (legacy method).
        
        NOTE: This is kept for backwards compatibility.
        For offline HIL workflow, use generate_trajectory_and_save_buffer() instead.
        
        Args:
            episode_id: Episode ID for filename
            episode_length: Maximum episode length
            deterministic: Use deterministic actions
            custom_save_path: Custom save path (optional)
            
        Returns:
            Path to saved JSON file, or None if failed
        """
        result = self.generate_trajectory_and_save_buffer(
            episode_id=episode_id,
            episode_length=episode_length,
            deterministic=deterministic
        )
        
        if result is None:
            return None
        
        return result['json_path']
    
    def generate_multiple_trajectories(
        self,
        num_trajectories: int,
        start_episode_id: int = 0,
        episode_length: int = 400,
        deterministic: bool = True
    ) -> List[str]:
        """Generate multiple trajectories.
        
        Args:
            num_trajectories: Number of trajectories to generate
            start_episode_id: Starting episode ID
            episode_length: Episode length
            deterministic: Use deterministic actions
            
        Returns:
            List of saved JSON file paths
        """
        saved_paths = []
        
        logger.info(f"Generating {num_trajectories} trajectories...")
        
        for i in range(num_trajectories):
            episode_id = start_episode_id + i
            logger.info(f"Generating trajectory {i+1}/{num_trajectories} (episode {episode_id})")
            
            save_path = self.generate_and_export(
                episode_id=episode_id,
                episode_length=episode_length,
                deterministic=deterministic
            )
            
            if save_path:
                saved_paths.append(save_path)
        
        logger.info(f"✓ Generated {len(saved_paths)}/{num_trajectories} trajectories")
        
        return saved_paths


def load_checkpoints_and_generate_trajectory(
    env,
    trainer_list: List[Any],
    checkpoint_paths: List[str],
    episode_id: int,
    layout_name: str,
    trajectory_dir: str = "human_interface/data/trajectories_for_human",
    episode_length: int = 400,
    device: str = "cpu"
) -> Optional[str]:
    """Convenience function: Load checkpoints and generate trajectory.
    
    Args:
        env: Overcooked environment
        trainer_list: List of trainers
        checkpoint_paths: List of checkpoint paths for each agent
        episode_id: Episode ID
        layout_name: Layout name
        trajectory_dir: Directory to save trajectory
        episode_length: Episode length
        device: Device to load checkpoints
        
    Returns:
        Path to saved trajectory JSON
    
    Example:
        save_path = load_checkpoints_and_generate_trajectory(
            env=env,
            trainer_list=[trainer0, trainer1],
            checkpoint_paths=[
                "policy_pool/random3/fcp/agent0/checkpoint_5000000.pth",
                "policy_pool/random3/fcp/agent1/checkpoint_5000000.pth"
            ],
            episode_id=42,
            layout_name="random3"
        )
    """
    generator = TrajectoryGenerator(
        env=env,
        trainer_list=trainer_list,
        layout_name=layout_name,
        trajectory_dir=trajectory_dir
    )
    
    # Load checkpoints
    success = generator.load_agent_checkpoints(checkpoint_paths, device=device)
    if not success:
        logger.error("Failed to load checkpoints")
        return None
    
    # Generate and export
    save_path = generator.generate_and_export(
        episode_id=episode_id,
        episode_length=episode_length,
        deterministic=True
    )
    
    return save_path

