"""
TrajectoryExporter: Export trajectory data to JSON for frontend visualization.

This module handles:
1. Converting rollout buffer data to frontend-compatible JSON
2. Parsing observations into Overcooked game state format
3. Saving trajectory JSON files
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from zsceval.utils.feedback.feedback_schema import TrajectoryData, validate_trajectory_schema
from zsceval.utils.feedback.layout_parser import LayoutParser
from zsceval.utils.feedback.obs_decoder import ObservationDecoder


class TrajectoryExporter:
    """Exporter for trajectory data to frontend JSON format.
    
    This class is used by Runner (typically env_id=0) to export
    trajectories to the human_interface/data/trajectories_for_human/ directory.
    """
    
    def __init__(self, trajectory_dir: Optional[str] = None, layout_name: str = "cramped_room"):
        """Initialize TrajectoryExporter.
        
        Args:
            trajectory_dir: Directory to save trajectory JSON files.
                          If None, uses default 'human_interface/data/trajectories_for_human/'
            layout_name: Overcooked layout name for static info
        """
        if trajectory_dir is None:
            self.trajectory_dir = Path("human_interface/data/trajectories_for_human")
        else:
            self.trajectory_dir = Path(trajectory_dir)
        
        # Create directory if it doesn't exist
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        self.layout_name = layout_name
        
        # Initialize layout parser and observation decoder
        self.layout_parser = LayoutParser()
        self.obs_decoder = ObservationDecoder(num_agents=2)
        
        logger.info(f"TrajectoryExporter initialized with directory: {self.trajectory_dir}")
    
    def export_from_buffer(
        self,
        buffer: Any,
        env_id: int,
        episode_id: int,
        env_info: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Export trajectory from rollout buffer.
        
        Args:
            buffer: SeparatedReplayBuffer or HILRolloutBuffer instance
            env_id: Environment ID to export (typically 0)
            episode_id: Episode number for filename
            env_info: Optional environment info for static data
            save_path: Optional custom save path (overrides default naming)
            
        Returns:
            Path to saved JSON file, or None if export failed
        """
        try:
            # Extract static info (from env or default)
            static_info = self._build_static_info(env_info)
            
            # Extract dynamic states from buffer
            dynamic_states = self._build_dynamic_states(buffer, env_id)
            
            # Create trajectory data
            trajectory = TrajectoryData(
                static_info=static_info,
                dynamic_states=dynamic_states
            )
            
            # Convert to dict
            trajectory_dict = trajectory.to_dict()
            
            # Validate schema
            validate_trajectory_schema(trajectory_dict)
            
            # Determine save path
            if save_path is None:
                save_path = self.trajectory_dir / f"rollout_{episode_id}.json"
            else:
                save_path = Path(save_path)
            
            # Save to file
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(trajectory_dict, f, indent=2)
            
            logger.info(f"[TrajectoryExporter] Exported trajectory to {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"[TrajectoryExporter] Failed to export trajectory: {e}")
            return None
    
    def _build_static_info(self, env_info: Optional[Dict]) -> Dict[str, Any]:
        """Build static info dictionary by parsing layout file.
        
        Args:
            env_info: Environment info dictionary (optional, can override layout_name)
            
        Returns:
            Static info dictionary parsed from layout file
        """
        # Get layout name from env_info or use default
        layout_name = self.layout_name
        if env_info is not None and "layoutName" in env_info:
            layout_name = env_info["layoutName"]
        
        # Parse layout file to get static info
        static_info = self.layout_parser.parse_layout(layout_name)
        
        # Override with additional env_info if provided
        if env_info is not None:
            static_info.update(env_info)
        
        return static_info
    
    def _build_dynamic_states(self, buffer: Any, env_id: int) -> List[Dict[str, Any]]:
        """Build dynamic states list from buffer data.
        
        Args:
            buffer: Rollout buffer instance
            env_id: Environment ID
            
        Returns:
            List of dynamic state dictionaries
        """
        dynamic_states = []
        
        # Get episode length from buffer
        episode_length = buffer.episode_length
        
        previous_actions = None
        
        for timestep in range(episode_length):
            # Extract data for this timestep
            obs = buffer.obs[timestep, env_id]  # shape: (obs_dim,)
            actions = buffer.actions[timestep, env_id] if hasattr(buffer, 'actions') else None
            rewards = buffer.rewards[timestep, env_id] if hasattr(buffer, 'rewards') else None
            
            # Parse observation to game state
            state = self._parse_obs_to_state(obs, timestep, actions, rewards, previous_actions)
            
            dynamic_states.append(state)
            
            # Update previous actions for next iteration
            previous_actions = actions
        
        return dynamic_states
    
    def _parse_obs_to_state(
        self, 
        obs: np.ndarray, 
        timestep: int,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        previous_actions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Parse observation array to game state dictionary.
        
        Uses ObservationDecoder to extract player positions, objects, etc.
        from the observation tensor.
        
        Args:
            obs: Observation array, shape (H, W, C)
            timestep: Current timestep
            actions: Action taken at this timestep (optional)
            rewards: Reward received at this timestep (optional)
            previous_actions: Action taken at previous timestep (optional)
            
        Returns:
            Dynamic state dictionary
        """
        # Use observation decoder
        state = self.obs_decoder.decode(
            obs=obs,
            timestep=timestep,
            actions=actions,
            rewards=rewards,
            previous_actions=previous_actions
        )
        
        return state
    
    def export_from_raw_data(
        self,
        obs_sequence: np.ndarray,
        actions_sequence: Optional[np.ndarray],
        rewards_sequence: Optional[np.ndarray],
        episode_id: int,
        env_info: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """Export trajectory from raw numpy arrays.
        
        This is useful when you have collected data but not in buffer format.
        
        Args:
            obs_sequence: Observation sequence, shape (T, obs_dim)
            actions_sequence: Action sequence, shape (T, action_dim) or None
            rewards_sequence: Reward sequence, shape (T, 1) or None
            episode_id: Episode number for filename
            env_info: Optional environment info for static data
            save_path: Optional custom save path
            
        Returns:
            Path to saved JSON file, or None if export failed
        """
        try:
            # Build static info
            static_info = self._build_static_info(env_info)
            
            # Build dynamic states
            dynamic_states = []
            T = obs_sequence.shape[0]
            
            previous_actions = None
            
            for t in range(T):
                obs = obs_sequence[t]
                actions = actions_sequence[t] if actions_sequence is not None else None
                rewards = rewards_sequence[t] if rewards_sequence is not None else None
                
                state = self._parse_obs_to_state(obs, t, actions, rewards, previous_actions)
                dynamic_states.append(state)
                
                # Update previous actions
                previous_actions = actions
            
            # Create trajectory
            trajectory = TrajectoryData(
                static_info=static_info,
                dynamic_states=dynamic_states
            )
            
            trajectory_dict = trajectory.to_dict()
            validate_trajectory_schema(trajectory_dict)
            
            # Save
            if save_path is None:
                save_path = self.trajectory_dir / f"rollout_{episode_id}.json"
            else:
                save_path = Path(save_path)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(trajectory_dict, f, indent=2)
            
            logger.info(f"[TrajectoryExporter] Exported raw data to {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"[TrajectoryExporter] Failed to export raw data: {e}")
            return None

