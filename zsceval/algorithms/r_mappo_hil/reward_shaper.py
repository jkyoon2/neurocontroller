"""
Reward Shaper (M1): Modify rewards in rollout buffer based on human feedback.

This module implements reward shaping by applying penalties to timesteps
around human-identified error points, using a calibration window.
"""

from typing import Dict, List

import numpy as np
from loguru import logger


class RewardShaper:
    """Applies reward shaping based on human feedback.
    
    Method M1: Modifies rewards in the rollout buffer by applying penalties
    to timesteps within a calibration window of human-identified errors.
    """
    
    def __init__(
        self,
        penalty_magnitude: float = -1.0,
        calibration_window: int = 5,
        enable_stats: bool = True
    ):
        """Initialize RewardShaper.
        
        Args:
            penalty_magnitude: Magnitude of penalty to apply (negative value)
            calibration_window: Number of timesteps before/after error to apply penalty
            enable_stats: Whether to track statistics
        """
        self.penalty_magnitude = penalty_magnitude
        self.calibration_window = calibration_window
        self.enable_stats = enable_stats
        
        # Statistics
        self.stats = {
            "total_errors": 0,
            "total_timesteps_modified": 0,
            "total_penalty_applied": 0.0,
        }
        
        logger.info(
            f"RewardShaper initialized: "
            f"penalty={penalty_magnitude}, window={calibration_window}"
        )
    
    def apply_shaping(
        self,
        buffer,
        error_timesteps: List[int],
        env_id: int
    ) -> Dict[str, int]:
        """Apply reward shaping to buffer based on error timesteps.
        
        Args:
            buffer: HILRolloutBuffer instance
            error_timesteps: List of timestep indices where errors occurred
            env_id: Environment ID to apply shaping to
            
        Returns:
            Dictionary with shaping statistics
        """
        if not error_timesteps:
            return {"timesteps_modified": 0, "penalty_applied": 0.0}
        
        timesteps_modified = 0
        total_penalty = 0.0
        
        episode_length = buffer.episode_length
        
        for t_error in error_timesteps:
            # Define window
            start = max(0, t_error - self.calibration_window)
            end = min(episode_length, t_error + self.calibration_window + 1)
            
            # Apply penalty to all timesteps in window
            for t in range(start, end):
                if t < episode_length:
                    # Modify reward
                    buffer.rewards[t, env_id, 0] += self.penalty_magnitude
                    
                    timesteps_modified += 1
                    total_penalty += abs(self.penalty_magnitude)
        
        # Update statistics
        if self.enable_stats:
            self.stats["total_errors"] += len(error_timesteps)
            self.stats["total_timesteps_modified"] += timesteps_modified
            self.stats["total_penalty_applied"] += total_penalty
        
        logger.debug(
            f"Reward shaping applied: {len(error_timesteps)} errors, "
            f"{timesteps_modified} timesteps modified, "
            f"total penalty: {total_penalty:.2f}"
        )
        
        return {
            "timesteps_modified": timesteps_modified,
            "penalty_applied": total_penalty
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get reward shaping statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "total_errors": 0,
            "total_timesteps_modified": 0,
            "total_penalty_applied": 0.0,
        }

