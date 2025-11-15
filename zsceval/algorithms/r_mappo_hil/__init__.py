"""
Human-in-the-Loop (HIL) R-MAPPO Algorithm

This package implements HIL extensions to R-MAPPO for incorporating
human feedback into the training loop.

Modules:
    - hil_trainer: Main HIL training class
    - reward_shaper: M1 - Reward shaping from human feedback
    - policy_constraint: M2 - Policy constraints at error points
    - state_augmentor: M3 - State representation augmentation via VAE
"""

from .hil_trainer import HIL_RMAPPO

__all__ = ["HIL_RMAPPO"]

