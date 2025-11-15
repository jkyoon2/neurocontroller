"""
Policy Constraint (M2): Add constraints to PPO objective at error points.

This module implements policy constraints by adding a penalty term to the
PPO loss function that discourages actions at human-identified error points.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from loguru import logger


class PolicyConstraint:
    """Applies policy constraints based on human feedback.
    
    Method M2: Adds a constraint term to the PPO objective that penalizes
    the probability of actions taken at error timesteps.
    """
    
    def __init__(
        self,
        constraint_coef: float = 0.1,
        constraint_type: str = "kl",  # "kl", "l2", or "entropy"
        enable_stats: bool = True
    ):
        """Initialize PolicyConstraint.
        
        Args:
            constraint_coef: Coefficient (λ) for constraint loss term
            constraint_type: Type of constraint to apply
            enable_stats: Whether to track statistics
        """
        self.constraint_coef = constraint_coef
        self.constraint_type = constraint_type
        self.enable_stats = enable_stats
        
        # Statistics
        self.stats = {
            "total_constraints_applied": 0,
            "avg_constraint_loss": 0.0,
        }
        
        logger.info(
            f"PolicyConstraint initialized: "
            f"coef={constraint_coef}, type={constraint_type}"
        )
    
    def compute_constraint_loss(
        self,
        action_log_probs: torch.Tensor,
        actions_batch: torch.Tensor,
        error_mask: torch.Tensor,
        dist: Optional[torch.distributions.Distribution] = None
    ) -> torch.Tensor:
        """Compute constraint loss for error actions.
        
        Args:
            action_log_probs: Log probabilities of actions, shape (batch_size, 1)
            actions_batch: Actions taken, shape (batch_size, 1)
            error_mask: Binary mask indicating error timesteps, shape (batch_size, 1)
            dist: Action distribution (for KL-based constraints)
            
        Returns:
            Constraint loss tensor
        """
        if error_mask.sum() == 0:
            # No errors in this batch
            return torch.tensor(0.0, device=action_log_probs.device)
        
        if self.constraint_type == "kl":
            # KL divergence from uniform distribution
            # Encourages policy to be more uncertain at error points
            if dist is not None:
                # Uniform distribution over actions
                num_actions = dist.probs.shape[-1]
                uniform_probs = torch.ones_like(dist.probs) / num_actions
                
                # KL(policy || uniform)
                kl_div = F.kl_div(
                    dist.logits.log_softmax(dim=-1),
                    uniform_probs,
                    reduction='none'
                ).sum(dim=-1, keepdim=True)
                
                # Apply only to error timesteps
                constraint_loss = (kl_div * error_mask).sum() / error_mask.sum()
            else:
                # Fallback: use negative log prob (penalize confident actions)
                constraint_loss = -(action_log_probs * error_mask).sum() / error_mask.sum()
        
        elif self.constraint_type == "l2":
            # L2 penalty on action log probabilities at error points
            # Penalizes confident (high probability) actions
            constraint_loss = (action_log_probs.pow(2) * error_mask).sum() / error_mask.sum()
        
        elif self.constraint_type == "entropy":
            # Entropy regularization at error points
            # Encourages exploration at error timesteps
            if dist is not None:
                entropy = dist.entropy().unsqueeze(-1)
                # Negative entropy (we want to maximize, so minimize negative)
                constraint_loss = -(entropy * error_mask).sum() / error_mask.sum()
            else:
                # Fallback
                constraint_loss = -(action_log_probs * error_mask).sum() / error_mask.sum()
        
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")
        
        # Update statistics
        if self.enable_stats:
            self.stats["total_constraints_applied"] += int(error_mask.sum().item())
            self.stats["avg_constraint_loss"] += constraint_loss.item()
        
        return constraint_loss * self.constraint_coef
    
    def get_stats(self) -> Dict[str, float]:
        """Get policy constraint statistics."""
        stats = self.stats.copy()
        if stats["total_constraints_applied"] > 0:
            stats["avg_constraint_loss"] /= stats["total_constraints_applied"]
        return stats
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "total_constraints_applied": 0,
            "avg_constraint_loss": 0.0,
        }

