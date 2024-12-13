import torch
import numpy as np
from typing import Union


class UniformRandomPolicy:
    """
    Uniform random action selection for continuous action spaces.
    Useful as a baseline exploration strategy.
    """

    def __init__(self,
                 action_dim: int,
                 action_low: Union[float, np.ndarray],
                 action_high: Union[float, np.ndarray],
                 device: torch.device):
        self.action_dim = action_dim
        self.device = device
        
        # Convert bounds to tensors
        if isinstance(action_low, (int, float)):
            self.action_low = torch.full(
                (action_dim,), 
                action_low, 
                device=device
            )
        else:
            self.action_low = torch.tensor(
                action_low,
                dtype=torch.float32,
                device=device
            )
            
        if isinstance(action_high, (int, float)):
            self.action_high = torch.full(
                (action_dim,),
                action_high,
                device=device
            )
        else:
            self.action_high = torch.tensor(
                action_high,
                dtype=torch.float32,
                device=device
            )
            
    def select_action(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate uniform random actions within the specified bounds.
        Returns actions of shape (batch_size, action_dim).
        """
        random_actions = torch.rand(
            (batch_size, self.action_dim),
            device=self.device
        )
        # Scale from [0,1] to [low,high]
        scaled_actions = (
            self.action_low + 
            (self.action_high - self.action_low) * random_actions
        )
        return scaled_actions
        
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """Alias for select_action with batch_size=1."""
        return self.select_action(batch_size=1).squeeze(0)
