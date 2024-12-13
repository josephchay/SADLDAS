import torch
import torch.nn as nn

from src.network import SpectralApproximator


class Critic(nn.Module):
    """
    Critic network for value estimation in reinforcement learning.
    
    Implements an ensemble-based Q-value estimation approach with multiple 
    spectral approximator networks and an additional network for uncertainty estimation.
    
    Attributes:
        input (nn.Linear): Input layer combining state and action
        nets (nn.ModuleList): List of spectral approximator networks for Q-value and uncertainty estimation
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32):
        """
        Initialize the Critic network.
        
        Args:
            state_dim (int): Dimension of the input state
            action_dim (int): Dimension of the input action
            hidden_dim (int, optional): Dimension of hidden layers. Defaults to 32.
        """

        super(Critic, self).__init__()

        # Input layer combines state and action
        self.input = nn.Linear(state_dim + action_dim, hidden_dim)

        # Create three Q-value networks (ensemble approach)
        # Different spectral approximator networks for robust value estimation
        qA = SpectralApproximator(hidden_dim, 1)
        qB = SpectralApproximator(hidden_dim, 1)
        qC = SpectralApproximator(hidden_dim, 1)

        # Additional network for uncertainty or variance estimation
        s2 = SpectralApproximator(hidden_dim, 1)

        # Store networks in a ModuleList for easy iteration and parameter tracking
        self.nets = nn.ModuleList([qA, qB, qC, s2])

    def forward(self, state: torch.Tensor, action: torch.Tensor, united: bool = False) -> torch.Tensor:
        """
        Forward pass of the Critic network.
        
        Args:
            state (torch.Tensor): Input state tensor
            action (torch.Tensor): Input action tensor
            united (bool, optional): If True, returns min Q-value and uncertainty. 
                                     If False, returns list of all Q-value estimates. 
                                     Defaults to False.
        
        Returns:
            torch.Tensor: Either a list of Q-value estimates or 
                          a tuple of (min Q-value, uncertainty estimate)
        """

        # Concatenate state and action along the last dimension
        x = torch.cat([state, action], -1)
        
        # Process combined state-action through input layer
        x = self.input(x)
        
        # Apply each spectral approximator network to the processed input
        xs = [net(x) for net in self.nets]
        if not united:
            # Return all Q-value estimates separately
            return xs
        
        # For united mode:
        # 1. Compute the minimum Q-value across the first three networks
        # 2. Use the fourth network as an uncertainty/variance estimate
        qmin = torch.min(torch.stack(xs[:3], dim=-1), dim=-1).values
        return qmin, xs[3]
