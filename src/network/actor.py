from typing import Type
import torch
import torch.nn as nn

from .layers import SpectralApproximator


def get_exploration_type(exploration_type: str = 'SDLD') -> Type:
    """
    Dynamically select and return an exploration strategy class based on the input type.
    
    Args:
        exploration_type (str): The type of exploration strategy to use.
        
    Returns:
        Type: A class representing the selected exploration strategy.
        
    Raises:
        ValueError: If an unrecognized exploration type is provided.
    """

    if exploration_type == 'SDLD':
        from exploration import SpectralDecompositionalLowDiscrepancyNoise
        return SpectralDecompositionalLowDiscrepancyNoise
    elif exploration_type == 'GA':
        from exploration import GANoise
        return GANoise
    elif exploration_type == 'OU':
        from exploration import OUNoise
        return OUNoise
    elif exploration_type == 'LDAS':
        from exploration import LDASPolicy
        return LDASPolicy
    elif exploration_type == 'Uniform':
        from exploration import UniformRandomPolicy
        return UniformRandomPolicy
    else:
        raise ValueError(f"Exploration type '{exploration_type}' not recognized")


class Actor(nn.Module):
    """
    Actor network for policy-based reinforcement learning with multiple exploration strategies.
    
    The network takes a state as input and outputs an action with added exploration noise.
    Supports different exploration strategies through a flexible design.
    
    Attributes:
        state_dim (int): Dimensionality of the input state
        action_dim (int): Dimensionality of the output action
        max_action (float): Maximum magnitude of the action
        exploration_type (str): Type of exploration strategy
        exploration (object): Instantiated exploration strategy
    """

    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        device: torch.device, 
        hidden_dim: int = 32, 
        max_action: float = 1.0, 
        exploration_type: str = 'SDLD', 
        exploration_kwargs: dict = None
    ):
        
        """
        Initialize the Actor network.
        
        Args:
            state_dim (int): Dimension of the input state
            action_dim (int): Dimension of the output action
            device (torch.device): Device to run the network on
            hidden_dim (int, optional): Dimension of hidden layers. Defaults to 32.
            max_action (float, optional): Maximum magnitude of action. Defaults to 1.0.
            exploration_type (str, optional): Type of exploration strategy. Defaults to 'SDLD'.
            exploration_kwargs (dict, optional): Additional arguments for exploration strategy.
        """

        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input = nn.Linear(state_dim, hidden_dim)

        self.net = nn.Sequential(
            SpectralApproximator(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = torch.mean(max_action).item()
        
        # Get exploration class and instantiate with user's parameters
        self.exploration_type = exploration_type
        exploration_class = get_exploration_type(exploration_type)
        exploration_kwargs = exploration_kwargs or {}
        self.exploration = exploration_class(**exploration_kwargs)
        self.exploration_type = exploration_type

        self.critic = None

        self.device = device

    def forward(self, state: torch.Tensor, mean: bool = False) -> torch.Tensor:
        """
        Forward pass of the actor network.
        
        Args:
            state (torch.Tensor): Input state tensor
            mean (bool, optional): If True, return action without exploration noise. Defaults to False.
        
        Returns:
            torch.Tensor: Action tensor with optional exploration noise
        
        Raises:
            ValueError: If critic is not set for certain exploration types
        """

        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = self.input(state)
        x = self.max_action * self.net(x)
        if mean:
            return x

        # Different calls based on exploration type
        match self.exploration_type:
            case 'SDLD':
                if self.critic is None:
                    raise ValueError("Critic network not set for SpectralQEnsembleDiscrepancyNoise. Use set_critic() method.")
                noise, _ = self.exploration.generate(self.critic, self, state)
            case 'GA':
                noise = self.exploration.generate(x)
            case 'OU':
                noise = self.exploration.generate(x)
            case 'LDAS':
                noise = self.exploration.generate(state)
            case 'Uniform':
                noise = self.exploration.select_action()
            case _:
                raise ValueError(f"Unknown exploration type: {self.exploration_type}")

        x += noise
        return x.clamp(-self.max_action, self.max_action)
    
    def set_critic(self, critic):
        """Set critic network for exploration noise generation."""

        self.critic = critic
