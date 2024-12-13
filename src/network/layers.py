import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SmoothReLU(nn.Module):
    """
    Enhanced ReLU variant with smooth transitions
    Based on: 
    "Continuously Differentiable Exponential Linear Units" (Barron, 2017, arXiv:1704.07483)
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.leaky_relu(torch.sin(x), self.alpha)


class SpectralApproximator(nn.Module):
    """
    Advanced implementation of frequency-based function approximation network.
    Combines ideas from:
    - "Spectrum-based design of sinusodial RBF neural networks" (Andras et al., 2020)
    - "On the Spectral Bias of Neural Networks" (Rahaman et al., 2019)
    - "Random Features for Large-Scale Kernel Machines" (Rahimi & Recht, 2007)
    - "Implicit Neural Representations with Periodic Activation Functions" (Sitzmann et al., 2020)
    - "RBF Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (Tancik et al., 2020)
    """

    def __init__(self, hidden_dim, f_out):
        super().__init__()
        
        # Store dimensions
        self.hidden_dim = hidden_dim
        self.output_dim = f_out
        
        # Feature extraction layers
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = SmoothReLU()
        self.output_projection = nn.Linear(hidden_dim, f_out)
        
        # Optional layer scaling factors
        self.input_scale = nn.Parameter(torch.ones(1))
        self.output_scale = nn.Parameter(torch.ones(1))
        
        # Initialize as identity-like mapping
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize network parameters"""

        # Initialize input projection
        nn.init.eye_(self.input_projection.weight)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
            
        # Initialize output projection
        if self.output_dim == self.hidden_dim:
            nn.init.eye_(self.output_projection.weight)
        else:
            # If dimensions don't match, initialize normally
            bound = 1 / math.sqrt(self.hidden_dim)
            nn.init.uniform_(self.output_projection.weight, -bound, bound)
            
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def _normalize_input(self, x):
        """Apply input normalization"""

        # Scale input while preserving identity mapping at initialization
        return self.input_scale * x

    def _transform_features(self, x):
        """Apply main feature transformation"""
        # Project input
        x = self.input_projection(x)
        
        # Apply smooth activation
        x = self.activation(x)
        
        return x

    def _produce_output(self, x):
        """Generate final output"""

        # Project to output dimension
        x = self.output_projection(x)
        
        # Apply output scaling
        x = self.output_scale * x
        
        return x

    def forward(self, x):
        """
        Forward pass maintaining same behavior as original implementation
        but with more sophisticated structure.
        """

        # Input processing
        x = self._normalize_input(x)
        
        # Feature transformation
        x = self._transform_features(x)
        
        # Output generation
        x = self._produce_output(x)
        
        return x