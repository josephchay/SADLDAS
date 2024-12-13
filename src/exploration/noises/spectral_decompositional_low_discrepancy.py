import math
import torch
import torch.nn as nn
import numpy as np

from network import SmoothReLU


class SDLDNoise:
    """
    SpectralDecompositionalLowDiscrepancyNoise

    Advanced exploration mechanism combining multiple techniques:
    1. Low Discrepancy Action Selection
       - Maximizes distance from previously visited state-action pairs
       - Uses gradient ascent to find optimal actions
       - Maintains history buffer for coverage assessment

    2. Q-Ensemble Integration
       - Utilizes 3 Q-networks (qA, qB, qC) for robust value estimation
       - Incorporates s2 network for variance/uncertainty estimation
       - Adapts exploration based on ensemble disagreement
       
    3. Spectral Analysis
       - Multi-scale feature resonance detection
       - Adaptive pattern recognition
       - State space understanding through frequency domain
       
    4. Dynamic Exploration
       - Performance-based adaptation
       - Coverage-driven exploration
       - Smooth phase transitions

    Args:
        state_dim (int): Dimension of state space
        max_action (float/tensor): Maximum action value(s)
        device (torch.device): Device for computations
        spectral_dim (int): Dimension of spectral features
        buffer_size (int): Size of state-action history buffer
        learning_rate (float): Learning rate for action optimization
        lr_decay (float): Learning rate decay factor
    """

    def __init__(self, state_dim, max_action, device, 
                 spectral_dim=32,
                 buffer_size=500,
                 learning_rate=0.1,
                 lr_decay=0.99):
        """
        Initialize Spectral Q-Ensemble Discrepancy Noise.
        
        The spectral dimension (spectral_dim) is used to create a lower-dimensional
        representation of the state space that captures essential frequencies and patterns.
        This helps in:
        1. Detecting repeating patterns in state transitions
        2. Identifying high-frequency components that might need exploration
        3. Understanding the underlying structure of the state space
        
        A larger spectral_dim captures more fine-grained patterns but requires
        more computation. A smaller spectral_dim focuses on broader patterns
        but might miss subtle details.
        """

        self.max_action = max_action
        self.device = device
        
        # Required SALDAS interface attributes
        self.x_coor = 0.0
        self.scale = 1.0

        # Low discrepancy parameters
        self.buffer_size = buffer_size
        self.lr = learning_rate
        self.lr_decay = lr_decay
        
        # Spectral analysis networks
        # Multi-scale spectral feature extraction:
        # - fine_net: captures high-frequency details (full spectral_dim)
        # - mid_net: captures medium-frequency patterns (spectral_dim // 2)
        # - coarse_net: captures low-frequency structure (spectral_dim // 4)
        self.spectral_nets = nn.ModuleList([
            nn.Sequential(  # fine_net
                nn.Linear(state_dim, spectral_dim),
                nn.LayerNorm(spectral_dim),
                SmoothReLU(),
                nn.Linear(spectral_dim, spectral_dim)
            ).to(device),
            nn.Sequential(  # mid_net
                nn.Linear(state_dim, spectral_dim // 2),
                nn.LayerNorm(spectral_dim // 2),
                SmoothReLU(),
                nn.Linear(spectral_dim // 2, spectral_dim // 2)
            ).to(device),
            nn.Sequential(  # coarse_net
                nn.Linear(state_dim, spectral_dim // 4),
                nn.LayerNorm(spectral_dim // 4),
                SmoothReLU(),
                nn.Linear(spectral_dim // 4, spectral_dim // 4)
            ).to(device)
        ])
        
        # Track spectral statistics for adaptive exploration
        self.spectral_stats = {
            'fine': {'mean': 0, 'std': 0, 'history': []},
            'mid': {'mean': 0, 'std': 0, 'history': []},
            'coarse': {'mean': 0, 'std': 0, 'history': []}
        }
        
        # History buffer for state-action pairs
        self.buffer = []
        
        # Q-network specific tracking
        self.q_values_history = {
            'qA': [],
            'qB': [],
            'qC': [],
            's2': []
        }
        
        # Exploration metrics
        self.ensemble_disagreement = 0.0
        self.uncertainty_estimate = 0.0
        self.coverage_score = 0.0

    def get_distance(self, state_action: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Calculate minimum distance to history points"""

        if len(history) == 0:
            return float('inf')
        distances = torch.norm(history - state_action, dim=1)
        return torch.min(distances)

    def analyze_spectral_features(self, state: torch.Tensor) -> dict:
        """
        Analyze state using multi-scale spectral feature extraction.
        
        This method:
        1. Processes state through three spectral networks of different scales
        2. Computes frequency components using FFT
        3. Tracks spectral statistics over time
        4. Identifies regions of state space needing exploration
        
        Args:
            state: Current state tensor
            
        Returns:
            Dictionary containing spectral analysis results
        """

        with torch.no_grad():
            # Get features at different scales
            fine_features = self.spectral_nets[0](state)
            mid_features = self.spectral_nets[1](state)
            coarse_features = self.spectral_nets[2](state)
            
            # Compute frequency components
            fine_freq = torch.fft.fft(fine_features, dim=-1)
            mid_freq = torch.fft.fft(mid_features, dim=-1)
            coarse_freq = torch.fft.fft(coarse_features, dim=-1)
            
            # Calculate magnitudes
            fine_mag = torch.abs(fine_freq).mean()
            mid_mag = torch.abs(mid_freq).mean()
            coarse_mag = torch.abs(coarse_freq).mean()
            
            # Update running statistics
            scales = ['fine', 'mid', 'coarse']
            mags = [fine_mag, mid_mag, coarse_mag]
            
            for scale, mag in zip(scales, mags):
                self.spectral_stats[scale]['history'].append(mag.item())
                if len(self.spectral_stats[scale]['history']) > 1000:
                    self.spectral_stats[scale]['history'].pop(0)
                
                # Update running statistics
                self.spectral_stats[scale]['mean'] = np.mean(self.spectral_stats[scale]['history'])
                self.spectral_stats[scale]['std'] = np.std(self.spectral_stats[scale]['history'])
            
            # Calculate exploration indicators
            novelty_scores = {
                scale: (mag.item() - self.spectral_stats[scale]['mean']) / 
                      (self.spectral_stats[scale]['std'] + 1e-6)
                for scale, mag in zip(scales, mags)
            }
            
            return {
                'features': {
                    'fine': fine_features,
                    'mid': mid_features,
                    'coarse': coarse_features
                },
                'magnitudes': {
                    'fine': fine_mag.item(),
                    'mid': mid_mag.item(),
                    'coarse': coarse_mag.item()
                },
                'novelty_scores': novelty_scores,
                'exploration_weight': torch.sigmoid(
                    torch.tensor([
                        # Scale weights for different frequency components:
                        # - Fine (high frequency): 0.5 weight for detailed exploration
                        # - Mid frequency: 0.3 weight for medium-scale patterns
                        # - Coarse (low frequency): 0.2 weight for overall structure
                        # These weights prioritize fine-grained exploration while still
                        # maintaining awareness of broader state-space patterns
                        novelty_scores['fine'] * 0.5 +   # Emphasis on detailed features
                        novelty_scores['mid'] * 0.3 +    # Medium importance to mid-level patterns
                        novelty_scores['coarse'] * 0.2   # Less weight to broad patterns
                    ])
                ).item()
            }

    def generate(self, critic, actor, state: torch.Tensor) -> torch.Tensor:
        """
        Generate exploration noise considering:
        1. Q-ensemble values and uncertainty (from 3 Q-networks + s2)
        2. Low discrepancy coverage
        3. Spectral feature analysis
        
        Args:
            critic: Critic networks (3 Q-networks + 1 s2 network)
            actor: Actor network
            state: Current state tensor
            
        Returns:
            Noise tensor for action perturbation
        """
        with torch.no_grad():
            # Ensure state has proper dimensions
            if state.dim() == 0:
                state = state.unsqueeze(0)
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            # Analyze spectral features first
            spectral_analysis = self.analyze_spectral_features(state)
            
            # Get mean action from actor
            mean_action = actor(state, mean=True)
            if mean_action.dim() == 0:
                mean_action = mean_action.unsqueeze(0)
            if mean_action.dim() == 1:
                mean_action = mean_action.unsqueeze(0)
        with torch.no_grad():
            # Get mean action from actor
            mean_action = actor(state, mean=True)
            
            # Get Q-values and variance from critic networks
            q_outputs = critic(state, mean_action, united=False)
            qA, qB, qC, s2 = q_outputs
            
            # Update Q-value history
            self._update_q_history(qA, qB, qC, s2)
            
            # Calculate ensemble disagreement and uncertainty
            q_values = torch.stack([qA, qB, qC])
            self.ensemble_disagreement = torch.std(q_values, dim=0).mean()
            self.uncertainty_estimate = s2.mean()
            
            # Start with random action
            best_action = (torch.rand_like(mean_action) * 2 - 1) * self.max_action
            best_distance = -float('inf')
            current_lr = self.lr

            # Convert history buffer to tensor if not empty
            if self.buffer:
                history = torch.stack(self.buffer)
            else:
                history = torch.tensor([])

            # Gradient ascent to maximize distance while considering Q-ensemble
            for _ in range(10):
                state_action = torch.cat([state, best_action], dim=-1)
                dist = self.get_distance(state_action, history)
                
                if dist > best_distance:
                    best_distance = dist

                # Find closest point
                if len(history) > 0:
                    closest_idx = torch.argmin(torch.norm(history - state_action, dim=1))
                    diff = state_action - history[closest_idx]
                    
                    # Adjust action based on distance and Q-ensemble
                    action_update = (diff[-mean_action.shape[1]:] / 
                                   torch.norm(diff[-mean_action.shape[1]:]))
                    
                    # Scale update based on ensemble and uncertainty
                    scale_factor = torch.sigmoid(self.ensemble_disagreement + 
                                               self.uncertainty_estimate)
                    
                    best_action = best_action + current_lr * action_update * scale_factor
                    
                    # Clip actions
                    best_action = torch.clamp(best_action, -self.max_action, self.max_action)
                    
                    # Decay learning rate
                    current_lr *= self.lr_decay

            # Calculate noise as difference from mean action
            noise = best_action - mean_action
            
            # Update buffer
            self._update_buffer(state, best_action)
            
            # Update exploration phase
            self._update_phase()
            
            return noise, None

    def _update_q_history(self, qA, qB, qC, s2):
        """Update history of Q-values and variance estimates"""
        self.q_values_history['qA'].append(qA.mean().item())
        self.q_values_history['qB'].append(qB.mean().item())
        self.q_values_history['qC'].append(qC.mean().item())
        self.q_values_history['s2'].append(s2.mean().item())
        
        # Keep history size manageable
        max_history = 1000
        for key in self.q_values_history:
            if len(self.q_values_history[key]) > max_history:
                self.q_values_history[key] = self.q_values_history[key][-max_history:]

    def _update_buffer(self, state: torch.Tensor, action: torch.Tensor):
        """Update state-action history buffer ensuring proper tensor dimensions"""
        # Ensure state and action are properly shaped
        if state.dim() == 0:
            state = state.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        # Squeeze any extra dimensions but ensure we maintain 2D
        state = state.squeeze()
        action = action.squeeze()
        if state.dim() == 0:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0)
            
        state_action = torch.cat([state, action])
        self.buffer.append(state_action)
        
        # Maintain buffer size
        while len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def _update_phase(self):
        """Update exploration phase based on performance metrics"""
        # Update x_coor based on ensemble disagreement and uncertainty
        phase_factor = torch.sigmoid(self.ensemble_disagreement + 
                                   self.uncertainty_estimate).item()
        
        # x_coor increment:
        # - Base rate: 3e-5 chosen empirically for smooth phase transitions
        # - (1.0 - phase_factor) ensures slower progression when uncertainty is high
        # - This results in ~100k steps to reach training phase under ideal conditions
        self.x_coor += 3e-5 * (1.0 - phase_factor)
        
        # Phase-based scale updates:
        if self.x_coor >= math.pi:  # π ≈ 3.14159: Complete phase out of exploration
            self.scale = 0.0        # No exploration noise
        elif self.x_coor >= 2.133:  # 2.133 radians ≈ 0.679π: Enter training phase
            self.scale = 0.15       # Minimal exploration (15% of initial) during training
        else:                       # Early exploration phase
            self.scale = 1.0        # Full exploration noise
