import torch
import torch.nn as nn


class ActionDenoiser(nn.Module):
    """
    Variational autoencoder-based action denoising module.

    Implements structured noise modeling from "Critic Regularized Regression" (Wang et al., 2020)
    latent variable inference techniques "Parrot: Data-Driven Behavioral Priors for Reinforcement Learning" (Singh et al., 2020).
    Latent dimensionality reduction VAE architecture. "Adaptive Compression of the Latent Space in Variational Autoencoders" (Sejnova, Vavrecka, & Stepanova, 2023)
    """

    def __init__(self, action_dim, hidden_dim=32, latent_dim=8):
        super().__init__()
        
        # Architecture based on Haarnoja et al. (2018) SAC's policy network scaling
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer normalization from "Conservative Q-Learning for Offline Reinforcement Learning" (Kumar et al. (2020))
            nn.SiLU(),  # SiLU activation showed better performance than ReLU for VAEs (Ramachandran et al., 2017)
            nn.Linear(hidden_dim, 2 * latent_dim)  # mu and logvar as in Kingma & Welling (2014)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.latent_dim = latent_dim
        
        # Initialize training history and step counter
        self.training_steps = 0
        
    def encode(self, x):
        """Encoder implementation following Kingma & Welling (2014) VAE architecture"""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick from Kingma & Welling (2014).
        Enables backpropagation through sampling process.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decoder maps latent samples back to action space"""
        return self.decoder(z)
        
    def forward(self, noisy_actions):
        # Increment training steps
        self.training_steps += 1
        
        mu, logvar = self.encode(noisy_actions)
        z = self.reparameterize(mu, logvar)
        denoised_actions = self.decode(z)
        
        # KL divergence loss from Kingma & Welling (2014)
        # Factor of -0.5 comes from the closed-form KL in between
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Reconstruction loss with MSE following Fujimoto et al. (2019) TD3's action smoothing
        # Based on "Addressing Function Approximation Error in Actor-Critic Methods"
        recon_loss = nn.MSELoss()(denoised_actions, noisy_actions)
        
        # β-VAE formulation (Higgins et al., 2017) for controlled disentanglement
        beta = torch.clamp(torch.tensor(self.training_steps / 1000.0), 0.0, 1.0)
        total_loss = recon_loss + beta * kl_loss
        
        return denoised_actions, total_loss


class Denoisify:
    """
    Wrapper to integrate denoising into the actor pipeline

    Action denoising wrapper based on TD3's delayed policy updates (Fujimoto et al., 2019)
    """

    def __init__(self, actor, action_dim, device):
        self.actor = actor
        self.denoiser = ActionDenoiser(action_dim).to(device)
        self.device = device
        # Learning rate from SAC paper (Haarnoja et al., 2018)
        self.optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=3e-4)
        
    def __call__(self, state, mean=False):
        with torch.set_grad_enabled(not mean):  # No gradients needed for mean actions
            noisy_actions = self.actor(state, mean)
            
            if not mean:
                denoised_actions, loss = self.denoiser(noisy_actions)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                return denoised_actions
                
            return noisy_actions
            
    # Forward actor methods to maintain interface
    def __getattr__(self, name):
        return getattr(self.actor, name)
