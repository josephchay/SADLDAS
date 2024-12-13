import torch
import torch.optim as optim
import copy

from src.losses import ReSMAE, ReaSMAE
from src.network import Actor, Critic
from src.network import Denoisify


class SADLDAS(object):
    """
    Key Features:
    - Adaptive actor and critic networks
    - Flexible exploration strategies
    - Variance-aware loss functions
    - Target network soft updates
    """

    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0, 
                 exploration_type='SDLD', exploration_kwargs=None):
        """
        Initialize the SADLDAS learning algorithm.
        
        Args:
            state_dim (int): Dimensionality of the state space
            action_dim (int): Dimensionality of the action space
            hidden_dim (int): Dimension of hidden layers in networks
            device (torch.device): Computational device (CPU/GPU)
            max_action (float, optional): Maximum magnitude of actions. Defaults to 1.0.
            exploration_type (str, optional): Type of exploration strategy. Defaults to 'GA'.
            exploration_kwargs (dict, optional): Additional exploration strategy parameters
        """

        # Initialize main actor network with exploration
        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action, 
                           exploration_type='GA', exploration_kwargs=exploration_kwargs,
                           ).to(device)

        # Initialize critic networks
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        # Link critic to actor for exploration feedback
        self.actor.set_critic(self.critic)

        # Set up optimizers with different learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=7e-4)
        
        # Apply denoising to the actor pipeline network
        self.actor = Denoisify(self.actor, action_dim, device)

        # Store key hyperparameters and learning state
        self.max_action = max_action
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy tracking variables for adaptive learning
        self.q_old_policy = 0.0
        self.s2_old_policy = 0.0
        self.tr_step = 0

    def select_action(self, state, replay_buffer=None, mean=False):
        """
        Select an action for the given state.
        
        Args:
            state (np.ndarray): Current environment state
            replay_buffer (optional): Replay buffer for state normalization
            mean (bool, optional): If True, returns deterministic action without exploration
        
        Returns:
            np.ndarray: Selected action
        """

        with torch.no_grad():
            state = torch.FloatTensor(state).reshape(-1, self.state_dim).to(self.device)
            if replay_buffer: 
                state = replay_buffer.normalize(state)
            action = self.actor(state, mean=mean)
        return action.cpu().data.numpy().flatten()

    def train(self, batch):
        """
        Perform a single training step with a batch of experiences.
        
        Args:
            batch (tuple): Batch of (state, action, reward, next_state, done) experiences
        
        Returns:
            dict: Training metrics and losses
        """

        # Increment training step counter
        self.tr_step += 1
        state, action, reward, next_state, done = batch

        # Update critic and get Q-values
        q_value, s2_value, critic_loss = self.critic_update(state, action, reward, next_state, done)
        
        # Update actor policy
        actor_loss = self.actor_update(state, next_state)

        # Get current learning rates from optimizers
        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        critic_lr = self.critic_optimizer.param_groups[0]['lr']

        # Return training metrics
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'q_value': q_value.mean().item(),
            's2_value': s2_value.mean().item(),
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'n_updates': self.tr_step,
        }

    def critic_update(self, state, action, reward, next_state, done):
        """
        Update the critic network using temporal difference learning.
        
        Implements a soft target network update and calculates loss using 
        a custom ReSMAE (Regularized Spectral Mean Absolute Error) approach.
        
        Args:
            state (torch.Tensor): Current states
            action (torch.Tensor): Current actions
            reward (torch.Tensor): Rewards
            next_state (torch.Tensor): Next states
            done (torch.Tensor): Episode termination flags
        
        Returns:
            tuple: Q-values, uncertainty estimates, and critic loss
        """

        with torch.no_grad():
            # Soft target network update
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(0.997 * target_param.data + 0.003 * param)

            # Compute next action using target actor (deterministic)
            next_action = self.actor(next_state, mean=True)
            
            # Get Q-value and uncertainty estimates from target critic
            q_next_target, s2_next_target = self.critic_target(next_state, next_action, united=True)
            
            # Compute target Q-values with discounting
            q_value = reward + (1 - done) * 0.99 * q_next_target
            
            # Compute variance estimate with reduced objective
            s2_value = 3e-3 * (3e-3 * torch.var(reward) + (1 - done) * 0.99 * s2_next_target)  # reduced objective to learn Bellman's sum of dumped variance
            
            # Get current policy's Q-values and uncertainty
            self.next_q_old_policy, self.next_s2_old_policy = self.critic(next_state, next_action, united=True)

        # Get individual Q-value estimates from critic
        out = self.critic(state, action, united=False)

        # Compute critic loss using ReSMAE on multiple Q-value estimates
        critic_loss = ReSMAE(q_value - out[0]) + ReSMAE(q_value - out[1]) + ReSMAE(q_value - out[2]) + ReSMAE(s2_value - out[3])

        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return q_value, s2_value, critic_loss

    def actor_update(self, state, next_state):
        """
        Update the actor network to maximize expected returns.
        
        Uses a policy gradient approach with custom ReaSMAE 
        (Regularized Approximate Spectral Mean Absolute Error) loss.
        
        Args:
            state (torch.Tensor): Current states
            next_state (torch.Tensor): Next states
        
        Returns:
            torch.Tensor: Actor loss
        """

        # Compute deterministic action
        action = self.actor(state, mean=True)
        
        # Get current policy's Q-values and uncertainty
        q_new_policy, s2_new_policy = self.critic(state, action, united=True)
        
        # Compute actor loss comparing current and previous policy
        actor_loss = -ReaSMAE(q_new_policy - self.q_old_policy) - ReaSMAE(s2_new_policy - self.s2_old_policy)

        # Add next state's policy contribution to loss
        next_action = self.actor(next_state, mean=True)
        next_q_new_policy, next_s2_new_policy = self.critic(next_state, next_action, united=True)
        actor_loss += -ReaSMAE(next_q_new_policy - self.next_q_old_policy.mean().detach()) - ReaSMAE(next_s2_new_policy - self.next_s2_old_policy.mean().detach())

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update policy tracking variables
        with torch.no_grad():
            self.q_old_policy = q_new_policy.mean().detach()
            self.s2_old_policy = s2_new_policy.mean().detach()

        return actor_loss
