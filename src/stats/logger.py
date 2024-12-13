import math
from pathlib import Path
import time
import json

import numpy as np

from src.utils import convert_tensor
from src.constants import STATS_ROLLING_WINDOW


class Logger:
    """Handles logging, plotting, and video recording for training"""

    def __init__(self, save_dir: Path, env_name: str, model_version: int):
        self.save_dir = save_dir
        self.env_name = env_name
        self.model_version = model_version

        self.start_time = time.time()

        # Initialize metrics
        self.metrics = {
            # Basic episode and environment metrics
            'rollout': {
                'episode_lengths': [],      # Length of each episode
                'episode_rewards': [],      # Total reward per episode
                'success_rate': [],         # Episode success rate if applicable
                'time': {
                    'episodes': 0,
                    'total_timesteps': 0,
                    'fps_history': [],
                    'start_time': self.start_time,
                }
            },

            # Exploration strategy metrics
            'exploration': {
                'type': 'unknown',         # Will be set to 'ou', 'uniform', or 'ldas'
                'theta': 0.0,              # For OU noise
                'sigma': 0.0,              # For OU noise
                'action_repeats': 0,       # For uniform/LDAS
                'epsilon': 0.0,            # For uniform/LDAS
                'epsilon_decay': 1.0,      # For uniform/LDAS
                'current_epsilon': 0.0,    # Current exploration rate after decay
            },

            # Action-related metrics (not exploration specific)
            'action': {
                # General action statistics
                'distribution': {
                    'mean': [],            # Mean action values
                    'std': [],             # Action standard deviation
                    'min': [],             # Minimum action values
                    'max': [],             # Maximum action values
                    'entropy': [],         # Action distribution entropy
                },
                # Action space utilization
                'space_coverage': {
                    'percentage': [],      # How much of action space is used
                    'density': [],         # Distribution of actions in space
                    'bounds': {
                        'min': [],         # Minimum action bounds
                        'max': [],         # Maximum action bounds
                    }
                }
            },

            # Phase-specific noise and exploration metrics
            'phase': {
                'noise': {
                    'scale': 0.0,          # Current noise magnitude
                    'x_coor': 0.0,         # Noise decay parameter
                    'mode': 'explore',     # Current mode (explore/train/none)
                },
                'exploration': {
                    'ratio': [],           # Exploration vs exploitation ratio
                    'random_actions': [],  # Count of random actions
                    'policy_actions': [],  # Count of policy-based actions
                }
            },

            # Training performance metrics
            'train': {
                # Actor metrics
                'actor': {
                    'losses': [],          # Actor loss history
                    'learning_rate': 3e-4, # Current learning rate
                    'gradients': {         # Gradient statistics
                        'mean': [],
                        'std': [],
                        'norm': [],
                    }
                },
                # Critic metrics
                'critic': {
                    'losses': [],          # Critic loss history
                    'learning_rate': 7e-4, # Current learning rate
                    'gradients': {         # Gradient statistics
                        'mean': [],
                        'std': [],
                        'norm': [],
                    },
                    # Q-value statistics per network
                    'q_values': {
                        'qA': [],          # First Q-network values
                        'qB': [],          # Second Q-network values
                        'qC': [],          # Third Q-network values
                        'min': [],         # Minimum Q across networks
                        'max': [],         # Maximum Q across networks
                        'mean': [],        # Mean Q-value
                        'std': [],         # Q-value standard deviation
                    },
                    's2_values': [],       # Variance predictions
                },
                # Overall training progress
                'progress': {
                    'n_updates': 0,        # Number of update steps
                    'td_errors': [],       # TD error history
                    'batch_stats': {       # Training batch statistics
                        'reward_mean': [],
                        'reward_std': [],
                        'value_mean': [],
                        'value_std': [],
                    }
                }
            },

            # Replay buffer metrics
            'buffer': {
                # Basic buffer statistics
                'capacity': {
                    'size': 0,             # Current buffer size
                    'max_size': 0,         # Maximum buffer size
                    'utilization': 0.0,    # Current utilization ratio
                },
                # Sample statistics
                'samples': {
                    'age': {               # Age of sampled transitions
                        'mean': [],
                        'std': [],
                        'distribution': [],
                    },
                    'weights': {           # Sample importance weights
                        'mean': [],
                        'std': [],
                        'min': [],
                        'max': [],
                    }
                },
                # State normalization statistics
                'normalization': {
                    'states': {
                        'min': None,       # State minimum values
                        'max': None,       # State maximum values
                        'mean': None,      # State mean values
                        'std': None,       # State standard deviations
                    },
                    'rewards': {
                        'min': [],         # Reward minimum values
                        'max': [],         # Reward maximum values
                        'mean': [],        # Reward mean values
                        'std': [],         # Reward standard deviations
                    }
                },
                # Buffer parameters
                'params': {
                    'fade_factor': 0.0,    # Current fade factor
                    'stall_penalty': 0.0,  # Current stall penalty
                }
            },
        }

    def log_training_step(self, episode: int, reward: float, length: int, training_info: dict):
        """Log comprehensive training metrics"""

        # Update rollout metrics
        self.metrics['rollout']['episode_lengths'].append(length)
        self.metrics['rollout']['episode_rewards'].append(float(reward))

        # Update time metrics
        self.metrics['rollout']['time']['episodes'] = episode
        self.metrics['rollout']['time']['total_timesteps'] += length

        # Calculate FPS
        time_elapsed = time.time() - self.metrics['rollout']['time']['start_time']
        fps = self.metrics['rollout']['time']['total_timesteps'] / max(time_elapsed, 1e-6)
        self.metrics['rollout']['time']['fps_history'].append(fps)

        # Update training metrics
        if training_info:
            if 'actor_loss' in training_info:
                self.metrics['train']['actor']['losses'].append(float(training_info['actor_loss']))
            if 'critic_loss' in training_info:
                self.metrics['train']['critic']['losses'].append(float(training_info['critic_loss']))
            if 'n_updates' in training_info:
                self.metrics['train']['progress']['n_updates'] = training_info['n_updates']

            # New metrics from training_info
            if 'q_value' in training_info:
                q_value = float(training_info['q_value'])
                self.metrics['train']['critic']['q_values']['mean'].append(q_value)
            if 's2_value' in training_info:
                self.metrics['train']['critic']['s2_values'].append(float(training_info['s2_value']))
            if 'actor_lr' in training_info:
                self.metrics['train']['actor']['learning_rate'] = training_info['actor_lr']
            if 'critic_lr' in training_info:
                self.metrics['train']['critic']['learning_rate'] = training_info['critic_lr']

        # Print and save metrics every episode
        self._print_metrics()
        self.save_metrics()

    def update_buffer_metrics(self, replay_buffer):
        """Update replay buffer metrics"""
        
        # Update capacity metrics
        self.metrics['buffer']['capacity']['size'] = len(replay_buffer)
        self.metrics['buffer']['capacity']['max_size'] = replay_buffer.capacity
        self.metrics['buffer']['capacity']['utilization'] = len(replay_buffer) / replay_buffer.capacity
        
        # Update parameters
        self.metrics['buffer']['params']['fade_factor'] = replay_buffer.fade_factor
        self.metrics['buffer']['params']['stall_penalty'] = replay_buffer.stall_penalty
        
        # Update normalization metrics
        if hasattr(replay_buffer, 'min_values'):
            self.metrics['buffer']['normalization']['states']['min'] = replay_buffer.min_values
            self.metrics['buffer']['normalization']['states']['max'] = replay_buffer.max_values

    def update_noise_metrics(self, noise):
        """Update noise and exploration metrics"""
        
        self.metrics['phase']['noise']['x_coor'] = noise.x_coor
        self.metrics['phase']['noise']['scale'] = noise.scale

        # Update mode based on x_coor thresholds
        if noise.x_coor >= math.pi:
            self.metrics['phase']['noise']['mode'] = 'none'
        elif noise.x_coor >= 2.133:
            self.metrics['phase']['noise']['mode'] = 'train'
        else:
            self.metrics['phase']['noise']['mode'] = 'explore'

    def _print_metrics(self):
        """Print formatted metrics table with current episode info"""

        try:
            # Get current episode info
            episode = self.metrics['rollout']['time']['episodes']
            reward = float(self.metrics['rollout']['episode_rewards'][-1]) if self.metrics['rollout']['episode_rewards'] else 0.0
            steps = int(self.metrics['rollout']['episode_lengths'][-1]) if self.metrics['rollout']['episode_lengths'] else 0

            # Calculate rolling window statistics
            window = min(STATS_ROLLING_WINDOW, len(self.metrics['rollout']['episode_lengths']))
            ep_len_mean = float(np.nanmean(self.metrics['rollout']['episode_lengths'][-window:])) if window > 0 else steps
            ep_rew_mean = float(np.nanmean(self.metrics['rollout']['episode_rewards'][-window:])) if window > 0 else reward

            # Calculate time metrics
            time_elapsed = max(time.time() - self.metrics['rollout']['time']['start_time'], 1e-6)
            timesteps = int(self.metrics['rollout']['time']['total_timesteps'])
            fps = float(timesteps / time_elapsed) if time_elapsed > 0 else 0.0
            
            # Calculate training metrics
            actor_loss = float(np.nanmean(self.metrics['train']['actor']['losses'][-100:])) if self.metrics['train']['actor']['losses'] else 0.0
            critic_loss = float(np.nanmean(self.metrics['train']['critic']['losses'][-100:])) if self.metrics['train']['critic']['losses'] else 0.0
            q_value = float(np.mean(self.metrics['train']['critic']['q_values']['mean'][-100:])) if self.metrics['train']['critic']['q_values']['mean'] else 0.0
            s2_value = float(np.mean(self.metrics['train']['critic']['s2_values'][-100:])) if self.metrics['train']['critic']['s2_values'] else 0.0

            # Print table header
            print(f"\n\n\nTraining Metrics for {self.env_name} (v{self.model_version})")
            print("\n" + "=" * 39)
            print(f"| {'Metric':<20} | {'Value':>12} |")
            print("-" * 39)

            # Current metrics
            print(f"| {'current/':<20} | {'':>12} |")  # Section header
            print(f"|    {'episode':<17} | {episode:>12d} |")
            print(f"|    {'reward':<17} | {reward:>12.2f} |")
            print(f"|    {'steps':<17} | {steps:>12d} |")
            
            # Rollout metrics
            print(f"| {'rollout/':<20} | {'':>12} |")  # Section header
            print(f"|    {'ep_len_mean':<17} | {ep_len_mean:>12.2f} |")
            print(f"|    {'ep_rew_mean':<17} | {ep_rew_mean:>12.2f} |")
            
            # Time metrics
            print(f"| {'time/':<20} | {'':>12} |")  # Section header
            print(f"|    {'episodes':<17} | {episode:>12d} |")
            print(f"|    {'fps':<17} | {fps:>12.0f} |")
            print(f"|    {'time_elapsed':<17} | {time_elapsed:>12.2f} |")
            print(f"|    {'total_timesteps':<17} | {timesteps:>12d} |")
            
            # Training metrics
            print(f"| {'train/':<20} | {'':>12} |")  # Section header
            print(f"|    {'actor_loss':<17} | {actor_loss:>12.2f} |")
            print(f"|    {'critic_loss':<17} | {critic_loss:>12.2f} |")
            print(f"|    {'actor_lr':<17} | {self.metrics['train']['actor']['learning_rate']:>12.6f} |")
            print(f"|    {'critic_lr':<17} | {self.metrics['train']['critic']['learning_rate']:>12.6f} |")
            print(f"|    {'q_value':<17} | {q_value:>12.2f} |")
            print(f"|    {'s2_value':<17} | {s2_value:>12.2f} |")
            print(f"|    {'n_updates':<17} | {int(float(self.metrics['train']['progress']['n_updates'])):>12d} |")

            # Buffer metrics
            if self.metrics['buffer']['capacity']['max_size'] > 0:
                print(f"| {'buffer/':<20} | {'':<12} |")
                print(f"|    {'size':<17} | {int(self.metrics['buffer']['capacity']['size']):>12d} |")
                print(f"|    {'utilization %':<17} | {float(self.metrics['buffer']['capacity']['utilization'])*100:>12.1f} |")
                print(f"|    {'fade_factor':<17} | {float(self.metrics['buffer']['params']['fade_factor']):>12.2f} |")

            # Noise metrics
            if self.metrics['phase']['noise']['mode'] != 'none':
                print(f"| {'noise/':<20} | {'':<12} |")
                print(f"|    {'mode':<17} | {str(self.metrics['phase']['noise']['mode']):>12s} |")
                print(f"|    {'scale':<17} | {float(self.metrics['phase']['noise']['scale']):>12.3f} |")
                print(f"|    {'x_coor':<17} | {float(self.metrics['phase']['noise']['x_coor']):>12.3f} |")

            print("=" * 39)

        except Exception as e:
            print(f"Error printing metrics: {e}")

    def save_metrics(self):
        """Save detailed metrics to JSON"""

        metrics_file = self.save_dir / 'metrics.json'

        try:
            save_data = {
                'training_info': {
                    'environment': self.env_name,
                    'total_episodes': self.metrics['rollout']['time']['episodes'],
                    'total_timesteps': self.metrics['rollout']['time']['total_timesteps'],
                    'time_elapsed': float(time.time() - self.metrics['rollout']['time']['start_time']),
                    'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'latest_metrics': {
                    'rollout': {
                        'ep_len_mean': float(np.mean(self.metrics['rollout']['episode_lengths'][-100:])),
                        'ep_rew_mean': float(np.mean(self.metrics['rollout']['episode_rewards'][-100:]))
                    },
                    'time': {
                        'fps': float(np.mean(self.metrics['rollout']['time']['fps_history'][-100:])),
                        'total_timesteps': self.metrics['rollout']['time']['total_timesteps']
                    },
                    'train': {
                        'actor': {
                            'loss': float(np.mean(self.metrics['train']['actor']['losses'][-100:])),
                            'learning_rate': self.metrics['train']['actor']['learning_rate']
                        },
                        'critic': {
                            'loss': float(np.mean(self.metrics['train']['critic']['losses'][-100:])),
                            'learning_rate': self.metrics['train']['critic']['learning_rate'],
                            'q_value': float(np.mean(self.metrics['train']['critic']['q_values']['mean'][-100:])),
                            's2_value': float(np.mean(self.metrics['train']['critic']['s2_values'][-100:]))
                        },
                        'progress': {
                            'n_updates': self.metrics['train']['progress']['n_updates']
                        }
                    },
                    'buffer': convert_tensor(self.metrics['buffer']),
                    'phase': convert_tensor(self.metrics['phase'])
                },
                'history': convert_tensor(self.metrics)  # Full history for plotting
            }

            with open(metrics_file, 'w') as f:
                json.dump(save_data, f, indent=4)

        except Exception as e:
            print(f"Error saving metrics: {e}")

    def log_eval_step(self, reward: float, step: int):
        """Log metrics from an evaluation episode"""

        try:
            # Add current evaluation metrics
            if not hasattr(self.metrics, 'eval'):
                self.metrics['eval'] = {
                    'rewards': [],
                    'lengths': []
                }
            
            self.metrics['eval']['rewards'].append(float(reward))
            self.metrics['eval']['lengths'].append(int(step))

            # Calculate evaluation stats
            avg_reward = np.mean(self.metrics['eval']['rewards'][-10:]) if self.metrics['eval']['rewards'] else 0
            avg_length = np.mean(self.metrics['eval']['lengths'][-10:]) if self.metrics['eval']['lengths'] else 0

            # Print table header
            print("\n")
            print(f"Evaluation Metrics for {self.env_name} (v{self.model_version})")
            print("\n" + "=" * 39)
            print(f"| {'Metric':<20} | {'Value':>12} |")
            print("-" * 39)

            # Print metrics
            print(f"| {'current/':<20} | {'':>12} |")  # Section header
            print(f"|    {'reward':<17} | {reward:>12.2f} |")
            print(f"|    {'steps':<17} | {step:>12d} |")
            print(f"| {'eval/':<20} | {'':>12} |")  # Section header
            print(f"|    {'avg_reward':<17} | {avg_reward:>12.2f} |")
            print(f"|    {'avg_length':<17} | {avg_length:>12.0f} |")
            print(f"|    {'total_evals':<17} | {len(self.metrics['eval']['rewards']):>12d} |")
            print("=" * 39)

        except Exception as e:
            print(f"Error logging evaluation metrics: {e}")

    def get_metrics(self):
        """Return the current training metrics"""

        return self.metrics
