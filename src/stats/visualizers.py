import math
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional

from src.constants import STATS_ROLLING_WINDOW


class Visualizer:
    """Handles all visualization and plotting functionality"""

    def __init__(self, save_dir: Path, env_name: str):
        self.save_dir = save_dir
        self.env_name = env_name

        # Set style for plots
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_training_metrics(self, metrics: dict):
        """Generate and save training curves"""

        episodes = range(len(metrics['rollout']['episode_rewards']))
        if not episodes:
            return

        try:
            plt.close('all')

            # Create figures for each visualization
            figs = []
            axs = []
            num_figs = 7
            for _ in range(num_figs):
                fig = plt.figure(figsize=(15, 8))
                ax = fig.add_subplot(111)
                figs.append(fig)
                axs.append(ax)

            # Create two separate figures
            self._plot_rewards_metrics(axs[0], episodes, metrics)
            self._plot_loss_metrics(axs[1], episodes, metrics)
            self._plot_action_metrics(axs[2], episodes, metrics)
            self._plot_buffer_metrics(axs[3], episodes, metrics)
            self._plot_q_networks_s2_variance_metrics(axs[4], episodes, metrics)
            self._plot_time_metrics(axs[5], episodes, metrics)
            self._plot_spectral_noise_hyperparameters(axs[5], episodes, metrics)

            plot_names = [
                'episode_metrics.png',
                'loss_metrics.png',
                'action_metrics.png',
                'buffer_metrics.png',
                'q_networks_s2_variance_metrics.png',
                'time_metrics.png',
                'spectral_noise_hyperparameters.png'
            ]

            for fig, name in zip(figs, plot_names):
                plt.figure(fig.number)
                plt.savefig(self.save_dir / name, dpi=300, bbox_inches='tight')

        finally:
            plt.close('all')
            
    def _plot_rewards_metrics(self, ax1, episodes: range, metrics):
        """Plot rewards metrics"""

        # Create twin axis for different scales
        ax1_twin = ax1.twinx()

        # Get raw data
        rewards = np.array(metrics['rollout']['episode_rewards'])
        lengths = np.array(metrics['rollout']['episode_lengths'])

        # Plot raw data
        rewards_line = ax1.plot(episodes, rewards, color='blue', 
                            label='Episode Reward', alpha=0.6, linewidth=1.5)
        lengths_line = ax1_twin.plot(episodes, lengths, color='green',
                                label='Episode Length', alpha=0.6, linewidth=1.5)

        if len(episodes) > 0:
            window = min(STATS_ROLLING_WINDOW, len(episodes))
            
            # For rewards
            rolling_indices = episodes  # Use all episodes
            rolling_reward_mean = np.zeros(len(rolling_indices))
            rolling_reward_std = np.zeros(len(rolling_indices))
            rolling_reward_min = np.zeros(len(rolling_indices))
            rolling_reward_max = np.zeros(len(rolling_indices))
            
            # Similarly for lengths
            rolling_length_mean = np.zeros(len(rolling_indices))
            rolling_length_std = np.zeros(len(rolling_indices))
            rolling_length_min = np.zeros(len(rolling_indices))
            rolling_length_max = np.zeros(len(rolling_indices))
            
            # Calculate rolling statistics
            for i, end_idx in enumerate(rolling_indices):
                # Use all data up to current index for first window points
                if i < window - 1:
                    reward_window = rewards[:i+1]
                    length_window = lengths[:i+1]
                else:
                    # Use fixed window size once enough data is available
                    reward_window = rewards[i-window+1:i+1]
                    length_window = lengths[i-window+1:i+1]
                
                # Calculate reward statistics
                rolling_reward_mean[i] = np.mean(reward_window)
                rolling_reward_std[i] = np.std(reward_window)
                rolling_reward_min[i] = np.min(rewards[:i+1])  # Global min up to current point
                rolling_reward_max[i] = np.max(rewards[:i+1])  # Global max up to current point
                
                # Calculate length statistics
                rolling_length_mean[i] = np.mean(length_window)
                rolling_length_std[i] = np.std(length_window)
                rolling_length_min[i] = np.min(lengths[:i+1])  # Global min up to current point
                rolling_length_max[i] = np.max(lengths[:i+1])  # Global max up to current point

            # Plot reward statistics
            reward_mean = ax1.plot(rolling_indices, rolling_reward_mean,
                                '--', color='darkblue', alpha=0.4, label='Rolling Reward',
                                linewidth=1.5)
            # Plot reward min/max envelope
            reward_minmax = ax1.fill_between(rolling_indices,
                            rolling_reward_min, rolling_reward_max,
                            color='#8BA3C7', alpha=0.14, label='Reward Min/Max')
            # Plot reward standard deviation band
            reward_std = ax1.fill_between(rolling_indices,
                            rolling_reward_mean - rolling_reward_std,
                            rolling_reward_mean + rolling_reward_std,
                            color='blue', alpha=0.2, label='Reward ±σ')

            # Plot length statistics
            length_mean = ax1_twin.plot(rolling_indices, rolling_length_mean,
                                    '--', color='darkgreen', alpha=0.4, label='Rolling Length',
                                    linewidth=1.5)
            # Plot length min/max envelope
            length_minmax = ax1_twin.fill_between(rolling_indices,
                                    rolling_length_min, rolling_length_max,
                                    color='#8BC7A3', alpha=0.14, label='Length Min/Max')
            # Plot length standard deviation band
            length_std = ax1_twin.fill_between(rolling_indices,
                                rolling_length_mean - rolling_length_std,
                                rolling_length_mean + rolling_length_std,
                                color='green', alpha=0.2, label='Length ±σ')

            # Add the lines and fills to the legend
            lines = [rewards_line[0], reward_mean[0], reward_minmax, reward_std,
                    lengths_line[0], length_mean[0], length_minmax, length_std]
            labels = ['Episode Reward', 'Rolling Reward', 'Reward Min/Max', 'Reward ±σ',
                    'Episode Length', 'Rolling Length', 'Length Min/Max', 'Length ±σ']

            # Add training start indicator line (where pure exploration ends)
            if 'train' in metrics and len(metrics['train']['actor']['losses']) > 0:
                start_idx = len(episodes) - len(metrics['train']['actor']['losses'])
                training_line = ax1.axvline(x=start_idx, color='gray', linestyle=':', 
                                        alpha=0.5, label='Training Initiation')
                lines.append(training_line)
                labels.append('Training Initiation')

        # Configure axes
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1_twin.set_ylabel('Step')
        ax1.tick_params(axis='y')
        ax1_twin.tick_params(axis='y')
        ax1.set_title(f'Learning Curve - {self.env_name}')
        ax1.grid(True, alpha=0.3)

        # Create legend on top left
        ax1.legend(lines, labels, loc='upper left')

    def _plot_time_metrics(self, ax1, episodes: range, metrics):
        """Plot time duration metrics and steps per episode"""
        
        # Create twin axis for different scales
        ax1_twin = ax1.twinx()

        # Get raw data
        fps_history = np.array(metrics['rollout']['time']['fps_history'])
        steps = np.array(metrics['rollout']['episode_lengths'])
        
        # Calculate time per episode from fps
        times_per_episode = steps / fps_history
        
        # Plot raw data
        time_line = ax1.plot(episodes, times_per_episode, color='orange', 
                            label='Episode Duration', alpha=0.6, linewidth=1.5)
        steps_line = ax1_twin.plot(episodes, steps, color='purple',
                                label='Steps per Episode', alpha=0.6, linewidth=1.5)
        
        lines = []
        lines.extend(time_line + steps_line)
        labels = ['Episode Duration', 'Steps per Episode']

        if len(episodes) > 0:
            window = min(STATS_ROLLING_WINDOW, len(episodes))
            rolling_indices = episodes

            # Initialize arrays for rolling statistics
            rolling_time_mean = np.zeros(len(rolling_indices))
            rolling_time_std = np.zeros(len(rolling_indices))
            rolling_time_min = np.zeros(len(rolling_indices))
            rolling_time_max = np.zeros(len(rolling_indices))
            
            rolling_steps_mean = np.zeros(len(rolling_indices))
            rolling_steps_std = np.zeros(len(rolling_indices))
            rolling_steps_min = np.zeros(len(rolling_indices))
            rolling_steps_max = np.zeros(len(rolling_indices))
            
            # Calculate rolling statistics
            for i, end_idx in enumerate(rolling_indices):
                if i < window - 1:
                    time_window = times_per_episode[:i+1]
                    steps_window = steps[:i+1]
                else:
                    time_window = times_per_episode[i-window+1:i+1]
                    steps_window = steps[i-window+1:i+1]
                
                # Calculate time statistics
                rolling_time_mean[i] = np.mean(time_window)
                rolling_time_std[i] = np.std(time_window)
                rolling_time_min[i] = np.min(times_per_episode[:i+1])
                rolling_time_max[i] = np.max(times_per_episode[:i+1])
                
                # Calculate steps statistics
                rolling_steps_mean[i] = np.mean(steps_window)
                rolling_steps_std[i] = np.std(steps_window)
                rolling_steps_min[i] = np.min(steps[:i+1])
                rolling_steps_max[i] = np.max(steps[:i+1])
            
            # Plot time statistics
            time_mean = ax1.plot(rolling_indices, rolling_time_mean,
                            '--', color='darkorange', alpha=0.4,
                            label='Rolling Duration', linewidth=1.5)
            time_minmax = ax1.fill_between(rolling_indices,
                                        rolling_time_min, rolling_time_max,
                                        color='#FFB347', alpha=0.14,
                                        label='Duration Min/Max')
            time_std = ax1.fill_between(rolling_indices,
                                    rolling_time_mean - rolling_time_std,
                                    rolling_time_mean + rolling_time_std,
                                    color='orange', alpha=0.2,
                                    label='Duration ±σ')
                                    
            # Plot steps statistics
            steps_mean = ax1_twin.plot(rolling_indices, rolling_steps_mean,
                                    '--', color='darkviolet', alpha=0.4,
                                    label='Rolling Steps', linewidth=1.5)
            steps_minmax = ax1_twin.fill_between(rolling_indices,
                                            rolling_steps_min, rolling_steps_max,
                                            color='#B08BC7', alpha=0.14,
                                            label='Steps Min/Max')
            steps_std = ax1_twin.fill_between(rolling_indices,
                                        rolling_steps_mean - rolling_steps_std,
                                        rolling_steps_mean + rolling_steps_std,
                                        color='purple', alpha=0.2,
                                        label='Steps ±σ')

            # Add training start indicator if data exists
            if 'train' in metrics and len(metrics['train']['actor']['losses']) > 0:
                start_idx = len(episodes) - len(metrics['train']['actor']['losses'])
                training_line = ax1.axvline(x=start_idx, color='gray', linestyle=':',
                                        alpha=0.5, label='Training Initiation')
                lines.append(training_line)
                labels.append('Training Initiation')
                
            # Update lines and labels
            lines.extend([time_line[0], time_mean[0], time_minmax, time_std,
                        steps_line[0], steps_mean[0], steps_minmax, steps_std])
            labels.extend(['Episode Duration', 'Rolling Duration', 'Duration Min/Max', 'Duration ±σ',
                        'Steps per Episode', 'Rolling Steps', 'Steps Min/Max', 'Steps ±σ'])

        # Configure axes
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Time (seconds)')
        ax1_twin.set_ylabel('Steps')
        ax1.tick_params(axis='y')
        ax1_twin.tick_params(axis='y')
        ax1.set_title(f'Episode Duration and Steps - {self.env_name}')
        ax1.grid(True, alpha=0.3)

        # Create legend
        ax1.legend(lines, labels, loc='upper left')

    def _plot_loss_metrics(self, ax: plt.Axes, episodes: range, metrics: dict):
        """Plot loss-related metrics including initial zero period"""

        total_episodes = len(episodes)
        
        # Create full arrays with zeros
        full_actor_losses = np.zeros(total_episodes)
        full_critic_losses = np.zeros(total_episodes)

        # Get actual loss values
        actor_losses = metrics['train']['actor']['losses']
        critic_losses = metrics['train']['critic']['losses']

        # Create twin axis for different scales
        ax2_twin = ax.twinx()

        lines = []

        # Plot complete loss arrays including zeros
        actor_line = ax.plot(episodes, full_actor_losses,
                            alpha=0.6, color='red', label='Actor Loss',
                            linewidth=1.5, marker='o', markersize=2)
        critic_line = ax2_twin.plot(episodes, full_critic_losses,
                                    alpha=0.6, color='purple', label='Critic Loss',
                                    linewidth=1.5, marker='o', markersize=2)

        lines.extend(actor_line + critic_line)
        labels = ['Actor Loss', 'Critic Loss']

        # If we have loss values, update the arrays
        if metrics['train']['progress']['n_updates'] > 0 and len(actor_losses) > 0:
            # Calculate start index
            start_idx = total_episodes - len(actor_losses)
            full_actor_losses[start_idx:] = actor_losses
            full_critic_losses[start_idx:] = critic_losses

            # Update plot data
            actor_line[0].set_ydata(full_actor_losses)
            critic_line[0].set_ydata(full_critic_losses)

            # Calculate statistics from the beginning
            window = min(STATS_ROLLING_WINDOW, len(episodes))
            
            # Initialize arrays for the entire episode range
            rolling_indices = range(start_idx, total_episodes)
            actor_mean = np.zeros(len(rolling_indices))
            actor_std = np.zeros(len(rolling_indices))
            actor_min = np.zeros(len(rolling_indices))
            actor_max = np.zeros(len(rolling_indices))
            
            critic_mean = np.zeros(len(rolling_indices))
            critic_std = np.zeros(len(rolling_indices))
            critic_min = np.zeros(len(rolling_indices))
            critic_max = np.zeros(len(rolling_indices))
            
            # Calculate rolling statistics
            for i, current_idx in enumerate(rolling_indices):
                # Use all data up to current index for first window points
                if i < window - 1:
                    actor_window = full_actor_losses[start_idx:current_idx+1]
                    critic_window = full_critic_losses[start_idx:current_idx+1]
                else:
                    # Use fixed window size once enough data is available
                    actor_window = full_actor_losses[current_idx-window+1:current_idx+1]
                    critic_window = full_critic_losses[current_idx-window+1:current_idx+1]
                
                # Calculate actor statistics
                actor_mean[i] = np.mean(actor_window)
                actor_std[i] = np.std(actor_window)
                actor_min[i] = np.min(full_actor_losses[start_idx:current_idx+1])  # Global min up to current point
                actor_max[i] = np.max(full_actor_losses[start_idx:current_idx+1])  # Global max up to current point
                
                # Calculate critic statistics
                critic_mean[i] = np.mean(critic_window)
                critic_std[i] = np.std(critic_window)
                critic_min[i] = np.min(full_critic_losses[start_idx:current_idx+1])  # Global min up to current point
                critic_max[i] = np.max(full_critic_losses[start_idx:current_idx+1])  # Global max up to current point

            # Plot actor statistics
            actor_mean_line = ax.plot(rolling_indices, actor_mean,
                                    '--', color='darkred', alpha=0.4, label='Rolling Actor Loss',
                                    linewidth=1.5)
            # Plot actor min/max envelope for actor
            actor_minmax = ax.fill_between(rolling_indices,
                                        actor_min, actor_max,
                                        color='#C78B8B', alpha=0.14, label='Actor Loss Min/Max')
            # Plot actor standard deviation band for actor
            actor_std_fill = ax.fill_between(rolling_indices,
                                        actor_mean - actor_std,
                                        actor_mean + actor_std,
                                        color='red', alpha=0.2, label='Actor Loss ±σ')

            # Plot critic statistics
            critic_mean_line = ax2_twin.plot(rolling_indices, critic_mean,
                                        '--', color='darkviolet', alpha=0.4, label='Rolling Critic Loss',
                                        linewidth=1.5)
            # Plot critic min/max envelope for critic
            critic_minmax = ax2_twin.fill_between(rolling_indices,
                                                critic_min, critic_max,
                                                color='#B08BC7', alpha=0.14, label='Critic Loss Min/Max')
            # Plot critic standard deviation band for critic
            critic_std_fill = ax2_twin.fill_between(rolling_indices,
                                                critic_mean - critic_std,
                                                critic_mean + critic_std,
                                                color='purple', alpha=0.2, label='Critic Loss ±σ')

            # Clear existing lines and labels
            lines.clear()
            labels.clear()

            # Add lines and fills to the legend
            lines.extend([actor_line[0], actor_mean_line[0], actor_minmax, actor_std_fill,
                        critic_line[0], critic_mean_line[0], critic_minmax, critic_std_fill])
            labels.extend(['Actor Loss', 'Rolling Actor Loss', 'Actor Loss Min/Max', 'Actor Loss ±σ',
                        'Critic Loss', 'Rolling Critic Loss', 'Critic Loss Min/Max', 'Critic Loss ±σ'])

            # Add training start indicator line
            training_line = ax.axvline(x=start_idx, color='gray', linestyle=':', 
                                    alpha=0.5, label='Training Initiation')
            lines.append(training_line)
            labels.append('Training Initiation')

            # Dynamically adjust y-axis limits
            valid_losses = np.concatenate([
                full_actor_losses[full_actor_losses != 0],
                full_critic_losses[full_critic_losses != 0]
            ])

            if len(valid_losses) > 0:
                padding = 0.05  # 5% padding
                y_min = np.min(valid_losses) * (1 - padding)
                y_max = np.max(valid_losses) * (1 + padding)
                ax.set_ylim(y_min, y_max)
                ax2_twin.set_ylim(y_min, y_max)

        # Set axis labels and styling
        ax.set_ylabel('Actor Loss')
        ax2_twin.set_ylabel('Critic Loss')
        ax.tick_params(axis='y')
        ax2_twin.tick_params(axis='y')
        ax.set_xlabel('Episode')
        ax.set_title(f'Training Losses - {self.env_name}')
        ax.grid(True, alpha=0.3)

        # Add legend
        ax.legend(lines, labels, loc='upper left')

    def _plot_action_metrics(self, ax: plt.Axes, episodes: range, metrics: dict):
        """
        Plot action distribution metrics showing mean and standard deviation of actions.
        Includes statistical tracking like rolling averages and min/max ranges.
        """
        
        total_episodes = len(episodes)
        
        # Create full arrays with zeros
        full_action_means = np.zeros(total_episodes)
        full_action_stds = np.zeros(total_episodes)
        
        # Get actual values from metrics
        if ('action' in metrics and 'distribution' in metrics['action']):
            action_means = metrics['action']['distribution'].get('mean', [])
            action_stds = metrics['action']['distribution'].get('std', [])
            
            if len(action_means) > 0:  # Only proceed if we have actual values
                start_idx = total_episodes - len(action_means)
                
                # Fill arrays with actual values
                full_action_means[start_idx:] = action_means
                full_action_stds[start_idx:] = action_stds
        
        lines = []
        labels = []
        
        # Plot base values
        mean_line = ax.plot(episodes, full_action_means,
                        alpha=0.6, color='orange', label='Action Mean',
                        linewidth=1.5, marker='o', markersize=2)
        std_line = ax.plot(episodes, full_action_stds,
                        alpha=0.6, color='red', label='Action Std',
                        linewidth=1.5, marker='s', markersize=2)
        
        # Add base lines to legend
        lines.extend(mean_line + std_line)
        labels.extend(['Action Mean', 'Action Std'])
        
        if np.any(full_action_means != 0):
            start_idx = np.where(full_action_means != 0)[0][0]
            window = min(STATS_ROLLING_WINDOW, len(episodes))
            rolling_indices = range(start_idx, total_episodes)
            
            # Initialize arrays for statistics
            mean_means = np.zeros(len(rolling_indices))  # Mean of means
            mean_stds = np.zeros(len(rolling_indices))   # Std of means
            mean_mins = np.zeros(len(rolling_indices))   # Min of means
            mean_maxs = np.zeros(len(rolling_indices))   # Max of means
            
            std_means = np.zeros(len(rolling_indices))   # Mean of stds
            std_stds = np.zeros(len(rolling_indices))    # Std of stds
            std_mins = np.zeros(len(rolling_indices))    # Min of stds
            std_maxs = np.zeros(len(rolling_indices))    # Max of stds
            
            # Calculate statistics using rolling window
            for i, current_idx in enumerate(rolling_indices):
                if i < window - 1:
                    mean_window = full_action_means[start_idx:current_idx+1]
                    std_window = full_action_stds[start_idx:current_idx+1]
                else:
                    mean_window = full_action_means[current_idx-window+1:current_idx+1]
                    std_window = full_action_stds[current_idx-window+1:current_idx+1]
                
                # Calculate stats for means
                mean_means[i] = np.mean(mean_window)
                mean_stds[i] = np.std(mean_window)
                mean_mins[i] = np.min(mean_window)
                mean_maxs[i] = np.max(mean_window)
                
                # Calculate stats for stds
                std_means[i] = np.mean(std_window)
                std_stds[i] = np.std(std_window)
                std_mins[i] = np.min(std_window)
                std_maxs[i] = np.max(std_window)
            
            # Plot mean statistics
            mean_avg_line = ax.plot(rolling_indices, mean_means,
                                '--', color='darkorange', label='Rolling Mean',
                                linewidth=1.5)[0]
            mean_fill = ax.fill_between(rolling_indices, mean_mins, mean_maxs,
                                    color='#FFB347', alpha=0.1, label='Mean Min/Max')
            mean_std_band = ax.fill_between(rolling_indices,
                                        mean_means - mean_stds,
                                        mean_means + mean_stds,
                                        color='orange', alpha=0.2, label='Mean ±σ')
            
            # Plot std statistics
            std_avg_line = ax.plot(rolling_indices, std_means,
                                '--', color='darkred', label='Rolling Std',
                                linewidth=1.5)[0]
            std_fill = ax.fill_between(rolling_indices, std_mins, std_maxs,
                                    color='#FF6B6B', alpha=0.1, label='Std Min/Max')
            std_std_band = ax.fill_between(rolling_indices,
                                        std_means - std_stds,
                                        std_means + std_stds,
                                        color='red', alpha=0.2, label='Std ±σ')
            
            # Add statistics to legend
            lines.extend([mean_avg_line, std_avg_line,
                        mean_fill, std_fill,
                        mean_std_band, std_std_band])
            labels.extend(['Rolling Mean', 'Rolling Std',
                        'Mean Min/Max', 'Std Min/Max',
                        'Mean ±σ', 'Std ±σ'])
            
            # Add training initiation line
            training_line = ax.axvline(x=start_idx, color='gray', linestyle=':',
                                    alpha=0.5, label='Training Initiation')
            lines.append(training_line)
            labels.append('Training Initiation')
            
            # Adjust y-axis limits
            valid_values = np.concatenate([
                full_action_means[full_action_means != 0],
                full_action_stds[full_action_stds != 0]
            ])
            
            if len(valid_values) > 0:
                padding = 0.05
                min_val = np.min(valid_values) * (1 - padding)
                max_val = np.max(valid_values) * (1 + padding)
                value_range = max_val - min_val
                # Add extra range to ensure negative values are visible
                ax.set_ylim(min_val - value_range * 0.5, max_val + value_range * 0.5)
        
        # Set axis labels and styling
        ax.set_xlabel('Episode')
        ax.set_ylabel('Action Distribution')
        ax.set_title(f'Action Distribution Metrics - {self.env_name}')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(lines, labels, loc='upper left', fontsize='small')

    def _plot_buffer_metrics(self, ax: plt.Axes, episodes: range, metrics: dict):
        """Plot replay buffer metrics including utilization and parameters"""
        
        total_episodes = len(episodes)
        
        # Create full arrays with zeros
        full_utilization = np.zeros(total_episodes)
        full_fade_factor = np.zeros(total_episodes)
        
        # Get buffer metrics
        buffer_size = metrics['buffer']['capacity']['size']
        buffer_capacity = metrics['buffer']['capacity']['max_size']
        utilization = metrics['buffer']['capacity']['utilization']
        fade_factor = metrics['buffer']['params']['fade_factor']
        
        # Create twin axis
        ax_twin = ax.twinx()
        
        lines = []
        
        # Plot complete arrays including zeros
        util_line = ax.plot(episodes, full_utilization,
                        alpha=0.6, color='cyan', label='Buffer Utilization',
                        linewidth=1.5, marker='o', markersize=2)
        fade_line = ax_twin.plot(episodes, full_fade_factor,
                                alpha=0.6, color='magenta', label='Fade Factor',
                                linewidth=1.5, marker='o', markersize=2)
        
        lines.extend(util_line)
        lines.extend(fade_line)
        labels = ['Buffer Utilization', 'Fade Factor']
        
        if buffer_size > 0:
            # Set the values after training starts
            start_idx = max(0, total_episodes - len(metrics['train']['actor']['losses']))
            full_utilization[start_idx:] = utilization
            full_fade_factor[start_idx:] = fade_factor
            
            # Update plot data
            util_line[0].set_ydata(full_utilization)
            fade_line[0].set_ydata(full_fade_factor)
            
            # Add buffer capacity reference line
            capacity_line = ax.axhline(y=1.0, color='red', linestyle='--',
                                    alpha=0.5, label='Max Capacity')
            lines.append(capacity_line)
            labels.append('Max Capacity')
            
            # Add training start indicator
            training_line = ax.axvline(x=start_idx, color='gray', linestyle=':',
                                    alpha=0.5, label='Training Initiation')
            lines.append(training_line)
            labels.append('Training Initiation')
        
        # Set axis labels and styling
        ax.set_xlabel('Episode')
        ax.set_ylabel('Buffer Utilization (%)')
        ax_twin.set_ylabel('Fade Factor')
        ax.set_title(f'Replay Buffer Metrics - {self.env_name}')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)  # Utilization is a percentage
        
        # Add legend
        ax.legend(lines, labels, loc='upper left')

    def _plot_q_networks_s2_variance_metrics(self, ax: plt.Axes, episodes: range, metrics: dict):
        """
        Plot Q-network and S2 (variance) metrics.
        Shows individual Q-networks (qA, qB, qC) and their statistics.
        Rolling values represent moving averages over a window of episodes.
        """

        total_episodes = len(episodes)
        
        # Create full arrays with zeros for each Q-network
        full_qA_values = np.zeros(total_episodes)
        full_qB_values = np.zeros(total_episodes)
        full_qC_values = np.zeros(total_episodes)
        full_s2_values = np.zeros(total_episodes)
        
        # Get actual values from training metrics - keeping working structure
        if ('train' in metrics and 'critic' in metrics['train'] and 
            'q_values' in metrics['train']['critic']):
            
            # Access the Q-network outputs
            qA_values = metrics['train']['critic']['q_values'].get('qA', [])
            qB_values = metrics['train']['critic']['q_values'].get('qB', [])
            qC_values = metrics['train']['critic']['q_values'].get('qC', [])

            # For backward compatibility
            if not qA_values and 'mean' in metrics['train']['critic']['q_values']:
                mean_q = metrics['train']['critic']['q_values']['mean']
                qA_values = mean_q
                qB_values = mean_q
                qC_values = mean_q
                
            s2_values = metrics['train']['critic'].get('s2_values', [])

            if len(qA_values) > 0:  # Only proceed if we have actual values
                start_idx = total_episodes - len(qA_values)
                
                # Fill the arrays with actual values
                full_qA_values[start_idx:] = qA_values
                full_qB_values[start_idx:] = qB_values
                full_qC_values[start_idx:] = qC_values
                full_s2_values[start_idx:] = s2_values
        
        # Create twin axis for S2 values
        ax_twin = ax.twinx()
        
        lines = []
        labels = []

        # First plot base values (keeping working structure)
        qA_line = ax.plot(episodes, full_qA_values,
                        alpha=0.6, color='blue', label='Q-Network A',
                        linewidth=1.5, marker='o', markersize=2)
        qB_line = ax.plot(episodes, full_qB_values,
                        alpha=0.6, color='red', label='Q-Network B',
                        linewidth=1.5, marker='s', markersize=2)
        qC_line = ax.plot(episodes, full_qC_values,
                        alpha=0.6, color='purple', label='Q-Network C',
                        linewidth=1.5, marker='^', markersize=2)
        s2_line = ax_twin.plot(episodes, full_s2_values,
                            alpha=0.6, color='green', label='S2-Value',
                            linewidth=1.5, marker='d', markersize=2)

        # Add base lines to legend
        lines.extend(qA_line + qB_line + qC_line + s2_line)
        labels.extend(['Q-Network A', 'Q-Network B', 'Q-Network C', 'S2-Value'])
        
        # If we have valid data, add statistics and fills
        if np.any(full_qA_values != 0):
            start_idx = np.where(full_qA_values != 0)[0][0]
            window = min(STATS_ROLLING_WINDOW, len(episodes))
            rolling_indices = range(start_idx, total_episodes)
            
            # Initialize arrays for statistics
            qA_means = np.zeros(len(rolling_indices))
            qA_mins = np.zeros(len(rolling_indices))
            qA_maxs = np.zeros(len(rolling_indices))
            qA_stds = np.zeros(len(rolling_indices))
            
            qB_means = np.zeros(len(rolling_indices))
            qB_mins = np.zeros(len(rolling_indices))
            qB_maxs = np.zeros(len(rolling_indices))
            qB_stds = np.zeros(len(rolling_indices))
            
            qC_means = np.zeros(len(rolling_indices))
            qC_mins = np.zeros(len(rolling_indices))
            qC_maxs = np.zeros(len(rolling_indices))
            qC_stds = np.zeros(len(rolling_indices))
            
            s2_means = np.zeros(len(rolling_indices))
            s2_mins = np.zeros(len(rolling_indices))
            s2_maxs = np.zeros(len(rolling_indices))
            s2_stds = np.zeros(len(rolling_indices))

            # Calculate statistics using rolling window
            for i, current_idx in enumerate(rolling_indices):
                if i < window - 1:
                    qA_window = full_qA_values[start_idx:current_idx+1]
                    qB_window = full_qB_values[start_idx:current_idx+1]
                    qC_window = full_qC_values[start_idx:current_idx+1]
                    s2_window = full_s2_values[start_idx:current_idx+1]
                else:
                    qA_window = full_qA_values[current_idx-window+1:current_idx+1]
                    qB_window = full_qB_values[current_idx-window+1:current_idx+1]
                    qC_window = full_qC_values[current_idx-window+1:current_idx+1]
                    s2_window = full_s2_values[current_idx-window+1:current_idx+1]
                
                # Calculate statistics for QA
                qA_means[i] = np.mean(qA_window)
                qA_mins[i] = np.min(qA_window)
                qA_maxs[i] = np.max(qA_window)
                qA_stds[i] = np.std(qA_window)
                
                # Calculate statistics for QB
                qB_means[i] = np.mean(qB_window)
                qB_mins[i] = np.min(qB_window)
                qB_maxs[i] = np.max(qB_window)
                qB_stds[i] = np.std(qB_window)
                
                # Calculate statistics for QC
                qC_means[i] = np.mean(qC_window)
                qC_mins[i] = np.min(qC_window)
                qC_maxs[i] = np.max(qC_window)
                qC_stds[i] = np.std(qC_window)
                
                # Calculate statistics for S2
                s2_means[i] = np.mean(s2_window)
                s2_mins[i] = np.min(s2_window)
                s2_maxs[i] = np.max(s2_window)
                s2_stds[i] = np.std(s2_window)
            
            # Plot Q-Network A statistics
            qA_mean_line = ax.plot(rolling_indices, qA_means, '--', color='darkblue', alpha=0.4,
                                label='Rolling Q-A', linewidth=1.5)[0]
            qA_minmax = ax.fill_between(rolling_indices, qA_mins, qA_maxs,
                                    color='#8BA3C7', alpha=0.14, label='Q-A Min/Max')
            qA_std = ax.fill_between(rolling_indices, qA_means - qA_stds, qA_means + qA_stds,
                                color='blue', alpha=0.2, label='Q-A ±σ')

            # Plot Q-Network B statistics
            qB_mean_line = ax.plot(rolling_indices, qB_means, '--', color='darkred', alpha=0.4,
                                label='Rolling Q-B', linewidth=1.5)[0]
            qB_minmax = ax.fill_between(rolling_indices, qB_mins, qB_maxs,
                                    color='#C78B8B', alpha=0.14, label='Q-B Min/Max')
            qB_std = ax.fill_between(rolling_indices, qB_means - qB_stds, qB_means + qB_stds,
                                color='red', alpha=0.2, label='Q-B ±σ')

            # Plot Q-Network C statistics
            qC_mean_line = ax.plot(rolling_indices, qC_means, '--', color='darkviolet', alpha=0.4,
                                label='Rolling Q-C', linewidth=1.5)[0]
            qC_minmax = ax.fill_between(rolling_indices, qC_mins, qC_maxs,
                                    color='#B08BC7', alpha=0.14, label='Q-C Min/Max')
            qC_std = ax.fill_between(rolling_indices, qC_means - qC_stds, qC_means + qC_stds,
                                color='purple', alpha=0.2, label='Q-C ±σ')

            # Plot S2 statistics
            s2_mean_line = ax_twin.plot(rolling_indices, s2_means, '--', color='darkgreen', alpha=0.4,
                                    label='Rolling S2', linewidth=1.5)[0]
            s2_minmax = ax_twin.fill_between(rolling_indices, s2_mins, s2_maxs,
                                        color='#8BC7A3', alpha=0.14, label='S2 Min/Max')
            s2_std = ax_twin.fill_between(rolling_indices, s2_means - s2_stds, s2_means + s2_stds,
                                        color='green', alpha=0.2, label='S2 ±σ')
            
            # Clear existing lines and labels
            lines.clear()
            labels.clear()

            # Add mean lines and fills to legend
            lines.extend([qA_line[0], qA_mean_line, qA_minmax, qA_std,
                        qB_line[0], qB_mean_line, qB_minmax, qB_std,
                        qC_line[0], qC_mean_line, qC_minmax, qC_std,
                        s2_line[0], s2_mean_line, s2_minmax, s2_std])
            labels.extend(['Q-Network A', 'Rolling Q-A', 'Q-A Min/Max', 'Q-A ±σ',
                        'Q-Network B', 'Rolling Q-B', 'Q-B Min/Max', 'Q-B ±σ',
                        'Q-Network C', 'Rolling Q-C', 'Q-C Min/Max', 'Q-C ±σ',
                        'S2-Value', 'Rolling S2', 'S2 Min/Max', 'S2 ±σ'])
            
            # Add training initiation line
            training_line = ax.axvline(x=start_idx, color='gray', linestyle=':',
                                    alpha=0.5, label='Training Initiation')
            lines.append(training_line)
            labels.append('Training Initiation')
            
            # Adjust y-axis limits
            valid_q_values = np.concatenate([
                full_qA_values[full_qA_values != 0],
                full_qB_values[full_qB_values != 0],
                full_qC_values[full_qC_values != 0]
            ])
            valid_s2_values = full_s2_values[full_s2_values != 0]
            
            if len(valid_q_values) > 0 and len(valid_s2_values) > 0:
                # Basic Q-values axis limits
                q_min_val = np.min(valid_q_values)
                q_max_val = np.max(valid_q_values)
                ax.set_ylim(q_min_val, q_max_val)

                # Basic S2 axis limits
                s2_min_val = np.min(valid_s2_values)
                s2_max_val = np.max(valid_s2_values)
                ax_twin.set_ylim(s2_min_val, s2_max_val)

        # Set axis labels and styling
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q-Values')
        ax_twin.set_ylabel('S2-Value (Variance)')
        ax.set_title(f'Q-Networks and Variance Metrics - {self.env_name}')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(lines, labels, loc='upper left', fontsize='small')

    def _plot_spectral_noise_hyperparameters(self, ax: plt.Axes, episodes: range, metrics: dict):
        """Plot spectral noise hyperparameter metrics with three axes (left, right, and color)"""
        
        if 'exploration' not in metrics or 'spectral' not in metrics['exploration']:
            return

        # Create twin axis for second metric
        ax2 = ax.twinx()
        
        # Get hyperparameter data
        spectral_data = metrics['exploration']['spectral']
        steps = np.array(episodes)

        # First metric: Feature extraction dimension impact (left y-axis)
        feature_metrics = np.array(spectral_data.get('feature_effectiveness', []))
        if len(feature_metrics) > 0:
            line1 = ax.plot(steps, feature_metrics, 
                        color='blue', label='Feature Extraction', 
                        alpha=0.8, linewidth=1.5)
            ax.set_ylabel('Feature Extraction Effectiveness', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')

        # Second metric: Learning rate impact (right y-axis) 
        learning_metrics = np.array(spectral_data.get('learning_progress', []))
        if len(learning_metrics) > 0:
            line2 = ax2.plot(steps, learning_metrics,
                            color='red', label='Learning Progress',
                            alpha=0.8, linewidth=1.5)
            ax2.set_ylabel('Learning Rate Progress', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

        # Third metric: Buffer utilization (color intensity)
        buffer_metrics = np.array(spectral_data.get('buffer_utilization', []))
        if len(buffer_metrics) > 0:
            # Create color map based on buffer utilization
            colors = plt.cm.viridis(buffer_metrics / np.max(buffer_metrics))
            scatter = ax.scatter(steps, feature_metrics,
                            c=buffer_metrics, cmap='viridis',
                            alpha=0.3, s=20)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Buffer Utilization')

        # Combine legends
        lines = []
        labels = []
        if len(feature_metrics) > 0:
            lines.extend(line1)
            labels.append('Feature Extraction')
        if len(learning_metrics) > 0:  
            lines.extend(line2)
            labels.append('Learning Progress')

        ax.legend(lines, labels, loc='upper left')
        
        # Set title and labels
        ax.set_title(f'Spectral Noise Hyperparameters - {self.env_name}')
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.3)

        # Add training start indicator if available
        if 'train' in metrics and len(metrics['train']['actor']['losses']) > 0:
            start_idx = len(episodes) - len(metrics['train']['actor']['losses'])
            ax.axvline(x=start_idx, color='gray', linestyle=':', 
                    alpha=0.5, label='Training Initiated')

    def __del__(self):
        """Ensure all plots are closed when the visualizer is destroyed"""

        plt.close('all')


class ComparisonVisualizer:
    """Handles visualization of multiple training runs rewards metrics"""
    
    def __init__(self, base_save_dir: Path):
        self.base_save_dir = Path(base_save_dir)
        
        # Set style for plots
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Color scheme for different runs
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
    def load_metrics(self, metrics_files: List[Path]) -> List[Dict]:
        """Load metrics from multiple JSON files"""
        metrics_list = []
        
        for file_path in metrics_files:
            try:
                with open(file_path, 'r') as f:
                    metrics = json.load(f)
                    metrics_list.append(metrics)
            except Exception as e:
                print(f"Error loading metrics from {file_path}: {e}")
                
        return metrics_list
    
    def plot_rewards_comparison(self, metrics_list: List[Dict], labels: Optional[List[str]] = None,
                          rolling_window: int = 50, save_path: Optional[Path] = None,
                          x_axis: str = 'episode', plot_elements: Optional[List[str]] = None):
        """Plot comparative reward metrics from multiple runs
        
        Args:
            metrics_list: List of metrics dictionaries
            labels: Optional list of labels for each run
            rolling_window: Window size for rolling statistics
            save_path: Optional path to save the figure
            x_axis: Whether to plot against 'episode' or 'step'
            plot_elements: List of elements to plot. Options: ['raw', 'mean', 'std', 'training_start']
                        If None, all elements are plotted
        """
        
        if not metrics_list:
            print("No metrics provided for visualization")
            return
            
        if labels is None:
            labels = [f"Run {i+1}" for i in range(len(metrics_list))]
            
        # Create figure and axis
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Store lines and labels for legend
        lines = []
        legend_labels = []
        
        # Plot each run
        for idx, (metrics, label) in enumerate(zip(metrics_list, labels)):
            color = self.colors[idx % len(self.colors)]
            alpha_base = 0.6
            alpha_stats = 0.2
            
            # Get reward data
            rewards = np.array(metrics['history']['rollout']['episode_rewards'])
            
            # Determine x-axis values based on selection
            if x_axis == 'step':
                # Calculate cumulative steps
                steps = np.array(metrics['history']['rollout']['episode_lengths'])
                x_values = np.cumsum(steps)
                ax.set_xlabel('Steps')
            else:  # default to episode
                x_values = np.arange(len(rewards))
                ax.set_xlabel('Episode')

            # Default to all elements if none specified
            if plot_elements is None:
                plot_elements = ['raw', 'mean', 'std', 'training_start']

            # Plot raw data only if selected
            if 'raw' in plot_elements:
                reward_line = ax.plot(x_values, rewards, color=color, 
                                    alpha=alpha_base, label=f"{label}",
                                    linewidth=1.0, marker='o', markersize=2)
                lines.append(reward_line[0])
                legend_labels.append(f"{label}")

            if len(x_values) > 0 and ('mean' in plot_elements or 'std' in plot_elements):
                window = min(rolling_window, len(x_values))
                
                # Calculate rolling statistics
                rolling_mean = np.zeros(len(x_values))
                rolling_std = np.zeros(len(x_values))
                
                # Calculate rolling statistics
                for i in range(len(x_values)):
                    if i < window - 1:
                        data_window = rewards[:i+1]
                    else:
                        data_window = rewards[i-window+1:i+1]
                    
                    rolling_mean[i] = np.mean(data_window)
                    rolling_std[i] = np.std(data_window)
                
                # Plot rolling mean if selected
                if 'mean' in plot_elements:
                    mean_line = ax.plot(x_values, rolling_mean,
                                    '--', color=color, alpha=alpha_base*1.5,
                                    linewidth=2.0)
                    lines.append(mean_line[0])
                    legend_labels.append(f"{label} (Rolling Mean)")
                    
                # Plot std bands if selected
                if 'std' in plot_elements:
                    std_fill = ax.fill_between(x_values,
                                            rolling_mean - rolling_std,
                                            rolling_mean + rolling_std,
                                            color=color, alpha=alpha_stats)
                    lines.append(std_fill)
                    legend_labels.append(f"{label} (±σ)")
        
        # Add training start indicator at the end (only for the first run that has the data)
        if 'training_start' in plot_elements:
            for metrics in metrics_list:
                if 'train' in metrics['history'] and len(metrics['history']['train']['actor']['losses']) > 0:
                    start_idx = len(metrics['history']['rollout']['episode_rewards']) - len(metrics['history']['train']['actor']['losses'])
                    if x_axis == 'step':
                        start_x = np.cumsum(metrics['history']['rollout']['episode_lengths'])[start_idx]
                    else:
                        start_x = start_idx
                    training_line = ax.axvline(x=start_x, color='gray', linestyle=':', alpha=0.7)
                    lines.append(training_line)
                    legend_labels.append('Training Initiation')
                    break  # Only add one training start line
        
        # Configure axes
        ax.set_ylabel('Reward')
        ax.set_title('Reward Comparison Across Training Runs')
        ax.grid(True, alpha=0.3)
        
        # Create legend at top left
        ax.legend(lines, legend_labels, loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            # Ensure the directory exists
            save_dir = self.base_save_dir / 'visualizations'
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Create full save path
            full_save_path = save_dir / save_path
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {full_save_path}")
            
        plt.close()

    def compare_runs(self, metrics_files: List[Path], labels: Optional[List[str]] = None,
                rolling_window: int = 50, x_axis: str = 'episode',
                plot_elements: Optional[List[str]] = None):
        """Generate rewards comparison
        
        Args:
            metrics_files: List of paths to metrics.json files
            labels: Optional list of labels for each run
            rolling_window: Window size for rolling statistics
            x_axis: Whether to plot against 'episode' or 'step'
            plot_elements: List of elements to plot. Options: ['raw', 'mean', 'std', 'training_start']
                        If None, all elements are plotted
        """
        
        # Load metrics
        metrics_list = self.load_metrics(metrics_files)
        
        # Generate plot
        self.plot_rewards_comparison(
            metrics_list,
            labels=labels,
            rolling_window=rolling_window,
            x_axis=x_axis,
            plot_elements=plot_elements,
            save_path="rewards_comparison.png"
        )
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(metrics_list, labels)
        
        # Save summary
        save_dir = self.base_save_dir / 'visualizations'
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "comparison_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
            
        return summary

    def _generate_summary_statistics(self, metrics_list: List[Dict],
                                   labels: Optional[List[str]] = None) -> Dict:
        """Generate summary statistics for comparison"""
        
        if labels is None:
            labels = [f"Run {i+1}" for i in range(len(metrics_list))]
            
        summary = {}
        
        for metrics, label in zip(metrics_list, labels):
            rewards = metrics['history']['rollout']['episode_rewards']
            
            # Calculate final performance (last 100 episodes)
            final_rewards = rewards[-100:]
            
            summary[label] = {
                "final_performance": {
                    "mean_reward": float(np.mean(final_rewards)),
                    "std_reward": float(np.std(final_rewards))
                },
                "overall_performance": {
                    "mean_reward": float(np.mean(rewards)),
                    "std_reward": float(np.std(rewards)),
                    "max_reward": float(np.max(rewards)),
                    "min_reward": float(np.min(rewards)),
                    "total_episodes": len(rewards)
                },
                "training_info": {
                    "environment": metrics['training_info']['environment'],
                    "total_timesteps": metrics['training_info']['total_timesteps'],
                    "time_elapsed": metrics['training_info']['time_elapsed']
                }
            }
            
        return summary