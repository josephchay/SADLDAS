import os
import copy
import json
import logging
import torch
import numpy as np
import pickle
import time
import math
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.constants import set_seeds, TrainingConfig, get_environment_recommendations, parse_training_args, EnvironmentFactory
from src.memory import ReplayBuffer
from src.network import SADLDAS
from src.evaluate import Evaluator
from src.stats import Logger, Visualizer

# Configure logging
logging.getLogger().setLevel(logging.CRITICAL)


class ModelManager:
    """
    Manages model versioning, checkpointing, and metrics tracking for the SADLDAS training process.
    
    This class handles:
    - Directory structure creation and management
    - Model saving and loading with versioning
    - Training metrics logging and visualization
    - Checkpoint management and best model tracking
    - Evaluation directory organization
    
    Attributes:
        base_dir (Path): Root directory for all experiment data
        env_dir (Path): Environment-specific directory
        version_dir (Path): Version-specific directory
        train_dir (Path): Training data directory
        model_dir (Path): Model storage directory
        checkpoints_dir (Path): Model checkpoints directory
        eval_dir (Path): Evaluation results directory
        best_reward (float): Best observed reward for model selection
        is_new_training (bool): Flag for new training runs
    """

    def __init__(self, env_name: str, save_dir: str = 'logs', checkpoint_frequency: int = 10):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Set up base directories
        self.base_dir = Path(os.path.join(project_root, save_dir))
        self.base_dir.mkdir(exist_ok=True)
        
        # Set up environment-specific directories
        self.env_dir = self.base_dir / env_name
        self.env_dir.mkdir(exist_ok=True)
        
        # Set up version-specific directories
        self.version = self._get_next_version()
        self.version_dir = self.env_dir / f'v{self.version}'
        self.version_dir.mkdir(exist_ok=True)

        # Set up training directories
        self.train_dir = self.version_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        
        self.model_dir = self.train_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)

        self.checkpoints_dir = self.model_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.best_dir = self.model_dir / 'best'
        self.best_dir.mkdir(exist_ok=True)

        self.recordings_dir = self.train_dir / 'recordings'
        self.recordings_dir.mkdir(exist_ok=True)

        # Set up evaluation directory (for evaluations during training)
        self.eval_dir = self.version_dir / 'eval'
        self.eval_dir.mkdir(exist_ok=True)

        self.eval_counter = self._get_next_eval_counter()
        self.current_eval_dir = self.eval_dir / f'{self.eval_counter}'
        self.current_eval_dir.mkdir(exist_ok=True)

        # Initialize metrics file and logger
        self.metrics_file = self.train_dir / 'training_metrics.txt'
        self._initialize_metrics_file()
        self.logger = Logger(self.train_dir, env_name, self.version)
        self.visualizer = Visualizer(self.train_dir, env_name)

        self.checkpoint_frequency = checkpoint_frequency

        # Track best model performance
        self.best_reward = float('-inf')

        print(f"Created new training version for {env_name}: v{self.version}")
        print(f"Training directory: {self.train_dir}")
        print(f"Evaluation directory: {self.current_eval_dir}")

        # Flag to track if this is a new training run
        self.is_new_training = True

    def _get_next_version(self) -> int:
        """
        Determines the next version number for experiment tracking.
        
        Scans existing version directories and increments the highest version number.
        Used to maintain separate versions of training runs for the same environment.
        
        Returns:
            int: Next available version number
        """

        existing_versions = [int(d.name[1:]) for d in self.env_dir.glob('v*')
                             if d.is_dir() and d.name[1:].isdigit()]
        return max(existing_versions, default=0) + 1

    def _get_next_eval_counter(self) -> int:
        """
        Manages evaluation numbering sequence.
        
        Tracks and increments evaluation run numbers within a training version.
        Used to organize multiple evaluation runs during training.
        
        Returns:
            int: Next available evaluation counter
        """

        existing_counters = [
            int(d.name)
            for d in self.eval_dir.glob('*')
            if d.is_dir() and d.name.isdigit()
        ]

        return max(existing_counters, default=0) + 1

    def increment_eval_counter(self) -> Path:
        """
        Creates new evaluation directory with incremented counter.
        
        Manages directory creation for new evaluation runs and maintains
        proper directory structure for evaluation results.
        
        Returns:
            Path: Path to the new evaluation directory
        """


        self.eval_counter += 1
        self.current_eval_dir = self.eval_dir / str(self.eval_counter)
        self.current_eval_dir.mkdir(exist_ok=True)

        return self.current_eval_dir

    def _initialize_metrics_file(self):
        """
        Sets up initial metrics logging file.
        
        Creates and initializes metrics file with:
        - Version information
        - Timestamp
        - Basic formatting
        Used for tracking training progress over time.
        """

        with open(self.metrics_file, 'w') as f:
            f.write("Training Metrics Log\n")
            f.write("===================\n")
            f.write(f"Version: v{self.version}\n")
            f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("===================\n\n")

    def _log_metrics(self, metrics: Dict, episode: int):
        """
        Records training metrics to log file.
        
        Args:
            metrics (Dict): Dictionary containing training metrics
            episode (int): Current episode number
            
        Logs:
        - Episode rewards (mean, max, min)
        - Episode steps
        - Buffer statistics
        - Timestamp
        """

        with open(self.metrics_file, 'a') as f:
            f.write(f"\nEpisode {episode} Metrics:\n")
            f.write("-----------------\n")

            # Log basic metrics
            if 'total_rewards' in metrics:
                recent_rewards = metrics['total_rewards'][-100:]
                f.write(f"Average Return (last 100): {np.mean(recent_rewards):.2f}\n")
                f.write(f"Max Return (last 100): {np.max(recent_rewards):.2f}\n")
                f.write(f"Min Return (last 100): {np.min(recent_rewards):.2f}\n")

            if 'total_steps' in metrics:
                recent_steps = metrics['total_steps'][-100:]
                f.write(f"Average Steps (last 100): {np.mean(recent_steps):.2f}\n")

            # Additional training statistics
            f.write(f"Replay Buffer Size: {metrics.get('buffer_size', 'N/A')}\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def save_models(self, algo: SADLDAS, replay_buffer: ReplayBuffer, metrics: Dict, episode: int):
        """
        Handles model checkpoint and best model saving.
        
        Args:
            algo (SADLDAS): Algorithm instance containing models
            replay_buffer (ReplayBuffer): Current replay buffer
            metrics (Dict): Training metrics
            episode (int): Current episode number
            
        Saves:
        - Regular checkpoints at specified frequency
        - Best models based on performance
        - Replay buffer state
        - Associated metrics and metadata
        """

        # Only save periodic checkpoints
        is_checkpoint_episode = episode % self.checkpoint_frequency == 0
        
        # Calculate current performance
        current_avg_reward = (np.mean(metrics['total_rewards'][-100:])
                            if len(metrics['total_rewards']) >= 100
                            else np.mean(metrics['total_rewards']))
        
        is_best_model = current_avg_reward > self.best_reward
        
        # Save checkpoint if it's either periodic or best performance
        if is_checkpoint_episode or is_best_model:
            episode_checkpoint_dir = self.checkpoints_dir / f'eps_{episode}'
            episode_checkpoint_dir.mkdir(exist_ok=True)

            # Save current model state
            torch.save(algo.actor.state_dict(), episode_checkpoint_dir / 'actor_model.pt')
            torch.save(algo.critic.state_dict(), episode_checkpoint_dir / 'critic_model.pt')
            torch.save(algo.critic_target.state_dict(), episode_checkpoint_dir / 'critic_target_model.pt')

            # Save replay buffer and additional data
            buffer_data = {
                'buffer': replay_buffer,
                'x_coor': algo.actor.exploration.x_coor,
                'total_rewards': metrics['total_rewards'][-1000:],  # Keep only recent history
                'total_steps': metrics['total_steps'][-1000:],
                'average_steps': metrics.get('average_steps', 0),
                'version': self.version,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            if is_best_model:
                # Save full replay buffer only for best model
                buffer_data['buffer'] = replay_buffer
                self.best_reward = current_avg_reward
                print(f"\n>>>>> New best model saved! Mean Reward: {self.best_reward:.2f} <<<<<")
                
                # Save best model
                torch.save(algo.actor.state_dict(), self.best_dir / 'actor_model.pt')
                torch.save(algo.critic.state_dict(), self.best_dir / 'critic_model.pt')
                torch.save(algo.critic_target.state_dict(), self.best_dir / 'critic_target_model.pt')

                # Save best replay buffer
                with open(self.best_dir / 'replay_buffer', 'wb') as file:
                    pickle.dump(buffer_data, file)

            with open(episode_checkpoint_dir / f'replay_buffer_{episode}', 'wb') as file:
                pickle.dump(buffer_data, file)

            # Update metrics log
            metrics['buffer_size'] = len(replay_buffer)
            self._log_metrics(metrics, episode)

    def load_models(self, algo: SADLDAS, device: torch.device, version: Optional[int] = None) -> Tuple[Optional[ReplayBuffer], Dict]:
        """
        Loads saved models and associated data.
        
        Args:
            algo (SADLDAS): Algorithm instance to load models into
            device (torch.device): Target device for loaded models
            version (Optional[int]): Specific version to load, uses latest if None
            
        Returns:
            Tuple containing:
            - ReplayBuffer: Loaded replay buffer or None
            - Dict: Loaded metrics dictionary
            
        Handles:
        - Model state loading
        - Replay buffer restoration
        - Metrics recovery
        - Version management
        - Error handling and fallbacks
        """

        metrics = {
            'total_rewards': [],
            'total_steps': [],
            'average_steps': 0
        }
        replay_buffer = None

        # For first run of a new experiment version, don't load any models
        if self.is_new_training:
            print("Starting new training experiment run...")
            return replay_buffer, metrics

        # Determine which version to load
        if version is not None:
            load_dir = self.env_dir / f'v{version}'
            if not load_dir.exists():
                print(f"Version v{version} not found. Starting new training expriment run...")
                return replay_buffer, metrics
        else:
            # Find latest version
            versions = [int(d.name[1:]) for d in self.env_dir.glob('v*')
                        if d.is_dir() and d.name[1:].isdigit()]
            if versions:
                version = max(versions)
                load_dir = self.env_dir / f'v{version}'

        if not load_dir:
            return replay_buffer, metrics

        try:
            # Check for new directory structure
            model_dir = load_dir / 'train/models/best'
            if not model_dir.exists():
                # Try old directory structure
                model_dir = load_dir / 'models/best'
                if not model_dir.exists():
                    print("No saved models found. Starting new training run...")
                    return replay_buffer, metrics

            print(f"Loading models from version v{version}...")
            
            # Load model states
            algo.actor.load_state_dict(torch.load(load_dir / 'models/best/actor_model.pt'))
            algo.critic.load_state_dict(torch.load(load_dir / 'models/best/critic_model.pt'))
            algo.critic_target.load_state_dict(torch.load(load_dir / 'models/best/critic_target_model.pt'))

            # Load replay buffer and metrics
            buffer_path = model_dir / 'replay_buffer'
            if buffer_path.exists():
                with open(buffer_path, 'rb') as file:
                    data = pickle.load(file)
                    replay_buffer = data['buffer']
                    algo.actor.exploration.x_coor = data['x_coor']
                    metrics['total_rewards'] = data['total_rewards']
                    metrics['total_steps'] = data['total_steps']
                    metrics['average_steps'] = data.get('average_steps', 0)

            print(f"Successfully loaded models from version v{version}")

        except Exception as e:
            print(f"Error loading models: {e}")
            print("Starting new training run...")
            return replay_buffer, metrics

        return replay_buffer, metrics


class Trainer:
    """
    Primary training orchestrator for SADLDAS (Spectral Analysis-Driven Low Discrepancy Action Selection).
    
    This class manages the complete training lifecycle including:
    - Environment interaction
    - Model initialization and management
    - Training loop execution
    - Metrics tracking and visualization
    - Policy evaluation
    
    Attributes:
        config (TrainingConfig): Configuration parameters for training
        device (torch.device): Device for computation (CPU/GPU)
        env (gym.Env): Training environment instance
        model_manager (ModelManager): Handles model saving/loading and versioning
        algo (SADLDAS): Main algorithm instance
        replay_buffer (ReplayBuffer): Experience replay buffer with fade factor
        learn (bool): Flag indicating if learning phase has started
        evaluator (Evaluator): Handles policy evaluation during training
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create environments
        self.env = EnvironmentFactory.create_environment(config.env_name)

        # Apply any environment-specific recommendations
        recommended_config = get_environment_recommendations(config.env_name)
        for key, value in recommended_config.items():
            if not hasattr(self.config, key) or getattr(self.config, key) is None:
                setattr(self.config, key, value)

        # Initialize components
        self.model_manager = ModelManager(self.config.env_name)
        self.setup_components()
        self.env = EnvironmentFactory.setup_recording_environment(
            self.env,
            str(self.model_manager.recordings_dir)
        )

        self.learn = False
        self.evaluator = None

    def setup_components(self):
        """Initialize algorithm and replay buffer"""

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.config.max_action = (self.config.max_action *
                      torch.FloatTensor(self.env.action_space.high).to(self.device)
                      if self.env.action_space.is_bounded()
                      else self.config.max_action * 1.0)

        self.algo = SADLDAS(
            state_dim=state_dim,
            action_dim=action_dim, 
            hidden_dim=self.config.hidden_dim,
            device=self.device, 
            max_action=self.config.max_action,
            exploration_type='GA',
            exploration_kwargs={
                'max_action': self.config.max_action,
                'burst': self.config.burst,
                'tr_noise': self.config.tr_noise,
            }
        )

        self.replay_buffer = ReplayBuffer(
            state_dim, 
            action_dim, 
            self.config.buffer_capacity,
            self.device, 
            self.config.fade_factor, 
            self.config.stall_penalty
        )

        self.metrics = {'total_rewards': [], 'total_steps': []}

    def train(self):
        """
        Executes the main training loop.
        
        The training process consists of:
        1. Loading any existing models and buffers
        2. Collecting experience through environment interaction
        3. Training the policy when sufficient data is collected
        4. Regular evaluation and model checkpointing
        5. Metrics logging and visualization
        
        Handles both exploration and exploitation phases, with smooth transitions
        between them guided by spectral analysis.
        """

        try:
            saved_buffer, saved_metrics = self.model_manager.load_models(
                self.algo, 
                self.device,
                self.config.version,
            )
            if saved_buffer is not None:
                self.replay_buffer = saved_buffer
                self.metrics = saved_metrics
                start_episode = len(self.metrics['total_steps'])
            else:
                start_episode = 0

            self.learn = len(self.replay_buffer) >= self.config.explore_time

            # Create evaluation config from training config
            eval_config = self.config.to_eval_config(
                save_dir=self.model_manager.version_dir,
                render=False  # No rendering during training
            )

            for episode in range(start_episode, self.config.num_episodes):
                try:
                    metrics = self.run_episode(episode)
                    self.update_metrics(metrics)

                    self.model_manager.logger.update_buffer_metrics(self.replay_buffer)
                    self.model_manager.logger.update_noise_metrics(self.algo.actor.exploration)

                    if self.learn:
                        self.model_manager.save_models(
                            self.algo,
                            self.replay_buffer,
                            self.metrics, 
                            episode,
                        )

                        if episode >= self.config.start_test and episode % 50 == 0:
                            eval_config.model_dir = self.model_manager.version_dir

                            try:
                                # Update evaluator with new directory
                                self.evaluator = Evaluator(
                                    env_name=self.config.env_name,
                                    save_dir=self.model_manager.version_dir,
                                    config=eval_config,
                                    logger=self.model_manager.logger,
                                    eval_dir=self.model_manager.increment_eval_counter(),
                                )

                                eval_returns = self.evaluator.evaluate(self.config.eval_episodes)
                                eval_summary = self.evaluator.get_evaluation_summary(eval_returns)
                                
                                # Save evaluation results
                                eval_results = {
                                    'episode': episode,
                                    'returns': eval_returns,
                                    'summary': eval_summary
                                }
                                
                                # Save evaluation results to the current eval directory
                                eval_results_path = self.evaluator.eval_dir / 'eval_results.json'
                                with open(eval_results_path, 'w') as f:
                                    json.dump(eval_results, f, indent=4)
                            
                            except Exception as eval_error:
                                print(f"Error during evaluation: {eval_error}")
                                print("Continuing training despite evaluation error")

                except Exception as e:
                    print(f"Error during episode {episode}: {e}")
                    print("Attempting to continue with next episode")
                    
        except Exception as e:
            print(f"Critical error in training loop: {e}")

    def run_episode(self, episode: int) -> Dict:
        """
        Executes a single training episode.
        
        Args:
            episode (int): Current episode number
            
        Returns:
            Dict containing episode metrics:
                - total_reward: Cumulative episode reward
                - episode_steps: Number of steps in episode
                
        The episode execution includes:
        1. Environment reset
        2. Initial action warmup
        3. Step-by-step interaction with environment
        4. Training updates based on collected experience
        5. Metrics collection and logging
        """

        # initialize environment
        state = self.env.reset()[0]

        # Pre-processing steps
        rb_len = len(self.replay_buffer)
        tr_between_ep = self.get_training_frequency(rb_len)

        rewards = []
        # Initial random actions
        action = self.get_initial_action()

        for _ in range(2):
            next_state, reward, done, info, _ = self.env.step(action)
            rewards.append(reward)
            state = next_state

        # Training
        training_info = self.train_between_episodes(tr_between_ep) if self.learn else {}

        episode_steps = 0
        for steps in range(1, self.config.limit_step + 1):
            episode_steps += 1

            # Check if we should start learning
            if len(self.replay_buffer) >= self.config.explore_time and not self.learn:
                self.learn = self.start_learning()

            # Take action and update
            metrics = self.step(state)
            rewards.extend(metrics['rewards'])
            state = metrics['next_state']

            if metrics['done']:
                break

        episode_reward = sum(rewards)
        self.model_manager.logger.log_training_step(episode, episode_reward, episode_steps, training_info)
        self.model_manager.visualizer.plot_training_metrics(self.model_manager.logger.get_metrics())

        return {
            'total_reward': episode_reward,
            'episode_steps': episode_steps
        }

    def step(self, state: np.ndarray) -> Dict:
        """
        Executes a single environment step with training.
        
        Args:
            state (np.ndarray): Current environment state
            
        Returns:
            Dict containing step information:
                - next_state: Resulting state after action
                - rewards: List of rewards received
                - training_info: Training metrics if learning is active
                - done: Episode termination flag
                
        Performs:
        1. Action selection using spectral analysis
        2. Environment interaction
        3. Reward adjustment
        4. Experience storage
        5. Training updates if in learning phase
        """

        action = self.algo.select_action(state, self.replay_buffer)
        next_state, reward, done, info, _ = self.env.step(action)

        # Environment-specific reward adjustments
        reward = self.adjust_reward(next_state, reward, done)

        self.replay_buffer.add(state, action, reward + 1.0, next_state, done)

        training_info = self.train_steps() if self.learn else {}

        return {
            'next_state': next_state,
            'rewards': [reward],
            'training_info': training_info,
            'done': done,
        }

    def adjust_reward(self, next_state: np.ndarray, reward: float, done: bool) -> float:
        """
        Applies environment-specific reward adjustments.
        
        Args:
            next_state (np.ndarray): Resulting state after action
            reward (float): Original environment reward
            done (bool): Episode termination flag
            
        Returns:
            float: Adjusted reward value
            
        Modifications include:
        1. Ant environment angle penalties
        2. Humanoid forward progress rewards
        3. LunarLander crash penalty reduction
        4. BipedalWalker failure penalty adjustment
        """

        env_id = self.env.spec.id

        if "Ant" in env_id:
            if next_state[1] < getattr(self.config, 'angle_limit', 0.4):
                done = True
            if next_state[1] > 1e-3:
                reward += math.log(next_state[1])
        elif "Humanoid-" in env_id:
            reward += next_state[0]
        elif "LunarLander" in env_id:
            if reward == -100.0:
                reward = -50.0
        elif "BipedalWalkerHardcore" in env_id:
            if reward == -100.0:
                reward = -25.0

        return reward

    def train_steps(self):
        """
        Performs multiple training iterations using sampled experience.
        
        Returns:
            Dict containing averaged training metrics
            
        Process:
        1. Samples batches from replay buffer
        2. Updates policy and value networks
        3. Tracks and averages training metrics
        4. Applies spectral regularization
        """

        training_info = {}

        for _ in range(self.config.tr_per_step):
            step_info = self.algo.train(self.replay_buffer.sample())
            # Update or average the training metrics
            for key, value in step_info.items():
                if key in training_info:
                    training_info[key] = 0.5 * (training_info[key] + value)  # Running average
                else:
                    training_info[key] = value

        return training_info

    def start_learning(self) -> bool:
        """
        Initializes the learning phase of training.
        
        Returns:
            bool: True if learning initialization successful
            
        Steps:
        1. Computes state normalization statistics
        2. Performs initial uniform sampling training
        3. Transitions to prioritized sampling
        4. Sets up spectral analysis components
        """

        self.replay_buffer.find_min_max()
        print("\n>>>>> Initiated Training (Policy Learning) <<<<<")

        # Initial training info dictionaries
        uniform_info = {}
        nonuniform_info = {}

        # Initial training with uniform sampling
        for _ in range(64):
            step_info = self.algo.train(self.replay_buffer.sample(uniform=True))
            # Update uniform training metrics
            for key, value in step_info.items():
                if key in uniform_info:
                    uniform_info[key] = 0.5 * (uniform_info[key] + value)
                else:
                    uniform_info[key] = value

        for _ in range(64):
            step_info = self.algo.train(self.replay_buffer.sample())
            # Update non-uniform training metrics
            for key, value in step_info.items():
                if key in nonuniform_info:
                    nonuniform_info[key] = 0.5 * (nonuniform_info[key] + value)
                else:
                    nonuniform_info[key] = value

        return True

    def train_between_episodes(self, tr_between_ep: int):
        """
        Conducts training updates between episodes.
        
        Args:
            tr_between_ep (int): Number of training iterations to perform
            
        Returns:
            Dict containing averaged training metrics
            
        Performs multiple training iterations with:
        1. Experience sampling using fade factor
        2. Policy and value network updates
        3. Metrics averaging and tracking
        """

        training_info = {}

        for _ in range(tr_between_ep):
            step_info = self.algo.train(self.replay_buffer.sample())
            # Update or average the training metrics
            for key, value in step_info.items():
                if key in training_info:
                    training_info[key] = 0.5 * (training_info[key] + value)
                else:
                    training_info[key] = value
        return training_info

    def get_initial_action(self) -> np.ndarray:
        """Get initial random action for environment warmup"""
        action = 0.3 * self.config.max_action.to('cpu').numpy() * np.random.uniform(-1.0, 1.0, size=self.env.action_space.shape[0])
        return action

    def get_training_frequency(self, rb_len: int) -> int:
        """Calculate training frequency based on replay buffer length"""
        rb_len_threshold = 5000 * self.config.tr_between_ep_init
        tr_between_ep = self.config.tr_between_ep_init

        if not self.config.tr_between_ep_const:
            if self.config.tr_between_ep_init >= 100 and rb_len >= 350000:
                tr_between_ep = rb_len // 5000
            elif self.config.tr_between_ep_init < 100 and rb_len >= rb_len_threshold:
                tr_between_ep = rb_len // 5000

        return tr_between_ep

    def update_metrics(self, metrics: Dict):
        """Update training metrics and generate plots"""
        self.metrics['total_rewards'].append(metrics['total_reward'])
        self.metrics['total_steps'].append(metrics['episode_steps'])

        # Calculate averages
        average_reward = np.mean(self.metrics['total_rewards'][-100:])
        average_steps = np.mean(self.metrics['total_steps'][-100:])


def run_single_trial(config: TrainingConfig, trial_num: int) -> Dict:
    """Run a single training trial"""

    print(f"\nStarting Trial {trial_num + 1}/{config.num_trials}")

    # Create trial-specific seed config by offsetting the base seeds
    if config.seed_config:
        trial_seed_config = copy.deepcopy(config.seed_config)
        trial_seed_config.torch_seed += trial_num
        trial_seed_config.numpy_seed += trial_num
        trial_seed_config.python_seed += trial_num
        set_seeds(trial_seed_config)

    # Start training
    trainer = Trainer(config)
    trainer.train()

    # Return trial metrics
    return copy.deepcopy(trainer.model_manager.logger.get_metrics())


def main():
    """Entry point for training"""

    # Parse command line arguments
    config = parse_training_args()

    # Set base seeds if provided
    if config.seed_config:
        set_seeds(config.seed_config)

    for trial in range(config.num_trials):
        trial_data = run_single_trial(config, trial)


if __name__ == "__main__":
    main()
