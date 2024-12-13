from typing import List, Dict, Optional
import numpy as np
import torch
from pathlib import Path

from src.constants import EvaluationConfig, parse_evaluate_args, set_seeds, EnvironmentFactory
from src.stats import Logger
from src.network import SADLDAS
from src.memory import ReplayBuffer


class Evaluator:
    """Handles policy evaluation and logging with support for standalone usage"""

    def __init__(
        self,
        env_name: str,
        save_dir: Path,
        config: EvaluationConfig,
        logger: Optional[Logger] = None,
        eval_dir: Optional[Path] = None,
        load_model: bool = True,
    ):
        """
        Initialize evaluator with consistent directory structure.
        
        Args:
            env_name: Name of the Gymnasium environment
            save_dir: Base directory for all results (e.g., SALDAS_standalone_eval_logs)
            config: Evaluation configuration
            logger: Logger instance (optional)
            eval_dir: Specific evaluation directory (optional, used during training)
            load_model: Whether to load model files. (default: True)
        """

        self.env_name = env_name  # Store env_name as instance attribute
        self.save_dir = save_dir
        self.config = config
        
        # Set up environment
        self.env = EnvironmentFactory.create_environment(env_name, 'human') if config.render else EnvironmentFactory.create_environment(env_name)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        
        # Set up directory structure if not provided
        self.eval_dir = self._setup_directories(save_dir) if eval_dir is None else eval_dir

        # Set up components
        self._setup_components()
        
        # Initialize logger if not provided
        self.logger = logger or Logger(
            save_dir=self.eval_dir,
            env_name=self.env_name,
            model_version=config.model_version or 0
        )

        print(f"\nEvaluation results will be saved to: {self.eval_dir}")
        
        # Copy model files to evaluation directory
        if load_model and config.model_dir:
            self._load_and_copy_model(Path(config.model_dir))

    def _setup_components(self):
        """Initialize algorithm and replay buffer based on environment"""

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Configure max action
        if hasattr(self.env.action_space, 'high'):
            max_action = (self.config.network_config.max_action * 
                         torch.FloatTensor(self.env.action_space.high).to(self.device))
        else:
            max_action = (self.config.network_config.max_action * 
                         torch.ones(action_dim).to(self.device))

        # Initialize algorithm
        self.algo = SADLDAS(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.network_config.hidden_dim,
            device=self.device,
            max_action=max_action,
            burst=self.config.noise_config.burst,
            tr_noise=self.config.noise_config.tr_noise
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=self.config.buffer_config.buffer_capacity,
            device=self.device,
            fade_factor=self.config.buffer_config.fade_factor,
            stall_penalty=self.config.buffer_config.stall_penalty
        )

    def _setup_directories(self, base_dir: Path) -> Path:
        """Set up directory structure similar to training."""

        # Environment directory
        env_dir = base_dir / self.env_name
        env_dir.mkdir(exist_ok=True, parents=True)
        
        # Version directory
        version = self._get_next_version(env_dir)
        version_dir = env_dir / f'v{version}'
        version_dir.mkdir(exist_ok=True)
        
        # Evaluation directory
        eval_base_dir = version_dir / 'eval'
        eval_base_dir.mkdir(exist_ok=True)
        
        # Get next evaluation number
        existing_numbers = [
            int(d.name) 
            for d in eval_base_dir.glob('*')
            if d.is_dir() and d.name.isdigit()
        ]
        next_number = max(existing_numbers, default=0) + 1
        
        # Create numbered eval directory
        eval_dir = eval_base_dir / str(next_number)
        eval_dir.mkdir(exist_ok=True)
        
        return eval_dir

    def _get_next_version(self, env_dir: Path) -> int:
        """Get next version number for the environment directory."""
        existing_versions = [
            int(d.name[1:]) for d in env_dir.glob('v*')
            if d.is_dir() and d.name[1:].isdigit()
        ]
        return max(existing_versions, default=0) + 1

    def _load_and_copy_model(self, model_path: Path):
        """Load model and copy files to evaluation directory."""
        
        try:
            print(f"\nLoading models from path: {model_path}")
            # Find source model directory
            if (model_path / 'train/models/best').exists():
                source_dir = model_path / 'train/models/best'
                print("Using new directory structure (train/models/best)")
            elif (model_path / 'models/best').exists():
                source_dir = model_path / 'models/best'
                print("Using old directory structure (models/best)")
            else:
                raise FileNotFoundError(f"Could not find best model directory in {model_path}")
            
            # Create models directory in eval dir
            models_dir = self.eval_dir / 'models'
            models_dir.mkdir(exist_ok=True)
            print(f"Created models directory: {models_dir}")
            
            # Load and copy model files
            for model_file in ['actor_model.pt', 'critic_model.pt', 'critic_target_model.pt']:
                source_file = source_dir / model_file
                if not source_file.exists():
                    raise FileNotFoundError(f"Model file not found: {source_file}")
                print(f"Found model file: {source_file}")
                
                # Load model with weights_only=True
                if 'actor' in model_file:
                    self.algo.actor.load_state_dict(
                        torch.load(source_file, map_location=self.device, weights_only=True)
                    )
                elif 'critic_target' in model_file:
                    self.algo.critic_target.load_state_dict(
                        torch.load(source_file, map_location=self.device, weights_only=True)
                    )
                else:
                    self.algo.critic.load_state_dict(
                        torch.load(source_file, map_location=self.device, weights_only=True)
                    )
                
                # Copy file to eval directory
                import shutil
                dest_file = models_dir / model_file
                shutil.copy2(source_file, dest_file)
                print(f"Copied {model_file} to: {dest_file}")
            
            print(f"\nSuccessfully loaded and copied all models")
            print(f"Model files copied to: {models_dir}")
            print(f"Starting evaluation of {self.config.num_episodes} episodes...")
            print("-" * 50)
            
        except Exception as e:
            raise RuntimeError(f"Error loading/copying models: {str(e)}")

    def evaluate(self, episodes: int = 10, render: bool = True) -> List[float]:
        """
        Evaluate the current policy
        
        Args:
            episodes: Number of evaluation episodes
            render: Whether to render the environment
        
        Returns:
            List of episode returns
        """

        if episodes == 0:
            print("Skipping evaluation...")
            return []

        print(f"Running evaluation for {episodes} episodes...")
        episode_returns = []

        for episode in range(episodes):
            episode_reward, last_step = self.run_evaluation_episode(render=render)
            episode_returns.append(episode_reward)
            
            if self.logger:
                self.logger.log_eval_step(episode_reward, last_step)
            else:
                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {last_step}")

        # Print summary at the end
        summary = self.get_evaluation_summary(episode_returns)
        print("\nEvaluation Summary:")
        print(f"Mean Return: {summary['mean_return']:.2f} ± {summary['std_return']:.2f}")
        print(f"Min/Max Return: {summary['min_return']:.2f}/{summary['max_return']:.2f}")
        print(f"Episodes: {summary['num_episodes']}")

        return episode_returns

    def run_evaluation_episode(self, render: bool = True) -> tuple[float, int]:
        """
        Run a single evaluation episode
        
        Args:
            render: Whether to render the environment
        
        Returns:
            Tuple of (episode_reward, number_of_steps)
        """

        state = self.env.reset()[0]
        episode_reward = 0
        last_step = 0

        for step in range(1, self.config.limit_eval + 1):
            # Select action
            action = self.algo.select_action(
                state, self.replay_buffer, mean=True
            )
            
            # Take step in environment
            next_state, reward, done, info, _ = self.env.step(action)
            
            if render:
                self.env.render()
                
            episode_reward += reward
            state = next_state
            last_step = step
            
            if done:
                break

        return episode_reward, last_step

    @staticmethod
    def get_evaluation_summary(returns: List[float]) -> Dict:
        """
        Generate summary statistics for evaluation episode returns
        
        Args:
            returns: List of episode returns
            
        Returns:
            Dictionary containing summary statistics
        """
        if not returns:
            return {}

        return {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'min_return': float(np.min(returns)),
            'max_return': float(np.max(returns)),
            'num_episodes': len(returns)
        }

    def close(self):
        """Clean up resources"""
        if self.env:
            self.env.close()


def main():
    """Main function for standalone evaluation"""

    # Parse arguments
    config = parse_evaluate_args()
    
    # Set seeds if provided
    if config.seed_config:
        set_seeds(config.seed_config)
    
    # Create evaluator
    try:
        # For standalone evaluation, we always create a new version directory
        save_dir = Path("logs/standalone/evaluations") if config.save_dir is None else Path(config.save_dir)

        evaluator = Evaluator(
            env_name=config.env_name,
            save_dir=save_dir,
            config=config
        )
        
        # Run evaluation
        returns = evaluator.evaluate(
            episodes=config.num_episodes,
            render=config.render
        )
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    finally:
        if 'evaluator' in locals():
            evaluator.close()


if __name__ == "__main__":
    main()
