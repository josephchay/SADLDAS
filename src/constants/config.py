from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


STATS_ROLLING_WINDOW = 100


EXPLORATION_TYPES = [
    'SDLD',  # Spectral Decompositional Low Discrepancy noise
    'GA',  # Gaussian noise
    'OU',  # Ornstein-Uhlenbeck noise
    'LDAS'  # Low-discrepancy Action-Selection Policy
    'Uniform'  # Uniform Random Policy
]


@dataclass
class SeedConfig:
    """Configuration for random seeds"""

    torch_seed: int = 830143436
    numpy_seed: int = 167430301
    python_seed: int = 2193498338

    @classmethod
    def add_seed_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add seed arguments to parser"""
        parser.add_argument('--torch-seed', type=int, default=cls.torch_seed,
                        help='PyTorch random seed')
        parser.add_argument('--numpy-seed', type=int, default=cls.numpy_seed,
                        help='NumPy random seed')
        parser.add_argument('--python-seed', type=int, default=cls.python_seed,
                        help='Python random seed')
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'SeedConfig':
        """Create SeedConfig from parsed arguments"""
        return cls(
            torch_seed=args.torch_seed,
            numpy_seed=args.numpy_seed,
            python_seed=args.python_seed
        )
    
    @classmethod
    def remove_from_dict(cls, args_dict: dict) -> None:
        """Remove seed arguments from dictionary"""
        for seed_arg in ['torch_seed', 'numpy_seed', 'python_seed']:
            args_dict.pop(seed_arg, None)


@dataclass
class BaseConfig:
    """Base configuration with common parameters"""

    # Environment
    env_name: str = "LunarLanderContinuous-v3"

    # Network parameters
    hidden_dim: int = 256
    max_action: float = 1.0

    # Buffer parameters
    buffer_capacity: str = "full"  # Choices: ["short", "medium", "full"]
    fade_factor: float = 7.0
    stall_penalty: float = 0.07

    # Noise parameters
    burst: bool = False
    tr_noise: bool = True

    # Seeds
    seed_config: Optional[SeedConfig] = None

    @classmethod
    def add_common_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add common arguments to parser"""
        # Environment
        parser.add_argument('--env-name', type=str, default="LunarLanderContinuous-v3",
                        help='Environment name')
        
        # Network parameters
        parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden layer dimension')
        parser.add_argument('--max-action', type=float, default=1.0,
                        help='Maximum action value')
        
        # Buffer parameters
        parser.add_argument('--buffer-capacity', type=str, default="full",
                        choices=["short", "medium", "full"],
                        help='Replay buffer capacity')
        parser.add_argument('--fade-factor', type=float, default=7.0,
                        help='Buffer fade factor')
        parser.add_argument('--stall-penalty', type=float, default=0.07,
                        help='Stall penalty')
        
        # Noise parameters
        parser.add_argument('--burst', action='store_true',
                        help='Enable burst noise')
        parser.add_argument('--tr-noise', action='store_true',
                        help='Enable training noise')


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training parameters"""

    # Training process
    explore_time: int = 5000
    tr_between_ep_init: int = 15
    tr_between_ep_const: bool = False
    tr_per_step: int = 3
    num_episodes: int = 1_000
    angle_limit: float = 0.0

    # Episode limits
    limit_step: int = 2000
    limit_eval: int = 2000
    
    # Evaluation during training
    start_test: int = 250
    eval_episodes: int = 10

    # Multi-trail settings
    num_trials: int = 1
    version: Optional[int] = None

    def to_eval_config(self, save_dir: Path, render: bool = False) -> 'EvaluationConfig':
        """Convert TrainingConfig to EvaluationConfig"""
        return EvaluationConfig(
            env_name=self.env_name,
            hidden_dim=self.hidden_dim,
            max_action=self.max_action,
            buffer_capacity=self.buffer_capacity,
            fade_factor=self.fade_factor,
            stall_penalty=self.stall_penalty,
            burst=self.burst,
            tr_noise=self.tr_noise,
            model_dir=save_dir,
            model_version=self.version,
            num_episodes=self.eval_episodes,
            render=render,
            record_video=False,
            save_dir=save_dir,
            limit_eval=self.limit_eval,
            seed_config=self.seed_config
        )


@dataclass
class EvaluationConfig(BaseConfig):
    """Configuration for standalone evaluation"""
    # Model loading
    model_dir: Optional[Path] = None
    model_version: Optional[int] = None
    
    # Evaluation parameters
    num_episodes: int = 10
    render: bool = True
    record_video: bool = False
    save_dir: Optional[Path] = None
    
    # Episode limits
    limit_eval: int = 2000


@dataclass
class TuningConfig(BaseConfig):
    """Configuration for hyperparameter tuning"""
    # Tuning settings
    n_trials: int = 100
    n_startup_trials: int = 10
    study_name: Optional[str] = None
    storage: Optional[str] = None
    n_jobs: int = 1
    
    # Training limits for tuning
    max_episodes: int = 1000
    eval_episodes: int = 5
    
    # Parameter ranges for tuning
    hidden_dim_range: Tuple[int, int] = (32, 512)
    max_action_range: Tuple[float, float] = (0.1, 2.0)
    fade_factor_range: Tuple[float, float] = (1.0, 10.0)
    stall_penalty_range: Tuple[float, float] = (0.01, 0.2)
    tr_between_ep_init_range: Tuple[int, int] = (5, 100)
    explore_time_range: Tuple[int, int] = (1000, 10000)
    tr_per_step_range: Tuple[int, int] = (1, 10)
    
    # Base seed
    base_seed: Optional[int] = None
    
    def to_training_config(self) -> TrainingConfig:
        """Convert TuningConfig to TrainingConfig"""
        seed_config = None
        if self.base_seed is not None:
            seed_config = SeedConfig(
                torch_seed=self.base_seed,
                numpy_seed=self.base_seed + 1,
                python_seed=self.base_seed + 2
            )
            
        return TrainingConfig(
            env_name=self.env_name,
            hidden_dim=self.hidden_dim,
            max_action=self.max_action,
            buffer_capacity=self.buffer_capacity,
            fade_factor=self.fade_factor,
            stall_penalty=self.stall_penalty,
            burst=self.burst,
            tr_noise=self.tr_noise,
            num_episodes=self.max_episodes,
            eval_episodes=self.eval_episodes,
            seed_config=seed_config
        )


ENVIRONMENT_RECOMMENDATIONS: Dict[str, Dict[str, Any]] = {
    # You can update here if you want to have specific recommendations for environments :)

    'Pendulum-v1': {},
    'MountainCarContinuous-v0': {
        
    },
    'HalfCheetah-v5': {},
    'Walker2d-v5': {
        
    },
    'Humanoid-v5': {
        
    },
    'HumanoidStandup-v5': {
        
    },
    'Ant-v5': {
        
    },
    'BipedalWalker-v3': {
        
    },
    'BipedalWalkerHardcore-v3': {
        
    },
    'LunarLanderContinuous-v3': {
        
    },
    'Pusher-v5': {
        
    },
    'Swimmer-v5': {
        
    }
}


def get_environment_recommendations(env_name: str) -> Dict[str, Any]:
    """Get recommended configuration for a given environment"""

    return ENVIRONMENT_RECOMMENDATIONS.get(env_name, {})


def parse_training_args() -> TrainingConfig:
    """Parse command line arguments for training"""

    parser = argparse.ArgumentParser(description='SALDAS Training Arguments')
    
    BaseConfig.add_common_args(parser)
    SeedConfig.add_seed_args(parser)
    
    # Training process
    parser.add_argument('--explore-time', type=int, default=5000,
                       help='Exploration time before training')
    parser.add_argument('--tr-between-ep-init', type=int, default=15,
                       help='Initial training iterations between episodes')
    parser.add_argument('--tr-between-ep-const', action='store_true',
                       help='Keep training iterations between episodes constant')
    parser.add_argument('--tr-per-step', type=int, default=3,
                       help='Training iterations per step')
    parser.add_argument('--num-episodes', type=int, default=10_000_000,
                       help='Number of training episodes')
    parser.add_argument('--angle-limit', type=float, default=0.0,
                       help='Angle limit for specific environments')
    
    # Episode limits
    parser.add_argument('--limit-step', type=int, default=2000,
                       help='Maximum steps per episode')
    parser.add_argument('--limit-eval', type=int, default=2000,
                       help='Maximum steps per evaluation episode')

    # Evaluation parameters
    parser.add_argument('--start-test', type=int, default=250,
                       help='Start testing after N episodes')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    # Multi-trial settings
    parser.add_argument('--num-trials', type=int, default=1,
                       help='Number of training trials')
    parser.add_argument('--version', type=int, default=None,
                       help='Specific version to load')

    # Parse arguments
    args = parser.parse_args()
    training_args = vars(args)
    
    # Handle seeds
    seed_config = SeedConfig.from_args(args)
    SeedConfig.remove_from_dict(training_args)
    training_args['seed_config'] = seed_config
    
    return TrainingConfig(**training_args)


def parse_evaluate_args() -> EvaluationConfig:
    """Parse command line arguments for evaluation"""
    parser = argparse.ArgumentParser(description='SALDAS Evaluation Arguments')
    
    BaseConfig.add_common_args(parser)
    SeedConfig.add_seed_args(parser)
    
    # Model loading
    parser.add_argument('--model-dir', type=Path, required=True,
                       help='Directory containing saved model files')
    parser.add_argument('--model-version', type=int,
                       help='Specific model version to evaluate')
    
    # Evaluation parameters
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable environment rendering')
    parser.add_argument('--record-video', action='store_true',
                       help='Record evaluation videos')
    parser.add_argument('--save-dir', type=Path,
                       help='Directory to save evaluation results')
    
    # Episode limits
    parser.add_argument('--limit-eval', type=int, default=2000,
                       help='Maximum steps per evaluation episode')

    args = parser.parse_args()
    eval_args = vars(args)
    
    # Handle seeds
    seed_config = SeedConfig.from_args(args)
    SeedConfig.remove_from_dict(eval_args)
    eval_args['seed_config'] = seed_config
    
    # Handle render flag
    eval_args['render'] = not args.no_render
    del eval_args['no_render']
    
    return EvaluationConfig(**eval_args)


def parse_tuning_args() -> TuningConfig:
    """Parse command line arguments for tuning"""

    parser = argparse.ArgumentParser(description='SALDAS Hyperparameter Tuning Arguments')

    BaseConfig.add_common_args(parser)
    SeedConfig.add_seed_args(parser)

    # Tuning settings
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--n-startup-trials', type=int, default=10,
                       help='Number of random trials before optimization')
    parser.add_argument('--study-name', type=str, default=None,
                       help='Name for the optimization study')
    parser.add_argument('--storage', type=str, default=None,
                       help='Storage URL for the study')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs')

    # Training limits
    parser.add_argument('--max-episodes', type=int, default=1000,
                       help='Maximum episodes per trial')
    parser.add_argument('--eval-episodes', type=int, default=5,
                       help='Number of evaluation episodes per trial')
    
    # Parameter ranges for network
    parser.add_argument('--hidden-dim-min', type=int, default=32,
                       help='Minimum hidden dimension for tuning')
    parser.add_argument('--hidden-dim-max', type=int, default=512,
                       help='Maximum hidden dimension for tuning')
    parser.add_argument('--max-action-min', type=float, default=0.1,
                       help='Minimum max action for tuning')
    parser.add_argument('--max-action-max', type=float, default=2.0,
                       help='Maximum max action for tuning')
    
    # Parameter ranges for buffer
    parser.add_argument('--fade-factor-min', type=float, default=1.0,
                       help='Minimum fade factor for tuning')
    parser.add_argument('--fade-factor-max', type=float, default=10.0,
                       help='Maximum fade factor for tuning')
    parser.add_argument('--stall-penalty-min', type=float, default=0.01,
                       help='Minimum stall penalty for tuning')
    parser.add_argument('--stall-penalty-max', type=float, default=0.2,
                       help='Maximum stall penalty for tuning')
    
    # Parameter ranges for training process
    parser.add_argument('--tr-between-ep-init-min', type=int, default=5,
                       help='Minimum initial training iterations between episodes')
    parser.add_argument('--tr-between-ep-init-max', type=int, default=100,
                       help='Maximum initial training iterations between episodes')
    parser.add_argument('--explore-time-min', type=int, default=1000,
                       help='Minimum exploration time')
    parser.add_argument('--explore-time-max', type=int, default=10000,
                       help='Maximum exploration time')
    parser.add_argument('--tr-per-step-min', type=int, default=1,
                       help='Minimum training iterations per step')
    parser.add_argument('--tr-per-step-max', type=int, default=10,
                       help='Maximum training iterations per step')
    
    # Seeds
    parser.add_argument('--base-seed', type=int, default=None,
                       help='Base seed for trials')

    args = parser.parse_args()    
    tuning_args = vars(args)
    
    # Handle seeds
    seed_config = SeedConfig.from_args(args)
    SeedConfig.remove_from_dict(tuning_args)
    tuning_args['seed_config'] = seed_config

    # Create parameter ranges from min/max arguments
    param_ranges = {
        'hidden_dim_range': (args.hidden_dim_min, args.hidden_dim_max),
        'max_action_range': (args.max_action_min, args.max_action_max),
        'fade_factor_range': (args.fade_factor_min, args.fade_factor_max),
        'stall_penalty_range': (args.stall_penalty_min, args.stall_penalty_max),
        'tr_between_ep_init_range': (args.tr_between_ep_init_min, args.tr_between_ep_init_max),
        'explore_time_range': (args.explore_time_min, args.explore_time_max),
        'tr_per_step_range': (args.tr_per_step_min, args.tr_per_step_max),
    }
    
    # Update tuning_args with ranges
    tuning_args.update(param_ranges)
    
    # Remove min/max arguments
    for key in list(tuning_args.keys()):
        if key.endswith('_min') or key.endswith('_max'):
            del tuning_args[key]
    
    return TuningConfig(**tuning_args)


class EnvironmentFactory:
    """Factory for creating and managing environments"""

    @staticmethod
    def create_environment(env_name: str, render_mode: str = "rgb_array") -> gym.Env:
        """Create a Gymnasium environment"""

        env = gym.make(env_name, render_mode=render_mode)
        return env

    @staticmethod
    def setup_recording_environment(env: gym.Env, recordings_dir: str) -> gym.Env:
        """Setup environment for recording videos"""

        env = RecordEpisodeStatistics(env)
        env = RecordVideo(
            env,
            recordings_dir,
            episode_trigger=lambda _: True,
            name_prefix='recording',
            disable_logger=True,
        )

        return env
