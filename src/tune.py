from pathlib import Path
from src.constants import (
    parse_tuning_args,
    get_environment_recommendations,
)
from src.hyperparameter import Tuner


def main():
    """Main entry point for hyperparameter tuning"""
    # Parse arguments into TuningConfig
    tuning_config = parse_tuning_args()
    
    # Get environment recommendations
    env_recommendations = get_environment_recommendations(tuning_config.env_name)
    
    # Convert to base training config
    base_config = tuning_config.to_training_config()

    # Apply environment recommendations
    for key, value in env_recommendations.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
    
    # Create storage string for SQLite database if not provided
    if tuning_config.storage is None:
        storage_path = Path("tuning_storage")
        storage_path.mkdir(exist_ok=True)
        tuning_config.storage = f"sqlite:///{storage_path}/{tuning_config.study_name}.db"
    
    # Initialize and run tuner
    print(f"\nStarting hyperparameter optimization for {tuning_config.env_name}")
    print(f"Study name: {tuning_config.study_name}")
    print(f"Number of trials: {tuning_config.n_trials}")
    print(f"Storage: {tuning_config.storage}")
    print(f"Number of parallel jobs: {tuning_config.n_jobs}")
    print("\nParameter ranges:")
    print(f"Hidden dimension: {tuning_config.hidden_dim_range}")
    print(f"Max action: {tuning_config.max_action_range}")
    print(f"Fade factor: {tuning_config.fade_factor_range}")
    print(f"Stall penalty: {tuning_config.stall_penalty_range}")
    print(f"Training between episodes: {tuning_config.tr_between_ep_init_range}")
    print(f"Exploration time: {tuning_config.explore_time_range}")
    print(f"Training per step: {tuning_config.tr_per_step_range}")
    
    tuner = Tuner(
        base_config=base_config,
        tuning_config=tuning_config,
        n_trials=tuning_config.n_trials,
        n_startup_trials=tuning_config.n_startup_trials,
        study_name=tuning_config.study_name,
        storage_name=tuning_config.storage,
        n_jobs=tuning_config.n_jobs
    )
    
    try:
        tuner.tune()
        
        # Print best parameters
        print("\nOptimization completed!")
        print("\nBest parameters:")
        for key, value in tuner.study.best_params.items():
            print(f"{key}: {value}")
        print(f"\nBest value: {tuner.study.best_value:.2f}")
        
        # Save configuration
        config_path = tuner.results_dir / 'best_config.json'
        print(f"\nBest configuration saved to: {config_path}")
        
        # Print parameter importances
        print("\nParameter importance (top 5):")
        importances = tuner.get_param_importance()
        for param, importance in list(importances.items())[:5]:
            print(f"{param}: {importance:.3f}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        print("Partial results are still saved.")
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
    finally:
        if hasattr(tuner, 'results_dir'):
            print("\nResults saved to:", tuner.results_dir)


if __name__ == "__main__":
    main()
