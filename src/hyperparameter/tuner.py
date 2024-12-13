from typing import Dict, Any, Optional
import optuna
import numpy as np
from pathlib import Path
import json
import datetime
import copy

from src.constants import TrainingConfig, TuningConfig, SeedConfig
from src.train import Trainer


class Tuner:
    """Hyperparameter tuning framework using Optuna."""

    def __init__(
        self,
        base_config: TrainingConfig,
        tuning_config: TuningConfig,
        n_trials: int = 100,
        n_startup_trials: int = 10,
        study_name: Optional[str] = None,
        storage_name: Optional[str] = None,
        n_jobs: int = 1
    ):
        """
        Initialize tuner with configurations.
        
        Args:
            base_config: Base training configuration
            tuning_config: Tuning-specific configuration
            n_trials: Number of optimization trials
            n_startup_trials: Number of random trials before optimization
            study_name: Name for the optimization study
            storage_name: Storage URL for the study
            n_jobs: Number of parallel jobs
        """

        self.base_config = base_config
        self.tuning_config = tuning_config
        self.n_trials = n_trials
        self.study_name = study_name if study_name else f"{base_config.env_name}_optimization"
        self.n_jobs = n_jobs

        # Setup results directory
        self.results_dir = Path("tuning_results") / self.base_config.env_name
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Create study with improved sampler settings
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=n_startup_trials,
                multivariate=True,
                constant_liar=True
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=100
            )
        )

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters using ranges from tuning config."""
        params = {}
        
        # Network parameters (log scale for capacity-related parameters)
        params['hidden_dim'] = trial.suggest_int('hidden_dim', 
            *self.tuning_config.hidden_dim_range, log=True)
        params['max_action'] = trial.suggest_float('max_action', 
            *self.tuning_config.max_action_range)
            
        # Buffer parameters
        params['fade_factor'] = trial.suggest_float('fade_factor', 
            *self.tuning_config.fade_factor_range)
        params['stall_penalty'] = trial.suggest_float('stall_penalty', 
            *self.tuning_config.stall_penalty_range, log=True)
            
        # Training process parameters
        params['tr_between_ep_init'] = trial.suggest_int('tr_between_ep_init',
            *self.tuning_config.tr_between_ep_init_range)
        params['explore_time'] = trial.suggest_int('explore_time',
            *self.tuning_config.explore_time_range, log=True)
        params['tr_per_step'] = trial.suggest_int('tr_per_step',
            *self.tuning_config.tr_per_step_range)
            
        # Categorical parameters without probabilities
        params['buffer_capacity'] = trial.suggest_categorical('buffer_capacity',
            ["short", "medium", "full"])
        params['burst'] = trial.suggest_categorical('burst', [True, False])
        params['tr_noise'] = trial.suggest_categorical('tr_noise', [True, False])

        return params

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Objective value (mean reward - std penalty)
        """

        # Get parameter suggestions
        params = self.suggest_params(trial)

        # Create trial config
        trial_config = copy.deepcopy(self.base_config)
        for key, value in params.items():
            setattr(trial_config, key, value)

        # Set unique seeds for this trial
        if self.tuning_config.base_seed is not None:
            trial_seed = self.tuning_config.base_seed + trial.number
            trial_config.seed_config = SeedConfig(
                torch_seed=trial_seed,
                numpy_seed=trial_seed + 1000,
                python_seed=trial_seed + 2000
            )

        try:
            # Run training with early stopping checks
            trainer = Trainer(trial_config)
            best_reward = float('-inf')
            stagnant_steps = 0
            
            for episode in range(trial_config.num_episodes):
                metrics = trainer.model_manager.logger.get_metrics()
                
                if len(metrics['rollout']['episode_rewards']) >= 100:
                    current_reward = np.mean(metrics['rollout']['episode_rewards'][-100:])
                    
                    # Report intermediate value for pruning
                    trial.report(current_reward, episode)
                    
                    # Handle pruning
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    
                    # Check for improvement
                    if current_reward > best_reward:
                        best_reward = current_reward
                        stagnant_steps = 0
                    else:
                        stagnant_steps += 1
                    
                    # Early stopping
                    if stagnant_steps >= 50:  # No improvement for 50 episodes
                        break

            # Calculate final metrics
            final_metrics = trainer.model_manager.logger.get_metrics()
            rewards = final_metrics['rollout']['episode_rewards'][-100:]
            
            # Compute objective value considering both performance and stability
            mean_reward = np.mean(rewards)
            reward_std = np.std(rewards)
            objective_value = mean_reward - 0.1 * reward_std  # Penalize instability
            
            # Log comprehensive metrics
            trial.set_user_attr('mean_reward', float(mean_reward))
            trial.set_user_attr('std_reward', float(reward_std))
            trial.set_user_attr('max_reward', float(np.max(rewards)))
            trial.set_user_attr('min_reward', float(np.min(rewards)))
            trial.set_user_attr('final_episodes', len(final_metrics['rollout']['episode_rewards']))
            
            return objective_value

        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    def tune(self):
        """Run hyperparameter tuning with improved logging and visualization."""

        print(f"\nStarting hyperparameter tuning for {self.base_config.env_name}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Results will be saved to: {self.results_dir}\n")

        # Optimize with parallel jobs if specified
        optimize_kwargs = {
            'n_trials': self.n_trials,
            'callbacks': [self._save_tuning_state],
            'gc_after_trial': True
        }
        
        if self.n_jobs > 1:
            optimize_kwargs['n_jobs'] = self.n_jobs

        self.study.optimize(
            self.objective,
            **optimize_kwargs
        )

        # Save final results and visualizations
        self._save_results()
        self._create_visualizations()

    def _save_tuning_state(self, study: optuna.Study, trial: optuna.Trial):
        """
        Save detailed intermediate results.
        
        Args:
            study: Optuna study object
            trial: Current trial object
        """

        state = {
            "study_name": study.study_name,
            "environment": self.base_config.env_name,
            "n_trials": len(study.trials),
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number,
            "current_trial": trial.number,
            "current_params": trial.params,
            "current_value": trial.value if trial.value is not None else None,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.results_dir / "tuning_state.json", "w") as f:
            json.dump(state, f, indent=4)

    def _save_results(self):
        """Save comprehensive final results."""

        results = {
            "environment": self.base_config.env_name,
            "study_name": self.study.study_name,
            "n_trials": len(self.study.trials),
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "param_importance": self.get_param_importance(),
            "all_trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": t.value,
                    "user_attrs": t.user_attrs,
                    "state": t.state,
                    "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
                    "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
                }
                for t in self.study.trials if t.value is not None
            ]
        }

        # Save results
        with open(self.results_dir / "final_results.json", "w") as f:
            json.dump(results, f, indent=4)

        # Save best config
        best_config = copy.deepcopy(self.base_config)
        for key, value in self.study.best_params.items():
            setattr(best_config, key, value)

        with open(self.results_dir / "best_config.json", "w") as f:
            json.dump(best_config.__dict__, f, indent=4)

    def _create_visualizations(self):
        """Create optimization visualizations using plotly."""

        try:
            # Parameter importance plot
            importance_fig = optuna.visualization.plot_param_importances(self.study)
            importance_fig.write_html(str(self.results_dir / "param_importance.html"))

            # Optimization history plot
            history_fig = optuna.visualization.plot_optimization_history(self.study)
            history_fig.write_html(str(self.results_dir / "optimization_history.html"))

            # Parameter relationships plot
            parallel_fig = optuna.visualization.plot_parallel_coordinate(self.study)
            parallel_fig.write_html(str(self.results_dir / "param_relationships.html"))

            # Slice plot for most important parameters
            slice_fig = optuna.visualization.plot_slice(self.study)
            slice_fig.write_html(str(self.results_dir / "param_slices.html"))

        except Exception as e:
            print(f"Error creating visualizations: {e}")

    def get_param_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance.
        
        Returns:
            Dictionary mapping parameter names to their importance scores
        """

        try:
            return optuna.importance.get_param_importances(self.study)
        except Exception:
            return {}

    def get_best_trial(self) -> Optional[optuna.Trial]:
        """
        Get the best trial.
        
        Returns:
            Best trial object or None if no trials completed
        """

        return self.study.best_trial if len(self.study.trials) > 0 else None

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters.
        
        Returns:
            Dictionary of best parameters
        """

        return self.study.best_params if len(self.study.trials) > 0 else {}
