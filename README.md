# Stable Adaptive Decompositional Low-Discrepancy Action Selection (SADLDAS) for Reinforcement Learning 

#### Official Implementation

---

## About

Exploration in learned policies is crucial for its ability to generalize the knowledge of reinforcement learning (RL) agents effectively across different environments. Efficient state-action space coverage and exploration-exploitation trade-off still remains a challenge in RL, particularly for continuous action spaces.

We present to you SADLDAS, where our core contributions include: (1) A novel noise injection mechanism that outperforms existing noise and strategy techniques. (2) An off-policy architecture utilizing three Q-networks with S2 variance estimation, to maintain stability. (3) Integration of DDPG, and TD3 for performance validation. (4) Integration of SpectralBAN RL neural network layer with spectral approximation accompanied with sine transformation and leaky activations. (5) Novel rectified symmetrical and asymmetrical Smooth Mean Absolute (SMA) error loss functions for regression.

This codebase is the official implementation of the presented paper.

Regarding comments, in order to avoid redundancy or unorganized / unreadble code, only crucial information and uncommon, new, novel implementations found within the codebase are guided with codeblocks and comments. Rest assured those that are not commented are indeed more self-explanatory. You will find core logic and theoretical components that we have contributed well documented with comments.

## Development

### Prerequisites

Programming Languages
- [Python](https://www.python.org/)-3.12.7 - Download [here](https://www.python.org/downloads/)

Python Interpreter
- [Anaconda](https://www.anaconda.com/) - Download [here](https://www.anaconda.com/download) 

### Setting up the repository

```bash
git clone https://github.com/josephchay/SADLDAS.git
cd SALDAS
```

### Setup the project

```bash
pip install -e .
```

### Train an agent

#### Basic Training

To train the LunarLanderContinuous-v3 agent with default settings, run the following command:

```bash
python src/train.py
```

or as recommended
    
```bash
train
```
Check out `src/constants/config.py` for the default configurations and recommended configurations tailored for each environment.

##### Train with a specific environment

```bash
train --env_name <env_name>
```

##### Training Parameters

###### Core Configurations

Check out `src/constants/config.py` for all configuration arguments.

```bash
# Network Configuration
--hidden-dim 256            # Hidden layer dimension
--max-action 1.0            # Maximum action value

# Training Control
--explore-time 5000         # Initial exploration steps
--num-episodes 1_000        # Total training episodes
--tr-per-step 3             # Training iterations per step
--tr-between-ep-init 15     # Initial training iterations between episodes
--tr-between-ep-const       # Keep training iterations constant

# Episode Limits
--limit-step 2000           # Maximum steps per episode
--limit-eval 2000           # Maximum steps per evaluation episode

# Buffer Configuration
--buffer-capacity full      # Buffer capacity (options: short, medium, full)
--fade-factor 7.0           # Buffer fade factor
--stall-penalty 0.07        # Stall penalty

# Noise Settings
--burst                     # Enable burst noise
--tr-noise                  # Enable training noise

# Evaluation
--start-test 250            # Start testing after N episodes
--eval-episodes 10          # Number of evaluation episodes

# Multi-trial Training
--num-trials 1              # Number of training trials
```

#### Example Commands

Training the LunarLanderContinuous-v3 with default settings:

```bash
train --env-name LunarLanderContinuous-v3
```

Training the BipedalWalker with recommended settings:

```bash
train \
    --env-name BipedalWalker-v3 \
    --tr-between-ep-init 40 \
    --burst \
    --tr-noise False \
    --limit-step 1000000
```

Multiple training trials with custom evaluation:

```bash
train \
    --num-trials 5 \
    --eval-episodes 20 \
    --start-test 100
```

### Results
A `logs` directory will be created in the root directory of the project, with each training named after a version number (`v1`, `v2`, `v3`, etc.).
This directory contains realtime training metrics, statistics visualization, recordings, and the model checkpoints.

### Evaluation

Evaluation occurs periodically during training (controlled by `--start-test` and every 50 episodes).

#### Standalone

To evaluate a trained model directly:
```bash
evaluate \
    --env-name "LunarLanderContinuous-v3" \
    --model-dir "logs/LunarLanderContinuous-v3/v1" \
    --num-episodes 10
```

Detailed evaluation with custom parameters (checkout `parse_evaluate_args` and `EvaluationConfig` in `config.py` for available parameters):

```bash
evaluate \
    --env-name "LunarLanderContinuous-v3" \
    --model-dir "logs/LunarLanderContinuous-v3/v1" \
    --model-version 1 \
    --num-episodes 20 \
    --hidden-dim 256 \
    --max-action 1.0 \
    --buffer-capacity full \
    --fade-factor 7.0 \
    --stall-penalty 0.07 \
    --limit-eval 700 \
    --no-render \
    --save-dir "custom_eval_results"
```

Evaluate with limited episodes, custom save directory, and video recording:

```bash
evaluate \
    --env-name "Humanoid-v5" \
    --model-dir "logs/Humanoid-v5/v1" \
    --num-episodes 5 \
    --record-video \
    --save-dir "SADLDAS-standalone-eval-logs"
    --tr-between-ep-init 200
```

### Environment Recommendations
Each supported environment comes with recommended configurations in `src/constants/config.py` under the `ENVIRONMENT_RECOMMENDATIONS` dictionary.`. These are automatically applied but can be
overridden via command line arguments.

### HyperParameter Tuning

The tuner is in the `src/hyperparameter` directory.
To fine the optimal hyperparameters run:

```bash
tune \
    --n-trials 200 \                  # Try 200 different configurations
    --n-startup-trials 20 \           # First 20 trials are random exploration
    --study-name "lunar_lander_continuous_study" # Name to save results
    --env-name LunarLanderContinuous-v3
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Sample Results

Go to `src/results` directory to view multiple sample rendering results outcome from successful training of our agent!

Regarding Humanoid, future enhancements can be integrated to make it even better!
