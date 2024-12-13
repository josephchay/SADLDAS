# Statistics Logging and Visualizations

## About

This directory contains files specialized in logging performance and crucial metrics into the terminal during training, evaluation, and tuning.

## Logger

The `logger.py` file contains the `Logger` class focusing on logging information metrics under crucial categories namely: current, rollout, time, train / eval, buffer.

## Visualizers

The `visualizers` python file contains the class `Visualizer` focuses on plotting graphs based on metrics information as a component of training, evaluation, and tuning. Meanwhile the `ComparisonVisualizer` serves as a standalone script for comparing multiple results based on saved trials and runs acting as a tooling for plotting multiple results under difference constraints allowing robust plotting comparisons.
Examples include comparing the performance of injections of different noise types on any environment.

### Visualizing Training Results

#### Comparing Multiple Training Runs

The `ComparisonVisualizer` allows you to compare results from multiple training runs. You can customize the visualization with various options:
A basic example would be

```python
from pathlib import Path
from src.stats import ComparisonVisualizer

# Initialize visualizer
base_dir = Path("../logs/standalone")
viz = ComparisonVisualizer(base_save_dir=base_dir)

# Define paths to metrics files
metrics_files = [
    Path("../logs/LunarLanderContinuous-v3/exp1/train/metrics.json"),  # (example) where you used our default novel SDLDNoise 
    Path("../logs/LunarLanderContinuous-v3/exp2/train/metrics.json"),  # (example) where you used OU noise
]

# Basic usage with default settings
summary = viz.compare_runs(
    metrics_files=metrics_files,
    labels=["SDLD", "OU"]
)

# Advanced usage with customization
summary = viz.compare_runs(
    metrics_files=metrics_files,
    labels=["SDLD", "OU"]
    x_axis='step',           # Plot against timesteps instead of episodes
    rolling_window=100,      # Adjust smoothing window
    plot_elements=['raw', 'mean', 'training_start']  # Select what to display
)
```

### Available Options

#### X-Axis Options
- `x_axis='episode'`: Plot against training episodes (default)
- `x_axis='step'`: Plot against total environment steps

#### Plot Elements
You can choose which elements to display by passing a list to `plot_elements`. Available options:
- `'raw'`: Raw reward data points
- `'mean'`: Rolling mean of rewards
- `'std'`: Standard deviation bands
- `'training_start'`: Vertical line indicating start of training

Examples:
```python
# Only raw data and mean
viz.compare_runs(
    metrics_files=metrics_files,
    labels=labels,
    plot_elements=['raw', 'mean']
)

# Complete visualization
viz.compare_runs(
    metrics_files=metrics_files,
    labels=labels,
    plot_elements=['raw', 'mean', 'std', 'training_start']
)

# Just mean and standard deviation
viz.compare_runs(
    metrics_files=metrics_files,
    labels=labels,
    plot_elements=['mean', 'std']
)
```

#### Other Parameters
- `rolling_window`: Size of window for calculating rolling statistics (default: 50)
- `labels`: List of labels for each run (optional, defaults to "Run 1", "Run 2", etc.)

### Output
The visualizer will:
1. Generate a plot comparing the specified metrics
2. Save the plot to `{base_dir}/visualizations/rewards_comparison.png`
3. Generate and save summary statistics to `{base_dir}/visualizations/comparison_summary.json`
4. Return the summary statistics as a dictionary

### Direct Plotting
You can also use the plotting method directly for more control:

```python
viz.plot_rewards_comparison(
    metrics_list=viz.load_metrics(metrics_files),
    labels=labels,
    x_axis='step',
    plot_elements=['raw', 'mean'],
    save_path="custom_comparison.png"
)
```

This gives you the same customization options but allows you to specify a custom save path and skip generating the summary statistics.
