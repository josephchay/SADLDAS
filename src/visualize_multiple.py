from pathlib import Path
from src.constants import STATS_ROLLING_WINDOW
from src.stats import ComparisonVisualizer


# Initialize the visualizer with a simple base directory
viz = ComparisonVisualizer(
    base_save_dir=Path("../logs/standalone"),
)

# Generate comparison
summary = viz.compare_runs(
    metrics_files=[
        Path("../logs/LunarLanderContinuous-v3/v131/train/metrics.json"),
        Path("../logs/LunarLanderContinuous-v3/v134/train/metrics.json"),
    ],
    labels=[  # optional
        "Noise 1", 
        "Noise 2",
    ],
    plot_elements=[  # optional - customize what to show
        'raw',
        'mean',
        'training_start'
    ],
    rolling_window=STATS_ROLLING_WINDOW,
)
