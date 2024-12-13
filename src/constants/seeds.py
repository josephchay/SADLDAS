import torch
import numpy as np
import random

from src.constants import SeedConfig


def set_seeds(seed_config: SeedConfig = None):
    """Set random seeds for reproducibility"""

    if seed_config is None:
        print("Seeding not performed! No seed config provided.")
        return  # Skip seeding if no seed config provided

    # Skip seeding if any seed is None
    if any(seed is None for seed in [seed_config.torch_seed,
                                     seed_config.numpy_seed,
                                     seed_config.python_seed]):
        print("Seeding not performed! Some seeds are missing.")
        return

    torch.manual_seed(seed_config.torch_seed)
    np.random.seed(seed_config.numpy_seed)
    random.seed(seed_config.python_seed)

    # Additional seeds for cuda if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_config.torch_seed)
        torch.cuda.manual_seed_all(seed_config.torch_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
