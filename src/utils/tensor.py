import numpy as np
import torch


def convert_tensor(obj):
    """Helper function to convert tensors to Python native types"""

    if torch.is_tensor(obj):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensor(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensor(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj
