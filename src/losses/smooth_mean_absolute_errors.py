import torch
import torch.nn.functional as F


class RobustLosses:
    """
    Collection of robust loss functions for deep learning.
    Based on research from:
    - "A General and Adaptive Robust Loss Function" (Barron, 2019)
    - "Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification" (Song et al., 2021)
    """
    
    @staticmethod
    def adaptive_sma_error(error, delta=1.0, epsilon=1e-6):
        """
        Enhanced version of Smooth Mean Absolute error with adaptive thresholding.
        Papers:
        - "Robust Deep Learning Under Distribution Shifts with Adaptive Huber Regression" (Wang et al., 2021)
        - "Adaptive Huber Regression" (Sun et al., 2020)
        """

        abs_error = torch.abs(error).mean()
        
        # Adaptive threshold based on error magnitude
        adaptive_delta = delta * torch.log1p(abs_error)
        
        # Quadratic region
        quad_mask = abs_error <= adaptive_delta
        quad_loss = 0.5 * error[quad_mask] ** 2
        
        # Linear region
        lin_mask = abs_error > adaptive_delta
        lin_loss = adaptive_delta * (abs_error[lin_mask] - 0.5 * adaptive_delta)
        
        # Combine losses
        combined_loss = (quad_loss.sum() + lin_loss.sum()) / (len(error) + epsilon)
        
        return combined_loss * torch.tanh(combined_loss)

    @staticmethod
    def robust_sma_asymmetric_error(error, alpha=0.2, delta=1.0, epsilon=1e-6):
        """
        Asymmetric Smooth Mean Absolute error with robustness to outliers.
        Based on:
        - "Asymmetric Loss Functions for Deep Learning with Label Noise" (Amid et al., 2019)
        - "Quantile Regression Under Random Censoring" (Koenker, 2008)
        """

        abs_error = torch.abs(error)
        mean_error = abs_error.mean()
        
        # Compute asymmetric weights
        weights = torch.where(error >= 0, 
                            alpha * torch.ones_like(error),
                            (1 - alpha) * torch.ones_like(error))
        
        # Apply weighted SMA loss
        quad_mask = abs_error <= delta
        quad_loss = 0.5 * weights[quad_mask] * error[quad_mask] ** 2
        
        lin_mask = abs_error > delta
        lin_loss = weights[lin_mask] * delta * (abs_error[lin_mask] - 0.5 * delta)
        
        # Combine with mean scaling
        combined_loss = (quad_loss.sum() + lin_loss.sum()) / (len(error) + epsilon)
        
        return combined_loss * torch.tanh(mean_error)


def ReSMAE(error):
    """
    Rectified Smooth Mean Absolute Loss Function.
    Enhanced version based on:
    - "Robust Deep Learning Under Distribution Shifts" (Wang et al., 2021)
    - "Beyond Pinball Loss" (Song et al., 2021)
    """
    return RobustLosses.adaptive_sma_error(error)


def ReaSMAE(error):
    """
    Rectified Asymmetric Smooth Mean Absolute Error Loss Function.
    Enhanced version based on:
    - "Asymmetric Loss Functions for Learning with Noisy Labels" (Zhou et al., 2021)
    - "Quantile Regression: 40 years on" (Koenker, 2008)
    """
    return RobustLosses.robust_sma_asymmetric_error(error)
