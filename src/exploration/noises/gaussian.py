import math
import torch


class GANoise:
    def __init__(self, max_action, burst=False, tr_noise=True):
        self.x_coor = 0.0
        self.tr_noise = tr_noise
        self.scale = 1.0 if burst else 0.15
        self.max_action = max_action

    def generate(self, x):
        if self.tr_noise and self.x_coor >= 2.133:
            return (0.07 * torch.randn_like(x)).clamp(-0.175, 0.175)
        if self.x_coor >= math.pi:
            return 0.0

        with torch.no_grad():
            eps = self.scale * self.max_action * (math.cos(self.x_coor) + 1.0)
            lim = 2.5 * eps
            self.x_coor += 3e-5
        return (eps * torch.randn_like(x)).clamp(-lim, lim)
