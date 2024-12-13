import math
import torch


class OUNoise:
    def __init__(self, action_dim, device, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.device = device
        self.x_coor = 0.0
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.state = torch.ones(self.action_dim).to(self.device) * self.mu
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dim).to(self.device) * self.mu

    def generate(self, x):
        if self.x_coor >= math.pi:
            return 0.0
        with torch.no_grad():
            eps = (math.cos(self.x_coor) + 1.0)
            self.x_coor += 7e-4
            x = self.state
            dx = self.theta * (self.mu - x) + self.sigma * torch.randn_like(x)
            self.state = x + dx
        return eps * self.state
