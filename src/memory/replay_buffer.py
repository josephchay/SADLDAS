import math
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device, fade_factor=7.0, stall_penalty=0.03):
        capacity_dict = {"short": 100000, "medium": 300000, "full": 500000}
        self.capacity, self.length, self.device = capacity_dict[capacity], 0, device
        self.batch_size = min(max(128, self.length // 500), 1024)  # in order for sample to describe population
        self.random = np.random.default_rng()
        self.indices, self.indexes, self.probs, self.step = [], np.array([]), np.array([]), 0
        self.fade_factor = fade_factor
        self.stall_penalty = stall_penalty

        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)

        self.raw = True

    def find_min_max(self):
        self.min_values = torch.min(self.states, dim=0).values
        self.max_values = torch.max(self.states, dim=0).values

        self.min_values[torch.isinf(self.min_values)] = -1e+3
        self.max_values[torch.isinf(self.max_values)] = 1e+3

        self.min_values = 2.0 * (torch.floor(10.0 * self.min_values) / 10.0).reshape(1, -1).to(self.device)
        self.max_values = 2.0 * (torch.ceil(10.0 * self.max_values) / 10.0).reshape(1, -1).to(self.device)

        self.raw = False

    def normalize(self, state):
        if self.raw:
            return state
        state[torch.isneginf(state)] = -1e+3
        state[torch.isposinf(state)] = 1e+3
        state = 4.0 * (state - self.min_values) / (self.max_values - self.min_values) - 2.0
        state[torch.isnan(state)] = 0.0
        return state

    def add(self, state, action, reward, next_state, done):
        if self.length < self.capacity:
            self.length += 1
            self.indices.append(self.length - 1)
            self.indexes = np.array(self.indices)

        idx = self.length - 1

        # moving is life, stalling is dangerous
        delta = np.mean(np.abs(next_state - state)).clip(1e-1, 10.0)
        reward -= self.stall_penalty * math.log10(1.0 / delta)

        if self.length == self.capacity:
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
            self.dones = torch.roll(self.dones, shifts=-1, dims=0)

        self.states[idx, :] = torch.FloatTensor(state).to(self.device)
        self.actions[idx, :] = torch.FloatTensor(action).to(self.device)
        self.rewards[idx, :] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[idx, :] = torch.FloatTensor(next_state).to(self.device)
        self.dones[idx, :] = torch.FloatTensor([done]).to(self.device)

        self.batch_size = min(max(128, self.length // 500), 1024)

        self.step += 1

    def generate_probs(self, uniform=False):
        if uniform:
            return np.ones(self.length) / self.length
        if self.step > self.capacity:
            return self.probs

        def fade(norm_index):
            return np.tanh(self.fade_factor * norm_index ** 2)  # linear / -> non-linear _/‾

        weights = 1e-7 * (fade(self.indexes / self.length))  # weights are based solely on the history, highly squashed
        self.probs = weights / np.sum(weights)
        return self.probs

    def sample(self, uniform=False):
        indices = self.random.choice(self.indexes, p=self.generate_probs(uniform), size=self.batch_size)

        return (
            self.normalize(self.states[indices]),
            self.actions[indices],
            self.rewards[indices],
            self.normalize(self.next_states[indices]),
            self.dones[indices]
        )

        # return (
        #     self.states[indices],
        #     self.actions[indices],
        #     self.rewards[indices],
        #     self.next_states[indices],
        #     self.dones[indices]
        # )

    def __len__(self):
        return self.length
