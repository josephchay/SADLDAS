import torch


class LDASPolicy:
    """
    Selects actions to maximize distance from previously visited state-action pairs
    Implemented based on the paper: https://www.mdpi.com/2673-9909/2/2/14
    """

    def __init__(self, action_dim, max_action, device, 
                 buffer_size=500,
                 learning_rate=0.01,
                 lr_decay=0.99):
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        self.buffer_size = buffer_size
        self.lr = learning_rate
        self.lr_decay = lr_decay

        # Initialize history buffer
        self.buffer = []

    def get_distance(self, state_action, history):
        """Calculate minimum distance to history"""

        if len(history) == 0:
            return float('inf')
        distances = torch.norm(history - state_action, dim=1)
        return torch.min(distances)

    def generate(self, state):
        """Generate action based on maximizing distance from history"""

        if len(self.buffer) == 0:
            action = (torch.rand(self.action_dim, device=self.device) * 2 - 1) * self.max_action
            self._update_buffer(state, action)
            return action

        # Start with random action
        best_action = (torch.rand(self.action_dim, device=self.device) * 2 - 1) * self.max_action
        best_distance = -float('inf')
        current_lr = self.lr

        # Gradient ascent to maximize distance
        history = torch.stack(self.buffer)
        for _ in range(10):  # Limited iterations
            state_action = torch.cat([state, best_action])
            dist = self.get_distance(state_action, history)
            
            if dist > best_distance:
                best_distance = dist

            # Move away from closest point
            closest_idx = torch.argmin(torch.norm(history - state_action, dim=1))
            diff = state_action - history[closest_idx]
            best_action = best_action + current_lr * diff[-self.action_dim:] / torch.norm(diff[-self.action_dim:])

            # Clip and decay learning rate
            best_action = torch.clamp(best_action, -self.max_action, self.max_action)
            current_lr *= self.lr_decay

        self._update_buffer(state, best_action)
        return best_action

    def _update_buffer(self, state, action):
        """Update history buffer"""

        self.buffer.append(torch.cat([state, action]))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
