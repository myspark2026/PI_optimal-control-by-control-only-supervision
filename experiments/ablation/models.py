"""The two 3-by-256 tanh architectures used in the clean-label appendix."""
import numpy as np
import torch
from torch import nn


class ValueNetwork(nn.Module):
    def __init__(self, dimension, mean=None, std=None):
        super().__init__()
        self.register_buffer("x_mean", torch.as_tensor(
            np.zeros(dimension) if mean is None else mean, dtype=torch.float32).clone())
        self.register_buffer("x_std", torch.as_tensor(
            np.ones(dimension) if std is None else std, dtype=torch.float32).clone())
        self.fc1 = nn.Linear(dimension, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)
        self.activation = nn.Tanh()

    def raw(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.out(self.activation(self.fc3(x)))

    def forward(self, x):
        origin = torch.zeros((1, self.x_mean.numel()), dtype=x.dtype, device=x.device)
        z = self.raw((x - self.x_mean) / (self.x_std + 1e-8))
        z = z - self.raw((origin - self.x_mean) / (self.x_std + 1e-8))
        # Forward output is nonnegative; the input derivative is that of z.
        return z + (torch.relu(z) - z).detach()


class PolicyNetwork(nn.Module):
    def __init__(self, dimension, controls, mean=None, std=None):
        super().__init__()
        self.register_buffer("x_mean", torch.as_tensor(
            np.zeros(dimension) if mean is None else mean, dtype=torch.float32).clone())
        self.register_buffer("x_std", torch.as_tensor(
            np.ones(dimension) if std is None else std, dtype=torch.float32).clone())
        self.network = nn.Sequential(
            nn.Linear(dimension, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, controls),
        )

    def forward(self, x):
        return self.network((x - self.x_mean) / (self.x_std + 1e-8))
