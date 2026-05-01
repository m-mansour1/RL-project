"""RL agents: tabular Q-Learning and DQN."""

from __future__ import annotations

from collections import defaultdict, deque
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


class QLearningAgent:
    """Tabular Q-learning with quantile discretization of selected feature columns."""

    def __init__(self, feature_columns: List[str], action_size: int = 2, bins: int = 7, alpha: float = 0.05, gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.995):
        self.feature_columns = feature_columns
        self.action_size = action_size
        self.bins = bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(lambda: np.zeros(action_size, dtype=float))
        self.bin_edges: Dict[str, np.ndarray] = {}

    def fit_bins(self, df: pd.DataFrame) -> None:
        for col in self.feature_columns:
            values = df[col].dropna().values
            quantiles = np.linspace(0, 1, self.bins + 1)[1:-1]
            self.bin_edges[col] = np.unique(np.quantile(values, quantiles))

    def state_to_key(self, raw_feature_vector: np.ndarray, position: int) -> Tuple[int, ...]:
        keys = []
        for i, col in enumerate(self.feature_columns):
            keys.append(int(np.digitize(raw_feature_vector[i], self.bin_edges[col])))
        keys.append(int(position))
        return tuple(keys)

    def select_action(self, key: Tuple[int, ...], greedy: bool = False) -> int:
        if (not greedy) and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.action_size))
        return int(np.argmax(self.Q[key]))

    def update(self, s, a, r, s_next, done) -> None:
        best_next = 0.0 if done else np.max(self.Q[s_next])
        target = r + self.gamma * best_next
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


if torch is not None:
    class DQN(nn.Module):
        """MLP DQN for flattened rolling-window states."""

        def __init__(self, input_dim: int, output_dim: int = 2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

        def forward(self, x):
            return self.net(x)
else:
    class DQN:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DQN.")


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        if torch is None:
            raise ImportError("PyTorch is required for replay sampling.")
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.asarray(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
