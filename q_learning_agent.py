"""
Tabular Q-Learning agent with epsilon-greedy exploration.

The 6 continuous state features are each binned into B equal-width bins.
Position (feature index 6) is already discrete {0, 1}.
Q-table shape: (B, B, B, B, B, B, 2, 3)
Theoretical state count for B=4: 4^6 * 2 = 8,192 states.
"""

import numpy as np


class QLearningAgent:
    N_ACTIONS = 3  # Hold, Buy, Sell

    def __init__(
        self,
        n_bins: int = 4,
        gamma: float = 0.99,
        alpha: float = 0.1,
        eps_start: float = 1.0,
        eps_decay: float = 0.9995,
        eps_min: float = 0.01,
    ):
        self.n_bins = n_bins
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        # Inner edges for 6 continuous features (n_bins - 1 internal cut-points)
        self._edges = [
            np.linspace(0.0,   1.0,  n_bins + 1)[1:-1],   # RSI/100
            np.linspace(-0.05, 0.05, n_bins + 1)[1:-1],   # MACD_norm
            np.linspace(-0.5,  0.5,  n_bins + 1)[1:-1],   # SMA20_ratio
            np.linspace(-0.5,  0.5,  n_bins + 1)[1:-1],   # SMA200_ratio
            np.linspace(0.0,   1.0,  n_bins + 1)[1:-1],   # BB_PctB
            np.linspace(-0.2,  0.2,  n_bins + 1)[1:-1],   # log_return
        ]

        shape = (n_bins,) * 6 + (2, self.N_ACTIONS)
        self.Q = np.zeros(shape, dtype=np.float64)

    # ------------------------------------------------------------------
    def _discretize(self, state: np.ndarray) -> tuple:
        idx = tuple(
            int(np.clip(np.digitize(state[i], self._edges[i]), 0, self.n_bins - 1))
            for i in range(6)
        )
        return idx + (int(state[6]),)  # append discrete position

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        return int(np.argmax(self.Q[self._discretize(state)]))

    def greedy_action(self, state: np.ndarray) -> int:
        return int(np.argmax(self.Q[self._discretize(state)]))

    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        s  = self._discretize(state)
        ns = self._discretize(next_state)
        td_target = reward if done else reward + self.gamma * float(np.max(self.Q[ns]))
        self.Q[s][action] += self.alpha * (td_target - self.Q[s][action])
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
