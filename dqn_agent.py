"""
Deep Q-Network agent — pure NumPy, no deep-learning framework.

Architecture: 7 → 64 (ReLU) → 3
Training:     Adam optimizer, experience replay (capacity 3000),
              hard target-network copy every 200 steps.
"""

import numpy as np
from collections import deque


# ──────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 3000):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buf.append(
            (state.copy(), int(action), float(reward), next_state.copy(), float(done))
        )

    def sample(self, batch_size: int):
        idx   = np.random.choice(len(self._buf), batch_size, replace=False)
        batch = [self._buf[i] for i in idx]
        s  = np.array([t[0] for t in batch], dtype=np.float32)
        a  = np.array([t[1] for t in batch], dtype=np.int32)
        r  = np.array([t[2] for t in batch], dtype=np.float32)
        ns = np.array([t[3] for t in batch], dtype=np.float32)
        d  = np.array([t[4] for t in batch], dtype=np.float32)
        return s, a, r, ns, d

    def __len__(self):
        return len(self._buf)


# ──────────────────────────────────────────────────────────────────────
class MLP:
    """
    Two-layer MLP with He initialization and Adam optimizer.
    Forward and backward passes implemented in NumPy.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        n_actions: int = 3,
        lr: float = 1e-3,
    ):
        self.lr = lr
        # He initialization
        s1 = np.sqrt(2.0 / input_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = (np.random.randn(input_dim,  hidden_dim) * s1).astype(np.float32)
        self.b1 = np.zeros(hidden_dim,  dtype=np.float32)
        self.W2 = (np.random.randn(hidden_dim, n_actions)  * s2).astype(np.float32)
        self.b2 = np.zeros(n_actions,   dtype=np.float32)

        # Adam state (moment vectors for each parameter tensor)
        self._params = [self.W1, self.b1, self.W2, self.b2]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t  = 0
        self._b1 = 0.9
        self._b2 = 0.999
        self._eps = 1e-8

        # Cached activations for backward
        self._x = self._z1 = self._h1 = self._q = None

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x  = x
        self._z1 = x @ self.W1 + self.b1          # (B, hidden)
        self._h1 = np.maximum(0.0, self._z1)       # ReLU
        self._q  = self._h1 @ self.W2 + self.b2   # (B, actions)
        return self._q

    def backward_and_step(self, dq: np.ndarray):
        """Backprop through the network and update weights with Adam."""
        dW2 = self._h1.T @ dq                                # (hidden, actions)
        db2 = dq.sum(0)                                       # (actions,)
        dh1 = dq @ self.W2.T                                  # (B, hidden)
        dz1 = dh1 * (self._z1 > 0).astype(np.float32)        # ReLU gradient
        dW1 = self._x.T @ dz1                                 # (input, hidden)
        db1 = dz1.sum(0)                                      # (hidden,)

        grads = [dW1, db1, dW2, db2]
        self._t += 1
        for i, (p, g) in enumerate(zip(self._params, grads)):
            self._m[i] = self._b1 * self._m[i] + (1 - self._b1) * g
            self._v[i] = self._b2 * self._v[i] + (1 - self._b2) * g ** 2
            m_hat = self._m[i] / (1 - self._b1 ** self._t)
            v_hat = self._v[i] / (1 - self._b2 ** self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self._eps)

    def copy_weights_from(self, src: "MLP"):
        self.W1[:] = src.W1
        self.b1[:] = src.b1
        self.W2[:] = src.W2
        self.b2[:] = src.b2


# ──────────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    DQN with experience replay and hard target-network updates.
    Network: 7 → 64 (ReLU) → 3, implemented entirely in NumPy.
    """

    N_ACTIONS = 3
    STATE_DIM  = 7

    def __init__(
        self,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 3000,
        batch_size: int = 64,
        target_freq: int = 200,
        eps_start: float = 1.0,
        eps_decay: float = 0.9997,
        eps_min: float = 0.01,
    ):
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.target_freq = target_freq
        self.epsilon     = eps_start
        self.eps_decay   = eps_decay
        self.eps_min     = eps_min

        self.online = MLP(self.STATE_DIM, 64, self.N_ACTIONS, lr)
        self.target = MLP(self.STATE_DIM, 64, self.N_ACTIONS, lr)
        self.target.copy_weights_from(self.online)

        self.buffer  = ReplayBuffer(buffer_size)
        self._steps  = 0
        self.losses: list = []

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.N_ACTIONS)
        q = self.online.forward(state[np.newaxis])[0]
        return int(np.argmax(q))

    def greedy_action(self, state: np.ndarray) -> int:
        q = self.online.forward(state[np.newaxis])[0]
        return int(np.argmax(q))

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        B = self.batch_size

        q_all    = self.online.forward(s)           # (B, 3)
        q_next   = self.target.forward(ns)          # (B, 3)
        max_next = q_next.max(axis=1)               # (B,)

        targets  = r + self.gamma * max_next * (1.0 - d)   # Bellman targets
        q_pred   = q_all[np.arange(B), a]                  # selected Q-values
        td_err   = q_pred - targets                         # TD errors

        loss = float(np.mean(td_err ** 2))

        # Gradient: only the selected action gets a non-zero gradient
        dq = np.zeros_like(q_all)
        dq[np.arange(B), a] = 2.0 * td_err / B
        self.online.backward_and_step(dq)

        self._steps  += 1
        self.epsilon  = max(self.eps_min, self.epsilon * self.eps_decay)

        if self._steps % self.target_freq == 0:
            self.target.copy_weights_from(self.online)

        self.losses.append(loss)
        return loss
