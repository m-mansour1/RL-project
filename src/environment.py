"""Gym-like trading environment for discrete long/cash RL trading."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import TradingCosts


class TradingEnv:
    """Daily long/cash trading environment.

    MDP:
    - State: rolling feature window flattened + current position.
    - Action: 0=cash, 1=long.
    - Transition: next trading day.
    - Reward: position * next-day log return - realistic friction cost.
    - Frictions: commission + half spread + slippage; minimum notional trade threshold.
    """

    def __init__(self, df: pd.DataFrame, feature_columns: List[str], window_size: int = 20, initial_cash: float = 10_000.0, costs: TradingCosts | None = None):
        if "Log_Return" not in df.columns:
            raise ValueError("TradingEnv requires a Log_Return column.")
        if len(df) <= window_size + 2:
            raise ValueError("Not enough data for selected window_size.")

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.costs = costs or TradingCosts()
        self.features = self.df[feature_columns].astype(np.float32).values
        self.log_returns = self.df["Log_Return"].astype(np.float32).values
        self.close = self.df["Close"].astype(np.float32).values
        self.dates = self.df["Date"].values
        self.action_space_n = 2
        self.state_dim = self.window_size * len(self.feature_columns) + 1
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = self.window_size
        self.position = 0
        self.portfolio_value = self.initial_cash
        self.done = False
        self.total_reward = 0.0
        self.trades = 0
        self.rewards: List[float] = []
        self.portfolio_values: List[float] = [self.initial_cash]
        self.positions: List[int] = [self.position]
        self.actions: List[int] = []
        self.costs_paid: List[float] = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        window = self.features[self.t - self.window_size : self.t]
        state = window.flatten()
        state = np.append(state, self.position)
        return state.astype(np.float32)

    def _compute_cost_rate(self, old_position: int, new_position: int) -> Tuple[float, bool]:
        trade_size = abs(new_position - old_position)
        if trade_size == 0:
            return 0.0, False
        notional = self.portfolio_value * trade_size
        if notional < self.costs.min_trade_notional:
            return 0.0, False
        return self.costs.total_rate * trade_size, True

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished environment.")
        if action not in (0, 1):
            raise ValueError("Action must be 0=cash or 1=long.")

        old_position = self.position
        desired_position = int(action)
        cost_rate, executed = self._compute_cost_rate(old_position, desired_position)
        new_position = desired_position if executed or desired_position == old_position else old_position

        next_log_return = float(self.log_returns[self.t + 1])
        reward = float(new_position * next_log_return - cost_rate)

        self.portfolio_value *= float(np.exp(reward))
        self.position = new_position
        self.total_reward += reward
        self.trades += int(executed)
        self.rewards.append(reward)
        self.portfolio_values.append(self.portfolio_value)
        self.positions.append(self.position)
        self.actions.append(action)
        self.costs_paid.append(cost_rate)

        self.t += 1
        self.done = self.t >= len(self.df) - 2
        next_state = self._get_state()
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "trade_executed": executed,
            "cost_rate": cost_rate,
            "market_log_return": next_log_return,
            "date": str(self.dates[self.t]),
        }
        return next_state, reward, self.done, info

    def get_history(self) -> Dict[str, np.ndarray]:
        start_idx = self.window_size
        dates = self.df["Date"].iloc[start_idx : start_idx + len(self.portfolio_values)].values
        close = self.close[start_idx : start_idx + len(self.portfolio_values)]
        return {
            "dates": dates,
            "portfolio_values": np.asarray(self.portfolio_values, dtype=float),
            "positions": np.asarray(self.positions, dtype=int),
            "rewards": np.asarray(self.rewards, dtype=float),
            "close": close,
            "costs_paid": np.asarray(self.costs_paid, dtype=float),
        }
