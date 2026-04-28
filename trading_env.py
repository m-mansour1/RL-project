"""
Custom daily stock trading environment for AAPL.

State (7-dim):
    [RSI/100, MACD/Close, SMA_20/Close-1, SMA_200/Close-1, BB_PctB, log_return, position]

Actions:
    0 = Hold  |  1 = Buy (go invested)  |  2 = Sell (go cash)

Reward:
    position_after_action * log(C_t / C_{t-1})  -  0.001 * I(trade_executed)

Portfolio:
    PV_{t+1} = PV_t * exp(position_after_action * log_return_t)
"""

import numpy as np
import pandas as pd


class TradingEnv:
    N_ACTIONS = 3
    STATE_DIM = 7

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.n = len(df)
        self.step_idx = 0
        self.position = 0
        self.portfolio_value = 1.0
        self.portfolio_history: list = []
        self.action_history: list = []
        self.reward_history: list = []

    # ------------------------------------------------------------------
    def _build_state(self, idx: int) -> np.ndarray:
        row = self.df.iloc[idx]
        close = float(row["Close"]) + 1e-10

        rsi        = np.clip(float(row["RSI"]) / 100.0, 0.0, 1.0)
        macd_norm  = np.clip(float(row["MACD"]) / close, -0.05, 0.05)
        sma20_r    = np.clip(float(row["SMA_20"])  / close - 1.0, -0.5, 0.5)
        sma200_r   = np.clip(float(row["SMA_200"]) / close - 1.0, -0.5, 0.5)
        bb_pctb    = np.clip(float(row["BB_PctB"]), 0.0, 1.0)
        log_ret    = np.clip(float(row["Log_Return"]), -0.2, 0.2)
        pos        = float(self.position)

        return np.array([rsi, macd_norm, sma20_r, sma200_r, bb_pctb, log_ret, pos],
                        dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.step_idx = 0
        self.position = 0
        self.portfolio_value = 1.0
        self.portfolio_history = [1.0]
        self.action_history = []
        self.reward_history = []
        return self._build_state(0)

    # ------------------------------------------------------------------
    def step(self, action: int):
        # The agent observes state at row t and acts; reward comes from row t+1.
        # This avoids look-ahead bias (today's return is a state feature, not the reward).
        next_row = self.df.iloc[self.step_idx + 1]  # always valid: done fires at step_idx == n-2
        log_ret = np.clip(float(next_row["Log_Return"]), -0.2, 0.2)

        # Execute action
        traded = False
        if action == 1 and self.position == 0:   # Buy
            self.position = 1
            traded = True
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            traded = True
        # action == 0 (Hold): position unchanged

        reward = float(self.position) * log_ret - 0.001 * float(traded)
        self.portfolio_value *= np.exp(float(self.position) * log_ret)

        self.portfolio_history.append(self.portfolio_value)
        self.action_history.append(action)
        self.reward_history.append(reward)

        self.step_idx += 1
        done = self.step_idx >= self.n - 1   # last usable step is n-2 (needs row n-1 for reward)
        next_state = self._build_state(min(self.step_idx, self.n - 1))
        return next_state, reward, done, {}

    # ------------------------------------------------------------------
    def get_portfolio_value(self) -> float:
        return self.portfolio_value
