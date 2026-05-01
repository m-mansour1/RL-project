"""Central configuration for the RL technical-indicator ablation study."""

from dataclasses import dataclass
from typing import List, Dict


@dataclass(frozen=True)
class TradingCosts:
    commission_rate: float = 0.0005      # 5 bps
    half_spread_rate: float = 0.0002     # 2 bps half spread
    slippage_rate: float = 0.0003        # 3 bps
    min_trade_notional: float = 10.0     # ignore tiny trades below this amount

    @property
    def total_rate(self) -> float:
        return self.commission_rate + self.half_spread_rate + self.slippage_rate


@dataclass(frozen=True)
class ExperimentConfig:
    initial_cash: float = 10_000.0
    window_size: int = 20
    train_end: str = "2017-01-01"
    val_end: str = "2021-01-01"

    # Runtime defaults. Increase these for final report-quality runs.
    seeds: tuple = (42, 123, 2024)
    dqn_episodes: int = 80
    q_episodes: int = 300

    # DQN hyperparameters
    gamma: float = 0.99
    dqn_lr: float = 5e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    target_update_freq: int = 500
    warmup_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 30_000
    grad_clip: float = 1.0

    # Q-learning hyperparameters
    q_alpha: float = 0.05
    q_gamma: float = 0.99
    q_epsilon_start: float = 1.0
    q_epsilon_min: float = 0.05
    q_epsilon_decay: float = 0.995
    q_bins: int = 7

    # Evaluation
    annualization_factor: int = 252
    validation_interval: int = 10
    max_steps_per_episode: int | None = None


RAW_REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "F1_returns_only": ["Log_Return"],
    "F2_returns_volume": ["Log_Return", "Volume_z"],
    "F3_momentum": ["Log_Return", "Volume_z", "RSI_norm", "MACD_z", "MACD_Signal_z", "MACD_Hist_z"],
    "F4_trend": ["Log_Return", "Volume_z", "SMA_20_ratio", "SMA_50_ratio", "SMA_200_ratio", "EMA_20_ratio", "EMA_50_ratio"],
    "F5_volatility": ["Log_Return", "Volume_z", "BB_Width_z", "BB_PctB_clipped"],
    "F6_all_indicators": [
        "Log_Return", "Volume_z",
        "RSI_norm", "MACD_z", "MACD_Signal_z", "MACD_Hist_z",
        "SMA_20_ratio", "SMA_50_ratio", "SMA_200_ratio", "EMA_20_ratio", "EMA_50_ratio",
        "BB_Width_z", "BB_PctB_clipped"
    ],
}

# Keep tabular states small. Q-learning is intentionally used on interpretable low-dimensional states.
Q_LEARNING_FEATURE_GROUPS = {
    "Q_F1_returns_only": ["Log_Return"],
    "Q_F3_momentum": ["Log_Return", "RSI_norm", "MACD_Hist_z"],
    "Q_F5_volatility": ["Log_Return", "BB_PctB_clipped", "BB_Width_z"],
}
