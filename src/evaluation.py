"""Evaluation metrics and plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .environment import TradingEnv
from .utils import ensure_dir


def trading_metrics(history: Dict[str, np.ndarray], initial_cash: float = 10_000.0, annualization: int = 252) -> Dict[str, float]:
    pv = np.asarray(history["portfolio_values"], dtype=float)
    rewards = np.asarray(history["rewards"], dtype=float)
    positions = np.asarray(history["positions"], dtype=int)
    if len(pv) < 2:
        return {}
    simple_returns = pv[1:] / pv[:-1] - 1
    cumulative_return = pv[-1] / initial_cash - 1
    n_days = max(1, len(simple_returns))
    annual_return = (pv[-1] / initial_cash) ** (annualization / n_days) - 1
    annual_vol = float(np.std(simple_returns) * np.sqrt(annualization)) if len(simple_returns) > 1 else 0.0
    sharpe = float((np.mean(simple_returns) / np.std(simple_returns)) * np.sqrt(annualization)) if np.std(simple_returns) > 1e-12 else 0.0
    running_max = np.maximum.accumulate(pv)
    drawdown = (pv - running_max) / running_max
    max_drawdown = float(np.min(drawdown))
    num_trades = int(np.sum(np.abs(np.diff(positions)) > 0))
    turnover = float(num_trades / n_days)
    win_rate = float(np.mean(simple_returns > 0)) if len(simple_returns) else 0.0
    return {
        "final_portfolio_value": float(pv[-1]),
        "cumulative_return": float(cumulative_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "num_trades": float(num_trades),
        "turnover": float(turnover),
        "win_rate": float(win_rate),
        "mean_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
        "std_reward": float(np.std(rewards)) if len(rewards) else 0.0,
        "total_reward": float(np.sum(rewards)) if len(rewards) else 0.0,
    }


def evaluate_policy(env: TradingEnv, policy_fn) -> Dict[str, Any]:
    state = env.reset()
    done = False
    while not done:
        action = int(policy_fn(state))
        state, reward, done, info = env.step(action)
    history = env.get_history()
    return {"metrics": trading_metrics(history, initial_cash=env.initial_cash), "history": history}


def buy_and_hold(df, feature_columns, window_size, initial_cash, costs):
    env = TradingEnv(df, feature_columns, window_size, initial_cash, costs)
    return evaluate_policy(env, lambda state: 1)


def cash_baseline(df, feature_columns, window_size, initial_cash, costs):
    env = TradingEnv(df, feature_columns, window_size, initial_cash, costs)
    return evaluate_policy(env, lambda state: 0)


def sma_crossover_policy(df, short_col="SMA_20", long_col="SMA_200"):
    signal = (df[short_col] > df[long_col]).astype(int).values
    idx = {"t": 0}
    def policy(_state):
        action = int(signal[min(idx["t"], len(signal) - 1)])
        idx["t"] += 1
        return action
    return policy


def plot_learning_curves(logs: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    if logs.empty:
        return
    plt.figure(figsize=(10, 6))
    for (agent, feature_group), sub in logs.groupby(["agent", "feature_group"]):
        ordered = sub.sort_values("episode")
        if "episode_reward" in ordered:
            y = ordered["episode_reward"].rolling(10, min_periods=1).mean()
            plt.plot(ordered["episode"], y, label=f"{agent}-{feature_group}")
    plt.xlabel("Episode")
    plt.ylabel("Reward, 10-episode moving average")
    plt.title("Learning curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=200)
    plt.close()


def plot_portfolio_comparison(histories: Dict[str, Dict[str, np.ndarray]], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    plt.figure(figsize=(11, 6))
    for label, h in histories.items():
        dates = pd.to_datetime(h["dates"])
        plt.plot(dates, h["portfolio_values"], label=label)
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.title("Test portfolio value comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "portfolio_comparison.png", dpi=200)
    plt.close()


def plot_drawdowns(histories: Dict[str, Dict[str, np.ndarray]], output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    plt.figure(figsize=(11, 6))
    for label, h in histories.items():
        pv = h["portfolio_values"]
        dd = (pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv)
        plt.plot(pd.to_datetime(h["dates"]), dd, label=label)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.title("Test drawdown comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "drawdowns.png", dpi=200)
    plt.close()


def plot_positions(history: Dict[str, np.ndarray], label: str, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    dates = pd.to_datetime(history["dates"])
    close = history["close"]
    positions = history["positions"]
    n = min(len(dates), len(close), len(positions))
    plt.figure(figsize=(11, 6))
    plt.plot(dates[:n], close[:n], label="Close")
    pos_scaled = positions[:n] * (np.nanmax(close[:n]) - np.nanmin(close[:n])) * 0.1 + np.nanmin(close[:n])
    plt.plot(dates[:n], pos_scaled, label="Position scaled")
    plt.xlabel("Date")
    plt.title(f"Behavior visualization: {label}")
    plt.legend()
    plt.tight_layout()
    safe = label.replace("/", "_").replace(" ", "_")
    plt.savefig(output_dir / f"positions_{safe}.png", dpi=200)
    plt.close()


def plot_metric_bars(results: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    if results.empty:
        return
    for metric in ["cumulative_return", "sharpe", "max_drawdown", "turnover"]:
        if metric not in results.columns:
            continue
        summary = results.groupby(["agent", "feature_group"])[metric].mean().reset_index()
        summary["label"] = summary["agent"] + "-" + summary["feature_group"].astype(str)
        plt.figure(figsize=(12, 6))
        plt.bar(summary["label"], summary[metric])
        plt.xticks(rotation=75, ha="right")
        plt.title(f"Mean test {metric}")
        plt.tight_layout()
        plt.savefig(output_dir / f"metric_{metric}.png", dpi=200)
        plt.close()
