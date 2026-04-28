"""
Train Q-Learning and DQN on AAPL daily data, then evaluate vs Buy-and-Hold.

Usage:
    python train.py

Outputs:
    results/results.pkl   — episode rewards, test portfolio histories, metrics
"""

import os
import pickle

import numpy as np
import pandas as pd

from trading_env    import TradingEnv
from q_learning_agent import QLearningAgent
from dqn_agent      import DQNAgent

# ── Reproducibility ────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths / constants ──────────────────────────────────────────────────
DATA_PATH   = "AAPL_dataset.csv"
RESULTS_DIR = "results"
SPLIT_DATE  = "2020-07-01"       # 80 % train / 20 % test split
QL_EPISODES  = 40
DQN_EPISODES = 40


# ── Data helpers ───────────────────────────────────────────────────────
def load_and_split(path: str = DATA_PATH, split_date: str = SPLIT_DATE):
    cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "RSI", "MACD", "SMA_20", "SMA_200", "BB_PctB", "Log_Return",
    ]
    df = pd.read_csv(path, usecols=cols)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    train = df[df["Date"] <  split_date].reset_index(drop=True)
    test  = df[df["Date"] >= split_date].reset_index(drop=True)
    return train, test


def buy_and_hold_pv(df: pd.DataFrame) -> list:
    """
    Passive portfolio that holds AAPL for the entire period.
    Uses forward returns (rows 1..n-1) to match the env's reward formulation,
    where position chosen at row t earns log_return from row t+1.
    """
    pv   = 1.0
    hist = [pv]
    for lr in df["Log_Return"].values[1:]:    # rows 1..n-1, matching env steps 0..n-2
        lr = np.clip(float(lr), -0.2, 0.2)
        pv *= np.exp(lr)
        hist.append(pv)
    return hist


def compute_metrics(pv: list) -> dict:
    arr = np.array(pv, dtype=np.float64)
    total_ret = float(arr[-1] - 1.0)

    # Daily log-returns for Sharpe / drawdown
    log_r = np.diff(np.log(np.clip(arr, 1e-10, None)))
    sharpe = (
        float(log_r.mean() / log_r.std() * np.sqrt(252))
        if log_r.std() > 1e-10 else 0.0
    )
    running_max = np.maximum.accumulate(arr)
    max_dd = float(((arr - running_max) / (running_max + 1e-10)).min())

    return {"total_return": total_ret, "sharpe": sharpe, "max_drawdown": max_dd}


# ── Training loops ─────────────────────────────────────────────────────
def train_qlearning(
    train_df,
    n_episodes: int = QL_EPISODES,
    gamma: float = 0.99,
    n_bins: int = 4,
    verbose: bool = True,
):
    env   = TradingEnv(train_df)
    agent = QLearningAgent(n_bins=n_bins, gamma=gamma)
    ep_rewards, ep_pv = [], []

    for ep in range(n_episodes):
        state      = env.reset()
        ep_reward  = 0.0
        done       = False

        while not done:
            action                        = agent.select_action(state)
            next_state, reward, done, _   = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state      = next_state
            ep_reward += reward

        ep_rewards.append(ep_reward)
        ep_pv.append(env.get_portfolio_value())

        if verbose:
            print(
                f"  [QL] ep {ep+1:3d}/{n_episodes}  "
                f"reward={ep_reward:+.4f}  "
                f"pv={env.get_portfolio_value():.4f}  "
                f"ε={agent.epsilon:.4f}"
            )

    return agent, ep_rewards, ep_pv


def train_dqn(
    train_df,
    n_episodes: int = DQN_EPISODES,
    gamma: float = 0.99,
    verbose: bool = True,
):
    env   = TradingEnv(train_df)
    agent = DQNAgent(gamma=gamma)
    ep_rewards, ep_pv, ep_losses = [], [], []

    for ep in range(n_episodes):
        state      = env.reset()
        ep_reward  = 0.0
        ep_loss    = []
        done       = False

        while not done:
            action                      = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                ep_loss.append(loss)
            state      = next_state
            ep_reward += reward

        ep_rewards.append(ep_reward)
        ep_pv.append(env.get_portfolio_value())
        mean_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        ep_losses.append(mean_loss)

        if verbose:
            print(
                f"  [DQN] ep {ep+1:3d}/{n_episodes}  "
                f"reward={ep_reward:+.4f}  "
                f"pv={env.get_portfolio_value():.4f}  "
                f"loss={mean_loss:.6f}  "
                f"ε={agent.epsilon:.4f}"
            )

    return agent, ep_rewards, ep_pv, ep_losses


# ── Evaluation ─────────────────────────────────────────────────────────
def evaluate(agent, test_df):
    env   = TradingEnv(test_df)
    state = env.reset()
    done  = False
    while not done:
        action        = agent.greedy_action(state)
        state, _, done, _ = env.step(action)
    return env.portfolio_history, env.action_history


# ── Entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading and splitting data …")
    train_df, test_df = load_and_split()
    print(f"  Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")

    print("\n=== Training Q-Learning ===")
    ql_agent, ql_ep_rewards, ql_ep_pv = train_qlearning(train_df)

    print("\n=== Training DQN ===")
    dqn_agent, dqn_ep_rewards, dqn_ep_pv, dqn_ep_losses = train_dqn(train_df)

    print("\n=== Evaluating on test set ===")
    ql_pv,  ql_actions  = evaluate(ql_agent,  test_df)
    dqn_pv, dqn_actions = evaluate(dqn_agent, test_df)
    bnh_pv              = buy_and_hold_pv(test_df)

    # Align lengths (env processes n-1 steps; all give n elements)
    n = min(len(ql_pv), len(dqn_pv), len(bnh_pv))
    ql_pv  = ql_pv[:n]
    dqn_pv = dqn_pv[:n]
    bnh_pv = bnh_pv[:n]

    ql_metrics  = compute_metrics(ql_pv)
    dqn_metrics = compute_metrics(dqn_pv)
    bnh_metrics = compute_metrics(bnh_pv)

    print(f"\n{'Method':<18} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10}")
    print("─" * 50)
    for name, m in [
        ("Q-Learning",  ql_metrics),
        ("DQN",         dqn_metrics),
        ("Buy-and-Hold", bnh_metrics),
    ]:
        print(
            f"{name:<18} "
            f"{m['total_return']*100:>9.2f}%  "
            f"{m['sharpe']:>9.3f}  "
            f"{m['max_drawdown']*100:>9.2f}%"
        )

    # Save everything for plotting / experiments
    results = {
        "ql_ep_rewards":  ql_ep_rewards,
        "ql_ep_pv":       ql_ep_pv,
        "dqn_ep_rewards": dqn_ep_rewards,
        "dqn_ep_pv":      dqn_ep_pv,
        "dqn_ep_losses":  dqn_ep_losses,
        "ql_pv":          ql_pv,
        "ql_actions":     ql_actions,
        "dqn_pv":         dqn_pv,
        "dqn_actions":    dqn_actions,
        "bnh_pv":         bnh_pv,
        "ql_metrics":     ql_metrics,
        "dqn_metrics":    dqn_metrics,
        "bnh_metrics":    bnh_metrics,
        "test_dates":     test_df["Date"].astype(str).tolist(),
    }
    out_path = os.path.join(RESULTS_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved → {out_path}")
