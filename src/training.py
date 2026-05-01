"""Training loops for Q-Learning and DQN."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Any, Dict

import numpy as np
import pandas as pd

from .agents import QLearningAgent, DQN, ReplayBuffer
from .environment import TradingEnv
from .evaluation import evaluate_policy
from .utils import ensure_dir, set_seed

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


def _latest_features_for_q(env: TradingEnv, q_features: List[str]) -> np.ndarray:
    if list(q_features) == list(env.feature_columns):
        return env.features[env.t]

    q_indices = getattr(env, "_q_feature_indices", None)
    q_indices_key = getattr(env, "_q_feature_indices_key", None)
    if q_indices is None or q_indices_key != tuple(q_features):
        q_indices = [env.feature_columns.index(col) for col in q_features]
        env._q_feature_indices = q_indices
        env._q_feature_indices_key = tuple(q_features)
    return env.features[env.t, q_indices]


def train_q_learning(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_columns: List[str], q_features: List[str], config, costs, seed: int) -> Tuple[QLearningAgent, pd.DataFrame, Dict[str, Any]]:
    set_seed(seed)
    agent = QLearningAgent(
        feature_columns=q_features,
        bins=config.q_bins,
        alpha=config.q_alpha,
        gamma=config.q_gamma,
        epsilon_start=config.q_epsilon_start,
        epsilon_min=config.q_epsilon_min,
        epsilon_decay=config.q_epsilon_decay,
    )
    agent.fit_bins(train_df)
    logs = []
    train_env = TradingEnv(train_df, feature_columns, config.window_size, config.initial_cash, costs)

    for ep in range(config.q_episodes):
        train_env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            raw = _latest_features_for_q(train_env, q_features)
            s = agent.state_to_key(raw, train_env.position)
            a = agent.select_action(s)
            _, r, done, _ = train_env.step(a)
            raw_next = _latest_features_for_q(train_env, q_features)
            s_next = agent.state_to_key(raw_next, train_env.position)
            agent.update(s, a, r, s_next, done)
            ep_reward += r
            steps += 1
            if config.max_steps_per_episode is not None and steps >= config.max_steps_per_episode:
                done = True
        agent.decay_epsilon()
        logs.append({
            "agent": "Q-Learning",
            "feature_group": "+".join(q_features),
            "seed": seed,
            "episode": ep,
            "episode_reward": ep_reward,
            "epsilon": agent.epsilon,
        })

    val_env = TradingEnv(val_df, feature_columns, config.window_size, config.initial_cash, costs)
    def q_policy(_state, env=val_env, ag=agent, qf=q_features):
        raw = _latest_features_for_q(env, qf)
        key = ag.state_to_key(raw, env.position)
        return ag.select_action(key, greedy=True)
    val_result = evaluate_policy(val_env, q_policy)
    return agent, pd.DataFrame(logs), val_result


def linear_epsilon(step: int, start: float, final: float, decay_steps: int) -> float:
    frac = min(1.0, step / max(1, decay_steps))
    return start + frac * (final - start)


def select_dqn_action(policy_net, state: np.ndarray, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return int(np.random.randint(2))
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return int(policy_net(s).argmax(dim=1).item())


def optimize_dqn(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, grad_clip) -> float:
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        target = rewards + gamma * next_q * (1 - dones)
    loss = F.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


def train_dqn(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_columns: List[str], feature_group_name: str, config, costs, seed: int, checkpoint_dir: str | Path) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    if torch is None:
        raise ImportError("PyTorch is required to train DQN. Install torch first.")
    set_seed(seed)
    checkpoint_dir = ensure_dir(checkpoint_dir)
    env = TradingEnv(train_df, feature_columns, config.window_size, config.initial_cash, costs)
    policy_net = DQN(env.state_dim, 2)
    target_net = DQN(env.state_dim, 2)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.dqn_lr)
    replay_buffer = ReplayBuffer(config.replay_capacity)
    logs = []
    global_step = 0
    best_val_sharpe = -np.inf
    best_path = Path(checkpoint_dir) / f"dqn_{feature_group_name}_seed{seed}.pt"

    for ep in range(config.dqn_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_losses = []
        steps = 0
        while not done:
            epsilon = linear_epsilon(global_step, config.epsilon_start, config.epsilon_final, config.epsilon_decay_steps)
            action = select_dqn_action(policy_net, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            if len(replay_buffer) >= max(config.batch_size, config.warmup_steps):
                ep_losses.append(optimize_dqn(policy_net, target_net, optimizer, replay_buffer, config.batch_size, config.gamma, config.grad_clip))
            if global_step % config.target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
            global_step += 1
            steps += 1
            if config.max_steps_per_episode is not None and steps >= config.max_steps_per_episode:
                done = True

        row = {
            "agent": "DQN",
            "feature_group": feature_group_name,
            "seed": seed,
            "episode": ep,
            "episode_reward": ep_reward,
            "epsilon": linear_epsilon(global_step, config.epsilon_start, config.epsilon_final, config.epsilon_decay_steps),
            "loss": float(np.mean(ep_losses)) if ep_losses else np.nan,
            "final_portfolio_value": env.portfolio_value,
            "num_trades": env.trades,
        }
        if (ep + 1) % config.validation_interval == 0 or ep == config.dqn_episodes - 1:
            val_env = TradingEnv(val_df, feature_columns, config.window_size, config.initial_cash, costs)
            val_result = evaluate_policy(val_env, lambda s: select_dqn_action(policy_net, s, epsilon=0.0))
            row["val_sharpe"] = val_result["metrics"].get("sharpe", np.nan)
            row["val_cumulative_return"] = val_result["metrics"].get("cumulative_return", np.nan)
            if row["val_sharpe"] > best_val_sharpe:
                best_val_sharpe = row["val_sharpe"]
                torch.save(policy_net.state_dict(), best_path)
        logs.append(row)

    if best_path.exists():
        policy_net.load_state_dict(torch.load(best_path, map_location="cpu"))
    val_env = TradingEnv(val_df, feature_columns, config.window_size, config.initial_cash, costs)
    val_result = evaluate_policy(val_env, lambda s: select_dqn_action(policy_net, s, epsilon=0.0))
    return policy_net, pd.DataFrame(logs), val_result
