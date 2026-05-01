"""End-to-end ablation experiment runner."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .config import ExperimentConfig, TradingCosts, FEATURE_GROUPS, Q_LEARNING_FEATURE_GROUPS
from .environment import TradingEnv
from .evaluation import (
    evaluate_policy, buy_and_hold, cash_baseline, sma_crossover_policy,
    plot_learning_curves, plot_portfolio_comparison, plot_drawdowns, plot_positions, plot_metric_bars
)
from .features import prepare_dataset
from .training import train_dqn, train_q_learning, select_dqn_action, _latest_features_for_q
from .utils import ensure_dir, save_json, set_seed

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def run_baselines(test_df, feature_columns, config, costs) -> Dict[str, Any]:
    results = {}
    results["BuyHold"] = buy_and_hold(test_df, feature_columns, config.window_size, config.initial_cash, costs)
    results["Cash"] = cash_baseline(test_df, feature_columns, config.window_size, config.initial_cash, costs)
    sma_env = TradingEnv(test_df, feature_columns, config.window_size, config.initial_cash, costs)
    results["SMA20_200"] = evaluate_policy(sma_env, sma_crossover_policy(test_df))
    return results


def collect_result_row(agent, feature_group, seed, split, metrics):
    row = {"agent": agent, "feature_group": feature_group, "seed": seed, "split": split}
    row.update(metrics)
    return row


def run_full_ablation(dataset_path: str, output_dir: str = "results", fast_mode: bool = False) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    plots_dir = ensure_dir(Path(output_dir) / "plots")
    models_dir = ensure_dir(Path(output_dir) / "models")
    tables_dir = ensure_dir(Path(output_dir) / "tables")

    config = ExperimentConfig()
    if fast_mode:
        config = ExperimentConfig(
            seeds=(42,),
            dqn_episodes=2,
            q_episodes=3,
            warmup_steps=50,
            validation_interval=1,
            max_steps_per_episode=250,
        )
    costs = TradingCosts()
    save_json({"experiment_config": config.__dict__, "trading_costs": costs.__dict__}, Path(output_dir) / "config.json")
    train_df, val_df, test_df = prepare_dataset(dataset_path, config.train_end, config.val_end)

    all_metric_rows = []
    all_logs = []
    representative_histories = {}

    baseline_features = FEATURE_GROUPS["F1_returns_only"]
    baselines = run_baselines(test_df, baseline_features, config, costs)
    for name, result in baselines.items():
        all_metric_rows.append(collect_result_row(name, "N/A", 0, "test", result["metrics"]))
        representative_histories[name] = result["history"]

    for seed in config.seeds:
        for q_group_name, q_features in Q_LEARNING_FEATURE_GROUPS.items():
            print(f"Training Q-Learning: seed={seed}, group={q_group_name}", flush=True)
            set_seed(seed)
            agent, logs, _ = train_q_learning(train_df, val_df, q_features, q_features, config, costs, seed)
            logs["feature_group"] = q_group_name
            all_logs.append(logs)
            test_env = TradingEnv(test_df, q_features, config.window_size, config.initial_cash, costs)
            def q_policy(_state, env=test_env, ag=agent, qf=q_features):
                raw = _latest_features_for_q(env, qf)
                key = ag.state_to_key(raw, env.position)
                return ag.select_action(key, greedy=True)
            test_result = evaluate_policy(test_env, q_policy)
            all_metric_rows.append(collect_result_row("Q-Learning", q_group_name, seed, "test", test_result["metrics"]))
            if seed == config.seeds[0]:
                representative_histories[f"Q-{q_group_name}"] = test_result["history"]

    for seed in config.seeds:
        for group_name, features in FEATURE_GROUPS.items():
            print(f"Training DQN: seed={seed}, group={group_name}", flush=True)
            set_seed(seed)
            model, logs, _ = train_dqn(train_df, val_df, features, group_name, config, costs, seed, models_dir)
            all_logs.append(logs)
            test_env = TradingEnv(test_df, features, config.window_size, config.initial_cash, costs)
            test_result = evaluate_policy(test_env, lambda s, net=model: select_dqn_action(net, s, epsilon=0.0))
            all_metric_rows.append(collect_result_row("DQN", group_name, seed, "test", test_result["metrics"]))
            if seed == config.seeds[0] and group_name in ("F1_returns_only", "F3_momentum", "F6_all_indicators"):
                representative_histories[f"DQN-{group_name}"] = test_result["history"]
                plot_positions(test_result["history"], f"DQN-{group_name}", plots_dir)

    metrics_df = pd.DataFrame(all_metric_rows)
    logs_df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    metrics_df.to_csv(tables_dir / "test_metrics_all_runs.csv", index=False)
    logs_df.to_csv(tables_dir / "training_logs.csv", index=False)
    metric_columns = metrics_df.select_dtypes(include=[np.number]).columns.difference(["seed"])
    metrics_df.groupby(["agent", "feature_group"])[metric_columns].agg(["mean", "std"]).to_csv(tables_dir / "summary_mean_std.csv")
    effects_df = compute_indicator_effects(metrics_df)
    effects_df.to_csv(tables_dir / "indicator_effects_vs_returns_only.csv", index=False)
    stats = statistical_tests(metrics_df)
    pd.DataFrame(stats).to_csv(tables_dir / "statistical_tests.csv", index=False)
    plot_learning_curves(logs_df, plots_dir)
    plot_portfolio_comparison(representative_histories, plots_dir)
    plot_drawdowns(representative_histories, plots_dir)
    plot_metric_bars(metrics_df, plots_dir)
    write_summary_text(metrics_df, effects_df, stats, Path(output_dir) / "technical_summary.md")
    return {"metrics": metrics_df, "logs": logs_df, "indicator_effects": effects_df, "stats": stats, "output_dir": str(output_dir)}


def compute_indicator_effects(metrics_df: pd.DataFrame) -> pd.DataFrame:
    dqn = metrics_df[(metrics_df["agent"] == "DQN") & (metrics_df["split"] == "test")].copy()
    baseline = dqn[dqn["feature_group"] == "F1_returns_only"]
    rows = []
    for _, base in baseline.iterrows():
        seed = base["seed"]
        same_seed = dqn[dqn["seed"] == seed]
        for _, row in same_seed.iterrows():
            if row["feature_group"] == "F1_returns_only":
                continue
            rows.append({
                "seed": seed,
                "feature_group": row["feature_group"],
                "delta_cumulative_return": row["cumulative_return"] - base["cumulative_return"],
                "delta_sharpe": row["sharpe"] - base["sharpe"],
                "drawdown_improvement": base["max_drawdown"] - row["max_drawdown"],
                "delta_turnover": row["turnover"] - base["turnover"],
            })
    return pd.DataFrame(rows)


def statistical_tests(metrics_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if wilcoxon is None:
        return [{"test": "wilcoxon", "status": "scipy_not_installed"}]
    dqn = metrics_df[(metrics_df["agent"] == "DQN") & (metrics_df["split"] == "test")].copy()
    out = []
    for metric in ["cumulative_return", "sharpe", "max_drawdown", "turnover"]:
        pivot = dqn.pivot_table(index="seed", columns="feature_group", values=metric)
        if "F1_returns_only" not in pivot.columns:
            continue
        for col in pivot.columns:
            if col == "F1_returns_only":
                continue
            pair = pivot[["F1_returns_only", col]].dropna()
            if len(pair) < 2:
                continue
            try:
                stat, p = wilcoxon(pair[col], pair["F1_returns_only"])
                out.append({"test": "wilcoxon_signed_rank", "metric": metric, "comparison": f"{col} vs F1_returns_only", "n_pairs": len(pair), "statistic": float(stat), "p_value": float(p)})
            except Exception as e:
                out.append({"test": "wilcoxon_signed_rank", "metric": metric, "comparison": f"{col} vs F1_returns_only", "status": f"failed: {e}"})
    return out


def write_summary_text(metrics_df, effects_df, stats, path: Path) -> None:
    lines = ["# Technical Summary\n", "Research question: Do technical indicators help RL trading under transaction costs?\n", "The study compares identical DQN agents across controlled feature groups and includes Q-Learning plus non-RL baselines.\n"]
    dqn = metrics_df[metrics_df["agent"] == "DQN"]
    if not dqn.empty:
        mean_table = dqn.groupby("feature_group")[["cumulative_return", "sharpe", "max_drawdown", "turnover"]].mean()
        lines += ["\n## Mean DQN test metrics\n", mean_table.to_string(), "\n"]
    if not effects_df.empty:
        eff = effects_df.groupby("feature_group")[["delta_cumulative_return", "delta_sharpe", "drawdown_improvement", "delta_turnover"]].mean()
        lines += ["\n## Mean indicator effects versus returns-only DQN\n", eff.to_string(), "\n"]
    if stats:
        lines += ["\n## Statistical tests\n", pd.DataFrame(stats).to_string(index=False), "\n"]
    lines.append("\nInterpretation rule: indicators help only if they improve out-of-sample Sharpe/return without unacceptable drawdown or turnover increase.\n")
    path.write_text("\n".join(lines), encoding="utf-8")
