"""
Ablation studies for Q-Learning (Experiment 3 from the proposal):
  - Discount factor γ ∈ {0.90, 0.95, 0.99}
  - Discretization bin count B ∈ {3, 4, 5}

Run after train.py:
    python experiments.py

Outputs:
    results/experiments.pkl
"""

import os
import pickle

import numpy as np

from train import load_and_split, buy_and_hold_pv, compute_metrics, train_qlearning, evaluate

# ── Reproducibility ────────────────────────────────────────────────────
SEED = 42

RESULTS_DIR = "results"
GAMMAS      = [0.90, 0.95, 0.99]
BIN_COUNTS  = [3, 4, 5]


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_df, test_df = load_and_split()
    bnh_pv      = buy_and_hold_pv(test_df)
    bnh_metrics = compute_metrics(bnh_pv)

    # ── Ablation 1: Discount factor ────────────────────────────────────
    print("=== Ablation: Discount Factor γ ===")
    abl_gamma: dict = {}
    for g in GAMMAS:
        np.random.seed(SEED)
        agent, ep_rewards, ep_pvs = train_qlearning(
            train_df, gamma=g, verbose=False
        )
        pv, actions = evaluate(agent, test_df)
        n = min(len(pv), len(bnh_pv))
        m = compute_metrics(pv[:n])
        abl_gamma[g] = {
            "metrics":    m,
            "pv":         pv[:n],
            "actions":    actions,
            "ep_rewards": ep_rewards,
        }
        print(
            f"  γ={g}  return={m['total_return']*100:+.2f}%  "
            f"sharpe={m['sharpe']:.3f}  maxdd={m['max_drawdown']*100:.2f}%"
        )

    # ── Ablation 2: Bin count ──────────────────────────────────────────
    print("\n=== Ablation: Bin Count B ===")
    abl_bins: dict = {}
    for b in BIN_COUNTS:
        np.random.seed(SEED)
        agent, ep_rewards, ep_pvs = train_qlearning(
            train_df, n_bins=b, verbose=False
        )
        pv, actions = evaluate(agent, test_df)
        n = min(len(pv), len(bnh_pv))
        m = compute_metrics(pv[:n])
        abl_bins[b] = {
            "metrics":    m,
            "pv":         pv[:n],
            "actions":    actions,
            "ep_rewards": ep_rewards,
        }
        print(
            f"  B={b}  return={m['total_return']*100:+.2f}%  "
            f"sharpe={m['sharpe']:.3f}  maxdd={m['max_drawdown']*100:.2f}%"
        )

    out = {
        "ablation_gamma": abl_gamma,
        "ablation_bins":  abl_bins,
        "bnh_pv":         bnh_pv,
        "bnh_metrics":    bnh_metrics,
        "gammas":         GAMMAS,
        "bin_counts":     BIN_COUNTS,
    }
    path = os.path.join(RESULTS_DIR, "experiments.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nExperiment results saved → {path}")
