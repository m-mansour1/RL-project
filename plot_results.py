"""
Generate all figures from saved results.
Figures are written to results/figures/.

Run after train.py and experiments.py:
    python plot_results.py
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = "results"
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")

ACTION_NAMES = ["Hold", "Buy", "Sell"]
CLR = {
    "Q-Learning":  "#2196F3",
    "DQN":         "#4CAF50",
    "Buy-and-Hold": "#FF9800",
}


# ── Helpers ────────────────────────────────────────────────────────────
def _load(name: str):
    with open(os.path.join(RESULTS_DIR, name), "rb") as f:
        return pickle.load(f)


def _smooth(x, w: int = 5):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _save(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 1: Training rewards ─────────────────────────────────────────
def fig_training_rewards(r: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Cumulative Reward per Training Episode", fontsize=13, fontweight="bold")

    for ax, key, label in [
        (axes[0], "ql_ep_rewards",  "Q-Learning"),
        (axes[1], "dqn_ep_rewards", "DQN"),
    ]:
        color = CLR[label]
        vals  = r[key]
        eps   = list(range(1, len(vals) + 1))

        ax.plot(eps, vals, color=color, alpha=0.35, linewidth=1)
        if len(vals) >= 5:
            s = _smooth(vals)
            # smooth has len = n - 4; centre on episodes 3 … n-2
            xs = list(range(3, len(vals) - 1))
            ax.plot(xs, s, color=color, linewidth=2.5, label="5-ep moving avg")

        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Figure 2: Portfolio trajectories ──────────────────────────────────
def fig_portfolio(r: dict):
    n  = min(len(r["ql_pv"]), len(r["dqn_pv"]), len(r["bnh_pv"]))
    xs = list(range(n))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(xs, r["bnh_pv"][:n], color=CLR["Buy-and-Hold"], linewidth=2,   label="Buy-and-Hold", zorder=3)
    ax.plot(xs, r["ql_pv"][:n],  color=CLR["Q-Learning"],  linewidth=2,   label="Q-Learning",   zorder=2)
    ax.plot(xs, r["dqn_pv"][:n], color=CLR["DQN"],         linewidth=2,   label="DQN",          zorder=1)

    ax.set_title("Out-of-Sample Portfolio Value  (Jul 2020 – Dec 2025)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Trading Day (test set)")
    ax.set_ylabel("Portfolio Value  (initial = 1.00)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    return fig


# ── Figure 3: Action distribution ─────────────────────────────────────
def fig_action_dist(r: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Action Distribution on Test Set", fontsize=13, fontweight="bold")

    for ax, key, label in [
        (axes[0], "ql_actions",  "Q-Learning"),
        (axes[1], "dqn_actions", "DQN"),
    ]:
        color   = CLR[label]
        actions = r[key]
        counts  = [actions.count(i) for i in range(3)]
        total   = len(actions)
        pcts    = [100 * c / total for c in counts]

        bars = ax.bar(ACTION_NAMES, pcts, color=color, edgecolor="white", width=0.5)
        for bar, p in zip(bars, pcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{p:.1f}%",
                ha="center", va="bottom", fontsize=11,
            )
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Frequency (%)")
        ax.set_ylim(0, max(pcts) * 1.25 + 5)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ── Figure 4: DQN loss ─────────────────────────────────────────────────
def fig_dqn_loss(r: dict):
    losses = r["dqn_ep_losses"]
    eps    = list(range(1, len(losses) + 1))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eps, losses, color=CLR["DQN"], linewidth=2,
            marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.5)
    ax.set_title("DQN Bellman Loss per Episode (Training)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Squared Bellman Error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ── Figure 5: Metrics table ────────────────────────────────────────────
def fig_metrics_table(r: dict):
    rows = []
    for name, key in [
        ("Q-Learning",  "ql_metrics"),
        ("DQN",         "dqn_metrics"),
        ("Buy-and-Hold", "bnh_metrics"),
    ]:
        m = r[key]
        rows.append([
            name,
            f"{m['total_return']*100:+.2f}%",
            f"{m['sharpe']:+.3f}",
            f"{m['max_drawdown']*100:.2f}%",
        ])

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Method", "Total Return", "Sharpe Ratio", "Max Drawdown"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ECEFF1")

    ax.set_title(
        "Test Set Performance Summary  (Jul 2020 – Dec 2025)",
        fontsize=12, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    return fig


# ── Figure 6: Ablation ─────────────────────────────────────────────────
def fig_ablation(e: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Q-Learning Ablation Studies", fontsize=13, fontweight="bold")

    # ── Panel A: discount factor ──
    ax  = axes[0]
    ax2 = ax.twinx()
    gs  = e["gammas"]
    xs  = [str(g) for g in gs]
    rets    = [e["ablation_gamma"][g]["metrics"]["total_return"] * 100 for g in gs]
    sharpes = [e["ablation_gamma"][g]["metrics"]["sharpe"] for g in gs]

    ax.bar(xs, rets, color="#2196F3", alpha=0.75, width=0.4, label="Return (%)")
    ax2.plot(xs, sharpes, color="#F44336", marker="o", linewidth=2, label="Sharpe")

    ax.set_xlabel("Discount Factor γ")
    ax.set_ylabel("Total Return (%)")
    ax2.set_ylabel("Sharpe Ratio")
    ax.set_title("Effect of Discount Factor")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel B: bin count ──
    ax  = axes[1]
    ax3 = ax.twinx()
    bs  = e["bin_counts"]
    xs2 = [str(b) for b in bs]
    rets2    = [e["ablation_bins"][b]["metrics"]["total_return"] * 100 for b in bs]
    sharpes2 = [e["ablation_bins"][b]["metrics"]["sharpe"] for b in bs]

    ax.bar(xs2, rets2, color="#4CAF50", alpha=0.75, width=0.4, label="Return (%)")
    ax3.plot(xs2, sharpes2, color="#F44336", marker="o", linewidth=2, label="Sharpe")

    ax.set_xlabel("Number of Bins B per Feature")
    ax.set_ylabel("Total Return (%)")
    ax3.set_ylabel("Sharpe Ratio")
    ax.set_title("Effect of Discretisation Bin Count")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)

    r = _load("results.pkl")
    e = _load("experiments.pkl")

    print("Generating figures …")
    _save(fig_training_rewards(r), "fig1_training_rewards.png")
    _save(fig_portfolio(r),        "fig2_portfolio_trajectories.png")
    _save(fig_action_dist(r),      "fig3_action_distribution.png")
    _save(fig_dqn_loss(r),         "fig4_dqn_loss.png")
    _save(fig_metrics_table(r),    "fig5_metrics_table.png")
    _save(fig_ablation(e),         "fig6_ablation.png")

    print(f"\nAll 6 figures saved to {FIG_DIR}/")
