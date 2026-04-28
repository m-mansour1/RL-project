"""
Single entry point — runs all experiments and produces all figures.

Usage:
    python run_all.py

Steps executed:
    1. train.py       — trains Q-Learning & DQN, evaluates on test set
    2. experiments.py — ablation on γ and bin count
    3. plot_results.py — saves 6 figures to results/figures/
"""

import subprocess
import sys


def run(script: str):
    sep = "=" * 60
    print(f"\n{sep}\n  {script}\n{sep}")
    subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    run("train.py")
    run("experiments.py")
    run("plot_results.py")
    print("\n✓ Done. Results and figures are in the results/ directory.")
