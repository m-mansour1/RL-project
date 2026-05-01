"""Run the full RL technical-indicator ablation study.

Example:
    python main.py --data data/AAPL_dataset.csv --output results
    python main.py --data data/AAPL_dataset.csv --output results_fast --fast
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ablation import run_full_ablation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/AAPL_dataset.csv", help="Path to OHLCV CSV dataset.")
    parser.add_argument("--output", type=str, default="results", help="Directory where outputs will be saved.")
    parser.add_argument("--fast", action="store_true", help="Run a very short smoke test.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    results = run_full_ablation(dataset_path=args.data, output_dir=args.output, fast_mode=args.fast)
    print("\nFinished RL indicator ablation study.")
    print(f"Outputs saved to: {results['output_dir']}")
    print("\nTop rows of metrics:")
    print(results["metrics"].head().to_string(index=False))


if __name__ == "__main__":
    main()
