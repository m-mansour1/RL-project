"""Feature engineering for OHLCV trading datasets.

Accepts a CSV with at least: Date, Open, High, Low, Close, Volume.
If common technical indicators already exist, they are reused. Missing indicators are computed.
Feature normalization is performed without using test-period statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List

import numpy as np
import pandas as pd

from .config import RAW_REQUIRED_COLUMNS


@dataclass
class Normalizer:
    """Train-only z-score normalizer for selected columns."""
    means: Dict[str, float]
    stds: Dict[str, float]

    def transform(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        out = df.copy()
        for col in columns:
            std = self.stds.get(col, 1.0)
            if not np.isfinite(std) or std == 0:
                std = 1.0
            out[col] = (out[col] - self.means.get(col, 0.0)) / std
        return out


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(20, window // 5)).mean()
    std = series.rolling(window=window, min_periods=max(20, window // 5)).std()
    return (series - mean) / std.replace(0, np.nan)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add/repair all features used in the ablation."""
    out = df.copy()

    if "Log_Return" not in out.columns:
        out["Log_Return"] = np.log(out["Close"] / out["Close"].shift(1))

    for w in (20, 50, 200):
        if f"SMA_{w}" not in out.columns:
            out[f"SMA_{w}"] = out["Close"].rolling(w).mean()
    for w in (20, 50):
        if f"EMA_{w}" not in out.columns:
            out[f"EMA_{w}"] = out["Close"].ewm(span=w, adjust=False).mean()

    if "RSI" not in out.columns:
        out["RSI"] = compute_rsi(out["Close"], 14)
    if "MACD" not in out.columns:
        ema12 = out["Close"].ewm(span=12, adjust=False).mean()
        ema26 = out["Close"].ewm(span=26, adjust=False).mean()
        out["MACD"] = ema12 - ema26
    if "MACD_Signal" not in out.columns:
        out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    if "MACD_Hist" not in out.columns:
        out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    if "BB_Middle" not in out.columns:
        out["BB_Middle"] = out["Close"].rolling(20).mean()
    if "BB_Upper" not in out.columns or "BB_Lower" not in out.columns:
        std20 = out["Close"].rolling(20).std()
        out["BB_Upper"] = out["BB_Middle"] + 2 * std20
        out["BB_Lower"] = out["BB_Middle"] - 2 * std20
    if "BB_Width" not in out.columns:
        out["BB_Width"] = (out["BB_Upper"] - out["BB_Lower"]) / out["BB_Middle"]
    if "BB_PctB" not in out.columns:
        denom = (out["BB_Upper"] - out["BB_Lower"]).replace(0, np.nan)
        out["BB_PctB"] = (out["Close"] - out["BB_Lower"]) / denom

    out["RSI_norm"] = out["RSI"] / 100.0
    out["MACD_z"] = rolling_zscore(out["MACD"])
    out["MACD_Signal_z"] = rolling_zscore(out["MACD_Signal"])
    out["MACD_Hist_z"] = rolling_zscore(out["MACD_Hist"])

    out["SMA_20_ratio"] = out["SMA_20"] / out["Close"] - 1
    out["SMA_50_ratio"] = out["SMA_50"] / out["Close"] - 1
    out["SMA_200_ratio"] = out["SMA_200"] / out["Close"] - 1
    out["EMA_20_ratio"] = out["EMA_20"] / out["Close"] - 1
    out["EMA_50_ratio"] = out["EMA_50"] / out["Close"] - 1

    out["BB_Width_z"] = rolling_zscore(out["BB_Width"])
    out["BB_PctB_clipped"] = out["BB_PctB"].clip(-1, 2)
    out["Volume_z"] = rolling_zscore(np.log1p(out["Volume"]))

    return out.replace([np.inf, -np.inf], np.nan)


def chronological_split(df: pd.DataFrame, train_end: str = "2017-01-01", val_end: str = "2021-01-01") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["Date"] < train_end].copy()
    val = df[(df["Date"] >= train_end) & (df["Date"] < val_end)].copy()
    test = df[df["Date"] >= val_end].copy()
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(f"Empty split produced. Sizes: train={len(train)}, val={len(val)}, test={len(test)}. Adjust split dates.")
    return train, val, test


def fit_train_normalizer(train_df: pd.DataFrame, columns: List[str]) -> Normalizer:
    means = {c: float(train_df[c].mean()) for c in columns if c in train_df.columns}
    stds = {c: float(train_df[c].std()) for c in columns if c in train_df.columns}
    return Normalizer(means, stds)


def prepare_dataset(path: str, train_end: str, val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_ohlcv(path)
    df = add_technical_indicators(df)

    model_cols = [
        "Log_Return", "Volume_z", "RSI_norm", "MACD_z", "MACD_Signal_z", "MACD_Hist_z",
        "SMA_20_ratio", "SMA_50_ratio", "SMA_200_ratio", "EMA_20_ratio", "EMA_50_ratio",
        "BB_Width_z", "BB_PctB_clipped",
    ]
    df = df.dropna(subset=model_cols + ["Close"]).reset_index(drop=True)
    train, val, test = chronological_split(df, train_end=train_end, val_end=val_end)

    standardize_cols = [
        "Log_Return", "Volume_z", "MACD_z", "MACD_Signal_z", "MACD_Hist_z",
        "SMA_20_ratio", "SMA_50_ratio", "SMA_200_ratio", "EMA_20_ratio", "EMA_50_ratio", "BB_Width_z",
    ]
    norm = fit_train_normalizer(train, standardize_cols)
    return (
        norm.transform(train, standardize_cols).reset_index(drop=True),
        norm.transform(val, standardize_cols).reset_index(drop=True),
        norm.transform(test, standardize_cols).reset_index(drop=True),
    )
