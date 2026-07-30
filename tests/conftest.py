"""Shared pytest fixtures and import-path setup.

`conftest.py` is discovered automatically by pytest — you never import it
yourself. Anything defined here as a fixture is available by name in any test
file in this directory tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `src` importable when running `pytest` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def business_days() -> pd.DatetimeIndex:
    """750 business days starting 2020-01-01 (~3 years).

    Long enough that the default `min_obs=500` short-history threshold doesn't
    fire on the healthy tickers — fixtures have to be consistent with the
    thresholds the code under test uses, or you end up weakening real checks to
    make an arbitrary fixture pass.
    """
    return pd.bdate_range("2020-01-01", periods=750)


@pytest.fixture
def clean_wide(business_days) -> pd.DataFrame:
    """A well-behaved wide price matrix: 4 tickers, no defects.

    Prices are generated as a geometric random walk with a fixed seed, so every
    test run sees identical data. Reproducibility is not a nicety — a test that
    fails one run in twenty is worse than no test.
    """
    rng = np.random.default_rng(42)
    n_days, tickers = len(business_days), ["AAA", "BBB", "CCC", "DDD"]
    shocks = rng.normal(loc=0.0003, scale=0.012, size=(n_days, len(tickers)))
    prices = 100 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(prices, index=business_days, columns=tickers)


@pytest.fixture
def defective_wide(clean_wide) -> pd.DataFrame:
    """`clean_wide` with four deliberately planted defects.

    Planted (positions chosen so they don't overlap):
      - AAA rows 10..12   -> NaN            (a 3-day gap, bridgeable)
      - BBB rows 50..70   -> NaN            (a 21-day gap, too long to bridge)
      - CCC row  100      -> +35% jump      (extreme return)
      - DDD rows 150..154 -> constant price (stale)
      - EEE               -> a new ticker with only the last 100 days
    """
    df = clean_wide.copy()
    df.iloc[10:13, df.columns.get_loc("AAA")] = np.nan
    df.iloc[50:71, df.columns.get_loc("BBB")] = np.nan
    df.iloc[100, df.columns.get_loc("CCC")] *= 1.35
    df.iloc[150:155, df.columns.get_loc("DDD")] = df.iloc[149, df.columns.get_loc("DDD")]

    short = pd.Series(np.nan, index=df.index, name="EEE")
    short.iloc[-100:] = 50.0 * np.exp(
        np.cumsum(np.random.default_rng(7).normal(0, 0.01, 100))
    )
    return df.join(short)


@pytest.fixture
def long_df(clean_wide) -> pd.DataFrame:
    """`clean_wide` melted back into tidy long format with full OHLCV columns."""
    out = (
        clean_wide.stack()
        .rename("adj_close")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker"})
    )
    out.columns = ["date", "ticker", "adj_close"]
    out["open"] = out["adj_close"] * 0.999
    out["high"] = out["adj_close"] * 1.004
    out["low"] = out["adj_close"] * 0.996
    out["close"] = out["adj_close"]
    out["volume"] = 1_000_000
    return out[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
