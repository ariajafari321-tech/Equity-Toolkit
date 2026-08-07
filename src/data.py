"""
Data layer: acquire, inspect, and clean daily equity price data.

DESIGN PRINCIPLE FOR THIS MODULE
--------------------------------
Every function here must be *causal*: the value this module produces for date `t`
may depend only on data from dates <= t. Never on data from t+1 onward.

This is not a stylistic preference. It is the difference between research and
self-deception. A cleaning step that uses future information (filling a gap with
a column-wide mean, winsorizing against the full-sample distribution, dropping a
ticker because it later gets delisted) will produce a backtest that is profitable
on paper and worthless in production. The failure is silent — there is no error
message, just a Sharpe ratio that is a lie.

`tests/test_data.py::test_cleaning_is_causal` enforces this mechanically. Read
that test before you write these functions.

YOUR JOB
--------
Every function body below is `raise NotImplementedError`. Replace them.
The docstrings are the specification. The tests are the acceptance criteria.
Do not change the function signatures — the tests depend on them.
"""

from __future__ import annotations


from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Columns every raw price file is expected to carry.
RAW_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

# Defect records are (ticker, date, kind, detail). `date` may be NaT for
# ticker-level defects that aren't tied to a specific day.
DEFECT_COLUMNS = ["ticker", "date", "kind", "detail"]


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #

def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    out_path: str | Path,
) -> Path:
    
    """Download daily OHLCV bars and write them to a single tidy CSV.

    Parameters
    ----------
    tickers
        Ticker symbols to fetch.
    start, end
        ISO date strings, e.g. "2015-01-01". `end` is exclusive, matching the
        convention of the underlying API.
    out_path
        Where to write the CSV.

    Returns
    -------
    Path to the file written.

    Notes
    -----
    Write the file in LONG (tidy) format — one row per (date, ticker) — with the
    columns in `RAW_COLUMNS`. Long format is the right storage layout because it
    handles tickers with different histories without padding, and it appends
    cleanly when you later extend the date range. You will reshape to wide for
    analysis; that is a view, not a storage decision.

    Use `adj_close` for anything return-related. `close` is the raw printed
    price, which jumps discontinuously on splits and dividends — a 2-for-1 split
    shows up as a -50% return that never happened. `adj_close` is restated to
    remove those. (It also means adjusted history *changes* over time as new
    dividends accrue, which is a real and subtle source of irreproducibility.
    Note the download date in your README.)

    Do NOT silently skip tickers that fail to download. Collect the failures and
    report them — a silently missing ticker is a survivorship-flavoured bug.
    """
    df = yf.download(tickers, start=start, end=end, auto_adjust=False).stack(level=1).reset_index()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df[RAW_COLUMNS]
    df.to_csv(out_path, index=False)
    return Path(out_path)


def load_raw(path: str | Path) -> pd.DataFrame:



    """Load the tidy CSV written by `download_prices`.

    Returns
    -------
    DataFrame with columns `RAW_COLUMNS`, where `date` is datetime64 (not string)
    and the five price/volume columns are numeric.

    Notes
    -----
    Parse dates on load rather than converting afterwards. Assert the columns you
    expect are present and fail loudly if they aren't — a data loader that
    silently returns a half-empty frame will cost you an afternoon later.
    """
    df = pd.read_csv(path, parse_dates=['date'])
    df = df[RAW_COLUMNS]
    return df


# --------------------------------------------------------------------------- #
# Reshaping
# --------------------------------------------------------------------------- #

def to_wide(long_df: pd.DataFrame, field: str = "adj_close") -> pd.DataFrame:
    
    """Pivot long-format data into a wide price matrix.

    Returns
    -------
    DataFrame indexed by date (sorted ascending), one column per ticker, values
    taken from `field`. Columns sorted alphabetically for determinism.

    Notes
    -----
    The union of all dates becomes the index, so a ticker that didn't trade on
    some date gets NaN there. That is correct and informative — do not fill it
    here. Distinguishing "no trade" from "missing data" is the job of
    `find_defects`.
    """

    df_pivoted = long_df.pivot(index="date", columns="ticker", values=field)
    df_pivoted = df_pivoted.sort_index().sort_index(axis=1)
    return df_pivoted


# --------------------------------------------------------------------------- #
# Inspection — the heart of this project
# --------------------------------------------------------------------------- #

def find_defects(
    wide: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    min_obs: int = 500,
    extreme_return: float = 0.20,
    stale_run: int = 3,
) -> pd.DataFrame:
    """Scan a wide price matrix and return every data quality problem found.

    Parameters
    ----------
    wide
        Wide price matrix, as produced by `to_wide`.
    volume
        Optional wide volume matrix on the same index and columns. If supplied,
        zero-volume days are detected too. If None, that check is skipped.
    min_obs, extreme_return, stale_run
        Thresholds — see the defect kinds below. Exposed as parameters rather
        than hard-coded because a sensible threshold for large-cap US equities
        is not sensible for small caps or crypto, and a function that hides its
        assumptions is a function you will misuse in six months.

    Returns
    -------
    DataFrame with columns `DEFECT_COLUMNS`. One row per defect. Empty
    DataFrame (with those columns) if the data is clean.

    The `kind` field must be one of:

    ``"missing"``
        A NaN price on a date where at least one other ticker has a price,
        **within that ticker's own trading history** — i.e. between its first
        and last non-NaN observation. Leading NaNs (the name hadn't listed yet)
        are not missing data, they are absence of data, and conflating the two
        floods your log with thousands of meaningless rows. That distinction is
        the whole reason `short_history` exists as a separate kind.

    ``"zero_volume"``
        A day with a price but zero or NaN volume. Often a stale quote carried
        forward by the vendor rather than a real trade. Requires `volume`.

    ``"extreme_return"``
        A one-day absolute log return above `extreme_return`. Sometimes real —
        earnings, biotech readouts, 2020-03-16. Often a bad print or an
        unadjusted corporate action. The point is not to delete these, it is to
        *look at them*. Detail should record the return value.

    ``"stale_price"``
        `stale_run` or more consecutive days at an identical price. In liquid
        names this essentially never happens naturally.

    ``"short_history"``
        Ticker-level. Fewer than `min_obs` non-NaN observations. `date` is NaT.
        These names will quietly wreck a correlation matrix — see the note in
        `analysis.correlation_matrix`.

    Notes
    -----
    This function is the actual deliverable of Milestone 1. Anyone can call
    `.pct_change()`. Knowing what is wrong with your data before you model it is
    the thing that separates a researcher from someone running a library.

    Design it so the returned frame is directly readable: a person should be able
    to sort by `kind`, look at the top 20 rows, and understand their dataset.

    Vectorize. If you find yourself writing `for date in wide.index:` you are
    doing it wrong — that loop will take minutes on 10 years x 30 tickers and
    milliseconds if you express it as column operations.
    """

    rows = []
    for ticker in wide.columns:
            count = wide[ticker].notna().sum()
    
            if count < min_obs:
                rows.append({"ticker": ticker, 
                             "date": pd.NaT, 
                             "kind": "short_history", 
                             "detail": f"{count} observations"
                        })

            series = wide[ticker]
            missing_today = series.isna()
            market_open = wide.notna().any(axis=1)
            first = series.first_valid_index()
            last = series.last_valid_index()
            inside_history = (series.index >= first) & (series.index <= last)
            missing = missing_today & market_open & inside_history
            for d in series.index[missing]:
                rows.append({"ticker": ticker, "date": d, "kind": "missing", "detail": "no price"})

            log_ret = np.log(series).diff()
            big = log_ret.abs() > extreme_return

            for d, r in log_ret[big].items():
                rows.append({
            "ticker": ticker,
            "date": d,
            "kind": "extreme_return",
            "detail": f"log return {r:.4f}",
        })



            same = (series.diff() == 0) & series.notna()
            grp = (~same).cumsum()
            run = same.groupby(grp).transform("sum")
            stale = run >= (stale_run - 1)
            for d in series.index[stale]:
                rows.append({"ticker": ticker, "date": d, "kind": "stale_price", "detail": "repeated price"})



    if not rows:
        return pd.DataFrame(columns=DEFECT_COLUMNS)
    return pd.DataFrame(rows)[DEFECT_COLUMNS]



def defect_summary(defects: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a defect log into a per-ticker, per-kind count table.

    Returns
    -------
    DataFrame indexed by ticker, one column per defect kind, values are counts.
    Missing combinations are 0, not NaN.

    Notes
    -----
    This is the table that goes in your README. It is how you show, in one
    glance, that you know your data.
    """
    summary = defects.groupby(["ticker", "kind"]).size().unstack(fill_value=0)
    return summary


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

def align_calendar(wide: pd.DataFrame, min_coverage: float = 0.8) -> pd.DataFrame:
    """Restrict to dates where enough tickers actually traded.

    Parameters
    ----------
    min_coverage
        Keep a date only if at least this fraction of tickers have a non-NaN
        price on it.

    Notes
    -----
    Why this exists: if your universe spans multiple exchanges or includes a
    ticker with a different holiday calendar, the union index from `to_wide`
    contains dates where one lonely name traded and everything else is NaN.
    Those rows produce garbage cross-sectional statistics.

    This is a *row filter*, not an imputation. Removing a date is causal;
    inventing a price is not.
    """
    coverage = wide.notna().sum(axis=1) / wide.shape[1]
    filtered_coverage = coverage>= min_coverage
    return wide.loc[filtered_coverage]


def clean_prices(wide: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Produce an analysis-ready price matrix.

    Parameters
    ----------
    max_gap
        Maximum number of consecutive missing days to bridge by carrying the
        last observation forward. Gaps longer than this are left as NaN.

    Notes
    -----
    Rules, and the reasoning behind each:

    1. **Forward-fill only.** Carrying yesterday's price into today uses only
       information available today. Back-filling, interpolating, or filling with
       any statistic computed over the whole column uses the future. Your
       Module 4 notebook used ``fillna(df["volume"].median())`` — on a time
       series that is exactly this mistake, and it is the reason this project
       exists.

    2. **Bound the fill.** Bridging a two-day gap is reasonable. Bridging a
       three-month gap manufactures three months of zero returns, which crushes
       your volatility estimate and inflates every Sharpe ratio downstream.
       Leave long gaps as NaN and let them propagate honestly.

    3. **Never fill leading NaNs.** A ticker that IPO'd in 2019 has no 2015
       price. There is nothing to carry forward, and inventing one is fabrication.

    The causality test in `tests/test_data.py` will catch violations of rule 1.
    Run it early and often.
    """
    cleaned_wide = wide.ffill(limit=max_gap, inplace=False)
    return cleaned_wide

