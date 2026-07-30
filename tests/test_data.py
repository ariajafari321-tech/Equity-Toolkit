"""Acceptance tests for `src.data`.

Read these before you write any implementation. In this project the tests ARE
the specification — the docstrings tell you why, the tests tell you exactly what.

Run them with, from the repo root:

    pytest -v

Right now every one fails with NotImplementedError. That is the correct starting
state. Make them go green one at a time, committing after each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    DEFECT_COLUMNS,
    align_calendar,
    clean_prices,
    defect_summary,
    find_defects,
    to_wide,
)


# --------------------------------------------------------------------------- #
# Reshaping
# --------------------------------------------------------------------------- #

def test_to_wide_shape_and_index(long_df, clean_wide):
    wide = to_wide(long_df, field="adj_close")

    assert list(wide.columns) == sorted(clean_wide.columns)
    assert wide.index.is_monotonic_increasing
    assert isinstance(wide.index, pd.DatetimeIndex)
    assert wide.shape == clean_wide.shape


def test_to_wide_preserves_values(long_df, clean_wide):
    wide = to_wide(long_df, field="adj_close")
    np.testing.assert_allclose(
        wide["AAA"].to_numpy(), clean_wide["AAA"].to_numpy(), rtol=1e-10
    )


def test_to_wide_does_not_fill_gaps(long_df):
    """A ticker missing on a date must stay NaN. Filling is a later decision."""
    trimmed = long_df[~((long_df.ticker == "AAA") & (long_df.date == long_df.date.min()))]
    wide = to_wide(trimmed, field="adj_close")
    assert pd.isna(wide["AAA"].iloc[0])


# --------------------------------------------------------------------------- #
# Defect detection
# --------------------------------------------------------------------------- #

def test_find_defects_returns_expected_schema(clean_wide):
    defects = find_defects(clean_wide)
    assert list(defects.columns) == DEFECT_COLUMNS


def test_find_defects_finds_nothing_in_clean_data(clean_wide):
    """A clean random walk should trip no detectors. If this fails, your
    thresholds are too tight and your defect log will be pure noise."""
    assert len(find_defects(clean_wide)) == 0


def test_find_defects_flags_missing_prices(defective_wide):
    defects = find_defects(defective_wide)
    aaa_missing = defects[(defects.ticker == "AAA") & (defects.kind == "missing")]
    assert len(aaa_missing) == 3


def test_find_defects_flags_extreme_return(defective_wide):
    defects = find_defects(defective_wide)
    extreme = defects[(defects.ticker == "CCC") & (defects.kind == "extreme_return")]
    assert len(extreme) >= 1


def test_find_defects_flags_stale_price(defective_wide):
    defects = find_defects(defective_wide)
    stale = defects[(defects.ticker == "DDD") & (defects.kind == "stale_price")]
    assert len(stale) >= 1


def test_find_defects_flags_short_history(defective_wide):
    defects = find_defects(defective_wide)
    short = defects[(defects.ticker == "EEE") & (defects.kind == "short_history")]
    assert len(short) == 1, "short_history is ticker-level: exactly one row"
    assert pd.isna(short.iloc[0]["date"]), "ticker-level defects carry NaT for date"


def test_defect_summary_is_a_count_table(defective_wide):
    summary = defect_summary(find_defects(defective_wide))
    assert summary.index.name == "ticker" or "AAA" in summary.index
    assert (summary.fillna(-1) >= 0).all().all(), "missing combinations must be 0, not NaN"


# --------------------------------------------------------------------------- #
# Calendar alignment
# --------------------------------------------------------------------------- #

def test_align_calendar_drops_thin_dates(clean_wide):
    df = clean_wide.copy()
    orphan = pd.Timestamp("2020-07-04")          # only one ticker "trades"
    df.loc[orphan] = np.nan
    df.loc[orphan, "AAA"] = 123.45
    df = df.sort_index()

    aligned = align_calendar(df, min_coverage=0.8)
    assert orphan not in aligned.index


def test_align_calendar_keeps_full_dates(clean_wide):
    aligned = align_calendar(clean_wide, min_coverage=0.8)
    assert len(aligned) == len(clean_wide)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

def test_clean_prices_bridges_short_gaps(defective_wide):
    """AAA has a 3-day gap; with max_gap=5 it should be fully bridged."""
    cleaned = clean_prices(defective_wide, max_gap=5)
    assert cleaned["AAA"].iloc[10:13].notna().all()


def test_clean_prices_carries_last_value_forward(defective_wide):
    cleaned = clean_prices(defective_wide, max_gap=5)
    last_good = defective_wide["AAA"].iloc[9]
    assert cleaned["AAA"].iloc[10] == pytest.approx(last_good)
    assert cleaned["AAA"].iloc[12] == pytest.approx(last_good)


def test_clean_prices_respects_max_gap(defective_wide):
    """BBB has a 21-day gap. With max_gap=5, the first 5 days get bridged and
    the remainder must stay NaN — manufacturing 21 days of zero returns would
    collapse BBB's measured volatility."""
    cleaned = clean_prices(defective_wide, max_gap=5)
    assert cleaned["BBB"].iloc[50:55].notna().all()
    assert cleaned["BBB"].iloc[55:71].isna().all()


def test_clean_prices_never_fills_leading_nans(defective_wide):
    """EEE starts 100 days before the end. There is no earlier price to carry
    forward, and inventing one is fabrication, not cleaning."""
    cleaned = clean_prices(defective_wide, max_gap=5)
    assert cleaned["EEE"].iloc[:-100].isna().all()


def test_cleaning_is_causal(defective_wide):
    """THE IMPORTANT ONE.

    Cleaning the full history and then truncating at date T must give exactly the
    same answer as truncating at T and then cleaning. If it doesn't, your cleaner
    used information from after T — which means every backtest built on it is
    reporting a performance you could not have achieved.

    A common way to fail this: measuring a gap's total length, then deciding
    whether to fill it. On the truncated series a gap that runs past T looks
    short; on the full series it looks long. The fix is to fill forward with a
    limit, which only ever looks backwards in time.
    """
    cutoff = defective_wide.index[len(defective_wide) // 2]

    clean_then_cut = clean_prices(defective_wide, max_gap=5).loc[:cutoff]
    cut_then_clean = clean_prices(defective_wide.loc[:cutoff], max_gap=5)

    pd.testing.assert_frame_equal(clean_then_cut, cut_then_clean)


def test_clean_prices_does_not_mutate_input(defective_wide):
    """A function that silently edits its argument will eventually cost you a
    day of debugging. Copy before you modify."""
    before = defective_wide.copy()
    clean_prices(defective_wide, max_gap=5)
    pd.testing.assert_frame_equal(defective_wide, before)
