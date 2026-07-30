"""Acceptance tests for `src.analysis`.

Most of these are *property* tests: instead of checking one hard-coded output,
they check a mathematical identity that must hold for any correct implementation.
That is a far stronger form of testing, and it is how you should think about
verifying quantitative code generally — you rarely know the right answer in
advance, but you always know properties it must satisfy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    TRADING_DAYS_PER_YEAR,
    average_pairwise_correlation,
    correlation_matrix,
    first_eigenvalue_share,
    log_returns,
    rolling_average_pairwise_correlation,
    rolling_volatility,
    simple_returns,
)


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #

def test_log_returns_of_constant_price_are_zero(business_days):
    flat = pd.DataFrame({"AAA": np.full(len(business_days), 100.0)}, index=business_days)
    r = log_returns(flat)
    assert r["AAA"].iloc[0] != r["AAA"].iloc[0], "first row must be NaN"
    np.testing.assert_allclose(r["AAA"].iloc[1:].to_numpy(), 0.0, atol=1e-12)


def test_log_returns_telescope(clean_wide):
    """Sum of daily log returns == log of the total price ratio.

    This is THE defining property of log returns and the reason they are the
    default in time-series work. If this fails you have computed something else.
    """
    r = log_returns(clean_wide)
    for ticker in clean_wide.columns:
        total = np.log(clean_wide[ticker].iloc[-1] / clean_wide[ticker].iloc[0])
        assert r[ticker].sum() == pytest.approx(total, rel=1e-10)


def test_simple_and_log_returns_agree_for_small_moves(clean_wide):
    """ln(1+x) ~ x for small x. Daily equity returns are small, which is why
    people sometimes get away with conflating the two. Know the approximation
    and know where it breaks: at +50% the two differ by about 10 points."""
    s, l = simple_returns(clean_wide), log_returns(clean_wide)
    diff = (s - l).abs().to_numpy()
    assert np.nanmax(diff) < 1e-3


def test_returns_preserve_index_and_columns(clean_wide):
    r = log_returns(clean_wide)
    assert r.shape == clean_wide.shape
    assert list(r.columns) == list(clean_wide.columns)
    pd.testing.assert_index_equal(r.index, clean_wide.index)


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #

def test_rolling_volatility_recovers_known_sigma(business_days):
    """Feed it iid normal returns with a known sigma and check it comes back.

    Calibrating an estimator against data whose truth you control is the single
    most useful debugging habit in quantitative work. Do it before you ever point
    a new estimator at real data.
    """
    sigma = 0.01
    rng = np.random.default_rng(0)
    n = 3000
    idx = pd.bdate_range("2010-01-01", periods=n)
    r = pd.DataFrame({"AAA": rng.normal(0, sigma, n)}, index=idx)

    vol = rolling_volatility(r, window=252, annualize=False)
    assert vol["AAA"].dropna().mean() == pytest.approx(sigma, rel=0.05)


def test_rolling_volatility_annualization_is_sqrt_time(clean_wide):
    """Variance scales with time, so standard deviation scales with sqrt(time).
    The ratio of annualized to daily vol must be exactly sqrt(252)."""
    r = log_returns(clean_wide)
    daily = rolling_volatility(r, window=63, annualize=False)
    annual = rolling_volatility(r, window=63, annualize=True)

    ratio = (annual / daily).to_numpy()
    ratio = ratio[~np.isnan(ratio)]
    np.testing.assert_allclose(ratio, np.sqrt(TRADING_DAYS_PER_YEAR), rtol=1e-10)


def test_rolling_volatility_leading_values_are_nan(clean_wide):
    r = log_returns(clean_wide)
    vol = rolling_volatility(r, window=21)
    assert vol["AAA"].iloc[:21].isna().all(), (
        "the first full window ends at row 21 because row 0 of returns is NaN"
    )


def test_rolling_volatility_is_non_negative(clean_wide):
    vol = rolling_volatility(log_returns(clean_wide), window=21)
    assert (vol.dropna() >= 0).all().all()


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #

def test_correlation_matrix_matches_pandas(clean_wide):
    """Your NumPy implementation must agree with the library to 10 decimals.

    Passing this means you can build a covariance matrix from the definition,
    which is the prerequisite for PCA, factor models, and portfolio optimization.
    """
    r = log_returns(clean_wide).dropna()
    mine = correlation_matrix(r)
    theirs = r.corr()

    np.testing.assert_allclose(mine.to_numpy(), theirs.to_numpy(), atol=1e-10)
    assert list(mine.columns) == list(theirs.columns)
    assert list(mine.index) == list(theirs.index)


def test_correlation_matrix_is_symmetric_with_unit_diagonal(clean_wide):
    c = correlation_matrix(log_returns(clean_wide).dropna()).to_numpy()
    np.testing.assert_allclose(np.diag(c), 1.0, atol=1e-12)
    np.testing.assert_allclose(c, c.T, atol=1e-12)


def test_correlation_of_identical_series_is_one(business_days):
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, len(business_days))
    r = pd.DataFrame({"AAA": x, "BBB": x}, index=business_days)
    assert correlation_matrix(r).loc["AAA", "BBB"] == pytest.approx(1.0)


def test_correlation_is_scale_invariant(business_days):
    """Correlation is invariant to affine rescaling — doubling one series must
    not change it. This is exactly what dividing by the standard deviations buys
    you, and it is why correlation is comparable across assets but covariance
    is not."""
    rng = np.random.default_rng(5)
    a = rng.normal(0, 0.01, len(business_days))
    b = rng.normal(0, 0.01, len(business_days))
    base = pd.DataFrame({"AAA": a, "BBB": b}, index=business_days)
    scaled = pd.DataFrame({"AAA": a, "BBB": 7.0 * b + 3.0}, index=business_days)

    assert correlation_matrix(base).loc["AAA", "BBB"] == pytest.approx(
        correlation_matrix(scaled).loc["AAA", "BBB"]
    )


def test_average_pairwise_correlation_excludes_diagonal():
    """With 3 assets all pairwise 0.5, the answer is 0.5. If you include the
    diagonal you get 0.667 — a bias of roughly 1/n that shrinks as the universe
    grows, which makes it easy to miss on a large panel and wrong everywhere."""
    c = pd.DataFrame(
        [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    assert average_pairwise_correlation(c) == pytest.approx(0.5)


def test_average_pairwise_correlation_counts_each_pair_once():
    c = pd.DataFrame(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    assert average_pairwise_correlation(c) == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------- #
# Rolling correlation
# --------------------------------------------------------------------------- #

def test_rolling_average_correlation_shape_and_leading_nans(clean_wide):
    r = log_returns(clean_wide).dropna()
    s = rolling_average_pairwise_correlation(r, window=63)

    assert isinstance(s, pd.Series)
    assert len(s) == len(r)
    assert s.iloc[:62].isna().all()
    assert s.iloc[62:].notna().all()


def test_rolling_average_correlation_within_bounds(clean_wide):
    r = log_returns(clean_wide).dropna()
    s = rolling_average_pairwise_correlation(r, window=63).dropna()
    assert (s >= -1.0).all() and (s <= 1.0).all()


def test_rolling_average_correlation_final_window_matches_static(clean_wide):
    """The last rolling value must equal the static calculation on the last
    `window` rows. Consistency between a rolling and a static estimator is the
    cheapest way to catch an off-by-one in the window."""
    r = log_returns(clean_wide).dropna()
    window = 63
    s = rolling_average_pairwise_correlation(r, window=window)
    expected = average_pairwise_correlation(correlation_matrix(r.iloc[-window:]))
    assert s.iloc[-1] == pytest.approx(expected, rel=1e-8)


# --------------------------------------------------------------------------- #
# Stretch goal
# --------------------------------------------------------------------------- #

def test_first_eigenvalue_share_is_a_fraction(clean_wide):
    r = log_returns(clean_wide).dropna()
    share = first_eigenvalue_share(correlation_matrix(r))
    n = r.shape[1]
    assert 1.0 / n <= share <= 1.0


def test_first_eigenvalue_share_is_one_for_perfectly_correlated_assets(business_days):
    """If every asset is the same asset, one component explains everything."""
    rng = np.random.default_rng(11)
    x = rng.normal(0, 0.01, len(business_days))
    r = pd.DataFrame({"AAA": x, "BBB": x, "CCC": x}, index=business_days)
    assert first_eigenvalue_share(correlation_matrix(r)) == pytest.approx(1.0, abs=1e-8)
