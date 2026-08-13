"""
Analysis layer: returns, volatility, and cross-sectional correlation.

This is where you learn NumPy properly. pandas is a convenience layer over NumPy;
the moment you need matrix algebra — covariance, eigendecomposition, portfolio
weights — you are in NumPy whether you notice or not. Several functions below
ask you to implement something twice: once with the library helper, once from the
definition. That is deliberate. You cannot reason about PCA, factor models, or
portfolio optimization if the covariance matrix is a black box to you.

YOUR JOB
--------
Replace every `raise NotImplementedError`. Signatures are fixed; tests depend on
them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Arithmetic returns: r_t = P_t / P_{t-1} - 1.

    The first row is NaN by construction — there is no prior price. Keep it as
    NaN rather than dropping it, so the index still aligns with `prices`.
    """
    returned_prices = prices.pct_change()
    return returned_prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Continuously compounded returns: r_t = ln(P_t) - ln(P_{t-1}).

    Why log returns are the default in quant research
    -------------------------------------------------
    1. **They add across time.** The log return over a month is the *sum* of the
       daily log returns, because ln(a/b) + ln(b/c) = ln(a/c). Simple returns
       compound multiplicatively, so aggregating them means products, which is
       both slower and numerically nastier. `tests/test_analysis.py` checks this
       telescoping property directly.

    2. **They are closer to symmetric.** A +50% then -50% simple return leaves
       you down 25%; the asymmetry makes the arithmetic mean of simple returns a
       biased description of growth. Log returns are symmetric around zero.

    3. **They fit the standard model.** Geometric Brownian motion — the basis of
       Black-Scholes and most of continuous-time finance — assumes log prices
       follow a random walk with normal increments. Working in logs means your
       data and your theory speak the same language.

    The catch, and know it before an interviewer asks: log returns do NOT add
    across *assets*. The log return of a portfolio is not the weighted average of
    its constituents' log returns. For cross-sectional portfolio construction you
    want simple returns. Time series → logs. Cross-section → simple.
    """
    log_info = np.log(prices) - np.log(prices.shift(1))
    return log_info


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #

def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Trailing standard deviation of returns over a rolling window.

    Parameters
    ----------
    window
        Number of observations. 21 ~ one trading month, 63 ~ one quarter.
    annualize
        If True, scale by sqrt(periods_per_year).

    Notes
    -----
    **Why sqrt.** Variance of a sum of independent increments adds linearly, so
    variance scales with time and standard deviation scales with its square root.
    Annualizing daily vol therefore means multiplying by sqrt(252), not 252. The
    "independent" assumption is doing real work here — returns are not perfectly
    independent, so this is an approximation, but it is the universal convention.

    **Why the window matters.** A 21-day window reacts quickly and is noisy; a
    252-day window is stable and stale. There is no correct answer, only a
    bias-variance tradeoff. Say which you chose and why in your findings note —
    that sentence is worth more than the chart.

    **Causality, again.** `rolling` uses trailing windows by default. If you ever
    reach for `center=True`, stop: a centred window at date t includes data from
    after t.

    **ddof.** pandas defaults to ddof=1 (sample std), NumPy's `np.std` defaults to
    ddof=0 (population). They differ. Pick one, know which, and be consistent —
    the test suite assumes ddof=1.
    """
    rolling_std = returns.rolling(window=window).std(ddof=1)
    if annualize:
        rolling_std *= np.sqrt(periods_per_year)
    return rolling_std


# --------------------------------------------------------------------------- #
# Correlation — implement twice, once from the definition
# --------------------------------------------------------------------------- #

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of asset returns.

    Returns
    -------
    Square DataFrame with tickers on both axes.

    Notes
    -----
    Implement this **from the definition using NumPy**, not by calling
    `DataFrame.corr()`. The test compares your output against the library
    version, so you get to verify your own linear algebra.

    Build it in this order — each step is a concept you will reuse constantly:

    1. Drop rows with any NaN, and note how many rows you lost. This is where
       `short_history` tickers do their damage: one ticker with 3 years of data
       in a 10-year panel silently truncates the sample for *every* pair.
       Complete-case deletion is the honest default; pairwise deletion gives you
       a matrix that can fail to be positive semi-definite, which will blow up
       any optimizer you feed it to later.

    2. De-mean each column. (Centering is what makes the next step a covariance
       rather than a raw second moment.)

    3. Covariance is `X.T @ X / (n - 1)` on the centred matrix. Write it as a
       matrix product. Understand why the shapes work: X is (n_obs, n_assets),
       so X.T @ X is (n_assets, n_assets).

    4. Correlation is covariance normalized by the outer product of the standard
       deviations. `np.outer(s, s)` gives you the divisor.

    5. Force exact symmetry and an exact unit diagonal at the end — floating
       point will leave you with 0.9999999999999998 on the diagonal, and later
       code (eigendecomposition, Cholesky) can be surprisingly unhappy about it.

    When this passes the test against `DataFrame.corr()`, you understand
    covariance matrices in a way that most people who use them do not.
    """
    clean = returns.dropna()
    centered_returns = clean.to_numpy() - clean.values.mean(axis=0)
    covariance_matrix = centered_returns.T @ centered_returns / (centered_returns.shape[0] - 1)
    std_dev = np.sqrt(np.diag(covariance_matrix))
    divisor = np.outer(std_dev, std_dev)
    corr = covariance_matrix / divisor
    corr = (corr + corr.T) / 2  # Force symmetry
    np.fill_diagonal(corr,1.0)
    return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)


def average_pairwise_correlation(corr: pd.DataFrame) -> float:
    """Mean of the off-diagonal entries of a correlation matrix.

    Notes
    -----
    The diagonal is all 1s and carries no information — including it biases the
    average upward by roughly 1/n. Use the strict upper triangle; each pair
    should be counted once, not twice. `np.triu_indices` with `k=1` is the tool.

    This single number is a decent proxy for "how much is everything moving
    together right now," which is the quantity your research question is about.
    """
    n = corr.shape[0]
    i, j = np.triu_indices(n,k=1)
    return corr.values[i, j].mean()


def rolling_average_pairwise_correlation(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """Average pairwise correlation computed over a rolling window.

    Returns
    -------
    Series indexed by date. NaN for the first `window - 1` dates.

    Notes
    -----
    This is the core measurement for the research question:

        *Does the average pairwise correlation of daily equity returns rise
        during high-volatility periods?*

    If it does, it has a large practical consequence: diversification fails
    exactly when you need it. A portfolio whose risk model assumes a static
    correlation matrix will understate drawdowns in a crisis, because the
    correlations it was calibrated on are calm-period correlations. This is
    roughly what happened to a lot of quant portfolios in August 2007 and again
    in March 2020.

    A loop over windows is acceptable here — this is genuinely O(n_windows)
    matrix computations and there is no clean vectorization. But profile it. If
    it takes more than a few seconds, your inner loop is doing something silly
    (recomputing a DataFrame view, or re-dropping NaNs every iteration).

    Once it works, ask the follow-up questions. That is what makes this research
    rather than a plot:
      - Does correlation *lead* volatility, lag it, or move contemporaneously?
      - Is the relationship there in both directions, or only in drawdowns?
      - How much of it survives if you exclude 2020?
      - Is the effect stronger within a sector than across sectors?
    """
    out = np.full(len(returns), np.nan)
    for i in range(window - 1, len(returns)):
        window_returns = returns.iloc[i - window + 1:i + 1]
        out[i] = average_pairwise_correlation(correlation_matrix(window_returns))
    return pd.Series(out, index=returns.index)


# --------------------------------------------------------------------------- #
# Stretch goal — attempt only after everything above passes
# --------------------------------------------------------------------------- #

def first_eigenvalue_share(corr: pd.DataFrame) -> float:
    """Fraction of total variance explained by the largest principal component.

    Returns
    -------
    Largest eigenvalue divided by the sum of eigenvalues (which equals n for a
    correlation matrix, since each variable has unit variance — verify that
    yourself, it is a good sanity check).

    Notes
    -----
    This is PCA, arriving two phases early, and it is worth meeting now because
    it makes the concept concrete before you meet the formalism.

    On equity returns the first principal component is almost always "the
    market" — every stock loads on it with the same sign. Its share of total
    variance is another measure of how much everything is moving as one thing.
    Plot it next to your average pairwise correlation; they should track each
    other closely, and understanding *why* they must is a genuinely good
    exercise in linear algebra intuition.

    Use `np.linalg.eigvalsh`, not `np.linalg.eigvals` — correlation matrices are
    symmetric, and the symmetric solver is both faster and guaranteed to return
    real eigenvalues rather than complex ones with zero imaginary part.
    """
    eigenvalues = np.linalg.eigvalsh(corr)
    return float(eigenvalues.max() / eigenvalues.sum())
