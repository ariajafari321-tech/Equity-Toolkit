# Equity Toolkit — Reference

Two halves. **Part 1–3** is what you built in `data.py` and why. **Part 4** is a primer for
`analysis.py`, so the vocabulary is familiar before you need it.

This is not a function dictionary — those are useless, and autocomplete already has one.
It's the *ideas*, and the handful of idioms that do the work.

---

# Part 1 — What you built

```
prices.csv
   │  load_raw            read the file, validate the columns
   ▼
long table  (one row per date+ticker)
   │  to_wide             reshape: dates down, tickers across
   ▼
wide matrix  (2,516 × 30)
   ├─ find_defects        inspect: report everything wrong
   │     └─ defect_summary   aggregate the report into a count table
   │
   │  align_calendar      drop dates where too few names traded
   │  clean_prices        bridge short gaps, causally
   ▼
analysis-ready matrix  →  analysis.py
```

The order matters and it's the argument of the whole project: **inspect before you clean, clean
before you model.** Most people skip straight to modelling and never learn what their data was.

## `download_prices`
Pulls OHLCV from the vendor, reshapes to long format, writes a CSV.

*Key idea:* store long, analyse wide. Long format handles tickers with different histories
without padding, and appends cleanly when you extend the date range. Wide is a *view* you
build when you need it, not a storage decision.

*Also:* `auto_adjust=False`, because you want `close` and `adj_close` as separate columns.
`adj_close` is restated for splits and dividends — without it a 2-for-1 split looks like a
−50% return that never happened.

## `load_raw`
Reads the CSV, parses dates, validates columns.

*Key idea:* **fail loudly at the boundary.** `df[RAW_COLUMNS]` raises if a column is missing.
`reindex` would have invented it full of NaN; `filter` would have silently dropped it. When you
want a guarantee, reach for the operation that *fails*, not the one that copes.

## `to_wide`
Long → wide. Dates become the index, tickers become columns.

*Key idea:* reshaping restructures, it doesn't decide anything. A ticker with no row for a date
comes out NaN and **stays** NaN. Distinguishing "didn't trade" from "missing data" is a later
function's job.

## `find_defects`
The heart of the project. Scans the matrix and returns a report — one row per problem.

Five detectors:

| kind | rule | why it matters |
|---|---|---|
| `missing` | NaN inside the ticker's own history, on a day others traded | a hole where there should be data |
| `extreme_return` | \|log return\| > 20% | real event, or bad print, or unadjusted corporate action |
| `stale_price` | ≥3 consecutive identical prices | a carried-forward quote pretending to be a trade |
| `short_history` | fewer than 500 observations | will silently truncate every correlation |
| `zero_volume` | price with no volume | (optional — needs a volume matrix) |

*Key idea:* **`missing` is restricted to the ticker's own trading history.** Without that, a name
that listed late produces hundreds of junk rows and the real problems drown. That distinction is
why `short_history` exists as its own kind — one row saying "this name is young" instead of 650
saying "no price today."

## `defect_summary`
Aggregates the report into a per-ticker, per-kind count table.

*Key idea:* `fill_value=0`. A zero means "we looked and found none." A NaN means "we didn't look."
In a data quality report those are completely different claims.

## `align_calendar`
Drops dates where fewer than `min_coverage` of tickers traded.

*Key idea:* a row filter, not an imputation. Removing a date uses only information about that
date. Inventing a price does not. This is why it's safe and `clean_prices` isn't.

## `clean_prices`
Forward-fills gaps, bounded by `max_gap`.

*Key idea, and the most important one in the project:* **causality.** Three rules —

1. **Forward-fill only.** Carrying yesterday's price into today uses only what you knew
   yesterday. Back-filling, interpolating, or filling with a column mean all reach into the future.
2. **Bound the fill.** Bridging three months manufactures three months of zero returns. Zero
   returns crush measured volatility, and low volatility inflates every Sharpe ratio downstream.
3. **Never fill leading NaNs.** Nothing to carry forward before a stock listed. Forward-fill
   gives you this for free.

**The trap you avoided:** "measure each gap, fill only the short ones." Standing inside a gap you
cannot know how long it is — the gap ends when the next price arrives, and that hasn't happened.
`ffill(limit=n)` never asks that question; it only ever looks backward.

**The test that catches it:**

```python
clean(full).loc[:T]  ==  clean(full.loc[:T])
```

Cleaning and truncating must commute. Keep this technique — **to test whether any pipeline uses
the future, run it on truncated history and check the answer is unchanged.** It catches look-ahead
in feature engineering, signal construction, normalization, anywhere.

---

# Part 2 — The idioms that did the work

## Masks

A mask is a boolean Series or array the same length as your data. Almost everything in `data.py`
was building masks and combining them.

```python
series.isna()                     # no price today
series.notna()                    # has a price
log_ret.abs() > threshold         # moved too much
(series.diff() == 0)              # same as yesterday
```

Combine with `&` (and), `|` (or), `~` (not). **Parenthesize every comparison** — `&` binds
tighter than `>=`, so `a >= b & c` parses wrong.

```python
missing = no_price & market_open & inside_history
```

Chained comparisons (`a <= b >= c`) do **not** work on Series. Python expands them with `and`,
which forces a whole Series to one True/False. Write the two comparisons separately and `&` them.

## Collapse vs elementwise — the one that kept biting

| Collapses (gives you a number or a shorter thing) | Elementwise (same shape out) |
|---|---|
| `.sum()` `.mean()` `.any()` `.all()` `.max()` | `.isna()` `.abs()` `.diff()` `+ - * /` `> <` |

You reached for `.sum()` three separate times when you needed the un-collapsed mask. That's the
single conceptual habit still worth drilling: **ask whether you want one answer or one answer per row.**

`axis=k` is the axis that **disappears**:

```python
wide.notna().sum(axis=1)     # collapses columns → one number per DATE
wide.notna().any(axis=1)     # did anyone trade that day?
returns.mean(axis=0)         # collapses time  → one number per ASSET
```

`keepdims=True` (NumPy) keeps the collapsed axis as length 1 so it broadcasts back correctly.
Use it whenever you reduce and then recombine with the original.

## Selection

```python
wide.loc[boolean_series]      # keep rows where True   ← explicit, preferred
series.index[mask]            # the dates where True
log_ret[mask]                 # the values where True
log_ret[mask].items()         # (date, value) pairs
```

Prefer `.loc[...]` over bare `wide[...]` for rows. Bare brackets on a DataFrame are ambiguous —
sometimes columns, sometimes rows — and that ambiguity is a classic bug source.

## Reshaping

```python
long.pivot(index=..., columns=..., values=...)    # long → wide
df.stack(level=1)                                  # a column level → into the index
df.reset_index()                                   # index levels → back to columns
series.unstack(fill_value=0)                       # an index level → up into columns
```

`stack` and `unstack` are inverses. You used `stack` on the yfinance MultiIndex and `unstack` in
`defect_summary` — same tool, opposite directions.

## Time series

```python
series.diff()                    # today minus yesterday (first row NaN)
np.log(series).diff()            # log return
series.ffill(limit=n)            # carry forward, bounded — CAUSAL
series.first_valid_index()       # date of first real value
series.last_valid_index()
```

## Run detection

The one genuinely clever idiom, and it generalizes far beyond this project:

```python
same = (series.diff() == 0) & series.notna()
grp  = (~same).cumsum()                        # run ID: ticks up only on change
run  = same.groupby(grp).transform("sum")      # every row gets its run's length
stale = run >= (stale_run - 1)
```

`(~mask).cumsum()` gives every consecutive stretch a shared ID, because a running total of
"changes so far" only increments when something changes. Then `transform` writes the group's
answer back to **every row in the group** — as opposed to `sum`, which would collapse to one row
per group. Collapse vs elementwise again, one level up.

`k` identical prices produce `k−1` zero-diffs, hence the `− 1`.

---

# Part 3 — The six rules that kept costing you time

1. `axis=k` is the axis that **disappears**.
2. `.sum()` `.any()` `.mean()` **collapse**; `.isna()` `>` `+` **don't**.
3. Combine masks with `&`, and **parenthesize every comparison**.
4. A method needs `()` to run. `s.first_valid_index` is the function; `s.first_valid_index()` is the date.
5. **`return` is the last line.** Nothing below it executes.
6. `assert` lives in tests. **Never** in `src/`.

Three more from the week:

7. Read tracebacks **bottom-up** — the last line is the real error.
8. An `AssertionError` means *your code* is wrong. `NameError` / `ImportError` mean the test couldn't run.
9. **Check nesting first** when a test returns empty or partial results. It was the cause every time.

---

# Part 4 — Primer for `analysis.py`

19 tests. Four groups. Here's the vocabulary and the maths, so none of it is new when you sit down.

## Returns

```python
prices.pct_change()              # simple:  P_t / P_{t-1} - 1
np.log(prices).diff()            # log:     ln(P_t) - ln(P_{t-1})
```

**Why log returns are the default in time-series work:**

1. **They add across time.** ln(a/b) + ln(b/c) = ln(a/c), so a month's return is the *sum* of
   daily log returns. Simple returns compound multiplicatively. There's a test for exactly this.
2. **Roughly symmetric.** +50% then −50% leaves you down 25%; logs are symmetric about zero.
3. **They match the standard model.** Geometric Brownian motion assumes log prices random-walk
   with normal increments — the basis of Black–Scholes and most continuous-time finance.

**The catch, and it's an interview question:** log returns do *not* add across **assets**. A
portfolio's log return isn't the weighted average of its holdings' log returns. So: **time series
→ logs. Cross-section → simple.**

## Volatility

```python
returns.rolling(window).std(ddof=1)
```

**Annualizing scales with √time**, not time. Variance of a sum of independent increments adds
linearly, so variance scales with time and standard deviation scales with its square root.
Daily → annual is `× √252`.

**`ddof` matters.** pandas defaults to `ddof=1` (sample); NumPy's `np.std` defaults to `ddof=0`
(population). They differ. The tests assume `ddof=1`.

**Window choice is a bias–variance tradeoff.** 21 days reacts fast and is noisy; 252 is stable and
stale. No right answer — but say which you chose and why. That sentence is worth more than the chart.

**Causality again:** `rolling` is trailing by default. Never use `center=True` — a centred window
at date *t* includes data from after *t*.

## Correlation — built from the definition

You'll implement this in NumPy rather than calling `.corr()`, then check against the library.
This is the covariance capstone from your workbook, promoted to production code.

```
1. drop rows with any NaN          → complete-case deletion
2. centre each column               X - X.mean(axis=0)
3. covariance = Xc.T @ Xc / (n-1)   ← matrix product, (n_obs, k) → (k, k)
4. correlation = cov / np.outer(s, s)   where s = sqrt(diag(cov))
5. force exact symmetry and unit diagonal
```

**Step 1 matters more than it looks.** One ticker with 3 years of history in a 10-year panel
truncates the sample for *every pair*. Complete-case deletion is the honest default; pairwise
deletion can produce a matrix that isn't positive semi-definite, which blows up any optimizer
you feed it to.

**Step 5 isn't pedantry.** Floating point leaves 0.9999999999999998 on the diagonal, and
eigendecomposition and Cholesky can be surprisingly unhappy about it.

New NumPy tools:

```python
X.T @ X                    # matrix product — (n,k)ᵀ(n,k) → (k,k)
np.outer(s, s)             # s[i] * s[j] matrix — the correlation divisor
np.diag(M)                 # the diagonal as a 1-D array
np.fill_diagonal(M, 1.0)   # set it, in place
np.triu_indices(n, k=1)    # strict upper triangle — every pair ONCE
np.linalg.eigvalsh(M)      # eigenvalues of a SYMMETRIC matrix
```

`np.triu_indices(n, k=1)` gives `n(n−1)/2` pairs, not `n(n−1)`. Correlation pairs are unordered —
corr(A,B) = corr(B,A) — which is why `average_pairwise_correlation` must not double-count, and why
it must exclude the diagonal. **The diagonal is 1 by construction and carries zero information:**
ρᵢᵢ = Σᵢᵢ/(σᵢσᵢ) = σᵢ²/σᵢ² = 1, true for any dataset including pure noise.

## Rolling correlation — the research question

```
Does the average pairwise correlation of daily equity returns
rise during high-volatility periods?
```

If it does, diversification fails exactly when you need it: a risk model calibrated on calm-period
correlations understates crisis drawdowns. Roughly what happened to a lot of quant books in
August 2007 and again in March 2020 — and you now have the March 2020 data to look at.

A loop over windows is acceptable here; there's no clean vectorization. But profile it.

**Stretch: `first_eigenvalue_share`.** The largest eigenvalue divided by the sum. On equity returns
the first principal component is almost always "the market" — every stock loads on it with the same
sign. That's PCA arriving early, and it should track your average pairwise correlation closely.
Working out *why* it must is a genuinely good linear-algebra exercise.

Sanity check: for a correlation matrix the eigenvalues sum to `n`, because each variable has unit
variance. Verify that yourself the first time.

## Estimation error — carry this from the last session

With `n` observations, the sample correlation of two *independent* series has standard error
≈ 1/√n. At n=250 that's ±0.063 — so "uncorrelated" assets routinely measure ±0.1.

With N assets you estimate N(N−1)/2 correlations from the same n rows. At N=500 that's 124,750
numbers from maybe 1,000 days. The matrix becomes mostly noise, and an optimizer fed that matrix
allocates confidently to relationships that don't exist.

This is the entire motivation for shrinkage estimators and factor models. You'll meet them in
Phase 5 — and now you'll know what problem they solve.
