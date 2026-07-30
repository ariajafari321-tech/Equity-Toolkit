# Equity Data Quality & Returns Toolkit

> **Status: skeleton.** Every function raises `NotImplementedError`. The tests are
> the specification. Rewrite this README in your own words once the project works —
> what's below the line is a template, and a README that reads as boilerplate is
> worse than a short honest one.

---

## Research question

**Does the average pairwise correlation of daily US equity returns rise during
high-volatility periods?**

If it does, diversification weakens exactly when you need it most: a risk model
calibrated on calm-period correlations will understate crisis drawdowns. This is
a real and well-documented effect, which is a feature for a first project — you
can check your answer against the literature, and if you get a different sign you
know to go looking for the bug rather than announcing a discovery.

## Layout

```
equity-toolkit/
├── src/
│   ├── data.py          # acquire, inspect, clean  <- write this first
│   └── analysis.py      # returns, volatility, correlation
├── tests/
│   ├── conftest.py      # fixtures (synthetic data with planted defects)
│   ├── test_data.py     # the spec for data.py
│   └── test_analysis.py # the spec for analysis.py
├── scripts/
│   └── download.py      # one-time data pull; the data itself is gitignored
├── notebooks/
│   └── findings.ipynb   # PRESENTATION ONLY — no logic lives here
└── data/raw/            # gitignored
```

Logic in `.py`, presentation in `.ipynb`. This is how research code is organized
at firms, and it is why your notebook can stay short and readable: it imports
tested functions instead of redefining them in cell 47.

## Setup

```bash
cd equity-toolkit
python -m venv .venv
source .venv/Scripts/activate      # Git Bash on Windows
pip install -r requirements.txt
pytest -v                          # everything fails; that's the starting line
```

A virtual environment is not optional ceremony. It pins this project's
dependencies so that upgrading a package for something else next semester doesn't
silently change your results. Reproducibility is a research skill, not a
software-engineering one.

## Workflow

1. Run `pytest -v`. Pick the topmost failing test.
2. Read the corresponding docstring in `src/`.
3. Implement until that one test goes green.
4. `git commit -m "..."` — one commit per test is the right granularity.
5. Repeat.

Suggested order: `to_wide` → `load_raw` → `download_prices` → `find_defects` →
`defect_summary` → `align_calendar` → `clean_prices` → returns → volatility →
correlation → rolling correlation.

---

## Findings

*(Fill this in when you have results. This section is what a recruiter reads.)*

### Data
- Universe: 30 large-cap US equities across 8 sectors
- Period: 2015-01-01 to 2025-01-01, daily
- Source: Yahoo Finance via `yfinance`, downloaded YYYY-MM-DD
- **Known biases:** survivorship (universe selected from currently-listed names);
  adjusted-close restatement means the download date affects historical values

### Data quality
*(paste your `defect_summary` table here — this is the part almost no student
project has, and the part that shows you know your data)*

### Result
*(one chart, three sentences: what you measured, what you found, how confident
you are and why)*

### Limitations
*(be specific. "More data would help" is not a limitation, it's a filler
sentence. "The 2020 drawdown accounts for X% of the observed effect; excluding
it the relationship weakens to Y" is a limitation.)*
