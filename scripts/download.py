"""One-time data pull. Run from the repo root:

    python scripts/download.py

Writes data/raw/prices.csv. That file is gitignored on purpose — this script is
what gets committed, because a script is reproducible and a 40MB CSV is not.

A NOTE ON THE UNIVERSE BELOW, WHICH IS NOT A THROWAWAY DETAIL
-------------------------------------------------------------
These 30 tickers are large, liquid US names that all still exist today. That
last clause is a bias, and you should be able to name it in an interview:
selecting your universe by "companies that are around now" is **survivorship
bias**. Every firm that went bankrupt, got acquired, or was delisted over your
sample window is invisible to you, so your measured returns are biased upward
and your measured risk is biased downward.

You cannot fix this with free data — properly point-in-time index constituent
data is expensive (CRSP, Compustat). What you CAN do, and what separates a
researcher from someone running a script, is:

  1. Know that the bias exists.
  2. Say so explicitly in your README.
  3. Reason about its direction and rough size for your specific question.

For the correlation-versus-volatility question this project asks, survivorship
bias is relatively benign — it distorts the *level* of returns much more than the
*co-movement* of returns. Argue that in your findings note. Being able to say
"here is my bias, here is why it does or doesn't threaten this particular
conclusion" is exactly the skill the job is.
"""

from __future__ import annotations

from pathlib import Path

from src.data import download_prices

# 30 large-cap US names, deliberately spread across sectors so cross-sectional
# correlation is a meaningful quantity rather than 30 flavours of one tech bet.
UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO",
    # Financials
    "JPM", "BAC", "GS", "BRK-B",
    # Health care
    "JNJ", "UNH", "PFE", "ABBV",
    # Consumer
    "AMZN", "WMT", "HD", "PG", "KO", "MCD",
    # Energy & materials
    "XOM", "CVX", "COP", "LIN",
    # Industrials & utilities
    "CAT", "HON", "UNP", "NEE", "DUK",
    # Communication
    "GOOGL",
]

START = "2015-01-01"
END = "2025-01-01"
OUT = Path("data/raw/prices.csv")


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    path = download_prices(UNIVERSE, start=START, end=END, out_path=OUT)
    print(f"Wrote {path}")
