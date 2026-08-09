import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data import load_raw, to_wide, clean_prices
from src.analysis import log_returns, rolling_volatility

px = clean_prices(to_wide(load_raw("data/raw/prices.csv")))
r = log_returns(px)
vol = rolling_volatility(r, window=21)

print("returns:", r.shape)
print("vol:    ", vol.shape)
print(vol["NVDA"].dropna().describe())