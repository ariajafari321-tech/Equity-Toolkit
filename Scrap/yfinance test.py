import yfinance as yf

raw = yf.download(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01", auto_adjust=False)
print(raw.shape)
print(raw.columns)
print(raw.head())