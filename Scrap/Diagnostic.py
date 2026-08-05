import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.data import find_defects

# rebuild the fixture (copy from tests/conftest.py)
business_days = pd.bdate_range("2020-01-01", periods=750)
rng = np.random.default_rng(42)
shocks = rng.normal(0.0003, 0.012, size=(750, 4))
wide = pd.DataFrame(100*np.exp(np.cumsum(shocks, axis=0)),
                    index=business_days, columns=["AAA","BBB","CCC","DDD"])
wide.iloc[150:155, 3] = wide.iloc[149, 3]      # plant the stale run in DDD

d = find_defects(wide)
print(d["kind"].value_counts())
print(d.head(20))