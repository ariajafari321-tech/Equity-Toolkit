import sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_raw, to_wide, find_defects, defect_summary

raw = load_raw(ROOT / "data" / "raw" / "prices.csv")
wide = to_wide(raw)
defects = find_defects(wide)

print(defect_summary(defects))
print()
print("total defects:", len(defects))
print(defects[defects["kind"] == "extreme_return"].to_string(index=False))



