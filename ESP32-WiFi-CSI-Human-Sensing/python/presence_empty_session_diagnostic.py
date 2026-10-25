import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "activity" / "processed" / "activity_dataset.csv"

FEATURES = [
    "rssi_mean", "rssi_last", "temp_energy", "agg_mean", "agg_min",
    "sub_mean_0", "sub_mean_1", "sub_mean_3", "agg_max", "sub_mean_4"
]

df = pd.read_csv(DATA)

empty = df[df["activity"].str.upper() == "EMPTY"].copy()

print("=" * 70)
print("EMPTY SESSION DIAGNOSTIC")
print("=" * 70)
print(f"Dataset: {DATA}")
print(f"EMPTY rows: {len(empty)}")
print(f"EMPTY sessions: {empty['session_id'].nunique()}")
print()

print("Feature variation across EMPTY sessions:")
print("-" * 70)

session_means = empty.groupby("session_id")[FEATURES].mean()

summary = pd.DataFrame({
    "mean": session_means.mean(),
    "std_between_sessions": session_means.std(),
    "min_session": session_means.min(),
    "max_session": session_means.max(),
})

for feature in FEATURES:
    r = summary.loc[feature]
    print(
        f"{feature:15s} "
        f"mean={r['mean']:10.3f} "
        f"between_std={r['std_between_sessions']:10.3f} "
        f"range=[{r['min_session']:.3f}, {r['max_session']:.3f}]"
    )

print()
print("Individual EMPTY sessions:")
print("-" * 70)

overall_mean = session_means.mean()
overall_std = session_means.std().replace(0, 1e-9)

distances = ((session_means - overall_mean) / overall_std).pow(2).sum(axis=1).pow(0.5)

result = pd.DataFrame({
    "session_id": distances.index,
    "distance_from_empty_center": distances.values,
    "rows": empty.groupby("session_id").size().reindex(distances.index).values,
})

result = result.sort_values("distance_from_empty_center")

print(result.to_string(index=False))

print()
print("=" * 70)
print("Interpretation")
print("=" * 70)
print("Large distance = unusual EMPTY session.")
print("Small distance = representative EMPTY session.")
print()
print("No model was trained or modified.")
