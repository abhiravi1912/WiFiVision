import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "activity" / "processed" / "activity_dataset.csv"
LIVE = ROOT / "data" / "activity" / "processed" / "live_empty_diagnostic.csv"

df = pd.read_csv(DATASET)
live = pd.read_csv(LIVE)

feature_cols = [c for c in df.columns if c.startswith("sub_") or c in [
    "agg_mean", "agg_std", "agg_range", "agg_min", "agg_max",
    "temp_std", "temp_mean_abs_diff", "temp_energy", "temp_slope",
    "rssi_mean", "rssi_std", "rssi_last",
    "move_rel_activity", "move_peak_rel_diff"
] or c.startswith("spec_band_")]

empty = df[df["activity"] == "empty"][feature_cols]
present = df[df["activity"] != "empty"][feature_cols]
live_x = live[feature_cols]

rows = []

for col in feature_cols:
    e = empty[col].astype(float)
    p = present[col].astype(float)
    l = live_x[col].astype(float)

    pooled_std = np.sqrt((e.var() + p.var()) / 2)
    separation = abs(e.mean() - p.mean()) / (pooled_std + 1e-9)

    empty_mean = e.mean()
    present_mean = p.mean()
    live_mean = l.mean()

    empty_dist = abs(live_mean - empty_mean) / (e.std() + 1e-9)
    present_dist = abs(live_mean - present_mean) / (p.std() + 1e-9)

    rows.append({
        "feature": col,
        "separation": separation,
        "empty_mean": empty_mean,
        "present_mean": present_mean,
        "live_mean": live_mean,
        "live_vs_empty_sigma": empty_dist,
        "live_vs_present_sigma": present_dist
    })

out = pd.DataFrame(rows)

print("\n" + "=" * 72)
print(" PRESENCE FEATURE DIAGNOSTIC")
print("=" * 72)
print(f"Features analyzed : {len(feature_cols)}")
print(f"Recorded EMPTY    : {len(empty)}")
print(f"Recorded PRESENT  : {len(present)}")
print(f"Live EMPTY        : {len(live_x)}")

print("\nTOP FEATURES SEPARATING RECORDED EMPTY vs PRESENT")
print("-" * 72)

for _, r in out.sort_values("separation", ascending=False).head(20).iterrows():
    print(
        f"{r.feature:24s} "
        f"sep={r.separation:6.2f}  "
        f"live?E={r.live_vs_empty_sigma:7.2f}s  "
        f"live?P={r.live_vs_present_sigma:7.2f}s"
    )

print("\nTOP LIVE EMPTY FEATURES THAT ARE FAR FROM RECORDED EMPTY")
print("-" * 72)

for _, r in out.sort_values("live_vs_empty_sigma", ascending=False).head(20).iterrows():
    print(
        f"{r.feature:24s} "
        f"live?E={r.live_vs_empty_sigma:7.2f}s  "
        f"live?P={r.live_vs_present_sigma:7.2f}s  "
        f"sep={r.separation:6.2f}"
    )

print("\nTOP FEATURES WHERE LIVE IS CLOSER TO PRESENT THAN EMPTY")
print("-" * 72)

out["present_advantage"] = out["live_vs_empty_sigma"] - out["live_vs_present_sigma"]

for _, r in out.sort_values("present_advantage", ascending=False).head(20).iterrows():
    print(
        f"{r.feature:24s} "
        f"live?E={r.live_vs_empty_sigma:7.2f}s  "
        f"live?P={r.live_vs_present_sigma:7.2f}s  "
        f"sep={r.separation:6.2f}"
    )

print("=" * 72)
