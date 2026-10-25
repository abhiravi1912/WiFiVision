import pandas as pd
import numpy as np

PATH = r".\data\activity\processed\activity_dataset.csv"

df = pd.read_csv(PATH)
df = df[df["activity"].isin(["standing", "sitting"])].copy()

features = [
    "spec_band_1",
    "spec_band_2",
    "temp_mean_abs_diff",
    "rssi_mean",
    "rssi_last",
]

print("\n========================================")
print(" STANDING vs SITTING SESSION STABILITY")
print("========================================")

for activity in ["standing", "sitting"]:
    x = df[df["activity"] == activity]

    print(f"\n--- {activity.upper()} ---")
    print(f"Sessions: {x['session_id'].nunique()}")

    rows = []

    for sid, g in x.groupby("session_id"):
        row = {"session_id": sid}

        for f in features:
            row[f] = g[f].median()

        rows.append(row)

    s = pd.DataFrame(rows)

    for f in features:
        print(
            f"{f:24s} "
            f"session std={s[f].std():.4f}  "
            f"min={s[f].min():.4f}  "
            f"max={s[f].max():.4f}"
        )

print("\n========================================")
print(" SESSION MEDIANS")
print("========================================")

session_df = (
    df.groupby(["session_id", "activity"])[features]
      .median()
      .reset_index()
)

print(session_df.to_string(index=False))

print("\n========================================")
print(" DONE")
print("========================================")
