import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

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

# Use individual feature windows, not session medians.
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y = (df["activity"] == "standing").astype(int)
groups = df["session_id"]

gss = GroupShuffleSplit(n_splits=5, test_size=0.30, random_state=42)

scores = []

print("\n========================================")
print(" WINDOW-LEVEL STANDING vs SITTING TEST")
print("========================================")

for i, (train_idx, test_idx) in enumerate(gss.split(X, y, groups), 1):

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])

    acc = accuracy_score(y.iloc[test_idx], pred)
    f1 = f1_score(y.iloc[test_idx], pred, average="macro")

    scores.append((acc, f1))

    print(f"Split {i}: Accuracy={acc:.3f}  Macro F1={f1:.3f}")

print("\nAVERAGE")
print(f"Accuracy : {np.mean([x[0] for x in scores]):.3f}")
print(f"Macro F1 : {np.mean([x[1] for x in scores]):.3f}")

print("\n========================================")
print(" DONE")
print("========================================")
