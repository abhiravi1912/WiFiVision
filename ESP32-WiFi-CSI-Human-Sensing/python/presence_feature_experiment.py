import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "activity" / "processed" / "activity_dataset.csv"

df = pd.read_csv(DATA)

base_features = [c for c in df.columns if c.startswith("sub_") or c in [
    "agg_mean", "agg_std", "agg_range", "agg_min", "agg_max",
    "temp_std", "temp_mean_abs_diff", "temp_energy", "temp_slope",
    "rssi_mean", "rssi_std", "rssi_last",
    "move_rel_activity", "move_peak_rel_diff"
] or c.startswith("spec_band_")]

df["label"] = (df["activity"] != "empty").astype(int)

X = df[base_features]
y = df["label"]
groups = df["session_id"]

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=42)
train_idx, temp_idx = next(gss1.split(X, y, groups))

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_rel, test_rel = next(
    gss2.split(X.iloc[temp_idx], y.iloc[temp_idx], groups.iloc[temp_idx])
)

val_idx = temp_idx[val_rel]
test_idx = temp_idx[test_rel]

feature_sets = {}

# All features
feature_sets["ALL_180"] = base_features

# Remove subcarrier means that showed strong live-domain shift
bad = {
    "sub_mean_38", "sub_mean_39", "sub_mean_40", "sub_mean_41",
    "sub_mean_42", "sub_mean_43", "sub_mean_44", "sub_mean_45",
    "sub_mean_46", "sub_mean_47", "sub_mean_48"
}
feature_sets["NO_SHIFTED_MEANS"] = [f for f in base_features if f not in bad]

# Strong recorded separation features
tmp = df.iloc[train_idx]
empty = tmp[tmp["label"] == 0][base_features]
present = tmp[tmp["label"] == 1][base_features]

scores = {}
for f in base_features:
    pooled = np.sqrt((empty[f].var() + present[f].var()) / 2)
    scores[f] = abs(empty[f].mean() - present[f].mean()) / (pooled + 1e-9)

for n in [10, 20, 30, 50]:
    top = sorted(scores, key=scores.get, reverse=True)[:n]
    feature_sets[f"TOP_{n}"] = top

print("\n" + "=" * 72)
print(" PRESENCE FEATURE-SELECTION EXPERIMENT")
print("=" * 72)
print(f"Train sessions : {len(set(groups.iloc[train_idx]))}")
print(f"Val sessions   : {len(set(groups.iloc[val_idx]))}")
print(f"Test sessions  : {len(set(groups.iloc[test_idx]))}")

for name, features in feature_sets.items():

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="euclidean"
        ))
    ])

    pipe.fit(X.iloc[train_idx][features], y.iloc[train_idx])

    val_pred = pipe.predict(X.iloc[val_idx][features])
    test_pred = pipe.predict(X.iloc[test_idx][features])

    val_acc = accuracy_score(y.iloc[val_idx], val_pred)
    val_f1 = f1_score(y.iloc[val_idx], val_pred, average="macro")

    test_acc = accuracy_score(y.iloc[test_idx], test_pred)
    test_f1 = f1_score(y.iloc[test_idx], test_pred, average="macro")

    print(
        f"\n{name:20s} features={len(features):3d}"
        f"\n  VAL  acc={val_acc:.4f}  macroF1={val_f1:.4f}"
        f"\n  TEST acc={test_acc:.4f}  macroF1={test_f1:.4f}"
    )

print("\n" + "=" * 72)
