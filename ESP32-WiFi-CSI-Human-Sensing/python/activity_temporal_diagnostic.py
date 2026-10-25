import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

PATH = r".\data\activity\processed\activity_dataset.csv"

df = pd.read_csv(PATH)
df = df[df["activity"].isin(["standing", "sitting"])].copy()

meta = ["sample_id", "session_id", "timestamp", "activity"]
features = [c for c in df.columns if c not in meta]

df = df.sort_values(["session_id", "timestamp"])

def make_chunks(n):
    parts = []

    for sid, g in df.groupby("session_id"):
        g = g.reset_index(drop=True)

        for start in range(0, len(g) - n + 1, n):
            chunk = g.iloc[start:start+n]

            values = chunk[features].median(numeric_only=True).to_dict()
            values["session_id"] = sid
            values["activity"] = chunk["activity"].iloc[0]
            parts.append(values)

    return pd.DataFrame(parts)

print("\n========================================")
print(" TEMPORAL AGGREGATION DIAGNOSTIC")
print("========================================")

for n in [1, 5, 10, 20]:

    data = make_chunks(n)

    X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = (data["activity"] == "standing").astype(int)
    groups = data["session_id"]

    scores = []

    gss = GroupShuffleSplit(
        n_splits=5,
        test_size=0.30,
        random_state=42
    )

    for train_idx, test_idx in gss.split(X, y, groups):

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])

        scores.append((
            accuracy_score(y.iloc[test_idx], pred),
            f1_score(y.iloc[test_idx], pred, average="macro")
        ))

    print(
        f"\n{n:2d} windows/chunk "
        f"({n*8/60:.1f} sec approx)"
    )
    print(f"Accuracy : {np.mean([x[0] for x in scores]):.3f}")
    print(f"Macro F1 : {np.mean([x[1] for x in scores]):.3f}")

print("\n========================================")
print(" DONE")
print("========================================")
