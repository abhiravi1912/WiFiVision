import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "activity" / "processed" / "activity_dataset.csv"
MODEL_DIR = ROOT / "data" / "activity" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "rssi_mean", "rssi_last", "temp_energy", "agg_mean", "agg_min",
    "sub_mean_0", "sub_mean_1", "sub_mean_3", "agg_max", "sub_mean_4"
]

WINDOW = 20

df = pd.read_csv(DATA)
df["label"] = np.where(df["activity"].str.upper() == "EMPTY", "EMPTY", "PRESENT")

sessions = df["session_id"].unique()
train_s, test_s = train_test_split(
    sessions, test_size=0.20, random_state=42,
    stratify=df.groupby("session_id")["label"].first()
)
train_s, val_s = train_test_split(
    train_s, test_size=0.25, random_state=42,
    stratify=df[df["session_id"].isin(train_s)].groupby("session_id")["label"].first()
)

def make_temporal(data, session_ids):
    data = data[data["session_id"].isin(session_ids)]
    X, y = [], []

    for sid, g in data.groupby("session_id", sort=False):
        g = g.reset_index(drop=True)
        vals = g[FEATURES].to_numpy(dtype=np.float32)
        label = g["label"].iloc[0]

        n = len(vals) // WINDOW
        for i in range(n):
            w = vals[i * WINDOW:(i + 1) * WINDOW]

            # Temporal statistics: mean, std, min, max, range, delta
            feat = np.concatenate([
                w.mean(axis=0),
                w.std(axis=0),
                w.min(axis=0),
                w.max(axis=0),
                w.max(axis=0) - w.min(axis=0),
                w[-1] - w[0],
            ])

            X.append(feat)
            y.append(0 if label == "EMPTY" else 1)

    return np.asarray(X, dtype=np.float32), np.asarray(y)

X_train, y_train = make_temporal(df, train_s)
X_val, y_val = make_temporal(df, val_s)
X_test, y_test = make_temporal(df, test_s)

print("=" * 70)
print("TEMPORAL PRESENCE KNN")
print("=" * 70)
print(f"Features per window : {len(FEATURES)}")
print(f"Temporal window     : {WINDOW}")
print(f"Temporal features   : {X_train.shape[1]}")
print(f"Train windows       : {len(X_train)}")
print(f"Val windows         : {len(X_val)}")
print(f"Test windows        : {len(X_test)}")
print()

best = None

for k in [3, 5, 7, 9, 11, 15]:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            metric="euclidean"
        ))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred, average="macro")

    print(f"K={k:2d}  VAL acc={acc:.4f}  macroF1={f1:.4f}")

    if best is None or f1 > best[0]:
        best = (f1, k, model)

print()
print(f"Selected K={best[1]}")

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=best[1],
        weights="distance",
        metric="euclidean"
    ))
])

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
final_model.fit(X_trainval, y_trainval)

test_pred = final_model.predict(X_test)

print()
print("=" * 70)
print("TEST RESULT")
print("=" * 70)
print(f"Accuracy : {accuracy_score(y_test, test_pred):.4f}")
print(f"Macro F1 : {f1_score(y_test, test_pred, average='macro'):.4f}")
print()
print(classification_report(
    y_test,
    test_pred,
    target_names=["EMPTY", "PRESENT"],
    digits=4
))

model_path = MODEL_DIR / "presence_model_temporal_knn.pkl"
joblib.dump(final_model, model_path)

print(f"Saved: {model_path}")
print("=" * 70)
