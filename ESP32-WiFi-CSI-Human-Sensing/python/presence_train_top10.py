import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "activity" / "processed" / "activity_dataset.csv"
OUT = ROOT / "data" / "activity" / "models" / "presence_model_knn_top10.pkl"

features = [
    "rssi_mean", "rssi_last", "temp_energy", "agg_mean", "agg_min",
    "sub_mean_0", "sub_mean_1", "sub_mean_3", "agg_max", "sub_mean_4"
]

df = pd.read_csv(DATA)
X = df[features]
y = (df["activity"] != "empty").astype(int)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="euclidean"
    ))
])

model.fit(X, y)
joblib.dump(model, OUT)

print("Created:", OUT)
print("Features:", len(features))
print("Classes: EMPTY=0, PRESENT=1")
