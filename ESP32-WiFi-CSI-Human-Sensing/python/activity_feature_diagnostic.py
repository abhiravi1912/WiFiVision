import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

PATH = r".\data\activity\processed\activity_dataset.csv"

df = pd.read_csv(PATH)
df = df[df["activity"].isin(["standing", "sitting"])].copy()

meta = {"sample_id", "session_id", "timestamp", "activity"}
features = [c for c in df.columns if c not in meta]

# Collapse each session to its median feature values.
session_df = df.groupby(["session_id", "activity"])[features].median().reset_index()

standing = session_df[session_df.activity == "standing"]
sitting = session_df[session_df.activity == "sitting"]

print("\n========================================")
print(" STANDING vs SITTING FEATURE DIAGNOSTIC")
print("========================================")
print(f"Standing sessions: {len(standing)}")
print(f"Sitting sessions : {len(sitting)}")
print(f"Features         : {len(features)}")

# Cohen's d using session-level medians
results = []

for f in features:
    a = standing[f].dropna().to_numpy()
    b = sitting[f].dropna().to_numpy()

    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2))

    d = 0.0 if pooled == 0 else (np.mean(a)-np.mean(b))/pooled

    results.append({
        "feature": f,
        "effect": abs(d),
        "standing_median": np.median(a),
        "sitting_median": np.median(b)
    })

r = pd.DataFrame(results).sort_values("effect", ascending=False)

print("\nTOP 20 FEATURES BY STANDING/SITTING SEPARATION")
print(r.head(20).to_string(index=False))

# Session-based classifier
X = session_df[features]
y = (session_df.activity == "standing").astype(int)
groups = session_df.session_id

gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X.iloc[train_idx], y.iloc[train_idx])
pred = model.predict(X.iloc[test_idx])

print("\nSTANDING vs SITTING SESSION-LEVEL TEST")
print(f"Accuracy : {accuracy_score(y.iloc[test_idx], pred):.3f}")
print(f"Macro F1 : {f1_score(y.iloc[test_idx], pred, average='macro'):.3f}")

print("\nInterpretation:")
print("effect < 0.2  = very weak separation")
print("effect 0.2-0.5 = small separation")
print("effect 0.5-0.8 = moderate separation")
print("effect > 0.8  = strong separation")
print("Classifier near 0.50 means the current features contain little usable posture information.")
