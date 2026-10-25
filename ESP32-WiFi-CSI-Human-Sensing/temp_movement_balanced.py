import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

p = "data/activity/processed/activity_dataset.csv"
df = pd.read_csv(p)
df = df[df.activity.isin(["standing", "sitting", "moving"])].copy()
df["target"] = (df.activity == "moving").astype(int)

sessions = df[["session_id", "target"]].drop_duplicates("session_id")

train_sessions, test_sessions = train_test_split(
    sessions,
    test_size=0.40,
    random_state=42,
    stratify=sessions.target,
)

val_sessions, test_sessions = train_test_split(
    test_sessions,
    test_size=0.50,
    random_state=42,
    stratify=test_sessions.target,
)

train = df[df.session_id.isin(train_sessions.session_id)]
test = df[df.session_id.isin(test_sessions.session_id)]

# Equalize the number of training rows from STATIC and MOVING.
n = min(
    (train.target == 0).sum(),
    (train.target == 1).sum(),
)

static_train = train[train.target == 0].sample(
    n=n,
    random_state=42,
)

moving_train = train[train.target == 1].sample(
    n=n,
    random_state=42,
)

train = pd.concat([static_train, moving_train])

drop = ["sample_id", "session_id", "timestamp", "activity", "target"]

X = train.drop(columns=drop)
y = train.target

Xt = test.drop(columns=drop)
yt = test.target

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

model.fit(X, y)
pred = model.predict(Xt)

print(f"Balanced training rows: {len(train)}")
print(classification_report(
    yt,
    pred,
    target_names=["STATIC", "MOVING"],
    digits=4,
))
