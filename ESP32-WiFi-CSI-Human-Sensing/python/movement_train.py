from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "activity" / "processed" / "activity_dataset.csv"
MODEL_DIR = ROOT / "data" / "activity" / "models"

MODEL_PATH = MODEL_DIR / "movement_model.pkl"
MANIFEST_PATH = MODEL_DIR / "movement_model_manifest.json"
REPORT_PATH = MODEL_DIR / "movement_training_report.json"

RANDOM_STATE = 42


def split_sessions(df: pd.DataFrame):
    sessions = df[["session_id", "target"]].drop_duplicates("session_id")

    train_sessions, temp_sessions = train_test_split(
        sessions,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=sessions["target"],
    )

    val_sessions, test_sessions = train_test_split(
        temp_sessions,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_sessions["target"],
    )

    train_ids = set(train_sessions.session_id)
    val_ids = set(val_sessions.session_id)
    test_ids = set(test_sessions.session_id)

    return (
        df[df.session_id.isin(train_ids)].copy(),
        df[df.session_id.isin(val_ids)].copy(),
        df[df.session_id.isin(test_ids)].copy(),
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df = df[df["activity"].isin(["standing", "sitting", "moving"])].copy()

    if df.empty:
        raise ValueError("No standing/sitting/moving activity data found.")

    df["target"] = (df["activity"] == "moving").astype(int)

    if df["session_id"].nunique() < 3:
        raise ValueError("Need at least 3 sessions.")

    feature_columns = [
        c
        for c in df.columns
        if c not in ["sample_id", "session_id", "timestamp", "activity", "target"]
    ]

    train_df, val_df, test_df = split_sessions(df)

    X_train = train_df[feature_columns]
    y_train = train_df["target"]

    X_val = val_df[feature_columns]
    y_val = val_df["target"]

    X_test = test_df[feature_columns]
    y_test = test_df["target"]

    print("Movement dataset")
    print(f"Train: {len(train_df)} rows / {train_df.session_id.nunique()} sessions")
    print(f"Val:   {len(val_df)} rows / {val_df.session_id.nunique()} sessions")
    print(f"Test:  {len(test_df)} rows / {test_df.session_id.nunique()} sessions")
    print()

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)

    print("Validation")
    print(f"Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"Macro F1: {f1_score(y_val, val_pred, average='macro'):.4f}")
    print(
        classification_report(
            y_val,
            val_pred,
            target_names=["STATIC", "MOVING"],
            digits=4,
        )
    )

    test_pred = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")

    print("Test")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {test_macro_f1:.4f}")
    print(
        classification_report(
            y_test,
            test_pred,
            target_names=["STATIC", "MOVING"],
            digits=4,
        )
    )

    print("Confusion matrix")
    print(confusion_matrix(y_test, test_pred))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    manifest = {
        "task": "movement",
        "classes": ["STATIC", "MOVING"],
        "feature_columns": feature_columns,
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "random_state": RANDOM_STATE,
    }

    report = {
        "task": "movement",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_sessions": train_df.session_id.nunique(),
        "val_sessions": val_df.session_id.nunique(),
        "test_sessions": test_df.session_id.nunique(),
        "validation_accuracy": accuracy_score(y_val, val_pred),
        "validation_macro_f1": f1_score(y_val, val_pred, average="macro"),
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print()
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved manifest: {MANIFEST_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
