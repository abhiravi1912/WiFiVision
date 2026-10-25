
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "activity" / "processed" / "activity_dataset.csv"
MODEL_DIR = ROOT / "data" / "activity" / "models"

MODEL_PATH = MODEL_DIR / "presence_model_knn.pkl"
MANIFEST_PATH = MODEL_DIR / "presence_model_knn_manifest.json"
REPORT_PATH = MODEL_DIR / "presence_knn_training_report.json"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA)

    # Convert activity labels to binary presence labels.
    label_map = {
        "empty": "EMPTY",
        "standing": "PRESENT",
        "sitting": "PRESENT",
        "moving": "PRESENT",
    }

    df["presence"] = df["activity"].map(label_map)

    if df["presence"].isna().any():
        bad = sorted(df.loc[df["presence"].isna(), "activity"].unique())
        raise ValueError(f"Unknown activity labels: {bad}")

    # Use the same 180 CSI feature columns as the existing activity model.
    feature_cols = [
        c for c in df.columns
        if c.startswith("sub_mean_")
        or c.startswith("sub_std_")
        or c.startswith("sub_diff_")
        or c in [
            "agg_mean",
            "agg_std",
            "agg_range",
            "agg_min",
            "agg_max",
            "temp_std",
            "temp_mean_abs_diff",
            "temp_energy",
            "temp_slope",
            "spec_band_0",
            "spec_band_1",
            "spec_band_2",
            "spec_band_3",
            "rssi_mean",
            "rssi_std",
            "rssi_last",
            "move_rel_activity",
            "move_peak_rel_diff",
        ]
    ]

    if len(feature_cols) != 180:
        raise ValueError(f"Expected 180 features, found {len(feature_cols)}")

    # One label per session.
    sessions = (
        df[["session_id", "presence"]]
        .drop_duplicates("session_id")
        .reset_index(drop=True)
    )

    print()
    print("Presence rows:")
    print(df["presence"].value_counts())

    print()
    print("Presence sessions:")
    print(sessions["presence"].value_counts())

    # 60% train / 20% validation / 20% test, split by session.
    train_sessions, temp_sessions = train_test_split(
        sessions,
        test_size=0.40,
        random_state=42,
        stratify=sessions["presence"],
    )

    val_sessions, test_sessions = train_test_split(
        temp_sessions,
        test_size=0.50,
        random_state=42,
        stratify=temp_sessions["presence"],
    )

    train_ids = set(train_sessions["session_id"])
    val_ids = set(val_sessions["session_id"])
    test_ids = set(test_sessions["session_id"])

    train = df[df["session_id"].isin(train_ids)]
    val = df[df["session_id"].isin(val_ids)]
    test = df[df["session_id"].isin(test_ids)]

    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val = val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    y_train = train["presence"]
    y_val = val["presence"]
    y_test = test["presence"]

    print()
    print("Session split:")
    print(f"  Train: {len(train_ids)} sessions / {len(train)} rows")
    print(f"  Val:   {len(val_ids)} sessions / {len(val)} rows")
    print(f"  Test:  {len(test_ids)} sessions / {len(test)} rows")

    print()
    print("Train sessions:")
    print(train_sessions["presence"].value_counts())

    print()
    print("Validation sessions:")
    print(val_sessions["presence"].value_counts())

    print()
    print("Test sessions:")
    print(test_sessions["presence"].value_counts())

    # Evaluate several K values on validation data.
    candidates = [3, 5, 7, 9, 11, 15, 21]

    results = []

    for k in candidates:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=k,
                weights="distance",
                metric="euclidean",
            )),
        ])

        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        acc = accuracy_score(y_val, pred)
        macro_f1 = f1_score(y_val, pred, average="macro")

        results.append({
            "k": k,
            "validation_accuracy": float(acc),
            "validation_macro_f1": float(macro_f1),
        })

        print(
            f"K={k:2d} | "
            f"val_accuracy={acc:.4f} | "
            f"val_macro_f1={macro_f1:.4f}"
        )

    best = max(results, key=lambda x: x["validation_macro_f1"])
    best_k = best["k"]

    print()
    print(f"Selected KNN: k={best_k}")

    # Refit on train + validation, then evaluate once on untouched test.
    train_val = pd.concat([train, val], ignore_index=True)

    X_train_val = (
        train_val[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    y_train_val = train_val["presence"]

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=best_k,
            weights="distance",
            metric="euclidean",
        )),
    ])

    final_model.fit(X_train_val, y_train_val)

    test_pred = final_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")

    print()
    print("TEST RESULTS")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {test_macro_f1:.4f}")

    print()
    print(classification_report(
        y_test,
        test_pred,
        labels=["EMPTY", "PRESENT"],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test,
        test_pred,
        labels=["EMPTY", "PRESENT"],
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODEL_PATH)

    manifest = {
        "task": "binary_presence",
        "model_type": "KNeighborsClassifier",
        "classes": ["EMPTY", "PRESENT"],
        "source_activity_mapping": label_map,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "selected_k": int(best_k),
        "weights": "distance",
        "metric": "euclidean",
        "split_type": "session_based",
        "random_state": 42,
        "dataset": "data/activity/processed/activity_dataset.csv",
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    report = {
        "task": "binary_presence",
        "model": "KNN",
        "selected_k": int(best_k),
        "rows": {
            "total": int(len(df)),
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "sessions": {
            "total": int(len(sessions)),
            "train": int(len(train_ids)),
            "validation": int(len(val_ids)),
            "test": int(len(test_ids)),
        },
        "validation_results": results,
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "confusion_matrix": confusion_matrix(
            y_test,
            test_pred,
            labels=["EMPTY", "PRESENT"],
        ).tolist(),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("Saved:")
    print(MODEL_PATH)
    print(MANIFEST_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
