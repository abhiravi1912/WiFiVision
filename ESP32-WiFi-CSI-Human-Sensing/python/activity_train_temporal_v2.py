from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
ACTIVITIES = ["empty", "standing", "sitting", "moving"]

# 20 feature rows ~= 2.7 seconds with the current CSI feature cadence.
TEMPORAL_WINDOWS = 20
TEMPORAL_STRIDE = 1

# Rich temporal statistics for every original feature.
STAT_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "delta",
]

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "activity" / "processed" / "activity_dataset.csv"
MODEL_DIR = BASE_DIR / "data" / "activity" / "models"

MODEL_PATH = MODEL_DIR / "activity_model_temporal_v2.pkl"
MANIFEST_PATH = MODEL_DIR / "activity_model_temporal_v2_manifest.json"
REPORT_PATH = MODEL_DIR / "activity_temporal_v2_training_report.json"


def split_by_session(df: pd.DataFrame):
    """Split complete sessions before making overlapping temporal windows."""
    rng = np.random.default_rng(SEED)

    train_parts = []
    val_parts = []
    test_parts = []

    for activity in ACTIVITIES:
        activity_df = df[df["activity"] == activity].copy()
        sessions = activity_df["session_id"].dropna().unique()
        sessions = np.array(sessions, dtype=object)
        rng.shuffle(sessions)

        n_sessions = len(sessions)
        n_test = max(1, int(round(n_sessions * 0.20)))
        n_val = max(1, int(round(n_sessions * 0.20)))

        # Keep at least one session for training.
        if n_test + n_val >= n_sessions:
            n_test = max(1, int(np.floor(n_sessions * 0.15)))
            n_val = max(1, int(np.floor(n_sessions * 0.15)))

        test_sessions = set(sessions[:n_test])
        val_sessions = set(sessions[n_test:n_test + n_val])
        train_sessions = set(sessions[n_test + n_val:])

        train_parts.append(activity_df[activity_df["session_id"].isin(train_sessions)])
        val_parts.append(activity_df[activity_df["session_id"].isin(val_sessions)])
        test_parts.append(activity_df[activity_df["session_id"].isin(test_sessions)])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    return train, val, test


def build_temporal_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    stride: int,
):
    """
    Build overlapping windows entirely inside each session.

    Each window produces six statistics for every original feature:
      mean, std, min, max, range, last-first delta

    Label is the activity of the source session.
    """
    X_rows = []
    y_rows = []
    session_rows = []

    for session_id, session_df in df.groupby("session_id", sort=False):
        session_df = session_df.sort_values(
            ["timestamp", "sample_id"],
            kind="stable",
        )

        activity_values = session_df["activity"].dropna().unique()
        if len(activity_values) != 1:
            # A session must have one activity label.
            continue

        activity = str(activity_values[0])
        if activity not in ACTIVITIES:
            continue

        values = (
            session_df[feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )

        # Do not let NaNs silently contaminate a whole temporal window.
        if not np.isfinite(values).all():
            values = np.nan_to_num(
                values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        n_rows = len(values)
        if n_rows < window_size:
            continue

        for start in range(0, n_rows - window_size + 1, stride):
            window = values[start:start + window_size]

            mean = window.mean(axis=0)
            std = window.std(axis=0)
            min_v = window.min(axis=0)
            max_v = window.max(axis=0)
            range_v = max_v - min_v
            delta = window[-1] - window[0]

            temporal_vector = np.concatenate(
                [mean, std, min_v, max_v, range_v, delta]
            ).astype(np.float32, copy=False)

            X_rows.append(temporal_vector)
            y_rows.append(activity)
            session_rows.append(str(session_id))

    if not X_rows:
        raise RuntimeError(
            "No temporal windows were created. "
            "Check the dataset and TEMPORAL_WINDOWS setting."
        )

    return (
        np.vstack(X_rows),
        np.asarray(y_rows, dtype=object),
        np.asarray(session_rows, dtype=object),
    )


def evaluate_model(name, model, X, y):
    predictions = model.predict(X)

    accuracy = float(accuracy_score(y, predictions))
    macro_f1 = float(f1_score(y, predictions, average="macro"))

    report = classification_report(
        y,
        predictions,
        labels=ACTIVITIES,
        target_names=ACTIVITIES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=ACTIVITIES,
    ).tolist()

    print(f"\n{name}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro F1: {macro_f1:.4f}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def main():
    print("=" * 72)
    print("OVERLAPPING TEMPORAL ACTIVITY MODEL TRAINING V2")
    print("=" * 72)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Temporal window: {TEMPORAL_WINDOWS} rows")
    print(f"Temporal stride: {TEMPORAL_STRIDE} row")
    print("Temporal statistics: " + ", ".join(STAT_NAMES))
    print("Session split occurs BEFORE temporal aggregation.")
    print()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = {
        "sample_id",
        "session_id",
        "timestamp",
        "activity",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df[df["activity"].isin(ACTIVITIES)].copy()

    feature_columns = [
        c
        for c in df.columns
        if c not in {"sample_id", "session_id", "timestamp", "activity"}
    ]

    if not feature_columns:
        raise ValueError("No feature columns found.")

    print("Original class/session counts:")
    print(df.groupby("activity")["session_id"].nunique().reindex(ACTIVITIES).fillna(0).astype(int))
    print()
    print("Original class/row counts:")
    print(df["activity"].value_counts().reindex(ACTIVITIES).fillna(0).astype(int))
    print()

    print("Splitting by session BEFORE temporal aggregation...")
    train_df, val_df, test_df = split_by_session(df)

    print(
        f"Train rows: {len(train_df)} | "
        f"Validation rows: {len(val_df)} | "
        f"Test rows: {len(test_df)}"
    )
    print(
        f"Train sessions: {train_df['session_id'].nunique()} | "
        f"Validation sessions: {val_df['session_id'].nunique()} | "
        f"Test sessions: {test_df['session_id'].nunique()}"
    )

    print("\nBuilding overlapping temporal datasets...")
    X_train, y_train, train_session_ids = build_temporal_dataset(
        train_df, feature_columns, TEMPORAL_WINDOWS, TEMPORAL_STRIDE
    )
    X_val, y_val, val_session_ids = build_temporal_dataset(
        val_df, feature_columns, TEMPORAL_WINDOWS, TEMPORAL_STRIDE
    )
    X_test, y_test, test_session_ids = build_temporal_dataset(
        test_df, feature_columns, TEMPORAL_WINDOWS, TEMPORAL_STRIDE
    )

    print(f"Temporal training rows: {len(X_train)}")
    print(f"Temporal validation rows: {len(X_val)}")
    print(f"Temporal test rows: {len(X_test)}")
    print()
    print("Temporal class counts:")
    for name, labels in [
        ("train", y_train),
        ("validation", y_val),
        ("test", y_test),
    ]:
        counts = pd.Series(labels).value_counts().reindex(ACTIVITIES).fillna(0).astype(int)
        print(f"{name}:")
        print(counts.to_string())
    print()

    temporal_feature_names = []
    for stat in STAT_NAMES:
        temporal_feature_names.extend(
            [f"{stat}__{feature}" for feature in feature_columns]
        )

    print(f"Original features: {len(feature_columns)}")
    print(f"Temporal features: {len(temporal_feature_names)}")
    print()

    candidates = {
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(
                    n_neighbors=15,
                    weights="distance",
                    n_jobs=-1,
                )),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2,
        ),
    }

    validation_results = {}

    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        validation_results[name] = evaluate_model(
            name,
            model,
            X_val,
            y_val,
        )

    best_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["macro_f1"],
            validation_results[name]["accuracy"],
        ),
    )

    best_model = candidates[best_name]

    print("\n" + "=" * 72)
    print(f"BEST VALIDATION MODEL: {best_name.upper()}")
    print("=" * 72)

    # Refit the selected model on train + validation temporal windows.
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])

    print(f"Refitting {best_name} on train + validation windows: {len(X_fit)}")
    best_model.fit(X_fit, y_fit)

    test_predictions = best_model.predict(X_test)

    test_accuracy = float(accuracy_score(y_test, test_predictions))
    test_macro_f1 = float(f1_score(y_test, test_predictions, average="macro"))

    test_report = classification_report(
        y_test,
        test_predictions,
        labels=ACTIVITIES,
        target_names=ACTIVITIES,
        output_dict=True,
        zero_division=0,
    )

    test_matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=ACTIVITIES,
    ).tolist()

    print("\nTEMPORAL V2 TEST")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {test_macro_f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=ACTIVITIES,
            target_names=ACTIVITIES,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(np.asarray(test_matrix))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    manifest = {
        "model_type": best_name,
        "version": "temporal_v2",
        "activities": ACTIVITIES,
        "seed": SEED,
        "temporal_windows": TEMPORAL_WINDOWS,
        "temporal_stride": TEMPORAL_STRIDE,
        "temporal_statistics": STAT_NAMES,
        "original_feature_count": len(feature_columns),
        "temporal_feature_count": len(temporal_feature_names),
        "feature_columns": feature_columns,
        "temporal_feature_names": temporal_feature_names,
        "training_session_split_before_aggregation": True,
        "model_path": str(MODEL_PATH),
    }

    report = {
        "version": "temporal_v2",
        "dataset": str(DATASET_PATH),
        "activities": ACTIVITIES,
        "seed": SEED,
        "temporal_windows": TEMPORAL_WINDOWS,
        "temporal_stride": TEMPORAL_STRIDE,
        "temporal_statistics": STAT_NAMES,
        "original_feature_count": len(feature_columns),
        "temporal_feature_count": len(temporal_feature_names),
        "original_rows": {
            activity: int((df["activity"] == activity).sum())
            for activity in ACTIVITIES
        },
        "original_sessions": {
            activity: int(
                df.loc[df["activity"] == activity, "session_id"].nunique()
            )
            for activity in ACTIVITIES
        },
        "splits": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_sessions": int(train_df["session_id"].nunique()),
            "validation_sessions": int(val_df["session_id"].nunique()),
            "test_sessions": int(test_df["session_id"].nunique()),
        },
        "temporal_samples": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "validation_results": validation_results,
        "selected_model": best_name,
        "test": {
            "accuracy": test_accuracy,
            "macro_f1": test_macro_f1,
            "classification_report": test_report,
            "confusion_matrix": test_matrix,
        },
        "model_path": str(MODEL_PATH),
        "original_activity_model_untouched": True,
        "previous_temporal_model_untouched": True,
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved V2 model separately:")
    print(MODEL_PATH)
    print(MANIFEST_PATH)
    print(REPORT_PATH)
    print("\nOriginal activity_model.pkl was NOT modified.")
    print("Previous activity_model_temporal.pkl was NOT modified.")
    print("\nIMPORTANT: Do not use this model in live inference yet.")
    print("activity_inference.py must be updated to reproduce the exact")
    print("20-row overlapping temporal feature construction first.")


if __name__ == "__main__":
    main()
