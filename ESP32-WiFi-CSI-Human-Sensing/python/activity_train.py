import json
import logging
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


ACTIVITIES = ["empty", "standing", "sitting", "moving"]

RANDOM_SEED = 42

# Temporal configuration.
# The diagnostic showed:
# 20 windows ~= 2.7 seconds
TEMPORAL_WINDOWS = 20

# Require a complete temporal chunk.
# This prevents partial chunks from being treated as full windows.
DROP_INCOMPLETE_CHUNKS = True


def split_by_session(df):
    """
    Split complete sessions into train / validation / test.

    IMPORTANT:
    Splitting happens BEFORE temporal aggregation so that windows
    from the same physical session can never leak across splits.
    """
    rng = np.random.RandomState(RANDOM_SEED)

    train_sessions = []
    val_sessions = []
    test_sessions = []

    for activity in ACTIVITIES:
        sessions = df.loc[
            df["activity"] == activity,
            "session_id",
        ].unique()

        sessions = np.array(sessions)
        rng.shuffle(sessions)

        n = len(sessions)

        if n < 3:
            raise ValueError(
                f"Activity '{activity}' has only {n} sessions. "
                "At least 3 sessions are required."
            )

        n_test = max(1, round(n * 0.20))
        n_val = max(1, round(n * 0.20))

        if n_test + n_val >= n:
            n_val = 1
            n_test = 1

        test = sessions[:n_test]
        val = sessions[n_test:n_test + n_val]
        train = sessions[n_test + n_val:]

        train_sessions.extend(train)
        val_sessions.extend(val)
        test_sessions.extend(test)

    train_df = df[df["session_id"].isin(train_sessions)].copy()
    val_df = df[df["session_id"].isin(val_sessions)].copy()
    test_df = df[df["session_id"].isin(test_sessions)].copy()

    return train_df, val_df, test_df


def build_temporal_dataset(
    df,
    feature_columns,
    windows=TEMPORAL_WINDOWS,
):
    """
    Convert short feature windows into temporal feature vectors.

    For every complete group of `windows` consecutive rows within a
    session, calculate:

        mean(feature)
        std(feature)

    The resulting vector is:

        [mean_f1 ... mean_fn, std_f1 ... std_fn]

    A temporal chunk never crosses a session boundary.
    """

    rows = []

    # Sort so temporal order is deterministic.
    work = df.copy()

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        errors="coerce",
    )

    if work["timestamp"].isna().any():
        raise ValueError(
            "Dataset contains invalid timestamps."
        )

    work = work.sort_values(
        ["session_id", "timestamp", "sample_id"]
    )

    for session_id, session_df in work.groupby(
        "session_id",
        sort=False,
    ):
        session_df = session_df.reset_index(drop=True)

        n_rows = len(session_df)

        # Complete non-overlapping chunks.
        n_chunks = n_rows // windows

        if n_chunks == 0:
            continue

        for chunk_index in range(n_chunks):
            start = chunk_index * windows
            end = start + windows

            chunk = session_df.iloc[start:end]

            # Safety check.
            if len(chunk) != windows:
                continue

            labels = chunk["activity"].astype(str).str.lower()

            # A chunk should contain one activity only.
            unique_labels = labels.unique()

            if len(unique_labels) != 1:
                # This should normally never happen because the collector
                # creates one activity per session.
                continue

            activity = unique_labels[0]

            values = chunk[feature_columns].to_numpy(
                dtype=float
            )

            means = np.mean(values, axis=0)

            # ddof=0 avoids NaN for any valid chunk.
            stds = np.std(
                values,
                axis=0,
                ddof=0,
            )

            temporal_values = np.concatenate(
                [means, stds]
            )

            row = {
                "session_id": session_id,
                "chunk_index": chunk_index,
                "activity": activity,
            }

            for i, feature_name in enumerate(
                feature_columns
            ):
                row[f"mean__{feature_name}"] = means[i]

            for i, feature_name in enumerate(
                feature_columns
            ):
                row[f"std__{feature_name}"] = stds[i]

            rows.append(row)

    if not rows:
        raise ValueError(
            "No complete temporal chunks could be created."
        )

    temporal_df = pd.DataFrame(rows)

    temporal_feature_columns = [
        column
        for column in temporal_df.columns
        if column.startswith("mean__")
        or column.startswith("std__")
    ]

    return temporal_df, temporal_feature_columns


def train_and_evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
):
    """
    Train KNN and Random Forest candidates and select by
    validation macro F1.
    """

    candidates = {
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }

    results = {}

    best_name = None
    best_score = -1.0
    best_model = None

    print()
    print("Training candidates...")
    print()

    for name, model in candidates.items():
        print(f"Training: {name}")

        model.fit(
            X_train,
            y_train,
        )

        val_pred = model.predict(
            X_val
        )

        val_accuracy = accuracy_score(
            y_val,
            val_pred,
        )

        val_f1 = f1_score(
            y_val,
            val_pred,
            labels=ACTIVITIES,
            average="macro",
            zero_division=0,
        )

        results[name] = {
            "validation_accuracy": float(
                val_accuracy
            ),
            "validation_macro_f1": float(
                val_f1
            ),
        }

        print(
            f"  Validation accuracy : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"  Validation macro F1 : "
            f"{val_f1:.4f}"
        )

        print()

        if val_f1 > best_score:
            best_score = val_f1
            best_name = name
            best_model = model

    return (
        best_name,
        best_score,
        best_model,
        results,
    )


def main():
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "data"
        / "activity"
        / "processed"
        / "activity_dataset.csv"
    )

    model_dir = (
        project_root
        / "data"
        / "activity"
        / "models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print()
    print("========================================")
    print(" TEMPORAL ACTIVITY MODEL TRAINING")
    print("========================================")
    print()

    print(
        f"Dataset: {dataset_path}"
    )

    print(
        f"Temporal windows: {TEMPORAL_WINDOWS}"
    )

    print(
        "Approximate temporal duration: "
        "~2.7 seconds"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    df = pd.read_csv(
        dataset_path
    )

    required = {
        "sample_id",
        "session_id",
        "timestamp",
        "activity",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df["activity"] = (
        df["activity"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unknown = sorted(
        set(df["activity"])
        - set(ACTIVITIES)
    )

    if unknown:
        raise ValueError(
            f"Unknown activity labels found: "
            f"{unknown}"
        )

    df = df[
        df["activity"].isin(ACTIVITIES)
    ].copy()

    # Keep exactly the same feature selection philosophy
    # as the original trainer.
    feature_columns = [
        c
        for c in df.columns
        if c not in {
            "sample_id",
            "session_id",
            "timestamp",
            "activity",
        }
    ]

    if not feature_columns:
        raise ValueError(
            "No feature columns found."
        )

    X_check = df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X_check.isna().any().any():
        raise ValueError(
            "Dataset contains non-numeric "
            "or missing feature values."
        )

    if not np.isfinite(
        X_check.to_numpy()
    ).all():
        raise ValueError(
            "Dataset contains non-finite "
            "feature values."
        )

    print()
    print("Original class/session counts:")
    print(
        df.groupby("activity")[
            "session_id"
        ]
        .nunique()
        .to_string()
    )

    print()
    print("Original class/row counts:")
    print(
        df.groupby("activity")
        .size()
        .to_string()
    )

    # -------------------------------------------------
    # SPLIT FIRST, THEN TEMPORAL AGGREGATION
    # -------------------------------------------------

    print()
    print(
        "Splitting by session BEFORE "
        "temporal aggregation..."
    )

    train_df, val_df, test_df = (
        split_by_session(df)
    )

    print(
        f"Train rows: {len(train_df)} | "
        f"Validation rows: {len(val_df)} | "
        f"Test rows: {len(test_df)}"
    )

    print(
        f"Train sessions: "
        f"{train_df['session_id'].nunique()} | "
        f"Validation sessions: "
        f"{val_df['session_id'].nunique()} | "
        f"Test sessions: "
        f"{test_df['session_id'].nunique()}"
    )

    print()
    print(
        "Building temporal training dataset..."
    )

    train_temporal, temporal_features = (
        build_temporal_dataset(
            train_df,
            feature_columns,
            TEMPORAL_WINDOWS,
        )
    )

    print(
        f"Temporal training rows: "
        f"{len(train_temporal)}"
    )

    print()
    print(
        "Building temporal validation dataset..."
    )

    val_temporal, _ = (
        build_temporal_dataset(
            val_df,
            feature_columns,
            TEMPORAL_WINDOWS,
        )
    )

    print(
        f"Temporal validation rows: "
        f"{len(val_temporal)}"
    )

    print()
    print(
        "Building temporal test dataset..."
    )

    test_temporal, _ = (
        build_temporal_dataset(
            test_df,
            feature_columns,
            TEMPORAL_WINDOWS,
        )
    )

    print(
        f"Temporal test rows: "
        f"{len(test_temporal)}"
    )

    # -------------------------------------------------
    # TEMPORAL DATASET SUMMARY
    # -------------------------------------------------

    print()
    print(
        "Temporal class counts:"
    )

    print(
        train_temporal.groupby(
            "activity"
        ).size().to_string()
    )

    # -------------------------------------------------
    # MODEL MATRICES
    # -------------------------------------------------

    X_train = (
        train_temporal[
            temporal_features
        ]
        .to_numpy(dtype=float)
    )

    y_train = (
        train_temporal["activity"]
        .to_numpy()
    )

    X_val = (
        val_temporal[
            temporal_features
        ]
        .to_numpy(dtype=float)
    )

    y_val = (
        val_temporal["activity"]
        .to_numpy()
    )

    X_test = (
        test_temporal[
            temporal_features
        ]
        .to_numpy(dtype=float)
    )

    y_test = (
        test_temporal["activity"]
        .to_numpy()
    )

    print()
    print(
        f"Original features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Temporal features: "
        f"{len(temporal_features)}"
    )

    # -------------------------------------------------
    # TRAIN
    # -------------------------------------------------

    (
        best_name,
        best_score,
        best_model,
        results,
    ) = train_and_evaluate(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    print(
        "========================================"
    )
    print(
        " BEST TEMPORAL MODEL"
    )
    print(
        "========================================"
    )

    print(
        f"Model: {best_name}"
    )

    print(
        f"Validation macro F1: "
        f"{best_score:.4f}"
    )

    # -------------------------------------------------
    # TEST
    # -------------------------------------------------

    test_pred = best_model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        labels=ACTIVITIES,
        average="macro",
        zero_division=0,
    )

    print()
    print(
        "========================================"
    )
    print(
        " TEMPORAL TEST RESULTS"
    )
    print(
        "========================================"
    )

    print(
        f"Accuracy : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{test_f1:.4f}"
    )

    print()
    print(
        "Classification report:"
    )

    report_text = classification_report(
        y_test,
        test_pred,
        labels=ACTIVITIES,
        zero_division=0,
    )

    print(report_text)

    print(
        "Confusion matrix:"
    )

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=ACTIVITIES,
    )

    cm_df = pd.DataFrame(
        cm,
        index=ACTIVITIES,
        columns=ACTIVITIES,
    )

    print(
        cm_df
    )

    # -------------------------------------------------
    # SAVE TEMPORAL MODEL SEPARATELY
    # -------------------------------------------------

    model_path = (
        model_dir
        / "activity_model_temporal.pkl"
    )

    manifest_path = (
        model_dir
        / "activity_model_temporal_manifest.json"
    )

    report_path = (
        model_dir
        / "activity_temporal_training_report.json"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    manifest = {
        "model_type": best_name,
        "model_variant": "temporal",
        "activities": ACTIVITIES,
        "original_feature_columns": feature_columns,
        "temporal_feature_columns": temporal_features,
        "n_original_features": len(
            feature_columns
        ),
        "n_temporal_features": len(
            temporal_features
        ),
        "temporal_windows": TEMPORAL_WINDOWS,
        "approx_temporal_seconds": 2.7,
        "aggregation": [
            "mean",
            "std",
        ],
        "random_seed": RANDOM_SEED,
        "dataset": str(
            dataset_path
        ),
    }

    report = {
        "model_type": best_name,
        "model_variant": "temporal",
        "activities": ACTIVITIES,
        "n_original_features": len(
            feature_columns
        ),
        "n_temporal_features": len(
            temporal_features
        ),
        "temporal_windows": TEMPORAL_WINDOWS,
        "approx_temporal_seconds": 2.7,
        "aggregation": [
            "mean",
            "std",
        ],
        "train_rows_original": int(
            len(train_df)
        ),
        "validation_rows_original": int(
            len(val_df)
        ),
        "test_rows_original": int(
            len(test_df)
        ),
        "train_temporal_rows": int(
            len(train_temporal)
        ),
        "validation_temporal_rows": int(
            len(val_temporal)
        ),
        "test_temporal_rows": int(
            len(test_temporal)
        ),
        "train_sessions": int(
            train_df["session_id"].nunique()
        ),
        "validation_sessions": int(
            val_df["session_id"].nunique()
        ),
        "test_sessions": int(
            test_df["session_id"].nunique()
        ),
        "validation_accuracy": float(
            results[best_name][
                "validation_accuracy"
            ]
        ),
        "validation_macro_f1": float(
            results[best_name][
                "validation_macro_f1"
            ]
        ),
        "test_accuracy": float(
            test_accuracy
        ),
        "test_macro_f1": float(
            test_f1
        ),
        "candidate_results": results,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            test_pred,
            labels=ACTIVITIES,
            output_dict=True,
            zero_division=0,
        ),
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(
        "Saved:"
    )

    print(
        f"  Model    : {model_path}"
    )

    print(
        f"  Manifest : {manifest_path}"
    )

    print(
        f"  Report   : {report_path}"
    )

    print()
    print(
        "Original activity_model.pkl "
        "was NOT modified."
    )

    print()
    print(
        "Temporal training complete."
    )


if __name__ == "__main__":
    main()