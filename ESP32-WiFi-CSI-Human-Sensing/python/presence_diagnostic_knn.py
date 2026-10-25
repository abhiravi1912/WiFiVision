import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import load_config, setup_logging
from feature_extraction import FeatureConfig, FeatureExtractor
from preprocessing import PreprocessConfig, Preprocessor
from serial_receiver import SerialReceiver


class PresenceDiagnostic:
    def __init__(self, port, baud, config):
        self.root = Path(__file__).resolve().parents[1]
        self.model_dir = self.root / "data" / "activity" / "models"

        self.model = joblib.load(
            self.model_dir / "presence_model_knn.pkl"
        )

        with open(
            self.model_dir / "presence_model_knn_manifest.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.manifest = json.load(f)

        self.feature_columns = self.manifest["feature_columns"]

        self.preprocess_config = PreprocessConfig.from_config(config)
        self.preprocessor = Preprocessor(self.preprocess_config)

        self.feature_config = FeatureConfig.from_config(config)
        self.extractor = FeatureExtractor(
            self.feature_config,
            preprocess_config=self.preprocess_config,
        )

        self.receiver = None
        self.samples = 0
        self.features = 0
        self.results = []
        self.live_features = []

    def on_sample(self, sample):
        self.samples += 1

        try:
            amplitude = self.preprocessor.process(sample)

            if amplitude is None:
                return

            feature_vector = self.extractor.update(
                amplitude,
                sample.rssi,
                sample.ts_local,
            )

            if feature_vector is None:
                return

            self.features += 1

            values = np.asarray(
                feature_vector.values,
                dtype=np.float32,
            )

            if values.shape != (180,):
                return

            if not np.isfinite(values).all():
                values = np.nan_to_num(
                    values,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

            X = pd.DataFrame(
                [values],
                columns=self.feature_columns,
            )

            self.live_features.append(values.copy())

            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            # Get transformed vector and nearest training examples.
            scaler = self.model.named_steps["scaler"]
            knn = self.model.named_steps["knn"]

            X_scaled = scaler.transform(X)

            distances, indices = knn.kneighbors(
                X_scaled,
                n_neighbors=5,
            )

            classes = self.model.classes_

            neighbor_labels = []

            # KNN stores training labels in _y.
            for idx, dist in zip(
                indices[0],
                distances[0],
            ):
                label = knn._y[idx]
                neighbor_labels.append(
                    (str(label), float(dist))
                )

            self.results.append({
                "prediction": str(prediction),
                "probabilities": {
                    str(c): float(p)
                    for c, p in zip(classes, probabilities)
                },
                "neighbors": neighbor_labels,
            })

        except Exception:
            logging.exception("Diagnostic error")

    def run(self, seconds):
        self.receiver = SerialReceiver(
            port=self.port,
            baud=self.baud,
            callback=self.on_sample,
        )

        print()
        print("========================================")
        print(" LIVE PRESENCE KNN DIAGNOSTIC")
        print("========================================")
        print(f"Port       : {self.port}")
        print(f"Baud       : {self.baud}")
        print("Room       : EMPTY")
        print(f"Duration   : {seconds} seconds")
        print()
        print("IMPORTANT: Make sure nobody is in the sensing area.")
        print()

        self.receiver.start()

        end_time = time.time() + seconds

        try:
            while time.time() < end_time:
                elapsed = int(seconds - max(0, end_time - time.time()))
                if elapsed % 5 == 0:
                    print(
                        f"  collecting... {elapsed:02d}/{seconds}s | "
                        f"CSI={self.samples} | features={self.features}",
                        flush=True,
                    )
                time.sleep(1)
        finally:
            self.receiver.stop()
            self.receiver.join(timeout=3)

        print()
        print("========================================")
        print(" DIAGNOSTIC RESULT")
        print("========================================")
        print(f"CSI samples  : {self.samples}")
        print(f"Base features: {self.features}")
        print(f"Predictions  : {len(self.results)}")
        print()

        if not self.results:
            print("NO FEATURE WINDOWS CAPTURED.")
            return

        counts = {}
        empty_probs = []
        present_probs = []

        for r in self.results:
            p = r["prediction"]
            counts[p] = counts.get(p, 0) + 1

            empty_probs.append(
                r["probabilities"].get("EMPTY", 0.0)
            )
            present_probs.append(
                r["probabilities"].get("PRESENT", 0.0)
            )

        print("Prediction counts:")
        for label, count in sorted(counts.items()):
            print(f"  {label:8s}: {count}")

        print()
        print(
            f"Mean EMPTY probability   : "
            f"{np.mean(empty_probs):.4f}"
        )
        print(
            f"Mean PRESENT probability : "
            f"{np.mean(present_probs):.4f}"
        )

        print()
        print("First 10 nearest-neighbor results:")
        print()

        for i, r in enumerate(self.results[:10], 1):
            probs = " | ".join(
                f"{k}={v:.2f}"
                for k, v in r["probabilities"].items()
            )

            neighbors = " | ".join(
                f"{label}@{dist:.2f}"
                for label, dist in r["neighbors"]
            )

            print(
                f"{i:02d}. {r['prediction']:7s} | "
                f"{probs}"
            )
            print(
                f"    neighbors: {neighbors}"
            )

        output_path = (
            self.root
            / "data"
            / "activity"
            / "processed"
            / "live_empty_diagnostic.csv"
        )

        if self.live_features:
            live_df = pd.DataFrame(
                self.live_features,
                columns=self.feature_columns,
            )
            live_df.to_csv(output_path, index=False)

            print()
            print(f"Saved live EMPTY features:")
            print(f"  {output_path}")
            print(f"  Rows: {len(live_df)}")
            print(f"  Features: {len(live_df.columns)}")

        print()
        print("Interpretation:")
        print(
            "If EMPTY live windows have mostly PRESENT "
            "nearest neighbors, the training distribution "
            "does not represent the live empty-room signal."
        )
        print(
            "If EMPTY neighbors are mostly EMPTY but the "
            "prediction is still PRESENT, we have a KNN/"
            "probability/decision issue to fix."
        )
        print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        default="COM13",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
    )

    parser.add_argument(
        "--seconds",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--config",
        default=None,
    )

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    diagnostic = PresenceDiagnostic(
        args.port,
        args.baud,
        config,
    )

    # Needed by run()
    diagnostic.port = args.port
    diagnostic.baud = args.baud

    diagnostic.run(args.seconds)


if __name__ == "__main__":
    main()
