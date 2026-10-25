import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np

from config import load_config, setup_logging
from feature_extraction import FeatureConfig, FeatureExtractor
from preprocessing import PreprocessConfig, Preprocessor
from serial_receiver import SerialReceiver


class ActivityInference:
    """
    Live CSI activity inference using the Temporal V2 model.

    Temporal V2 must exactly match the training pipeline:
      - 20 consecutive 180-feature rows
      - overlapping windows, stride 1
      - mean, std, min, max, range, delta (last - first)
      - 1080 final features
    """

    TEMPORAL_WINDOWS = 20

    def __init__(self, port, baud, config):
        self.project_root = Path(__file__).resolve().parents[1]

        self.port = port
        self.baud = baud
        self.config = config

        self.model_dir = (
            self.project_root
            / "data"
            / "activity"
            / "models"
        )

        self.model_path = (
            self.model_dir
            / "activity_model_temporal_v2.pkl"
        )

        self.manifest_path = (
            self.model_dir
            / "activity_model_temporal_v2_manifest.json"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Temporal V2 model not found: {self.model_path}"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Temporal V2 manifest not found: {self.manifest_path}"
            )

        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self._validate_manifest()

        self.model = joblib.load(self.model_path)

        self.preprocess_config = PreprocessConfig.from_config(config)
        self.preprocessor = Preprocessor(self.preprocess_config)

        self.feature_config = FeatureConfig.from_config(config)
        self.extractor = FeatureExtractor(
            self.feature_config,
            preprocess_config=self.preprocess_config,
        )

        self.receiver = None
        self.sample_count = 0
        self.feature_count = 0
        self.temporal_prediction_count = 0

        # Rolling buffer of the original 180-feature rows.
        # V2 uses an overlapping window with stride 1, so every
        # new feature row after the first 20 produces a prediction.
        self.feature_buffer = deque(
            maxlen=self.TEMPORAL_WINDOWS
        )

        self.last_prediction = None
        self.last_print_time = 0.0

        # Temporary diagnostic counter.
        self.diagnostic_count = 0

    def _validate_manifest(self):
        expected_windows = 20
        expected_stride = 1
        expected_statistics = [
            "mean",
            "std",
            "min",
            "max",
            "range",
            "delta",
        ]
        expected_original_features = 180
        expected_temporal_features = 1080

        if self.manifest.get("version") != "temporal_v2":
            raise ValueError(
                "Loaded model manifest is not temporal_v2."
            )

        if self.manifest.get("temporal_windows") != expected_windows:
            raise ValueError(
                "Temporal V2 window size mismatch: "
                f"{self.manifest.get('temporal_windows')} != "
                f"{expected_windows}"
            )

        if self.manifest.get("temporal_stride") != expected_stride:
            raise ValueError(
                "Temporal V2 stride mismatch: "
                f"{self.manifest.get('temporal_stride')} != "
                f"{expected_stride}"
            )

        if self.manifest.get("temporal_statistics") != expected_statistics:
            raise ValueError(
                "Temporal V2 statistics mismatch."
            )

        if self.manifest.get("original_feature_count") != expected_original_features:
            raise ValueError(
                "Temporal V2 original feature count mismatch: "
                f"{self.manifest.get('original_feature_count')} != "
                f"{expected_original_features}"
            )

        if self.manifest.get("temporal_feature_count") != expected_temporal_features:
            raise ValueError(
                "Temporal V2 temporal feature count mismatch: "
                f"{self.manifest.get('temporal_feature_count')} != "
                f"{expected_temporal_features}"
            )

    @staticmethod
    def build_temporal_features(feature_rows):
        """
        Reproduce activity_train_temporal_v2.py exactly.

        Input:
            20 rows x 180 original features

        Output:
            1 x 1080 features in this exact order:
              all means,
              all stds,
              all mins,
              all maxes,
              all ranges,
              all deltas.
        """
        window = np.asarray(
            feature_rows,
            dtype=np.float32,
        )

        if window.shape != (20, 180):
            raise ValueError(
                "Invalid temporal window shape: "
                f"{window.shape}; expected (20, 180)"
            )

        if not np.isfinite(window).all():
            window = np.nan_to_num(
                window,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        mean = window.mean(axis=0)
        std = window.std(axis=0)
        min_v = window.min(axis=0)
        max_v = window.max(axis=0)
        range_v = max_v - min_v
        delta = window[-1] - window[0]

        temporal_vector = np.concatenate(
            [mean, std, min_v, max_v, range_v, delta]
        ).astype(np.float32, copy=False)

        return temporal_vector.reshape(1, -1)

    def on_sample(self, sample):
        self.sample_count += 1

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

            self.feature_count += 1

            values = np.asarray(
                feature_vector.values,
                dtype=np.float32,
            )

            if values.shape != (180,):
                raise ValueError(
                    "Unexpected base feature vector shape: "
                    f"{values.shape}; expected (180,)"
                )

            self.feature_buffer.append(values)

            # Temporary live-feature diagnostic.
            if self.diagnostic_count < 20:
                names = feature_vector.names

                diagnostic_names = [
                    "rssi_mean",
                    "rssi_std",
                    "temp_std",
                    "temp_mean_abs_diff",
                    "temp_energy",
                    "move_rel_activity",
                    "move_peak_rel_diff",
                ]

                diagnostic = {}

                for name in diagnostic_names:
                    if name in names:
                        idx = names.index(name)
                        diagnostic[name] = float(values[idx])

                print("[DIAGNOSTIC]", diagnostic)

                self.diagnostic_count += 1

            # V2 needs the first 20 base feature rows before
            # producing its first prediction.
            if len(self.feature_buffer) < self.TEMPORAL_WINDOWS:
                return

            X = self.build_temporal_features(
                list(self.feature_buffer)
            )

            if X.shape != (1, 1080):
                raise ValueError(
                    "Unexpected temporal feature shape: "
                    f"{X.shape}; expected (1, 1080)"
                )

            prediction = self.model.predict(X)[0]
            self.temporal_prediction_count += 1

            probabilities = None

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X)[0]

            now = time.time()

            if (
                prediction != self.last_prediction
                or now - self.last_print_time >= 1.0
            ):
                if probabilities is not None:
                    classes = self.model.classes_

                    pairs = sorted(
                        zip(classes, probabilities),
                        key=lambda x: x[1],
                        reverse=True,
                    )

                    probability_text = " | ".join(
                        f"{name.upper()}={prob:.2f}"
                        for name, prob in pairs
                    )

                    print(
                        f"[ONLINE] "
                        f"{str(prediction).upper():8s} | "
                        f"{probability_text}"
                    )
                else:
                    print(
                        f"[ONLINE] "
                        f"{str(prediction).upper()}"
                    )

                self.last_prediction = prediction
                self.last_print_time = now

        except Exception:
            logging.exception(
                "Error during activity inference"
            )

    def run(self):
        self.receiver = SerialReceiver(
            port=self.port,
            baud=self.baud,
            callback=self.on_sample,
        )

        print()
        print("========================================")
        print(" LIVE ACTIVITY INFERENCE - TEMPORAL V2")
        print("========================================")
        print(f"Port       : {self.port}")
        print(f"Baud       : {self.baud}")
        print(f"Model      : {self.model_path}")
        print(f"Window     : {self.TEMPORAL_WINDOWS} feature rows")
        print("Stride     : 1 row")
        print("Statistics : mean/std/min/max/range/delta")
        print("Features   : 180 -> 1080")
        print()
        print(
            "Classes: EMPTY / STANDING / SITTING / MOVING"
        )
        print()
        print("Starting receiver...")
        print(
            "First prediction appears after 20 feature rows."
        )
        print("Press Ctrl+C to stop.")
        print()

        try:
            self.receiver.start()

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print()
            print("Stopping inference...")

        finally:
            if self.receiver is not None:
                self.receiver.stop()
                self.receiver.join(timeout=3)

        print()
        print("Inference stopped.")
        print(f"CSI samples       : {self.sample_count}")
        print(f"Base features     : {self.feature_count}")
        print(f"Temporal predicts : {self.temporal_prediction_count}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Live CSI activity inference using Temporal V2."
    )

    parser.add_argument(
        "--port",
        default="COM15",
        help="ESP32 receiver serial port.",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
        help="Serial baud rate.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    inference = ActivityInference(
        port=args.port,
        baud=args.baud,
        config=config,
    )

    inference.run()


if __name__ == "__main__":
    main()
