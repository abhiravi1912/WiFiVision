import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np

from config import load_config, setup_logging
from feature_extraction import FeatureConfig, FeatureExtractor
from preprocessing import PreprocessConfig, Preprocessor
from serial_receiver import SerialReceiver


class PresenceInference:
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
            / "presence_model_knn.pkl"
        )

        self.manifest_path = (
            self.model_dir
            / "presence_model_knn_manifest.json"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Presence model not found: {self.model_path}"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Presence manifest not found: {self.manifest_path}"
            )

        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

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

        self.last_prediction = None
        self.last_print_time = 0.0
        self.prediction_history = []
        self.stable_state = None

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
                    f"Unexpected feature shape: {values.shape}; "
                    "expected (180,)"
                )

            if not np.isfinite(values).all():
                values = np.nan_to_num(
                    values,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

            X = values.reshape(1, -1)

            prediction = self.model.predict(X)[0]

            # Temporal majority vote over the last 5 feature windows.
            self.prediction_history.append(str(prediction))

            if len(self.prediction_history) > 5:
                self.prediction_history.pop(0)

            if len(self.prediction_history) >= 5:
                empty_votes = self.prediction_history.count("EMPTY")
                present_votes = self.prediction_history.count("PRESENT")

                if empty_votes >= 3:
                    stable_prediction = "EMPTY"
                elif present_votes >= 3:
                    stable_prediction = "PRESENT"
                else:
                    stable_prediction = self.stable_state or str(prediction)

                self.stable_state = stable_prediction
            else:
                stable_prediction = self.stable_state or str(prediction)

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
                        f"{name}={prob:.2f}"
                        for name, prob in pairs
                    )

                    print(
                        f"[ONLINE] "
                        f"{stable_prediction:7s} | "
                        f"raw={str(prediction):7s} | "
                        f"{probability_text}"
                    )
                else:
                    print(
                        f"[ONLINE] {str(prediction)}"
                    )

                self.last_prediction = prediction
                self.last_print_time = now

        except Exception:
            logging.exception(
                "Error during presence inference"
            )

    def run(self):
        self.receiver = SerialReceiver(
            port=self.port,
            baud=self.baud,
            callback=self.on_sample,
        )

        print()
        print("========================================")
        print(" LIVE PRESENCE INFERENCE - KNN")
        print("========================================")
        print(f"Port       : {self.port}")
        print(f"Baud       : {self.baud}")
        print(f"Model      : {self.model_path}")
        print("Features   : 180")
        print("Classes    : EMPTY / PRESENT")
        print()
        print("Starting receiver...")
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
        print(f"CSI samples  : {self.sample_count}")
        print(f"Base features: {self.feature_count}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Live CSI presence inference using KNN."
    )

    parser.add_argument(
        "--port",
        default="COM13",
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

    inference = PresenceInference(
        port=args.port,
        baud=args.baud,
        config=config,
    )

    inference.run()


if __name__ == "__main__":
    main()