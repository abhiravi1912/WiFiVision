import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np

from config import load_config, setup_logging
from feature_extraction import FeatureConfig, FeatureExtractor
from preprocessing import PreprocessConfig, Preprocessor
from serial_receiver import SerialReceiver


class ActivityInference:
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

        self.model_path = self.model_dir / "activity_model.pkl"

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Activity model not found: {self.model_path}"
            )

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

        # Temporary diagnostic counter.
        self.diagnostic_count = 0

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

            # Temporary live-feature diagnostic.
            if self.diagnostic_count < 20:
                names = feature_vector.names
                values = feature_vector.values

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

            X = np.asarray(
                feature_vector.values,
                dtype=float,
            ).reshape(1, -1)

            prediction = self.model.predict(X)[0]

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
        print(" LIVE ACTIVITY INFERENCE")
        print("========================================")
        print(f"Port  : {self.port}")
        print(f"Baud  : {self.baud}")
        print(f"Model : {self.model_path}")
        print()
        print(
            "Classes: EMPTY / STANDING / SITTING / MOVING"
        )
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
        print(f"CSI samples : {self.sample_count}")
        print(f"Features    : {self.feature_count}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Live CSI activity inference."
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