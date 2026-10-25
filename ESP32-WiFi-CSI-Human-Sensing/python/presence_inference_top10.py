import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import load_config
from feature_extraction import FeatureConfig, FeatureExtractor
from preprocessing import PreprocessConfig, Preprocessor
from serial_receiver import SerialReceiver


FEATURES = [
    "rssi_mean",
    "rssi_last",
    "temp_energy",
    "agg_mean",
    "agg_min",
    "sub_mean_0",
    "sub_mean_1",
    "sub_mean_3",
    "agg_max",
    "sub_mean_4",
]


class Top10Inference:
    def __init__(self, port, baud, config):
        self.root = Path(__file__).resolve().parents[1]
        model_dir = self.root / "data" / "activity" / "models"

        self.model = joblib.load(
            model_dir / "presence_model_knn_top10.pkl"
        )

        self.preprocess_config = PreprocessConfig.from_config(config)
        self.preprocessor = Preprocessor(self.preprocess_config)

        self.feature_config = FeatureConfig.from_config(config)
        self.extractor = FeatureExtractor(
            self.feature_config,
            preprocess_config=self.preprocess_config,
        )

        self.port = port
        self.baud = baud
        self.receiver = None
        self.last_prediction = None
        self.last_print_time = 0.0
        self.samples = 0
        self.features = 0

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

            names = feature_vector.names
            values = np.asarray(
                feature_vector.values,
                dtype=np.float32,
            )

            if values.shape != (180,):
                raise ValueError(
                    f"Unexpected feature shape: {values.shape}"
                )

            if not np.isfinite(values).all():
                values = np.nan_to_num(
                    values,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

            feature_map = dict(zip(names, values))

            X = pd.DataFrame(
                [[feature_map[name] for name in FEATURES]],
                columns=FEATURES,
            )

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
                    pairs = sorted(
                        zip(self.model.classes_, probabilities),
                        key=lambda x: x[1],
                        reverse=True,
                    )

                    probability_text = " | ".join(
                        f"{name}={prob:.2f}"
                        for name, prob in pairs
                    )

                    print(
                        f"[ONLINE] {str(prediction):7s} | "
                        f"{probability_text}"
                    )
                else:
                    print(f"[ONLINE] {prediction}")

                self.last_prediction = prediction
                self.last_print_time = now

        except Exception as exc:
            print(f"[ERROR] {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM15")
    parser.add_argument("--baud", type=int, default=921600)
    args = parser.parse_args()

    config = load_config()

    app = Top10Inference(
        args.port,
        args.baud,
        config,
    )

    receiver = SerialReceiver(
        port=args.port,
        baud=args.baud,
        callback=app.on_sample,
    )

    app.receiver = receiver

    print(f"[START] Top-10 presence KNN on {args.port} @ {args.baud}")
    print("[INFO] Stay in the room or leave the sensing area to test.")
    print("[INFO] Press Ctrl+C to stop.")

    try:
        receiver.start()

        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[STOP] Stopping...")

    finally:
        receiver.stop()
        print(
            f"[STATS] CSI samples={app.samples} "
            f"features={app.features}"
        )


if __name__ == "__main__":
    main()



