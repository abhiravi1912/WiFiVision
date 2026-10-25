
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config, setup_logging
from feature_extraction import FeatureConfig, FeatureExtractor
from preprocessing import PreprocessConfig, Preprocessor
from serial_receiver import SerialReceiver


class BaselinePresence:
    def __init__(self, port, baud, config):
        self.root = Path(__file__).resolve().parents[1]

        baseline_path = (
            self.root
            / "data"
            / "activity"
            / "processed"
            / "live_empty_diagnostic.csv"
        )

        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline file not found: {baseline_path}"
            )

        baseline_df = pd.read_csv(baseline_path)

        std = baseline_df.std()

        self.feature_columns = baseline_df.columns.tolist()

        usable = std[
            (std > 1e-6) &
            (std < std.median() * 20)
        ].index.tolist()

        self.usable_features = usable

        self.baseline = baseline_df[usable].median()

        mad = (
            baseline_df[usable] - self.baseline
        ).abs().median()

        self.mad = mad.replace(0, np.nan)

        self.threshold = 2.2661

        preprocess_config = PreprocessConfig.from_config(config)
        self.preprocessor = Preprocessor(preprocess_config)

        feature_config = FeatureConfig.from_config(config)
        self.extractor = FeatureExtractor(
            feature_config,
            preprocess_config=preprocess_config,
        )

        self.receiver = None
        self.samples = 0
        self.features = 0
        self.last_state = None
        self.last_print = 0.0
        self.present_count = 0
        self.empty_count = 0

    def calculate_score(self, values):
        row = pd.Series(
            values,
            index=self.feature_columns,
        )

        z = (
            (row[self.usable_features] - self.baseline).abs()
            / self.mad
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if len(z) == 0:
            return float("inf")

        return float(z.median())

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

            values = np.nan_to_num(
                values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            score = self.calculate_score(values)

            raw_state = (
                "EMPTY"
                if score <= self.threshold
                else "PRESENT"
            )

            if raw_state == "PRESENT":
                self.present_count += 1
                self.empty_count = 0
            else:
                self.empty_count += 1
                self.present_count = 0

            if self.present_count >= 4:
                state = "PRESENT"
            elif self.empty_count >= 4:
                state = "EMPTY"
            else:
                state = self.last_state or "EMPTY"

            now = time.time()

            if (
                state != self.last_state
                or now - self.last_print >= 1.0
            ):
                print(
                    f"[ONLINE] {state:7s} | "
                    f"score={score:.3f} | "
                    f"threshold={self.threshold:.3f}"
                )

                self.last_state = state
                self.last_print = now

        except Exception as exc:
            print(f"[ERROR] {exc}")

    def run(self):
        self.receiver = SerialReceiver(
            port=self.port,
            baud=self.baud,
            callback=self.on_sample,
        )

        print()
        print("========================================")
        print(" LIVE BASELINE PRESENCE")
        print("========================================")
        print(f"Port       : {self.port}")
        print(f"Baud       : {self.baud}")
        print(f"Features   : {len(self.usable_features)}")
        print(f"Threshold  : {self.threshold}")
        print()
        print("EMPTY baseline loaded.")
        print("Press Ctrl+C to stop.")
        print()

        self.receiver.start()

        try:
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print()
            print("Stopping...")

        finally:
            self.receiver.stop()
            self.receiver.join(timeout=3)

        print()
        print(f"CSI samples  : {self.samples}")
        print(f"Base features: {self.features}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        default="COM15",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=921600,
    )

    parser.add_argument(
        "--config",
        default=None,
    )

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    detector = BaselinePresence(
        args.port,
        args.baud,
        config,
    )

    detector.port = args.port
    detector.baud = args.baud

    detector.run()


if __name__ == "__main__":
    main()
