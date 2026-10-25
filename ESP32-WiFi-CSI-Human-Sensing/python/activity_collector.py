import argparse
import csv
import logging
import time
from pathlib import Path

from config import load_config, setup_logging
from feature_extraction import FeatureConfig, FeatureExtractor
from preprocessing import PreprocessConfig, Preprocessor
from serial_receiver import SerialReceiver


class ActivityCollector:
    def __init__(self, activity, port, baud, config):
        self.activity = activity
        self.port = port
        self.baud = baud
        self.config = config

        self.project_root = Path(__file__).resolve().parents[1]
        self.output_root = self.project_root / "data" / "activity"

        self.raw_dir = self.output_root / "raw"
        self.processed_dir = self.output_root / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.preprocess_config = PreprocessConfig.from_config(config)
        self.preprocessor = Preprocessor(self.preprocess_config)

        self.feature_config = FeatureConfig.from_config(config)
        self.extractor = FeatureExtractor(
            self.feature_config,
            preprocess_config=self.preprocess_config,
        )

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{activity}_{timestamp}"

        self.raw_path = self.raw_dir / f"{self.session_id}.csv"
        self.dataset_path = self.processed_dir / "activity_dataset.csv"

        self.receiver = None
        self.feature_file = None
        self.writer = None

        self.sample_count = 0
        self.feature_count = 0

    def setup_writer(self):
        file_exists = (
            self.dataset_path.exists()
            and self.dataset_path.stat().st_size > 0
        )

        self.feature_file = self.dataset_path.open(
            "a",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.writer(self.feature_file)

        if not file_exists:
            header = [
                "sample_id",
                "session_id",
                "timestamp",
                "activity",
            ] + self.extractor.feature_names()

            self.writer.writerow(header)
            self.feature_file.flush()

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

            row = [
                self.feature_count,
                self.session_id,
                feature_vector.timestamp,
                self.activity,
            ] + feature_vector.values.tolist()

            self.writer.writerow(row)

            if self.feature_count % 10 == 0:
                self.feature_file.flush()

        except Exception:
            logging.exception("Error processing CSI sample")

    def run(self, duration):
        self.setup_writer()

        self.receiver = SerialReceiver(
            port=self.port,
            baud=self.baud,
            callback=self.on_sample,
            record_path=self.raw_path,
        )

        logging.info("Starting activity collection")
        logging.info("Activity: %s", self.activity)
        logging.info("Session: %s", self.session_id)
        logging.info("Port: %s", self.port)
        logging.info("Duration: %s seconds", duration)

        print()
        print("========================================")
        print(" ACTIVITY DATA COLLECTION")
        print("========================================")
        print(f"Activity : {self.activity}")
        print(f"Session  : {self.session_id}")
        print(f"Port     : {self.port}")
        print(f"Duration : {duration} seconds")
        print()
        print("Starting in 3 seconds...")
        time.sleep(3)

        try:
            self.receiver.start()

            print("COLLECTING...")
            print("Perform the selected activity now.")
            print()

            start_time = time.time()

            while time.time() - start_time < duration:
                elapsed = int(time.time() - start_time)
                remaining = max(0, duration - elapsed)

                print(
                    f"\rElapsed: {elapsed:3d}s | "
                    f"Remaining: {remaining:3d}s | "
                    f"CSI samples: {self.sample_count:5d} | "
                    f"features: {self.feature_count:5d}",
                    end="",
                    flush=True,
                )

                time.sleep(1)

            print()
            print()
            print("Collection complete.")

        except KeyboardInterrupt:
            print()
            print("Collection stopped by user.")

        finally:
            if self.receiver is not None:
                self.receiver.stop()
                self.receiver.join(timeout=3)

            if self.feature_file is not None:
                self.feature_file.flush()
                self.feature_file.close()

        print()
        print("Saved:")
        print(f"  Raw CSI      : {self.raw_path}")
        print(f"  Feature data : {self.dataset_path}")
        print()
        print(f"CSI samples   : {self.sample_count}")
        print(f"Feature rows  : {self.feature_count}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Collect CSI activity training data."
    )

    parser.add_argument(
        "--activity",
        required=True,
        choices=[
    "empty",
    "standing",
    "sitting",
    "moving",
    "fall",
],
        help="Activity label for this recording.",
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
        "--duration",
        type=int,
        default=30,
        help="Recording duration in seconds.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    collector = ActivityCollector(
        activity=args.activity,
        port=args.port,
        baud=args.baud,
        config=config,
    )

    collector.run(args.duration)


if __name__ == "__main__":
    main()
