import sys
import serial
import time
import joblib
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from.features import FeatureExtractor

FEATURES = [
    "rssi_mean", "rssi_last", "temp_energy", "agg_mean", "agg_min",
    "sub_mean_0", "sub_mean_1", "sub_mean_3", "agg_max", "sub_mean_4"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM15")
    ap.add_argument("--baud", type=int, default=921600)
    args = ap.parse_args()

    model = joblib.load(
        ROOT / "data" / "activity" / "models" / "presence_model_knn_top10.pkl"
    )

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    ext = FeatureExtractor()
    rows = []
    start = time.time()

    print("=" * 50)
    print("TOP-10 LIVE PRESENCE TEST")
    print("=" * 50)
    print("Room: EMPTY")
    print("Keep the room empty for 30 seconds.")

    try:
        while time.time() - start < 30:
            line = ser.readline().decode("utf-8", "ignore").strip()

            if not line.startswith("#CSI"):
                continue

            try:
                sample = ext.parse_line(line)
                if sample is None:
                    continue
                features = ext.extract(sample)
            except Exception:
                continue

            if features is None:
                continue

            if hasattr(features, "to_dict"):
                row = features.to_dict()
            elif isinstance(features, dict):
                row = features
            else:
                row = dict(zip(ext.feature_names(), features))

            rows.append(row)

            if len(rows) % 20 == 0:
                x = pd.DataFrame([row])[FEATURES]
                pred = int(model.predict(x)[0])
                prob = model.predict_proba(x)[0]

                print(
                    f"[ONLINE] {'PRESENT' if pred else 'EMPTY'} | "
                    f"EMPTY={prob[0]:.2f} | PRESENT={prob[1]:.2f}"
                )

    finally:
        ser.close()

    print("\nSamples:", len(rows))

    if rows:
        x = pd.DataFrame(rows)[FEATURES]
        pred = model.predict(x)
        prob = model.predict_proba(x)

        print(
            f"Final counts: EMPTY={(pred == 0).sum()} "
            f"PRESENT={(pred == 1).sum()}"
        )

        print(
            f"Mean probabilities: EMPTY={prob[:,0].mean():.4f} "
            f"PRESENT={prob[:,1].mean():.4f}"
        )

if __name__ == "__main__":
    main()
