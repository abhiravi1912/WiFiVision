# ESP32 Wi-Fi CSI Human Activity Sensing

An experimental, device-free human-activity sensing system built with two classic ESP32 DevKit V1 boards and Python. It captures Wi-Fi Channel State Information (CSI) from a controlled 2.4 GHz link and classifies an environment as **empty**, **standing**, **sitting**, or **moving**.

> This is a research prototype, not a safety, medical, security, or occupancy-control product. CSI models are sensitive to the room, board placement, people, furniture, and Wi-Fi conditions. Retrain the system in the environment where it will be used.

## Features

- Dedicated ESP32 access point and controlled UDP traffic stream
- CSI capture on a second ESP32, streamed to the host as JSON over USB serial
- CSI validation, amplitude preprocessing, and 180 base features
- Four-class temporal Random Forest inference: `empty`, `standing`, `sitting`, and `moving`
- Labeled data collection, training, diagnostics, a legacy PyQt6 visualizer, and synthetic-data tests

## Architecture

```text
ESP32-A: AP + UDP echo                         ESP32-B: station + CSI capture
  CSI_SENSOR (fixed 2.4 GHz channel)  <----->  sends 60 UDP probes/second
                                                       |
                                              USB serial (921600 baud)
                                                       |
                                                       v
                                             Python preprocessing
                                             180 CSI feature values
                                                       |
                                                       v
                                      20-row temporal feature window (1080 values)
                                                       |
                                                       v
                              EMPTY / STANDING / SITTING / MOVING prediction
```

ESP32-A returns each probe as a unicast HT20 data frame. ESP32-B receives that frame, extracts CSI through the ESP-IDF CSI API, and sends the raw packet to the host. Model training and inference run on the computer, not on the ESP32.

## Hardware

| Item | Qty | Purpose |
| --- | ---: | --- |
| Classic ESP32 DevKit V1 / ESP-WROOM-32 | 2 | Access point and CSI receiver |
| USB data cable | 2 | Power, programming, and receiver serial data |
| Host computer | 1 | Data collection, training, and inference |

This project targets the **classic ESP32**. CSI support and APIs differ on ESP32-S2, C3, S3, and C6 boards; those boards are not drop-in replacements.

## Repository layout

```text
firmware/
  esp32_ap/                       ESP32-A access point and UDP echo firmware
  esp32_receiver/                 ESP32-B CSI receiver and serial streamer
python/
  activity_collector.py           Record labeled CSI sessions
  activity_train_temporal_v2.py   Train the current temporal activity model
  activity_inference.py           Run live four-class inference
  serial_receiver.py              Read, validate, and optionally record CSI packets
  preprocessing.py                CSI validation, subcarrier selection, filtering
  feature_extraction.py           Build the 180 base features
  main.py                         Legacy position/presence GUI workflow
data/activity/
  raw/                            Recorded raw CSI sessions
  processed/                      Feature dataset
  models/                         Trained activity, presence, and movement models
tests/                            Synthetic-data and GUI smoke tests
```

## Quick start

### 1. Flash the firmware

Install the Arduino IDE or Arduino CLI and the **ESP32 by Espressif Systems** board package. The project targets `DOIT ESP32 DEVKIT V1` (`esp32:esp32:esp32doit-devkit-v1`) and needs no extra Arduino libraries.

Before flashing, make the following values match in both sketches:

- `WIFI_SSID` and `WIFI_PASSWORD`
- `WIFI_CHANNEL`
- `SERIAL_BAUD` (default: `921600`)

```bash
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_ap
arduino-cli upload -p <AP_PORT> --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_ap

arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_receiver
arduino-cli upload -p <RECEIVER_PORT> --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_receiver
```

See [AP firmware notes](firmware/esp32_ap/README.md) and [receiver firmware notes](firmware/esp32_receiver/README.md) for details.

### 2. Set up Python

Python 3.10+ is recommended. Run the following from the repository root.

```bash
python3 -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, use `py` or `python` in place of `python3` when appropriate.

### 3. Verify CSI reception

Start ESP32-A, then ESP32-B. Close any serial monitor connected to ESP32-B, then list candidate ports and start the receiver:

```bash
python python/serial_receiver.py --list
python python/serial_receiver.py --port <RECEIVER_PORT> --baud 921600
```

Typical port names are `COM13` on Windows, `/dev/ttyUSB0` or `/dev/ttyACM0` on Linux, and `/dev/cu.usbserial-*` on macOS.

### 4. Run live activity inference

The repository includes a trained Temporal V2 activity model and its manifest in `data/activity/models/`.

```bash
python python/activity_inference.py --port <RECEIVER_PORT> --baud 921600
```

The first prediction is produced after 20 base-feature rows. The terminal then prints the predicted class and its probabilities.

## Collect and train your own data

Collect data after placing the boards in their final positions. Record each activity in multiple independent sessions, balance the classes, and keep one activity per session.

```bash
python python/activity_collector.py --activity empty    --port <RECEIVER_PORT> --duration 60
python python/activity_collector.py --activity standing --port <RECEIVER_PORT> --duration 60
python python/activity_collector.py --activity sitting  --port <RECEIVER_PORT> --duration 60
python python/activity_collector.py --activity moving   --port <RECEIVER_PORT> --duration 60

python python/activity_train_temporal_v2.py
```

The collector appends feature rows to `data/activity/processed/activity_dataset.csv` and saves raw packets in `data/activity/raw/`. Training uses **session-based** train/validation/test splits, so rows from one recording cannot be spread across different partitions. Generated models, manifests, and reports are saved in `data/activity/models/`.

The collector accepts a `fall` label, but the current Temporal V2 training and inference pipeline supports only `empty`, `standing`, `sitting`, and `moving`. Add fall to the relevant model class definitions before claiming fall detection.

## Included experiment results

The checked-in Temporal V2 training report records a session-held-out test accuracy of **84.4%** and macro F1 of **85.3%** across 5,115 temporal windows from 119 recorded sessions. It shows stronger separation for empty and moving than for sitting versus standing.

These results describe this dataset and setup only; they are not a guarantee for a new room, new people, or changed device placement.

The model uses 20 overlapping base-feature rows (about 2.7 seconds at the observed feature cadence). For each of 180 base features it computes mean, standard deviation, minimum, maximum, range, and last-minus-first delta, producing 1,080 classifier inputs.

## Configuration

[`config.yaml`](config.yaml) holds serial, CSI preprocessing, feature, GUI, and legacy-workflow settings. Keep these values aligned with firmware:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `baud_rate` | `921600` | Must match receiver `SERIAL_BAUD` |
| `sample_rate` | `60 Hz` | Must match receiver `TRAFFIC_RATE_HZ` |
| `features.window_size` | `64` | CSI samples used for one base feature vector |
| `features.stride` | `8` | CSI samples between base feature vectors |

`main.py` and `config.yaml` also retain the original presence/movement/approximate-position workflow, which expects models in `data/models/`. For the checked-in four-class activity model, use `activity_inference.py`.

## Tests

The tests use synthetic packets or an offscreen Qt window; they do not prove real-world sensing accuracy.

```bash
python tests/make_synthetic_dataset.py
python tests/test_real_packet_regression.py
python tests/inference_model_test.py
python tests/gui_smoke_test.py
```

## Limitations

- CSI changes when furniture, antenna orientation, people, or radio interference change; retraining may be required.
- The current model is not designed for identity recognition, health monitoring, safety decisions, or reliable fall detection.
- This single-link experiment does not estimate precise 2D or 3D human position.
- The ESP32 Wi-Fi stack and USB serial link can drop packets. The pipeline validates malformed input but cannot recover uncaptured information.

## License

Licensed under the [MIT License](LICENSE).

