# WiFi-Vision: Privacy-Preserving Indoor Human Activity Recognition Using Wi-Fi CSI

---

![Platform: ESP32](https://img.shields.io/badge/Platform-ESP32-E7352C?style=flat-square&labelColor=334155&logo=espressif&logoColor=white) ![Platform: Windows](https://img.shields.io/badge/Platform-Windows-0078D4?style=flat-square&labelColor=334155&logo=windows&logoColor=white) ![Platform: Linux](https://img.shields.io/badge/Platform-Linux-FCC624?style=flat-square&labelColor=334155&logo=linux&logoColor=black) ![Language: Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&labelColor=334155&logo=python&logoColor=white) ![Firmware: Arduino-ESP32](https://img.shields.io/badge/Firmware-Arduino--ESP32-00979D?style=flat-square&labelColor=334155&logo=arduino&logoColor=white) ![CSI API: ESP-IDF](https://img.shields.io/badge/CSI%20API-ESP--IDF-000000?style=flat-square&labelColor=334155&logo=espressif&logoColor=white)

![Sensing: Wi-Fi CSI](https://img.shields.io/badge/Sensing-Wi--Fi%20CSI-7C3AED?style=flat-square&labelColor=7E22CE) ![Hardware: ESP32 × 2](https://img.shields.io/badge/Hardware-ESP32%20%C3%97%202-E7352C?style=flat-square&labelColor=9A3412&logo=espressif&logoColor=white) ![Machine Learning: scikit-learn](https://img.shields.io/badge/Machine%20Learning-scikit--learn-F7931E?style=flat-square&labelColor=9A3412&logo=scikitlearn&logoColor=white) ![Models: KNN + Random Forest](https://img.shields.io/badge/Models-KNN%20%2B%20Random%20Forest-0F766E?style=flat-square&labelColor=115E59)

![Interface: Flask](https://img.shields.io/badge/Interface-Flask-000000?style=flat-square&labelColor=1E3A8A&logo=flask&logoColor=white) ![Web App: PWA](https://img.shields.io/badge/Web%20App-PWA-5A0FC8?style=flat-square&labelColor=1E3A8A&logo=pwa&logoColor=white) ![Project: WiFi-Vision](https://img.shields.io/badge/Project-WiFi--Vision-0369A1?style=flat-square&labelColor=1E3A8A) ![Status: Experimental](https://img.shields.io/badge/Status-Experimental-F59E0B?style=flat-square&labelColor=92400E)

WiFi-Vision combines an ESP32 Wi-Fi Channel State Information (CSI) sensing pipeline with a Flask Progressive Web App (PWA). Two classic ESP32 boards form a controlled wireless link; a companion Python project extracts CSI-derived features and predicts indoor activity; WiSense presents that state in a browser dashboard.

> **Research prototype.** The dashboard-to-inference bridge is implemented, but the physical sensing model is experimental and environment-specific. This is not a medical, safety, security, emergency, occupancy-control, or fall-detection product.

## Table of Contents

- [Executive Summary](#executive-summary)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Algorithmic Pipeline](#algorithmic-pipeline)
- [Hardware and Experimental Setup](#hardware-and-experimental-setup)
- [Software Stack](#software-stack)
- [Repository Structure](#repository-structure)
- [CSI Data and Dataset Collection](#csi-data-and-dataset-collection)
- [Machine-Learning Methodology](#machine-learning-methodology)
- [Dashboard and User Interface](#dashboard-and-user-interface)
- [Quick Start Guide](#quick-start-guide)
- [Configuration and Communication](#configuration-and-communication)
- [API Reference](#api-reference)
- [Testing and Evaluation](#testing-and-evaluation)
- [Troubleshooting](#troubleshooting)
- [Limitations and Constraints](#limitations-and-constraints)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License and Acknowledgments](#license-and-acknowledgments)

---

## Executive Summary

WiFi-Vision investigates device-free indoor activity recognition without cameras or wearables. A person's body changes Wi-Fi multipath propagation as it enters, sits, stands, or moves through the sensing area. Channel State Information captures those per-packet channel changes as a structured signal that can be processed into features for classification.

The implemented system combines embedded wireless communication, serial data acquisition, signal preprocessing, a trained four-class temporal model, and a mobile-friendly dashboard. It is intended for research, prototyping, and studying privacy-preserving sensing—not for a claim that every intended capability has been validated in real deployments.

## System Architecture

### Current integrated configuration

The checked-in firmware uses ESP32-A as a dedicated soft access point; an external router is not required for the CSI link. ESP32-B sends UDP probes, receives AP echoes, captures CSI through ESP-IDF Wi-Fi APIs exposed by Arduino-ESP32, and streams JSON Lines over USB serial.

```mermaid
flowchart LR
  A[ESP32-A<br/>Soft AP + UDP Echo] <-->|2.4 GHz probe / HT20 echo| B[ESP32-B<br/>Station + CSI callback]
  B -->|USB serial<br/>JSON Lines, 921600 baud| C[Python CSI pipeline]
  C --> D[Validation + preprocessing]
  D --> E[180 base features]
  E --> F[Temporal V2 activity inference]
  F -->|stdout prediction + probabilities| G[WiSense live_bridge.py]
  G --> H[Flask JSON API]
  H --> I[Desktop / mobile PWA dashboard]
  I -->|HTTP polling| H
```

WiSense deliberately does not open the serial port. Its bridge launches one `activity_inference.py` process and parses the process output, avoiding conflicts from two programs trying to read ESP32-B.

### Scope of the web integration

The live class-prediction bridge is implemented. The dashboard's `/api/csi` route currently exposes normalized `rssi_mean` diagnostic values from the bridge; it is **not** a raw CSI-vector or per-subcarrier stream. Flask runs with its development server, not a production deployment.

## Key Features

| Area | Capability | Status |
| --- | --- | --- |
| Wireless sensing | ESP32 soft AP, controlled UDP echo, station-side CSI callback | Implemented |
| Serial processing | JSON parsing, packet validation, reconnection, optional raw CSV recording | Implemented |
| Preprocessing | Valid-subcarrier selection, Hampel filtering, EMA smoothing, optional low-pass filter | Implemented/configurable |
| ML | Four-class temporal classification: empty, standing, sitting, moving | Implemented in companion CSI project |
| ML output | Dashboard confidence when inference emits probabilities | Implemented when model supports it |
| Dashboard | Flask routes, live status/prediction polling, responsive PWA, LAN/QR pairing | Implemented |
| CSI visualization | Normalized diagnostic RSSI signal panel | Implemented |
| Raw CSI dashboard streaming | Per-subcarrier CSI buffers and a true CSI spectrogram | Not implemented |
| Hand-movement recognition | Gesture/hand classification | Not implemented |
| Fall detection | Four-class model support | Not implemented |

## Algorithmic Pipeline

### 5.1 CSI Acquisition

ESP32-A exposes the `CSI_SENSOR` AP and listens on UDP port `8765`. ESP32-B joins it, sends probes at the configured rate (default: 60 Hz), and enables `esp_wifi_set_csi_*` callbacks. Each packet includes radio metadata and a packed signed-int8 I/Q CSI buffer.

The receiver serializes one JSON object per line with fields including `timestamp`, `seq`, `rssi`, `channel`, `cwb`, `sig_mode`, `rx_state`, `first_word_invalid`, `csi_len`, and `csi`. In the intended HT20 layout, 64 CSI slots occupy 128 bytes.

### 5.2 Data Parsing and Preprocessing

The parser verifies required JSON fields, I/Q-pair parity, declared CSI length, and signed-int8 bounds. It represents the packed data as:

```text
H[k] = I[k] + jQ[k]
A[k] = |H[k]| = sqrt(I[k]^2 + Q[k]^2)
```

The stateful preprocessor can reject receiver errors, trim the documented invalid four-byte prefix, reject non-20 MHz packets, optionally require HT frames, select usable subcarriers `0..26` and `32..58`, apply a Hampel filter, smooth with EMA or a moving average, and optionally apply a Butterworth low-pass filter. Raw phase support exists but phase features are disabled by default because packet phase offsets need more calibration.

### 5.3 Feature Extraction

The current feature extractor uses a CSI amplitude window (default: 64 samples; stride: 8) to form 180 base features:

- Per-subcarrier mean, standard deviation, and mean absolute successive difference
- Aggregate amplitude mean, standard deviation, range, minimum, and maximum
- Temporal mean-curve variability, energy, and linear slope
- Four FFT-band energies
- RSSI mean, standard deviation, and latest value
- Relative movement statistics

### 5.4 Activity Recognition

`activity_inference.py` loads the Temporal V2 artifact from `data/activity/models/`. It buffers 20 consecutive 180-feature rows and computes mean, standard deviation, minimum, maximum, range, and last-minus-first delta for every feature:

```text
180 base features × 6 temporal statistics = 1,080 model inputs
```

The trainer evaluates distance-weighted KNN and balanced Random Forest candidates on validation sessions, chooses the highest macro-F1 candidate, refits on train plus validation windows, and evaluates on held-out sessions. The checked-in V2 manifest declares a Random Forest model with labels `empty`, `standing`, `sitting`, and `moving`.

### 5.5 Output and Visualization

The bridge parses the inference process's `[ONLINE]` lines into activity, presence, confidence, timestamp, and process-health state. Flask serves those snapshots to the browser. The dashboard polls status/predictions every two seconds and signal data at about five hertz.

## Hardware and Experimental Setup

### Required equipment

| Component | Purpose |
| --- | --- |
| ESP32-WROOM-32 / classic ESP32 board #1 | Soft AP and UDP echo sender |
| ESP32-WROOM-32 / classic ESP32 board #2 | Station, CSI capture, USB serial streamer |
| Laptop or desktop | Processing, inference, and dashboard host |
| Two USB data cables | Flashing, power, and serial communication |
| 2.4 GHz sensing area | Stable AP-to-receiver radio link |

### Optional equipment

| Component | Purpose |
| --- | --- |
| Phone/tablet on the host LAN | View or install the PWA |
| External router | Host-to-phone LAN access only; not needed by the sensing link |
| CP210x/CH340 driver | Only if the board's USB-UART interface needs it |

Place the boards on opposite sides of the experimental area and keep their orientation and furniture layout fixed while collecting/training data. Start ESP32-A before ESP32-B.

## Software Stack

| Layer | Verified technology | Role |
| --- | --- | --- |
| Firmware | Arduino-ESP32 and ESP-IDF Wi-Fi CSI APIs | ESP32 sketches; no standalone `idf.py` project |
| CSI processing | Python, pyserial, NumPy, SciPy, pandas | Serial I/O, parsing, filtering, features, datasets |
| ML | scikit-learn and joblib | KNN/Random Forest training and model artifacts |
| Legacy visualization | PyQt6 and matplotlib | CSI project's desktop/debug tooling |
| Backend | Flask | HTML and JSON API server |
| Frontend | Jinja2, HTML, CSS, vanilla JavaScript | Dashboard and landing page |
| PWA | Service worker and Web App Manifest | Installable app shell |
| Browser libraries | Three.js and qrcode.js | Landing visual and mobile pairing |

## Repository Structure

```text
WiFi-Vision_2.0/
└── WiSense/
    ├── app.py                         # Flask pages, API, LAN information
    ├── requirements.txt               # Flask and NumPy
    ├── LIVE_INTEGRATION_POWERSHELL.txt
    ├── services/
    │   ├── live_bridge.py             # Starts/parses external inference process
    │   ├── csi_service.py             # Normalized diagnostic signal endpoint
    │   └── prediction_service.py      # Prediction and health endpoints
    ├── templates/                     # Landing and dashboard HTML
    └── static/                        # CSS, JS, PWA manifest, icons

ESP32-WiFi-CSI-Human-Sensing-main/      # Required companion checkout
├── firmware/esp32_ap/                  # Soft AP + UDP echo
├── firmware/esp32_receiver/            # CSI capture + serial JSON
├── python/
│   ├── csi_parser.py                   # I/Q JSON parser
│   ├── preprocessing.py                # Streaming signal processing
│   ├── feature_extraction.py           # 180 base features
│   ├── activity_collector.py           # Labeled-session collection
│   ├── activity_train_temporal_v2.py   # Four-class trainer
│   └── activity_inference.py           # Live inference bridge target
├── data/activity/                      # Data, model, manifests, reports
├── config.yaml                         # Processing configuration
└── tests/                              # Companion-project test scripts
```

Generated virtual environments and dependency caches are omitted.

## CSI Data and Dataset Collection

ESP32-B emits newline-delimited JSON. The companion project can record raw sessions to `data/activity/raw/`, append feature rows to `data/activity/processed/activity_dataset.csv`, and write trained artifacts/reports to `data/activity/models/`.

Each processed row contains `sample_id`, `session_id`, `timestamp`, `activity`, and numeric feature fields. Collect several independent sessions per supported label:

```bash
python python/activity_collector.py --activity empty    --port <RECEIVER_PORT> --duration 60
python python/activity_collector.py --activity standing --port <RECEIVER_PORT> --duration 60
python python/activity_collector.py --activity sitting  --port <RECEIVER_PORT> --duration 60
python python/activity_collector.py --activity moving   --port <RECEIVER_PORT> --duration 60
```

The trainer splits complete sessions before temporal windows are created, avoiding leakage from adjacent correlated rows. New recordings across varied but controlled conditions are essential for meaningful evaluation. Although the collector accepts `fall`, the current V2 model does not train or infer it.

## Machine-Learning Methodology

| Aspect | Implemented method |
| --- | --- |
| Input | 20 consecutive 180-feature CSI vectors |
| Representation | Six temporal statistics per feature; 1,080 values |
| Classes | empty, standing, sitting, moving |
| Candidates | Distance-weighted KNN and balanced Random Forest |
| Selection | Validation macro F1, then accuracy tie-breaker |
| Split | Session-level train/validation/test before temporal aggregation |
| Artifacts | Joblib model, JSON manifest, JSON training report |
| Live output | `predict` and optional `predict_proba` |

No general accuracy figure is claimed in this README. Consult the saved report associated with a specific data-collection and training run.

## Dashboard and User Interface

- `/` provides the product landing page.
- `/dashboard` provides presence/activity, confidence, hardware/processing status, and a signal canvas.
- `/api/lan-info` returns desktop and LAN URLs for the QR pairing flow.
- The included manifest and service worker enable installation on supported browsers.

For its first ten seconds, the bridge reports a `STARTUP` state with `Empty Room`. That is a startup placeholder, not a sensor measurement. Afterward, the dashboard reflects the most recent parsed inference line. A running bridge process does not by itself confirm CSI reception; check `stale`, `updated_at`, and errors returned by the API.

## Quick Start Guide

### Prerequisites

Two classic ESP32 boards, Arduino IDE or Arduino CLI with **ESP32 by Espressif Systems**, Python 3.10+ for the CSI project, Python/Flask for WiSense, and a free receiver serial port are required.

### Step 1: Clone the repositories

No canonical remote URL is declared in this checkout. Clone both projects from their actual hosting locations:

```bash
git clone <wifi-vision-repository-url>
git clone <esp32-csi-repository-url>
```

### Step 2: Configure and flash firmware

In the CSI project, make Wi-Fi credentials/channel and serial settings match at the top of both sketches. Build and flash them with Arduino CLI:

```bash
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_ap
arduino-cli upload -p <AP_PORT> --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_ap
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_receiver
arduino-cli upload -p <RECEIVER_PORT> --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_receiver
```

The sketches use ESP-IDF CSI APIs through Arduino-ESP32; `idf.py` is not used.

### Step 3: Install Python dependencies

CSI project:

```bash
cd <esp32-csi-root>
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

WiSense:

```bash
cd <wifi-vision-root>/WiSense
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Verify the receiver

Power ESP32-A, then ESP32-B. Close Arduino Serial Monitor, then list and test receiver ports:

```bash
cd <esp32-csi-root>
python python/serial_receiver.py --list
python python/serial_receiver.py --port <RECEIVER_PORT> --baud 921600
```

### Step 5: Start backend and frontend

WiSense serves both through Flask. Set bridge variables in the same terminal:

```powershell
$env:WIFI_VISION_CSI_ROOT = "C:\path\to\ESP32-WiFi-CSI-Human-Sensing-main"
$env:WIFI_VISION_CSI_PORT = "COM13"
$env:WIFI_VISION_CSI_BAUD = "921600"
# Optional if CSI Python is elsewhere:
$env:WIFI_VISION_CSI_PYTHON = "C:\path\to\python.exe"

cd C:\path\to\WiFi-Vision_2.0\WiSense
.\venv\Scripts\python.exe app.py
```

On macOS/Linux, export the equivalent variables and run `python app.py`.

### Step 6: Verify the system

Open `http://localhost:5000/dashboard`, inspect `/api/status`, wait beyond the startup state, and confirm that `/api/prediction` has a recent non-stale update. These checks validate individual layers, not recognition accuracy in a new environment.

## Configuration and Communication

| Setting | Default | Purpose |
| --- | ---: | --- |
| Receiver `SERIAL_BAUD` | 921600 | ESP32-B serial speed |
| Receiver `TRAFFIC_RATE_HZ` | 60 | Probe rate |
| AP `UDP_ECHO_PORT` | 8765 | Echo service |
| `config.yaml: baud_rate` | 921600 | Host serial rate |
| `config.yaml: sample_rate` | 60 | Nominal processing rate |
| `WIFI_VISION_CSI_ROOT` | Auto-discovery attempt | CSI project containing inference script |
| `WIFI_VISION_CSI_PORT` | COM13 | ESP32-B serial port |
| `WIFI_VISION_CSI_BAUD` | 921600 | Bridge serial rate |
| `WIFI_VISION_CSI_PYTHON` | CSI venv Python, else `py` | Inference executable |

The receiver transfers JSON Lines over USB serial; browser-to-dashboard communication is HTTP at port `5000`.

## API Reference

| Method | Route | Response |
| --- | --- | --- |
| GET | `/` | Landing-page HTML |
| GET | `/dashboard` | Dashboard HTML |
| GET | `/manifest.webmanifest`, `/manifest.json` | PWA manifest |
| GET | `/sw.js` | Service worker |
| GET | `/api/status` | Bridge/process and derived sender/receiver/CSI/prediction status |
| GET | `/api/prediction` | Activity, presence, confidence, freshness, process state, optional error |
| GET | `/api/csi` | Normalized diagnostic samples, count, metric, optional range |
| GET | `/api/lan-info` | Local/LAN URLs and port |

A typical prediction object contains `mode`, `activity`, `presence`, `confidence`, `updated_at`, `stale`, and `process_alive`. Actual values depend on the running process; `mode` can be `STARTUP`, `LIVE`, or `LIVE_WAITING`.

## Testing and Evaluation

WiSense has no automated test suite in this checkout. The companion CSI project includes:

```bash
python tests/make_synthetic_dataset.py
python tests/test_real_packet_regression.py
python tests/inference_model_test.py
python tests/gui_smoke_test.py
```

These cover synthetic, parser/inference, and GUI paths; they are not a hardware accuracy benchmark. The temporal trainer writes accuracy, macro F1, a classification report, and a confusion matrix to its model report. Evaluate a particular model using the report generated from its own sessions.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Firmware upload fails | Verify board type, serial port, USB data cable, and CP210x/CH340 driver |
| AP not visible | Check matching SSID, password, and channel in both sketches |
| No CSI packets | Start AP first; close Serial Monitor; check ESP32-B port and 921600 baud |
| Malformed serial packets | Confirm receiver JSON mode and matching baud; inspect raw receiver output |
| Bridge exits | Verify root/Python environment variables, model files, dependencies, and port availability |
| Dashboard remains waiting | Check Flask terminal output and run `activity_inference.py` directly for its error |
| Signal panel is empty | Resolve CSI/inference health; WiSense does not fabricate live samples |
| Phone cannot open dashboard | Use the same LAN and allow local port 5000 through the firewall |
| Venv fails after moving directories | Recreate the virtual environment and reinstall dependencies |

## Limitations and Constraints

- CSI depends on room geometry, device placement, antennas, furniture, people, and RF interference.
- Cross-room generalization is not established by this repository.
- The model does not implement hand recognition, identity recognition, medical monitoring, reliable fall detection, precise localization, or multi-person tracking.
- The bridge parses standard output rather than using an in-process data interface.
- The dashboard signal endpoint is not raw CSI.
- Flask debug/development serving is not production hardened.

## Future Enhancements

- Replace stdout parsing with a typed local interface.
- Add a shared raw/processed CSI buffer for the dashboard.
- Add repeatable hardware integration tests and calibration guidance.
- Evaluate cross-room and cross-placement generalization.
- Improve model monitoring, dashboard analytics, and production deployment controls.
- Investigate zone-level and multi-person sensing only with appropriate data and validation.

## Contributing

1. Fork the repository and create a focused branch.
2. Keep firmware, application, and documentation changes scoped.
3. Run applicable checks and manually verify affected web/API behavior.
4. Do not commit credentials, recordings, virtual environments, or generated artifacts unless explicitly intended.
5. Open a pull request with verification steps and hardware assumptions.

## License and Acknowledgments

No license file is present in the WiSense checkout, so no license is implied for this web application. The companion ESP32 CSI project includes an MIT License.

WiFi-Vision uses Arduino-ESP32's Wi-Fi stack and ESP-IDF CSI APIs through the firmware source, plus Flask, NumPy, and the browser libraries listed above. Consult each dependency's license before redistribution.
