# WiFi-Vision: ESP32 Wi-Fi CSI Activity Sensing

A privacy-first, device-free human-activity sensing prototype that combines an ESP32 Wi-Fi CSI acquisition pipeline with a responsive Flask/PWA dashboard.

The system creates a controlled Wi-Fi link between two classic ESP32 boards, extracts Channel State Information (CSI), classifies the environment as **Empty Room**, **Standing**, **Sitting**, or **Moving**, and displays the live result in a browser on desktop or mobile.

> Research prototype only. It is not designed for medical, safety, surveillance, emergency, or fall-detection use. Accuracy depends strongly on the room, device placement, people, furniture, and radio environment.

## How the two projects fit together

| Component | Location | Responsibility |
| --- | --- | --- |
| CSI sensing and ML | `ESP32-WiFi-CSI-Human-Sensing-main/` | Firmware, serial capture, preprocessing, feature extraction, training, and live four-class inference |
| WiFi-Vision dashboard | `WiFi-Vision_2.0/WiSense/` | Flask API, browser dashboard, PWA, mobile pairing, and a bridge to the live inference process |

```text
ESP32-A: AP + UDP echo  <-->  ESP32-B: CSI receiver
                                      |
                                   USB serial
                                      |
                                      v
                         activity_inference.py
                   CSI -> 180 features -> temporal model
                                      |
                             [ONLINE] prediction
                                      |
                                      v
                 WiSense live_bridge.py (one background process)
                                      |
                                      v
                     Flask API + desktop/mobile PWA dashboard
```

The dashboard does not duplicate CSI parsing or ML inference. Its bridge starts and reads the existing `activity_inference.py` process, which keeps the serial port owned by one process.

## Features

- Two-classic-ESP32 CSI sensing link using controlled UDP probe/echo traffic
- Four-class Temporal V2 classifier: empty, standing, sitting, moving
- 20-row temporal inference window with 1,080 model features
- Flask web dashboard with prediction, health status, and normalized live signal indicator
- Installable PWA with responsive mobile layout and QR/LAN pairing
- Session-based data collection and model evaluation workflow
- No camera, image, or wearable is required

## Hardware

| Item | Qty | Purpose |
| --- | ---: | --- |
| ESP32 DevKit V1 / ESP-WROOM-32 (classic ESP32) | 2 | Access point and CSI receiver |
| USB data cable | 2 | Programming, power, and serial data |
| Host computer | 1 | Python inference and web server |
| Phone or browser device | Optional | Dashboard viewing on the local network |

The firmware targets classic ESP32 CSI APIs. ESP32-S2, C3, S3, and C6 boards are not drop-in replacements.

## Project layout

```text
ESP32-WiFi-CSI-Human-Sensing-main/
  firmware/
    esp32_ap/                 ESP32-A access point + UDP echo
    esp32_receiver/           ESP32-B CSI receiver + JSON serial output
  python/
    activity_inference.py     Current live four-class model entry point
    activity_collector.py     Record labeled CSI sessions
    activity_train_temporal_v2.py
  data/activity/models/       Temporal model, manifests, reports

WiFi-Vision_2.0/
  WiSense/
    app.py                    Flask application and API
    services/live_bridge.py   Starts and reads the CSI inference process
    templates/                Landing page and dashboard
    static/                   CSS, JavaScript, PWA assets
```

## Setup

### 1. Flash the ESP32 firmware

Install the **ESP32 by Espressif Systems** Arduino board package. Configure the same Wi-Fi credentials, channel, and serial baud rate in both firmware sketches, then flash them.

```bash
arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_ap
arduino-cli upload -p <AP_PORT> --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_ap

arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_receiver
arduino-cli upload -p <RECEIVER_PORT> --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_receiver
```

The receiver uses `921600` baud by default. Match `SERIAL_BAUD` in firmware with the host-side configuration.

### 2. Install the CSI project dependencies

From the CSI project root (the directory that contains `python/activity_inference.py`):

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Install the dashboard dependencies

From `WiFi-Vision_2.0/WiSense`:

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS/Linux
# source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure the dashboard bridge

The dashboard needs the exact path to the CSI project root and the receiver serial port. In Windows PowerShell:

```powershell
$env:WIFI_VISION_CSI_ROOT = "C:\path\to\ESP32-WiFi-CSI-Human-Sensing-main\ESP32-WiFi-CSI-Human-Sensing-main"
$env:WIFI_VISION_CSI_PORT = "COM13"
$env:WIFI_VISION_CSI_BAUD = "921600"
```

Set `WIFI_VISION_CSI_PYTHON` only if the CSI project's Python executable is not at `.venv\Scripts\python.exe`.

### 5. Launch WiFi-Vision

With both boards powered and the receiver serial port free:

```powershell
cd C:\path\to\WiFi-Vision_2.0\WiSense
.\venv\Scripts\python.exe app.py
```

Open:

- Desktop: `http://localhost:5000/dashboard`
- Mobile device on the same LAN: `http://<HOST_LAN_IP>:5000/dashboard`

The dashboard starts one background inference process on demand. Do not also start `activity_inference.py` separately, because two processes cannot safely use the same serial port.

## API

| Route | Purpose |
| --- | --- |
| `GET /` | Product landing page |
| `GET /dashboard` | Live dashboard |
| `GET /api/status` | Bridge and inference-process health |
| `GET /api/prediction` | Latest activity prediction |
| `GET /api/csi` | Latest normalized bridge signal samples |
| `GET /api/lan-info` | Desktop and LAN URLs for mobile pairing |

## Data collection and training

For a new environment, record several independent sessions per label and retrain the model:

```bash
python python/activity_collector.py --activity empty    --port <RECEIVER_PORT> --duration 60
python python/activity_collector.py --activity standing --port <RECEIVER_PORT> --duration 60
python python/activity_collector.py --activity sitting  --port <RECEIVER_PORT> --duration 60
python python/activity_collector.py --activity moving   --port <RECEIVER_PORT> --duration 60

python python/activity_train_temporal_v2.py
```

The model is evaluated using session-based splits, which prevents rows from the same recording appearing in both training and test sets.

## Reported experiment result

The checked-in Temporal V2 report records **84.4% test accuracy** and **85.3% macro F1** across 5,115 held-out temporal windows from 119 sessions. The recorded experiment separates empty and moving more reliably than sitting and standing. Treat these as dataset-specific results, not deployment guarantees.

## Limitations

- Move the boards, change furniture, alter antenna orientation, or add RF interference, and the model may require retraining.
- The dashboard's signal graph reflects normalized diagnostic signal data from the bridge; it is not a raw CSI-subcarrier visualization.
- The current single-link setup does not estimate precise 2D or 3D position.
- The supported live model has four activity classes. Although the collector accepts a `fall` label, fall detection is not implemented in the current training/inference model.
- Keep the dashboard on a trusted local network; the included Flask development server is not a hardened production deployment.

## License

The CSI sensing repository includes an MIT License. Confirm the license for all WiFi-Vision assets before publishing a combined repository.

