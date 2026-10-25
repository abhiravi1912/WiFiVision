"""
csi_service.py
--------------
CSI endpoint for the dashboard.

The existing activity inference process owns the serial receiver, so this
service does not open COM13 a second time (which would conflict with the
inference process).

The waveform endpoint returns the latest normalized feature signal when the
bridge exposes one. At minimum, it reports the stream as LIVE without
fabricating CSI samples.
"""

from .live_bridge import bridge

DEMO = False


def get_csi_samples(n_samples: int = 64) -> dict:
    snap = bridge.snapshot()
    samples = snap.get("signal_samples", [])

    if n_samples and len(samples) > n_samples:
        samples = samples[-n_samples:]

    return {
        "mode": "LIVE" if snap.get("process_alive") else "LIVE_WAITING",
        "samples": samples,
        "n": len(samples),
        "metric": snap.get("signal_metric", "rssi_mean"),
        "min": snap.get("signal_min"),
        "max": snap.get("signal_max"),
    }
