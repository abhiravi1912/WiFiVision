"""
prediction_service.py
---------------------
Live WiFi-CSI prediction service.

The frontend keeps its existing API contract:
  GET /api/prediction

The actual classification is provided by the existing project's
python/activity_inference.py process through live_bridge.py.

Four frontend classes:
  Empty Room
  Standing
  Sitting
  Moving

The ML model's "moving" class is displayed as "Moving".
"""

from .live_bridge import bridge

DEMO = False

ACTIVITY_CLASSES = ["Empty Room", "Standing", "Moving", "Sitting"]


def get_prediction() -> dict:
    return bridge.snapshot()


def get_status() -> dict:
    return bridge.status()
