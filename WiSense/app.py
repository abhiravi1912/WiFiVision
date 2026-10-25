import os
import socket
from flask import Flask, render_template, jsonify, send_from_directory, make_response
from services.csi_service import get_csi_samples
from services.prediction_service import get_prediction, get_status

app = Flask(__name__)


def get_local_ip():
    """Detect local LAN IP for seamless mobile device pairing."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── PWA & Static App Routes ───────────────────────────────────────────────────

@app.route("/manifest.webmanifest")
@app.route("/manifest.json")
def serve_manifest():
    response = make_response(send_from_directory(os.path.join(app.root_path, "static"), "manifest.webmanifest"))
    response.headers["Content-Type"] = "application/manifest+json"
    return response


@app.route("/sw.js")
def serve_sw():
    response = make_response(send_from_directory(os.path.join(app.root_path, "static", "js"), "sw.js"))
    response.headers["Content-Type"] = "application/javascript"
    response.headers["Service-Worker-Allowed"] = "/"
    return response


# ── Pages ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ── API ────────────────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """System / hardware connection status."""
    data = get_status()
    return jsonify(data)


@app.route("/api/csi")
def api_csi():
    """CSI waveform samples."""
    data = get_csi_samples(n_samples=64)
    return jsonify(data)


@app.route("/api/prediction")
def api_prediction():
    """Current activity prediction."""
    data = get_prediction()
    return jsonify(data)


@app.route("/api/lan-info")
def api_lan_info():
    """Returns local network address for instant mobile device connection."""
    ip = get_local_ip()
    port = 5000
    return jsonify({
        "ip": ip,
        "port": port,
        "local_url": f"http://localhost:{port}",
        "mobile_url": f"http://{ip}:{port}",
        "dashboard_url": f"http://{ip}:{port}/dashboard"
    })


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host_ip = get_local_ip()
    print("=" * 60)
    print("  WiFi-Vision — Mobile & Cross-Platform Sensing WebApp")
    print("=" * 60)
    print(f"  * Desktop browser:  http://localhost:5000")
    print(f"  * Mobile iOS/Android: http://{host_ip}:5000")
    print(f"  * Mobile Dashboard:   http://{host_ip}:5000/dashboard")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)

