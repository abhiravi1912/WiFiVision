/**
 * signal.js  —  WiFi-Vision
 * CSI waveform renderers for:
 *   1. Landing page dashboard preview (#previewSignalCanvas)
 *   2. Activity section mini-wave (#activityWave)
 *
 * Fetches live data from /api/csi for the preview.
 * Activity wave uses parameterized synthetic patterns.
 */

"use strict";

/* ── Shared color constants ──────────────────────────────────── */
const SIG_ACCENT = "#E39A3B";
const SIG_GREEN  = "#7FA68A";
const SIG_MUTED  = "#2A2D2B";
const SIG_BG     = "#171A18";

/* ══════════════════════════════════════════════════════════════
   1. Preview signal graph  (landing page)
═══════════════════════════════════════════════════════════════ */
(function initPreviewSignal() {
  const canvas = document.getElementById("previewSignalCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let animFrame;
  let samples = [];
  let fetchPending = false;

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = rect.width  * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
  }

  /* ── Fetch CSI from Flask ──────────────────────────────────── */
  async function fetchCSI() {
    if (fetchPending) return;
    fetchPending = true;
    try {
      const resp = await fetch("/api/csi");
      const data = await resp.json();
      samples = data.samples || [];

      // Update prediction display while we're at it
      fetchPrediction();
    } catch (_) {
      // Silent fail — demo still draws
    } finally {
      fetchPending = false;
    }
  }

  /* ── Fetch prediction ──────────────────────────────────────── */
  async function fetchPrediction() {
    try {
      const resp = await fetch("/api/prediction");
      const data = await resp.json();

      const presEl  = document.getElementById("previewPresence");
      const actEl   = document.getElementById("previewActivity");
      const confEl  = document.getElementById("previewConfidence");
      const zoneEl  = document.getElementById("previewZone");

      if (presEl)  presEl.textContent  = data.presence ? "Present" : "Empty";
      if (actEl)   actEl.textContent   = data.activity  || "—";
      if (confEl)  confEl.textContent  = data.confidence
        ? Math.round(data.confidence * 100) + "%"
        : "—";
      if (zoneEl)  zoneEl.textContent  = data.zone || "—";
    } catch (_) {}
  }

  /* ── Draw waveform ─────────────────────────────────────────── */
  function draw(t) {
    const W = canvas.getBoundingClientRect().width  || canvas.width;
    const H = canvas.getBoundingClientRect().height || canvas.height;

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = SIG_BG;
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = SIG_MUTED;
    ctx.lineWidth   = 0.5;
    ctx.globalAlpha = 0.4;
    for (let y = 0; y <= 4; y++) {
      const yy = (y / 4) * H;
      ctx.beginPath();
      ctx.moveTo(0, yy);
      ctx.lineTo(W, yy);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    if (!samples.length) {
      // Fallback synthetic
      samples = Array.from({ length: 64 }, (_, i) => {
        const x = (i / 64) * Math.PI * 4;
        return 0.5 + 0.3 * Math.sin(x + t * 0.8) + 0.1 * Math.sin(x * 2.1 + t) + (Math.random() - 0.5) * 0.08;
      });
    }

    // Signal path
    const n = samples.length;
    const stepX = W / (n - 1);

    // Shadow/glow
    ctx.shadowColor = SIG_ACCENT;
    ctx.shadowBlur  = 6;

    ctx.beginPath();
    ctx.strokeStyle = SIG_ACCENT;
    ctx.lineWidth   = 1.5;
    ctx.lineJoin    = "round";
    ctx.lineCap     = "round";

    for (let i = 0; i < n; i++) {
      const x = i * stepX;
      const y = H - samples[i] * H * 0.8 - H * 0.1;
      if (i === 0) ctx.moveTo(x, y);
      else         ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under curve
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 0.08;
    ctx.fillStyle = SIG_ACCENT;
    ctx.lineTo(W, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  let lastFetch = 0;

  function loop(t) {
    animFrame = requestAnimationFrame(loop);

    if (t - lastFetch > 200) {
      fetchCSI();
      lastFetch = t;
    }

    draw(t / 1000);
  }

  resizeCanvas();
  window.addEventListener("resize", resizeCanvas, { passive: true });
  requestAnimationFrame(loop);
})();

/* ══════════════════════════════════════════════════════════════
   2. Activity mini-wave  (activity selector section)
═══════════════════════════════════════════════════════════════ */
const ACTIVITY_WAVE_PARAMS = {
  Moving:      { freq: 1.4, amp: 0.32, noise: 0.12, color: SIG_ACCENT },
  Standing:     { freq: 0.6, amp: 0.10, noise: 0.04, color: SIG_GREEN  },
  Sitting:      { freq: 0.3, amp: 0.06, noise: 0.02, color: SIG_MUTED  },
  "Empty Room": { freq: 0.1, amp: 0.02, noise: 0.01, color: "#3A3D3B"  },
};

let _currentActivity = "Moving";
let _actAnimFrame;

function drawActivityWave(canvas, activity) {
  _currentActivity = activity;
}

(function initActivityWave() {
  const canvas = document.getElementById("activityWave");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const N   = 64;

  function resize() {
    const dpr  = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width  = (rect.width  || 380) * dpr;
    canvas.height = (rect.height || 60)  * dpr;
    ctx.scale(dpr, dpr);
  }

  function draw(t) {
    const W = canvas.getBoundingClientRect().width  || 380;
    const H = canvas.getBoundingClientRect().height || 60;
    const p = ACTIVITY_WAVE_PARAMS[_currentActivity] || ACTIVITY_WAVE_PARAMS["Moving"];

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0D0F0E";
    ctx.fillRect(0, 0, W, H);

    const stepX = W / (N - 1);

    ctx.beginPath();
    ctx.strokeStyle = p.color;
    ctx.lineWidth   = 1.5;
    ctx.lineJoin    = "round";

    for (let i = 0; i < N; i++) {
      const x = i * stepX;
      const norm = i / N;
      const val = 0.5
        + p.amp * Math.sin(p.freq * t + norm * Math.PI * 5)
        + p.amp * 0.5 * Math.sin(p.freq * 1.7 * t + norm * Math.PI * 8)
        + (Math.random() - 0.5) * p.noise;

      const y = H - Math.max(0.05, Math.min(0.95, val)) * H * 0.8 - H * 0.1;
      if (i === 0) ctx.moveTo(x, y);
      else         ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function loop(t) {
    _actAnimFrame = requestAnimationFrame(loop);
    draw(t / 1000);
  }

  resize();
  window.addEventListener("resize", resize, { passive: true });
  requestAnimationFrame(loop);
})();

// Expose for main.js activity selector
window.WiFiVision = window.WiFiVision || {};
window.WiFiVision.drawActivityWave = drawActivityWave;
