/**
 * dashboard.js  —  WiFi-Vision
 *
 * Polls Flask APIs and updates all dashboard panels.
 * Designed so the demo data source can be replaced by real hardware
 * without changing this file — only the Flask services change.
 *
 * Data flow:
 *   ESP32 hw → Python services → Flask API → this file → DOM
 *
 * Poll endpoints:
 *   /api/status     — hardware connection state
 *   /api/prediction — presence / activity / confidence / zone
 *   /api/csi        — raw CSI-derived signal samples
 *
 * Mode values returned by the API:
 *   "DEMO"       — demo service running, no hardware
 *   "LIVE"       — real hardware connected
 *
 * Hardware status field values:
 *   "connected"  — device confirmed reachable
 *   "offline"    — device not responding
 *   "waiting"    — not yet checked / stream not started
 *   "DEMO"       — placeholder in demo mode
 *   "live"       — data stream active
 *   "active"     — engine running
 */

"use strict";

/* ── Poll intervals (ms) ─────────────────────────────────────── */
const POLL_PREDICTION_MS = 2000;
const POLL_STATUS_MS     = 5000;
const POLL_CSI_MS        = 200;   // ~5 Hz

/* ── Colour tokens (must match CSS custom properties) ─────────── */
const C_ACCENT = "#E39A3B";
const C_GREEN  = "#7FA68A";
const C_MUTED  = "#9A9D98";
const C_RED    = "#C47070";
const C_BORDER = "#2A2D2B";
const C_BG     = "#171A18";

/* ══════════════════════════════════════════════════════════════
   Lightweight API helper
   Returns parsed JSON or throws on network / HTTP error.
═══════════════════════════════════════════════════════════════ */
async function apiFetch(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`${path} → HTTP ${resp.status}`);
  return resp.json();
}

/* ══════════════════════════════════════════════════════════════
   Panel switcher (Desktop sidebar + Mobile bottom navigation)
═══════════════════════════════════════════════════════════════ */
(function initPanels() {
  const desktopNavBtns = document.querySelectorAll(".dash-nav-item");
  const mobileNavBtns  = document.querySelectorAll(".dash-mobile-nav-btn[data-panel]");
  const allNavBtns     = [...desktopNavBtns, ...mobileNavBtns];
  const panels         = document.querySelectorAll(".dash-panel");

  function switchPanel(target) {
    if (!target) return;

    // Haptic feedback for touch devices (iOS / Android)
    if (navigator.vibrate) {
      try { navigator.vibrate(10); } catch (_) {}
    }

    allNavBtns.forEach(btn => {
      const match = btn.dataset.panel === target;
      btn.classList.toggle("active", match);
      btn.setAttribute("aria-pressed", String(match));
    });

    panels.forEach(panel => {
      const isTarget = panel.id === `panel-${target}`;
      panel.classList.toggle("active", isTarget);
      panel.hidden = !isTarget;
    });

    // Scroll active panel to top on mobile
    const mainEl = document.querySelector(".dash-main");
    if (mainEl) mainEl.scrollTop = 0;
  }

  allNavBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      switchPanel(btn.dataset.panel);
    });
  });

  // Wire up Phone Link triggers (topbar + mobile bottom nav)
  const topbarConnectBtn    = document.getElementById("dashConnectBtn");
  const mobileNavConnectBtn = document.getElementById("mobileNavConnectBtn");

  const openConnect = () => {
    if (window.openMobileConnectModal) {
      window.openMobileConnectModal();
    }
  };

  if (topbarConnectBtn)    topbarConnectBtn.addEventListener("click", openConnect);
  if (mobileNavConnectBtn) mobileNavConnectBtn.addEventListener("click", openConnect);
})();

/* ══════════════════════════════════════════════════════════════
   Live clock in top bar
═══════════════════════════════════════════════════════════════ */
(function initClock() {
  const el = document.getElementById("dashTimestamp");
  if (!el) return;
  const tick = () => { el.textContent = new Date().toLocaleTimeString("en-GB"); };
  tick();
  setInterval(tick, 1000);
})();

/* ══════════════════════════════════════════════════════════════
   Mode badge — updates sidebar + topbar badge text
═══════════════════════════════════════════════════════════════ */
function updateModeBadges(mode) {
  const sidebarBadge = document.getElementById("dashModeBadge");
  const label = mode === "LIVE" ? "LIVE" : "DEMO MODE";

  document.querySelectorAll(".demo-badge").forEach(el => {
    el.textContent = label;
    el.style.color            = mode === "LIVE" ? C_GREEN  : C_ACCENT;
    el.style.background       = mode === "LIVE" ? "rgba(127,166,138,0.1)" : "rgba(227,154,59,0.1)";
    el.style.borderColor      = mode === "LIVE" ? "rgba(127,166,138,0.3)" : "rgba(227,154,59,0.3)";
  });

  if (sidebarBadge) {
    sidebarBadge.textContent  = label;
    sidebarBadge.style.color  = mode === "LIVE" ? C_GREEN  : C_ACCENT;
  }
}

/* ══════════════════════════════════════════════════════════════
   Hardware status row renderer
   Handles: connected | offline | waiting | DEMO | live | active
═══════════════════════════════════════════════════════════════ */
function applyHwStatus(el, rawValue) {
  if (!el) return;
  const v = (rawValue || "waiting").toLowerCase();
  el.className = "hw-status mono";     // reset classes

  if (v === "demo") {
    el.textContent = "DEMO";
    el.classList.add("demo");
  } else if (v === "connected" || v === "live" || v === "active") {
    el.textContent = v.toUpperCase();
    el.classList.add("connected");
  } else if (v === "offline") {
    el.textContent = "OFFLINE";
    el.style.color  = C_RED;
    el.style.border = `1px solid rgba(196,112,112,0.3)`;
  } else {
    // "waiting" or anything unknown
    el.textContent = "Waiting for data";
    el.style.color  = C_MUTED;
  }
}

/* ══════════════════════════════════════════════════════════════
   Status / hardware polling  (/api/status)
═══════════════════════════════════════════════════════════════ */
(function initStatusPolling() {
  const senderEl   = document.getElementById("senderStatus");
  const receiverEl = document.getElementById("receiverStatus");
  const csiEl      = document.getElementById("csiStatus");
  const predEl     = document.getElementById("predStatus");
  const sysLabel   = document.getElementById("sysLabel");
  const sysInd     = document.getElementById("sysIndicator");

  async function poll() {
    try {
      const d = await apiFetch("/api/status");

      // Update mode badges everywhere
      updateModeBadges(d.mode || "DEMO");

      // System online indicator
      if (sysLabel) sysLabel.textContent = d.mode === "LIVE" ? "SYSTEM LIVE" : "SYSTEM ONLINE";
      if (sysInd)   sysInd.style.background = d.mode === "LIVE" ? C_GREEN : C_ACCENT;

      // Per-device hardware status rows
      applyHwStatus(senderEl,   d.sender);
      applyHwStatus(receiverEl, d.receiver);
      applyHwStatus(csiEl,      d.csi);
      applyHwStatus(predEl,     d.prediction || "active");

    } catch (err) {
      console.warn("[WiFi-Vision] /api/status error:", err.message);
      // Show "waiting" on network failure rather than leaving stale values
      [senderEl, receiverEl, csiEl, predEl].forEach(el => applyHwStatus(el, "waiting"));
    }
  }

  poll();
  setInterval(poll, POLL_STATUS_MS);
})();

/* ══════════════════════════════════════════════════════════════
   Prediction polling  (/api/prediction)
   Updates presence, activity, confidence, zone.
═══════════════════════════════════════════════════════════════ */
(function initPredictionPolling() {
  const presEl    = document.getElementById("presenceValue");
  const actEl     = document.getElementById("activityValue");
  const confEl    = document.getElementById("confValue");
  const confBar   = document.getElementById("confBar");
  const zoneEl    = document.getElementById("zoneValue");
  const streamEl  = document.getElementById("streamValue");
  const silhEl    = document.getElementById("detectionSilhouette");

  async function poll() {
    try {
      const d = await apiFetch("/api/prediction");

      /* Presence */
      if (presEl) {
        const present = d.presence;
        presEl.textContent = present === true  ? "PRESENT"
                           : present === false ? "EMPTY"
                           : "Waiting for data";
        presEl.style.color = present === true  ? C_GREEN
                           : present === false ? C_MUTED
                           : C_MUTED;
      }

      /* Activity */
      if (actEl) {
        actEl.textContent = d.activity
          ? d.activity.toUpperCase()
          : "Waiting for data";
      }

      /* Confidence */
      if (d.confidence != null) {
        const pct = Math.round(d.confidence * 100);
        if (confEl)  confEl.textContent  = pct + "%";
        if (confBar) confBar.style.width = pct + "%";
      } else {
        if (confEl)  confEl.textContent  = "—";
        if (confBar) confBar.style.width = "0%";
      }

      /* Zone — only show when available */
      if (zoneEl) {
        zoneEl.textContent = (d.zone && d.zone !== "—") ? d.zone : "—";
      }

      /* CSI stream indicator */
      if (streamEl) {
        streamEl.textContent = d.mode === "LIVE" ? "LIVE" : "DEMO";
        streamEl.style.color = d.mode === "LIVE" ? C_GREEN : C_ACCENT;
      }

      /* Silhouette — only populated activities get a figure */
      if (silhEl && window.drawSilhouette) {
        window.drawSilhouette(silhEl, d.activity || "Empty Room");
      }

    } catch (err) {
      console.warn("[WiFi-Vision] /api/prediction error:", err.message);
      if (actEl)  actEl.textContent  = "Waiting for data";
      if (presEl) presEl.textContent = "Waiting for data";
    }
  }

  poll();
  setInterval(poll, POLL_PREDICTION_MS);
})();

/* ══════════════════════════════════════════════════════════════
   Confidence Score Graph  (Status Panel — confScoreCanvas)
   ─────────────────────────────────────────────────────────────
   Plots prediction confidence over time as a rolling line chart.
   Polls /api/prediction at POLL_PREDICTION_MS intervals.
   Runs continuously while the Status panel is visible.
═══════════════════════════════════════════════════════════════ */
(function initConfidenceGraph() {
  const canvas = document.getElementById("confScoreCanvas");
  if (!canvas) return;

  const ctx           = canvas.getContext("2d");
  const MAX_POINTS    = 60;             // ~2 minutes at 2-second poll
  let   history       = [];             // [{conf: 0-1, activity: str}]
  let   animFrameId;
  let   fetchBusy     = false;
  let   lastActivity  = "";
  let   lastConf      = null;

  /* Resize canvas to physical pixels */
  function resize() {
    const dpr    = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    if (!parent) return;
    const rect    = parent.getBoundingClientRect();
    canvas.width  = (rect.width  || 700) * dpr;
    canvas.height = (rect.height || 220) * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* Fetch latest confidence from /api/prediction */
  async function fetchConfidence() {
    if (fetchBusy) return;
    fetchBusy = true;
    try {
      const d = await apiFetch("/api/prediction");

      if (d.confidence != null && !d.stale) {
        history.push({ conf: d.confidence, activity: d.activity || "" });
        if (history.length > MAX_POINTS) history.shift();
        lastActivity = d.activity || "";
        lastConf     = d.confidence;
      }

      const labelEl = document.getElementById("confGraphActivity");
      if (labelEl) {
        if (lastConf != null) {
          const pct = Math.round(lastConf * 100);
          labelEl.textContent = `${pct}% · ${lastActivity || "—"}`;
        } else {
          labelEl.textContent = "— Waiting for data";
        }
      }

      const sampleEl = document.getElementById("confGraphSample");
      if (sampleEl) {
        sampleEl.textContent = lastConf != null
          ? `${history.length} points · last: ${Math.round(lastConf * 100)}%`
          : "Waiting for prediction data";
      }
    } catch (_) {
      /* Keep last history on network hiccup */
    } finally {
      fetchBusy = false;
    }
  }

  /* Choose line colour based on confidence level */
  function confColor(conf) {
    if (conf >= 0.75) return C_GREEN;
    if (conf >= 0.45) return C_ACCENT;
    return C_RED;
  }

  /* Draw one frame */
  function draw() {
    const parent = canvas.parentElement;
    if (!parent) return;
    const W    = parent.clientWidth  || 700;
    const H    = parent.clientHeight || 220;
    const padL = 48;
    const padR = 16;
    const padT = 20;
    const padB = 28;

    ctx.clearRect(0, 0, W, H);

    /* Dark background */
    ctx.fillStyle = C_BG;
    ctx.fillRect(0, 0, W, H);

    /* Grid lines */
    const yLevels = [0, 0.25, 0.5, 0.75, 1.0];
    ctx.globalAlpha = 0.5;
    yLevels.forEach(lvl => {
      const y = padT + (1 - lvl) * (H - padT - padB);
      ctx.lineWidth   = 0.5;
      ctx.strokeStyle = lvl === 0.5 ? "#3A3D3B" : C_BORDER;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    });

    ctx.strokeStyle = C_BORDER;
    ctx.lineWidth   = 0.5;
    for (let i = 0; i <= 5; i++) {
      const x = padL + (i / 5) * (W - padL - padR);
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB); ctx.stroke();
    }
    ctx.globalAlpha = 1;

    /* Y-axis percentage labels */
    ctx.font      = `400 10px 'IBM Plex Mono', monospace`;
    ctx.fillStyle = C_MUTED;
    ctx.textAlign = "right";
    yLevels.forEach(lvl => {
      const y = padT + (1 - lvl) * (H - padT - padB);
      ctx.fillText(`${Math.round(lvl * 100)}%`, padL - 6, y + 3.5);
    });

    /* No-data placeholder */
    if (!history.length) {
      ctx.font      = `400 12px 'IBM Plex Mono', monospace`;
      ctx.fillStyle = C_MUTED;
      ctx.textAlign = "center";
      ctx.fillText("Waiting for prediction data…", W / 2, H / 2);
      return;
    }

    const n     = history.length;
    const drawW = W - padL - padR;
    const drawH = H - padT - padB;
    const stepX = n > 1 ? drawW / (n - 1) : drawW;

    const pts = history.map((pt, i) => ({
      x: padL + i * stepX,
      y: padT + (1 - pt.conf) * drawH,
      conf: pt.conf,
    }));

    /* Area fill */
    const latestConf = history[n - 1].conf;
    const lineColor  = confColor(latestConf);
    const grad = ctx.createLinearGradient(0, padT, 0, H - padB);
    grad.addColorStop(0,   `rgba(127,166,138,0.18)`);
    grad.addColorStop(0.5, `rgba(227,154,59,0.08)`);
    grad.addColorStop(1,   `rgba(23,26,24,0)`);

    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.lineTo(pts[n - 1].x, H - padB);
    ctx.lineTo(pts[0].x,     H - padB);
    ctx.closePath();
    ctx.fillStyle   = grad;
    ctx.globalAlpha = 1;
    ctx.fill();

    /* Main confidence line */
    ctx.shadowColor = lineColor;
    ctx.shadowBlur  = 8;
    ctx.strokeStyle = lineColor;
    ctx.lineWidth   = 2;
    ctx.lineJoin    = "round";
    ctx.lineCap     = "round";

    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.stroke();
    ctx.shadowBlur = 0;

    /* Latest-point dot */
    const last = pts[n - 1];
    ctx.beginPath();
    ctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
    ctx.fillStyle   = lineColor;
    ctx.shadowColor = lineColor;
    ctx.shadowBlur  = 12;
    ctx.fill();
    ctx.shadowBlur = 0;

    /* Current confidence label */
    const pct = Math.round(latestConf * 100);
    ctx.font      = `600 15px 'IBM Plex Mono', monospace`;
    ctx.fillStyle = lineColor;
    ctx.textAlign = "right";
    ctx.fillText(`${pct}%`, W - padR - 4, padT + 14);

    /* Activity watermark */
    if (lastActivity) {
      ctx.font        = `700 26px 'IBM Plex Mono', monospace`;
      ctx.fillStyle   = lineColor;
      ctx.globalAlpha = 0.05;
      ctx.textAlign   = "center";
      ctx.fillText(lastActivity.toUpperCase(), W / 2, H / 2 + 10);
      ctx.globalAlpha = 1;
    }

    /* 50% dashed reference */
    const midY = padT + 0.5 * drawH;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = C_MUTED;
    ctx.lineWidth   = 0.6;
    ctx.globalAlpha = 0.3;
    ctx.beginPath(); ctx.moveTo(padL, midY); ctx.lineTo(W - padR, midY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  /* Animation loop */
  function loop() {
    animFrameId = requestAnimationFrame(loop);
    draw();
  }

  /* Start / stop when Status panel is shown/hidden */
  const statusPanel = document.getElementById("panel-status");
  let   pollId;

  if (statusPanel) {
    new MutationObserver(() => {
      if (!statusPanel.hidden) {
        resize();
        if (!animFrameId) requestAnimationFrame(loop);
        if (!pollId) pollId = setInterval(fetchConfidence, POLL_PREDICTION_MS);
      } else {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
        if (pollId) { clearInterval(pollId); pollId = null; }
      }
    }).observe(statusPanel, { attributes: true, attributeFilter: ["hidden"] });
  }

  /* Status panel starts visible — kick off immediately */
  resize();
  requestAnimationFrame(loop);
  fetchConfidence();
  pollId = setInterval(fetchConfidence, POLL_PREDICTION_MS);

  /* Pause when browser tab is backgrounded */
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      if (statusPanel && !statusPanel.hidden) requestAnimationFrame(loop);
    } else {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  });

  /* Resize observer */
  if (canvas.parentElement) {
    new ResizeObserver(() => { resize(); }).observe(canvas.parentElement);
  }
})();

/* ══════════════════════════════════════════════════════════════
   CSI Frequency vs Time Spectrogram  (Signal Panel — dashSignalCanvas)
   ─────────────────────────────────────────────────────────────
   Renders a rolling waterfall / spectrogram of CSI subcarrier
   amplitudes. Each new CSI frame is added as a new column on
   the right; old columns scroll left.

   X-axis = Time  (newest at right, scrolls left)
   Y-axis = Frequency / Subcarrier index (0 at top, 63 at bottom)
   Pixel colour intensity = amplitude (thermal heatmap palette)

   Data source: /api/csi  ->  { samples: [0.0 ... 1.0] x N }
   Fetch rate: POLL_CSI_MS (~5 Hz)
═══════════════════════════════════════════════════════════════ */
(function initCSISpectrogramGraph() {
  const canvas = document.getElementById("dashSignalCanvas");
  if (!canvas) return;

  const ctx         = canvas.getContext("2d");
  const N_SUB       = 64;    // number of subcarriers / frequency bins
  const HISTORY     = 200;   // time columns to keep
  let   columns     = [];    // array of Float32Array(N_SUB), newest last
  let   animFrameId;
  let   fetchBusy   = false;
  let   lastFetch   = 0;
  let   sampleCount = 0;

  /* Resize canvas to physical pixels */
  function resize() {
    const dpr    = window.devicePixelRatio || 1;
    const parent = canvas.parentElement;
    if (!parent) return;
    const rect    = parent.getBoundingClientRect();
    canvas.width  = (rect.width  || 700) * dpr;
    canvas.height = (rect.height || 360) * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* Fetch one CSI frame from /api/csi */
  async function fetchCSI() {
    if (fetchBusy) return;
    fetchBusy = true;
    try {
      const d   = await apiFetch("/api/csi");
      let   raw = d.samples || [];

      if (raw.length > N_SUB) raw = raw.slice(0, N_SUB);
      const col = new Float32Array(N_SUB);
      for (let i = 0; i < raw.length; i++) col[i] = Math.max(0, Math.min(1, raw[i]));
      columns.push(col);
      if (columns.length > HISTORY) columns.shift();
      sampleCount++;

      const countEl = document.getElementById("sigSampleCount");
      if (countEl) {
        const subN = raw.length || N_SUB;
        countEl.textContent = `${subN} subcarriers · ${sampleCount} frames`;
      }
    } catch (_) {
      /* keep last columns on hiccup */
    } finally {
      fetchBusy = false;
    }
  }

  /* Amplitude -> RGBA colour (thermal heatmap palette) */
  function ampToRGBA(amp) {
    const stops = [
      { t: 0.0,  r: 13,  g: 15,  b: 14  },
      { t: 0.35, r: 61,  g: 44,  b: 20  },
      { t: 0.6,  r: 227, g: 154, b: 59  },
      { t: 0.8,  r: 127, g: 166, b: 138 },
      { t: 1.0,  r: 220, g: 255, b: 245 },
    ];

    let lo = stops[0], hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
      if (amp >= stops[i].t && amp <= stops[i + 1].t) {
        lo = stops[i]; hi = stops[i + 1]; break;
      }
    }
    const t = lo.t === hi.t ? 0 : (amp - lo.t) / (hi.t - lo.t);
    return [
      Math.round(lo.r + (hi.r - lo.r) * t),
      Math.round(lo.g + (hi.g - lo.g) * t),
      Math.round(lo.b + (hi.b - lo.b) * t),
      255,
    ];
  }

  /* Draw axes labels */
  function drawAxes(W, H, padL, padR, padT, padB) {
    const plotH = H - padT - padB;
    const plotW = W - padL - padR;
    const nCols = columns.length;

    ctx.font      = `400 9px 'IBM Plex Mono', monospace`;
    ctx.fillStyle = C_MUTED;
    ctx.textAlign = "right";

    [0, 16, 32, 48, 63].forEach(sub => {
      const y = padT + (sub / (N_SUB - 1)) * plotH;
      ctx.fillText(`${sub}`, padL - 5, y + 3);
      ctx.globalAlpha = 0.3;
      ctx.strokeStyle = C_BORDER;
      ctx.lineWidth   = 0.5;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL - 3, y);
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    /* Rotated Y-axis label */
    ctx.save();
    ctx.translate(11, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font      = `400 9px 'IBM Plex Mono', monospace`;
    ctx.fillStyle = C_MUTED;
    ctx.textAlign = "center";
    ctx.fillText("Subcarrier", 0, 0);
    ctx.restore();

    /* X-axis time ticks */
    if (nCols > 0) {
      ctx.font      = `400 9px 'IBM Plex Mono', monospace`;
      ctx.fillStyle = C_MUTED;
      ctx.textAlign = "center";
      for (let i = 0; i <= 5; i++) {
        const x = padL + (i / 5) * plotW;
        const secsAgo = Math.round(((5 - i) / 5) * nCols * (POLL_CSI_MS / 1000));
        ctx.fillText(secsAgo > 0 ? `-${secsAgo}s` : "now", x, H - padB + 14);
      }
    }
  }

  /* Draw one frame */
  function draw() {
    const parent = canvas.parentElement;
    if (!parent) return;
    const W    = parent.clientWidth  || 700;
    const H    = parent.clientHeight || 360;
    const padL = 52;
    const padR = 16;
    const padT = 12;
    const padB = 28;

    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = C_BG;
    ctx.fillRect(0, 0, W, H);

    if (!columns.length) {
      ctx.font      = `400 12px 'IBM Plex Mono', monospace`;
      ctx.fillStyle = C_MUTED;
      ctx.textAlign = "center";
      ctx.fillText("Waiting for CSI data…", W / 2, H / 2);
      drawAxes(W, H, padL, padR, padT, padB);
      return;
    }

    /* Spectrogram body via ImageData */
    const plotW   = W - padL - padR;
    const plotH   = H - padT - padB;
    const nCols   = columns.length;
    const imgW    = Math.max(1, Math.round(plotW));
    const imgH    = Math.max(1, Math.round(plotH));
    const imgData = ctx.createImageData(imgW, imgH);
    const data    = imgData.data;

    for (let ci = 0; ci < nCols; ci++) {
      const col    = columns[ci];
      const xStart = Math.round((ci / nCols) * imgW);
      const xEnd   = Math.round(((ci + 1) / nCols) * imgW);

      for (let si = 0; si < N_SUB; si++) {
        const amp    = col[si];
        const yStart = Math.round((si / N_SUB) * imgH);
        const yEnd   = Math.round(((si + 1) / N_SUB) * imgH);
        const rgba   = ampToRGBA(amp);

        for (let px = xStart; px < xEnd; px++) {
          for (let py = yStart; py < yEnd; py++) {
            const idx = (py * imgW + px) * 4;
            if (idx + 3 < data.length) {
              data[idx]     = rgba[0];
              data[idx + 1] = rgba[1];
              data[idx + 2] = rgba[2];
              data[idx + 3] = rgba[3];
            }
          }
        }
      }
    }

    ctx.putImageData(imgData, padL, padT);

    /* Border around plot area */
    ctx.strokeStyle = C_BORDER;
    ctx.lineWidth   = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);

    drawAxes(W, H, padL, padR, padT, padB);

    /* Colour legend (top-right) */
    const legW  = 10, legH = 60;
    const legX  = W - padR - legW - 2;
    const legY  = padT + 8;
    const legGrad = ctx.createLinearGradient(0, legY, 0, legY + legH);
    legGrad.addColorStop(0,   "rgb(220,255,245)");
    legGrad.addColorStop(0.4, "rgb(127,166,138)");
    legGrad.addColorStop(0.7, "rgb(227,154,59)");
    legGrad.addColorStop(1,   "rgb(13,15,14)");
    ctx.fillStyle = legGrad;
    ctx.fillRect(legX, legY, legW, legH);
    ctx.strokeStyle = C_BORDER;
    ctx.lineWidth   = 0.5;
    ctx.strokeRect(legX, legY, legW, legH);

    ctx.font      = `400 8px 'IBM Plex Mono', monospace`;
    ctx.fillStyle = C_MUTED;
    ctx.textAlign = "right";
    ctx.fillText("Hi", legX - 2, legY + 8);
    ctx.fillText("Lo", legX - 2, legY + legH);
  }

  /* Animation loop */
  function loop(timestamp) {
    animFrameId = requestAnimationFrame(loop);

    if (timestamp - lastFetch > POLL_CSI_MS) {
      fetchCSI();
      lastFetch = timestamp;
    }
    draw();
  }

  /* Start / stop when Signal panel is shown/hidden */
  const panel = document.getElementById("panel-signal");
  if (panel) {
    new MutationObserver(() => {
      if (!panel.hidden) {
        resize();
        if (!animFrameId) requestAnimationFrame(loop);
      } else {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
      }
    }).observe(panel, { attributes: true, attributeFilter: ["hidden"] });
  }

  /* Pause when browser tab is backgrounded */
  document.addEventListener("visibilitychange", () => {
    const isVisible = !document.hidden;
    if (isVisible && panel && !panel.hidden) {
      requestAnimationFrame(loop);
    } else {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    }
  });

  /* Resize observer */
  if (canvas.parentElement) {
    new ResizeObserver(() => { resize(); }).observe(canvas.parentElement);
  }

  /* Initial setup (Signal panel starts hidden — canvas sizes on first reveal) */
  resize();
})();
