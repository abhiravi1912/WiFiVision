/**
 * main.js  —  WiFi-Vision
 * Shared utilities: navbar, scroll reveal, mobile menu
 */

"use strict";

/* ── Navbar scroll effect ────────────────────────────────────── */
(function initNavbar() {
  const navbar = document.getElementById("navbar");
  if (!navbar) return;

  let lastY = 0;
  window.addEventListener("scroll", () => {
    const y = window.scrollY;
    if (y > 40) {
      navbar.style.background = "rgba(13, 15, 14, 0.97)";
    } else {
      navbar.style.background = "rgba(13, 15, 14, 0.88)";
    }
    lastY = y;
  }, { passive: true });
})();

/* ── Mobile nav toggle ───────────────────────────────────────── */
(function initMobileNav() {
  const toggle = document.getElementById("navToggle");
  const links  = document.getElementById("navLinks");
  if (!toggle || !links) return;

  toggle.addEventListener("click", () => {
    const open = links.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });

  // Close on link click
  links.querySelectorAll("a").forEach(a => {
    a.addEventListener("click", () => {
      links.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    });
  });
})();

/* ── Scroll reveal ───────────────────────────────────────────── */
(function initReveal() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const elements = document.querySelectorAll(".reveal");
  if (!elements.length) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
  );

  elements.forEach((el) => observer.observe(el));
})();

/* ── Activity selector ───────────────────────────────────────── */
(function initActivitySelector() {
  const btns = document.querySelectorAll(".activity-btn");
  if (!btns.length) return;

  const descriptions = {
    "Moving":    "Body movement generates distinct multi-path CSI variation patterns across subcarriers.",
    "Standing":   "Subtle respiration and micro-movements create low-amplitude, high-frequency CSI oscillations.",
    "Sitting":    "Minimal body displacement produces a near-static, low-variance CSI signature.",
    "Empty Room": "No human movement results in a stable, low-noise CSI baseline across all subcarriers.",
  };

  const nameEl = document.getElementById("activityName");
  const descEl = document.getElementById("activityDesc");
  const waveEl = document.getElementById("activityWave");
  const silhEl = document.getElementById("activitySilhouette");

  function updateActivity(name) {
    if (nameEl) nameEl.textContent = name;
    if (descEl) descEl.textContent = descriptions[name] || "";
    if (silhEl) drawSilhouette(silhEl, name);
    if (waveEl && window.WiFiVision && window.WiFiVision.drawActivityWave) {
      window.WiFiVision.drawActivityWave(waveEl, name);
    }
  }

  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      btns.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-pressed", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-pressed", "true");
      updateActivity(btn.dataset.activity);
    });
  });

  // Draw initial state
  const firstActive = document.querySelector(".activity-btn.active");
  if (firstActive) updateActivity(firstActive.dataset.activity);
})();

/* ── Silhouette SVG renderer ─────────────────────────────────── */
function drawSilhouette(container, activity) {
  const poses = {
    Moving: `
      <svg viewBox="0 0 60 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="height:90px;color:#E39A3B">
        <circle cx="30" cy="12" r="9" fill="currentColor" opacity=".85"/>
        <path d="M18 42 Q18 26 30 26 Q42 26 42 42 L39 58 H25 Z" fill="currentColor" opacity=".75"/>
        <line x1="25" y1="55" x2="16" y2="72" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
        <line x1="35" y1="55" x2="44" y2="68" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
        <line x1="16" y1="72" x2="14" y2="90" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
        <line x1="44" y1="68" x2="48" y2="86" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
      </svg>`,
    Standing: `
      <svg viewBox="0 0 60 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="height:90px;color:#7FA68A">
        <circle cx="30" cy="11" r="9" fill="currentColor" opacity=".85"/>
        <path d="M20 38 Q20 24 30 24 Q40 24 40 38 L38 58 H22 Z" fill="currentColor" opacity=".75"/>
        <line x1="14" y1="30" x2="22" y2="46" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
        <line x1="46" y1="30" x2="38" y2="46" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
        <line x1="25" y1="58" x2="25" y2="88" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
        <line x1="35" y1="58" x2="35" y2="88" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
      </svg>`,
    Sitting: `
      <svg viewBox="0 0 80 90" fill="none" xmlns="http://www.w3.org/2000/svg" style="height:90px;color:#9A9D98">
        <circle cx="22" cy="11" r="9" fill="currentColor" opacity=".85"/>
        <path d="M10 36 Q10 22 22 22 Q34 22 34 36 L32 54 H12 Z" fill="currentColor" opacity=".75"/>
        <line x1="6"  y1="28" x2="12" y2="42" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
        <line x1="38" y1="28" x2="32" y2="42" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
        <!-- horizontal seat -->
        <line x1="12" y1="54" x2="60" y2="54" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
        <!-- legs down -->
        <line x1="60" y1="54" x2="60" y2="82" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
        <line x1="12" y1="54" x2="12" y2="82" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" opacity=".3"/>
      </svg>`,
    "Empty Room": `
      <svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg" style="height:80px;color:#2A2D2B">
        <rect x="4"  y="4"  width="72" height="72" rx="4" stroke="#2A2D2B" stroke-width="1.5"/>
        <line x1="20" y1="20" x2="60" y2="60" stroke="#2A2D2B" stroke-width="1.5"/>
        <line x1="60" y1="20" x2="20" y2="60" stroke="#2A2D2B" stroke-width="1.5"/>
      </svg>`,
  };

  container.innerHTML = poses[activity] || poses["Empty Room"];
}

// Expose for use from other modules
window.drawSilhouette = drawSilhouette;

/* ══════════════════════════════════════════════════════════════
   PWA Service Worker Registration & Install Prompts
   ═══════════════════════════════════════════════════════════════ */
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/sw.js")
      .then((reg) => {
        console.log("[WiFi-Vision PWA] Service worker registered, scope:", reg.scope);
      })
      .catch((err) => {
        console.warn("[WiFi-Vision PWA] Service worker registration error:", err);
      });
  });
}

let deferredPwaPrompt = null;
window.addEventListener("beforeinstallprompt", (e) => {
  // Prevent mini-infobar on mobile Chrome
  e.preventDefault();
  deferredPwaPrompt = e;

  // Unhide install buttons
  const navInstallBtn = document.getElementById("pwaInstallNavBtn");
  if (navInstallBtn) {
    navInstallBtn.style.display = "inline-flex";
    navInstallBtn.addEventListener("click", async () => {
      if (!deferredPwaPrompt) return;
      deferredPwaPrompt.prompt();
      const choice = await deferredPwaPrompt.userChoice;
      if (choice.outcome === "accepted") {
        console.log("[WiFi-Vision PWA] User accepted install prompt");
      }
      deferredPwaPrompt = null;
      navInstallBtn.style.display = "none";
    });
  }
});

/* ══════════════════════════════════════════════════════════════
   Mobile Connect Modal & QR Code (iOS & Android)
   ═══════════════════════════════════════════════════════════════ */
(function initMobileConnectModal() {
  const backdrop  = document.getElementById("mobileModalBackdrop");
  const closeBtn  = document.getElementById("modalCloseBtn");
  const urlInput  = document.getElementById("modalUrlInput");
  const copyBtn   = document.getElementById("modalCopyBtn");
  const feedback  = document.getElementById("modalCopyFeedback");
  const qrFrame   = document.getElementById("modalQrCode");
  const tabBtns   = document.querySelectorAll(".modal-tab-btn");
  const tabPanels = {
    ios: document.getElementById("tabContentIos"),
    android: document.getElementById("tabContentAndroid")
  };

  let qrInstance = null;
  let cachedUrl  = null;

  async function openModal() {
    if (!backdrop) return;
    backdrop.classList.add("open");
    backdrop.setAttribute("aria-hidden", "false");

    // Fetch LAN IP info if not yet loaded
    if (!cachedUrl) {
      try {
        const resp = await fetch("/api/lan-info");
        const data = await resp.json();
        // Point to /dashboard so mobile scans jump straight into the live monitor
        cachedUrl = data.dashboard_url || `${window.location.protocol}//${window.location.host}/dashboard`;
      } catch (_) {
        cachedUrl = `${window.location.protocol}//${window.location.host}/dashboard`;
      }

      if (urlInput) urlInput.value = cachedUrl;

      // Render QR code
      if (qrFrame && typeof QRCode !== "undefined") {
        qrFrame.innerHTML = "";
        try {
          qrInstance = new QRCode(qrFrame, {
            text: cachedUrl,
            width: 176,
            height: 176,
            colorDark: "#0D0F0E",
            colorLight: "#FFFFFF",
            correctLevel: QRCode.CorrectLevel.M
          });
        } catch (e) {
          console.warn("[WiFi-Vision] QR code generation error:", e);
        }
      }
    }
  }

  function closeModal() {
    if (!backdrop) return;
    backdrop.classList.remove("open");
    backdrop.setAttribute("aria-hidden", "true");
  }

  // Bind close triggers
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (backdrop) {
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeModal();
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && backdrop && backdrop.classList.contains("open")) {
      closeModal();
    }
  });

  // Bind copy button
  if (copyBtn && urlInput) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(urlInput.value);
        if (feedback) {
          feedback.classList.add("show");
          setTimeout(() => feedback.classList.remove("show"), 2500);
        }
      } catch (_) {
        urlInput.select();
        document.execCommand("copy");
        if (feedback) {
          feedback.classList.add("show");
          setTimeout(() => feedback.classList.remove("show"), 2500);
        }
      }
    });
  }

  // OS tab switching (iOS Safari vs Android Chrome)
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      tabBtns.forEach(b => b.classList.toggle("active", b === btn));
      if (tabPanels.ios)     tabPanels.ios.classList.toggle("active", tab === "ios");
      if (tabPanels.android) tabPanels.android.classList.toggle("active", tab === "android");
    });
  });

  // Wire up nav button
  const navBtn = document.getElementById("navPhoneLink");
  if (navBtn) navBtn.addEventListener("click", openModal);

  // Expose globally for dashboard triggers
  window.openMobileConnectModal = openModal;
  window.closeMobileConnectModal = closeModal;
})();

