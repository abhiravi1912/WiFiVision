/**
 * hero3d.js  —  WiFi-Vision
 * Three.js hero visualization: ESP32 → Wi-Fi → Human → Wi-Fi → ESP32
 */

"use strict";

(function initHero3D() {
  const canvas = document.getElementById("heroCanvas");
  if (!canvas || typeof THREE === "undefined") return;

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    // Simple static placeholder
    canvas.style.opacity = "0.4";
    return;
  }

  /* ── Scene setup ──────────────────────────────────────────────────── */
  const W = canvas.parentElement.clientWidth  || 500;
  const H = canvas.parentElement.clientHeight || 440;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.setClearColor(0x000000, 0);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, W / H, 0.1, 100);
  camera.position.set(0, 1.2, 10);
  camera.lookAt(0, 0, 0);

  /* ── Colors ───────────────────────────────────────────────────────── */
  const C_ACCENT  = new THREE.Color(0xE39A3B);
  const C_GREEN   = new THREE.Color(0x7FA68A);
  const C_BORDER  = new THREE.Color(0x2A2D2B);
  const C_TEXT    = new THREE.Color(0xF1EFE8);
  const C_MUTED   = new THREE.Color(0x9A9D98);

  /* ── Lighting ─────────────────────────────────────────────────────── */
  scene.add(new THREE.AmbientLight(0xffffff, 0.18));

  const amber = new THREE.PointLight(C_ACCENT, 1.2, 12);
  amber.position.set(0, 2, 2);
  scene.add(amber);

  const fill = new THREE.DirectionalLight(0xffffff, 0.3);
  fill.position.set(-3, 3, 5);
  scene.add(fill);

  /* ── ESP32 board helper ────────────────────────────────────────────── */
  function makeESP32(x) {
    const group = new THREE.Group();

    // PCB
    const pcb = new THREE.Mesh(
      new THREE.BoxGeometry(1.4, 0.08, 0.7),
      new THREE.MeshStandardMaterial({ color: 0x1A2E1A, roughness: 0.8, metalness: 0.1 })
    );
    group.add(pcb);

    // Chip
    const chip = new THREE.Mesh(
      new THREE.BoxGeometry(0.4, 0.06, 0.4),
      new THREE.MeshStandardMaterial({ color: 0x2C2C2C, roughness: 0.5, metalness: 0.6 })
    );
    chip.position.set(0.1, 0.07, 0);
    group.add(chip);

    // Antenna stub
    const ant = new THREE.Mesh(
      new THREE.BoxGeometry(0.06, 0.35, 0.06),
      new THREE.MeshStandardMaterial({ color: C_MUTED, roughness: 0.6 })
    );
    ant.position.set(0.55, 0.22, 0);
    group.add(ant);

    // LED dot
    const led = new THREE.Mesh(
      new THREE.SphereGeometry(0.05, 8, 8),
      new THREE.MeshStandardMaterial({
        color: C_ACCENT, emissive: C_ACCENT, emissiveIntensity: 1.5,
        roughness: 0.3
      })
    );
    led.position.set(-0.3, 0.09, 0.2);
    group.add(led);

    // Label plane (border only)
    const border = new THREE.Mesh(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1.5, 0.1, 0.8)),
      new THREE.LineBasicMaterial({ color: C_BORDER, opacity: 0.5, transparent: true })
    );
    group.add(border);

    group.position.set(x, 0, 0);
    return group;
  }

  const esp32L = makeESP32(-3.2);
  const esp32R = makeESP32( 3.2);
  scene.add(esp32L, esp32R);

  /* ── Human silhouette ─────────────────────────────────────────────── */
  function makeHuman() {
    const group = new THREE.Group();
    const mat   = new THREE.MeshStandardMaterial({
      color: C_GREEN, roughness: 0.7, metalness: 0.0, transparent: true, opacity: 0.75
    });

    // Head
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 10, 10), mat);
    head.position.set(0, 1.58, 0);
    group.add(head);

    // Body
    const body = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.22, 0.72, 8), mat);
    body.position.set(0, 1.02, 0);
    group.add(body);

    // Legs
    [-0.12, 0.12].forEach((x, i) => {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.07, 0.6, 7), mat);
      leg.position.set(x, 0.36, 0);
      group.add(leg);
    });

    // Arms
    [-1, 1].forEach(side => {
      const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.055, 0.5, 7), mat);
      arm.position.set(side * 0.34, 1.1, 0);
      arm.rotation.z = side * 0.3;
      group.add(arm);
    });

    return group;
  }

  const human = makeHuman();
  scene.add(human);

  /* ── Signal rings ──────────────────────────────────────────────────── */
  const RING_COUNT  = 5;
  const rings       = [];

  function makeRing(side) {
    const geometry = new THREE.RingGeometry(0.1, 0.14, 40);
    const material = new THREE.MeshBasicMaterial({
      color: C_ACCENT, side: THREE.DoubleSide,
      transparent: true, opacity: 0.0
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.y = Math.PI / 2;
    return mesh;
  }

  // Left-side rings (ESP32L → Human)
  for (let i = 0; i < RING_COUNT; i++) {
    const r = makeRing("left");
    scene.add(r);
    rings.push({ mesh: r, side: "left", phase: i / RING_COUNT });
  }

  // Right-side rings (Human → ESP32R)
  for (let i = 0; i < RING_COUNT; i++) {
    const r = makeRing("right");
    scene.add(r);
    rings.push({ mesh: r, side: "right", phase: i / RING_COUNT });
  }

  /* ── Floor grid ───────────────────────────────────────────────────── */
  const gridHelper = new THREE.GridHelper(10, 12, 0x1E211F, 0x1A1D1B);
  gridHelper.position.y = -0.1;
  scene.add(gridHelper);

  /* ── Animation clock ──────────────────────────────────────────────── */
  const clock = new THREE.Clock();
  let frame;

  function animate() {
    frame = requestAnimationFrame(animate);
    const t = clock.getElapsedTime();

    // Slow camera drift
    camera.position.x = Math.sin(t * 0.08) * 0.6;
    camera.position.y = 1.2 + Math.sin(t * 0.12) * 0.2;
    camera.lookAt(0, 0.8, 0);

    // Subtle human sway
    human.rotation.y = Math.sin(t * 0.4) * 0.08;

    // LED pulse on ESP32s
    const ledIntensity = 0.8 + 0.7 * Math.abs(Math.sin(t * 1.5));
    [esp32L, esp32R].forEach(esp => {
      const led = esp.children[3];
      if (led && led.material) led.material.emissiveIntensity = ledIntensity;
    });

    // ESP32 gentle tilt
    esp32L.rotation.z = Math.sin(t * 0.3 + 0.5) * 0.025;
    esp32R.rotation.z = Math.sin(t * 0.3)        * 0.025;

    // Animate signal rings
    rings.forEach(({ mesh, side, phase }) => {
      const p = ((t * 0.38 + phase) % 1.0);

      if (side === "left") {
        // Travel from -3.2 to 0
        mesh.position.x = -3.2 + p * 3.2;
      } else {
        // Travel from 0 to 3.2
        mesh.position.x = p * 3.2;
      }

      // Size and fade
      const scale = 0.3 + p * 1.8;
      mesh.scale.set(scale, scale, scale);

      // Fade in/out
      let alpha;
      if (p < 0.15)      alpha = p / 0.15;
      else if (p > 0.75) alpha = (1.0 - p) / 0.25;
      else               alpha = 1.0;

      mesh.material.opacity = alpha * 0.55;

      // Color shift as signal passes through human (center)
      const distToCenter = Math.abs(mesh.position.x);
      mesh.material.color.lerpColors(C_ACCENT, C_GREEN, Math.max(0, 1 - distToCenter / 2));
    });

    // Amber point light orbit
    amber.position.x = Math.sin(t * 0.5) * 2;
    amber.position.z = Math.cos(t * 0.5) * 2 + 2;

    renderer.render(scene, camera);
  }

  animate();

  /* ── Resize handler ────────────────────────────────────────────────── */
  const resizeObs = new ResizeObserver(() => {
    const w = canvas.parentElement.clientWidth;
    const h = canvas.parentElement.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
  resizeObs.observe(canvas.parentElement);

  /* ── Cleanup on page hide ──────────────────────────────────────────── */
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      cancelAnimationFrame(frame);
    } else {
      animate();
    }
  });

})();
