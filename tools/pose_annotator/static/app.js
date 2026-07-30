import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
function setStatus(msg, ok = false) {
  statusEl.textContent = msg;
  statusEl.className = ok ? "ok" : "";
}

const canvas = $("c3d");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1012);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);

const controls = new TrackballControls(camera, canvas);
controls.rotateSpeed = 3.0;
controls.zoomSpeed = 1.4;
controls.panSpeed = 1.0;
controls.staticMoving = false;
controls.dynamicDampingFactor = 0.12;
controls.noRoll = false;

scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.0));
const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(2, 4, 3);
scene.add(dir);

const pivotMarker = new THREE.Mesh(
  new THREE.SphereGeometry(1, 16, 16),
  new THREE.MeshBasicMaterial({ color: 0xf59e0b })
);
pivotMarker.scale.setScalar(0.02);
scene.add(pivotMarker);

let sceneData = null;
let boxMesh = null;
let transform = null;
let pointsCloud = null;
let syncingUI = false;
/** Last place the pose was edited: 'box' (3D/gizmo/align/yaw) or 'ui' (side panel). */
let poseEditSource = "box";

let allPositions = null;
let allColors = null;
let nAll = 0;
let fgPositions = null;
let pickMode = false;
let segmentBusy = false;

/** Scale calibration state */
let scalePickMode = null; // null | "plane" | "target"
let planeSamples = []; // THREE.Vector3[]
let planeCentroid = null;
let planeNormal = null;
let planeOffset = 0;
/** @type {{center:number[], euler_deg:number[], size:number[]}|null} */
let alignPoseBackup = null;
let targetPoint = null;
let planeHelper = null;
let planeSampleGroup = null;
let targetMarkerMesh = null;
let lastScalePreview = null;
/** true = unit fixed (meters), apply = refine numbers only */
let metricLocked = true;

/** @type {ViewSlot[]} */
let views = [];
let activeViewId = null;
let nextViewId = 1;

/**
 * @typedef {object} ViewSlot
 * @property {number} id
 * @property {number} frameIndex
 * @property {{u:number,v:number}[]} fgSeeds
 * @property {{u:number,v:number}[]} bgSeeds
 * @property {Uint8Array|null} maskU8
 * @property {number} maskW
 * @property {number} maskH
 * @property {HTMLCanvasElement|null} maskTint
 * @property {number} zoom
 * @property {number} panX
 * @property {number} panY
 * @property {HTMLElement|null} root
 * @property {HTMLSelectElement|null} select
 * @property {HTMLElement|null} stage
 * @property {HTMLImageElement|null} img
 * @property {HTMLCanvasElement|null} overlay
 * @property {HTMLElement|null} hint
 */

const raycaster = new THREE.Raycaster();
raycaster.params.Points = { threshold: 0.08 };
const pointer = new THREE.Vector2();

function setPivot(x, y, z, announce = true) {
  controls.target.set(x, y, z);
  pivotMarker.position.set(x, y, z);
  controls.update();
  if (announce) setStatus(`旋转中心 → (${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)})`, true);
}

function focusOnPoint(p, distanceFactor = 0.35) {
  const extent = sceneData ? sceneData.extent : 5;
  const dist = Math.max(extent * distanceFactor, 0.3);
  const d = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
  if (d.lengthSq() < 1e-8) d.set(0.6, 0.4, 0.6).normalize();
  camera.position.copy(p).addScaledVector(d, dist);
  setPivot(p.x, p.y, p.z, false);
}

function getActiveView() {
  return views.find((v) => v.id === activeViewId) || views[0] || null;
}

function setActiveView(id) {
  activeViewId = id;
  for (const v of views) {
    if (v.root) v.root.classList.toggle("active", v.id === id);
  }
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  const cw = Math.max(1, Math.floor(rect.width));
  const ch = Math.max(1, Math.floor(rect.height));
  renderer.setSize(cw, ch, false);
  camera.aspect = cw / ch;
  camera.updateProjectionMatrix();
  controls.handleResize();
  for (const v of views) layoutView(v);
}
window.addEventListener("resize", resize);

function eulerFromUI() {
  return new THREE.Euler(
    THREE.MathUtils.degToRad(+$("rx").value),
    THREE.MathUtils.degToRad(+$("ry").value),
    THREE.MathUtils.degToRad(+$("rz").value),
    "XYZ"
  );
}

function readFrameFromUI() {
  return {
    center: [+$("cx").value, +$("cy").value, +$("cz").value],
    euler_deg: [+$("rx").value, +$("ry").value, +$("rz").value],
    size: [+$("sx").value, +$("sy").value, +$("sz").value],
    class_id: +$("classId").value,
    class_name: $("className").value || "object",
  };
}

/** Read pose from the 3D box (includes quaternion for export). */
function readFrameFromBox() {
  if (!boxMesh) return readFrameFromUI();
  boxMesh.updateMatrixWorld(true);
  const e = new THREE.Euler().setFromQuaternion(boxMesh.quaternion, "XYZ");
  const q = boxMesh.quaternion;
  return {
    center: [boxMesh.position.x, boxMesh.position.y, boxMesh.position.z],
    euler_deg: [
      THREE.MathUtils.radToDeg(e.x),
      THREE.MathUtils.radToDeg(e.y),
      THREE.MathUtils.radToDeg(e.z),
    ],
    size: [Math.abs(boxMesh.scale.x), Math.abs(boxMesh.scale.y), Math.abs(boxMesh.scale.z)],
    class_id: +$("classId").value,
    class_name: $("className").value || "object",
    quaternion_wxyz: [q.w, q.x, q.y, q.z],
  };
}

/** Push boxMesh → side panel (center / euler / size). Always runs; blocks UI→box briefly. */
function writeUIFromBox() {
  if (!boxMesh) return;
  syncingUI = true;
  poseEditSource = "box";
  boxMesh.updateMatrixWorld(true);
  const p = boxMesh.position;
  const e = new THREE.Euler().setFromQuaternion(boxMesh.quaternion, "XYZ");
  boxMesh.rotation.copy(e);
  const s = boxMesh.scale;
  $("cx").value = p.x.toFixed(4);
  $("cy").value = p.y.toFixed(4);
  $("cz").value = p.z.toFixed(4);
  $("rx").value = THREE.MathUtils.radToDeg(e.x).toFixed(2);
  $("ry").value = THREE.MathUtils.radToDeg(e.y).toFixed(2);
  $("rz").value = THREE.MathUtils.radToDeg(e.z).toFixed(2);
  $("sx").value = Math.abs(s.x).toFixed(4);
  $("sy").value = Math.abs(s.y).toFixed(4);
  $("sz").value = Math.abs(s.z).toFixed(4);
  requestAnimationFrame(() => {
    syncingUI = false;
  });
  for (const v of views) drawViewOverlay(v);
}

/** Pull side panel → boxMesh (manual euler/center/size edits). */
function applyUIToBox() {
  if (!boxMesh || syncingUI) return;
  poseEditSource = "ui";
  const f = readFrameFromUI();
  boxMesh.position.set(...f.center);
  boxMesh.rotation.copy(eulerFromUI());
  boxMesh.quaternion.setFromEuler(boxMesh.rotation);
  boxMesh.scale.set(...f.size);
  boxMesh.updateMatrixWorld(true);
  for (const v of views) drawViewOverlay(v);
}

/**
 * Flush pose so box + side panel + export payload agree.
 * - If last edit was in the panel, apply panel → box first.
 * - Always refresh panel from box, then return box pose (+ quaternion).
 */
function commitPoseForSave() {
  if (!boxMesh) return readFrameFromUI();
  if (poseEditSource === "ui") applyUIToBox();
  writeUIFromBox();
  return readFrameFromBox();
}

const UNIT_CORNERS = [
  [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
  [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
];
const BOX_EDGES = [
  [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7],
];

function boxCornersWorldFromMesh() {
  boxMesh.updateMatrixWorld(true);
  const m = boxMesh.matrixWorld;
  const corners = UNIT_CORNERS.map((c) => {
    const v = new THREE.Vector3(c[0], c[1], c[2]).applyMatrix4(m);
    return [v.x, v.y, v.z];
  });
  const center = new THREE.Vector3().setFromMatrixPosition(m);
  return [[center.x, center.y, center.z], ...corners];
}

function projectPoint(K, w2c, Xw) {
  const x = w2c[0][0] * Xw[0] + w2c[0][1] * Xw[1] + w2c[0][2] * Xw[2] + w2c[0][3];
  const y = w2c[1][0] * Xw[0] + w2c[1][1] * Xw[1] + w2c[1][2] * Xw[2] + w2c[1][3];
  const z = w2c[2][0] * Xw[0] + w2c[2][1] * Xw[1] + w2c[2][2] * Xw[2] + w2c[2][3];
  if (z <= 1e-6) return null;
  const u = (K[0][0] * x) / z + K[0][2];
  const v = (K[1][1] * y) / z + K[1][2];
  return [u, v, z];
}

/* ---------- Multi 2D views ---------- */

function fillFrameSelect(sel, selected) {
  sel.innerHTML = "";
  sceneData.frames.forEach((fr, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = fr.stem;
    if (i === selected) opt.selected = true;
    sel.appendChild(opt);
  });
}

function createView(frameIndex = 0) {
  /** @type {ViewSlot} */
  const v = {
    id: nextViewId++,
    frameIndex,
    fgSeeds: [],
    bgSeeds: [],
    maskU8: null,
    maskW: 0,
    maskH: 0,
    maskTint: null,
    zoom: 1,
    panX: 0,
    panY: 0,
    root: null,
    select: null,
    stage: null,
    img: null,
    overlay: null,
    hint: null,
  };

  const root = document.createElement("div");
  root.className = "pv";
  root.dataset.vid = String(v.id);

  const bar = document.createElement("div");
  bar.className = "pv-bar";
  const sel = document.createElement("select");
  fillFrameSelect(sel, frameIndex);
  const btnReset = document.createElement("button");
  btnReset.type = "button";
  btnReset.textContent = "复位";
  btnReset.title = "重置缩放/平移";
  bar.append(sel, btnReset);

  const stage = document.createElement("div");
  stage.className = "pv-stage" + (pickMode ? " pick" : "");
  const img = document.createElement("img");
  img.alt = "frame";
  img.draggable = false;
  const ov = document.createElement("canvas");
  ov.className = "ov";
  const hint = document.createElement("div");
  hint.className = "pv-hint";
  stage.append(img, ov, hint);

  root.append(bar, stage);
  $("previews").appendChild(root);

  v.root = root;
  v.select = sel;
  v.stage = stage;
  v.img = img;
  v.overlay = ov;
  v.hint = hint;

  root.addEventListener("pointerdown", () => setActiveView(v.id));
  sel.addEventListener("change", () => {
    v.frameIndex = +sel.value;
    v.fgSeeds = [];
    v.bgSeeds = [];
    v.maskU8 = null;
    v.maskTint = null;
    v.zoom = 1;
    v.panX = 0;
    v.panY = 0;
    loadViewImage(v).then(() => tryLoadSavedMask(v));
    applyMultiViewFilter();
  });
  btnReset.addEventListener("click", () => {
    v.zoom = 1;
    v.panX = 0;
    v.panY = 0;
    layoutView(v);
  });

  bindViewInteractions(v);
  views.push(v);
  setActiveView(v.id);
  loadViewImage(v).then(() => tryLoadSavedMask(v));
  return v;
}

function removeActiveView() {
  if (views.length <= 1) {
    setStatus("至少保留 1 个 2D 视图");
    return;
  }
  const v = getActiveView();
  if (!v) return;
  v.root.remove();
  views = views.filter((x) => x.id !== v.id);
  setActiveView(views[views.length - 1].id);
  applyMultiViewFilter();
  setStatus(`已移除视图，剩余 ${views.length} 个`, true);
}

function loadViewImage(v) {
  const fr = sceneData.frames[v.frameIndex];
  const src = fr.image_rel.startsWith("frames/") ? "/" + fr.image_rel : fr.image_rel;
  return new Promise((resolve) => {
    const done = () => {
      layoutView(v);
      resolve();
    };
    if (v.img.src.endsWith(src) || v.img.getAttribute("src") === src) {
      if (v.img.complete && v.img.naturalWidth) {
        done();
        return;
      }
    }
    v.img.onload = done;
    v.img.onerror = () => resolve();
    v.img.src = src;
  });
}

async function tryLoadSavedMask(v) {
  try {
    const res = await fetch(`/api/mask?frame_index=${v.frameIndex}`);
    const data = await res.json();
    if (!data.ok || !data.mask_png_b64) return false;
    const decoded = await decodeMaskPngB64(data.mask_png_b64);
    v.maskU8 = decoded.maskU8;
    v.maskTint = decoded.tint;
    v.maskW = decoded.w;
    v.maskH = decoded.h;
    drawViewOverlay(v);
    applyMultiViewFilter();
    return true;
  } catch {
    return false;
  }
}

/** Fit image into stage with contain, then apply zoom/pan */
function baseContain(v) {
  const stage = v.stage;
  const img = v.img;
  if (!img.naturalWidth) return null;
  const sw = stage.clientWidth;
  const sh = stage.clientHeight;
  const scale = Math.min(sw / img.naturalWidth, sh / img.naturalHeight);
  const dw = img.naturalWidth * scale;
  const dh = img.naturalHeight * scale;
  const ox = (sw - dw) / 2;
  const oy = (sh - dh) / 2;
  return { sw, sh, scale, dw, dh, ox, oy, nw: img.naturalWidth, nh: img.naturalHeight };
}

function layoutView(v) {
  const base = baseContain(v);
  if (!base) return;
  const { dw, dh, ox, oy } = base;
  // Final transform: pan around stage, zoom about image center in stage space
  const cx = ox + dw / 2;
  const cy = oy + dh / 2;
  const z = v.zoom;
  const left = cx + v.panX - (dw * z) / 2;
  const top = cy + v.panY - (dh * z) / 2;
  const w = dw * z;
  const h = dh * z;

  v.img.style.width = `${w}px`;
  v.img.style.height = `${h}px`;
  v.img.style.left = `${left}px`;
  v.img.style.top = `${top}px`;

  v.overlay.width = Math.max(1, Math.floor(v.stage.clientWidth));
  v.overlay.height = Math.max(1, Math.floor(v.stage.clientHeight));
  v.overlay.style.width = `${v.overlay.width}px`;
  v.overlay.style.height = `${v.overlay.height}px`;
  v.overlay.style.left = "0";
  v.overlay.style.top = "0";

  drawViewOverlay(v);
}

function imageRectOnStage(v) {
  const base = baseContain(v);
  if (!base) return null;
  const { dw, dh, ox, oy } = base;
  const z = v.zoom;
  const cx = ox + dw / 2;
  const cy = oy + dh / 2;
  return {
    left: cx + v.panX - (dw * z) / 2,
    top: cy + v.panY - (dh * z) / 2,
    w: dw * z,
    h: dh * z,
    nw: base.nw,
    nh: base.nh,
  };
}

function stageToImageUV(v, clientX, clientY) {
  const rect = v.stage.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const ir = imageRectOnStage(v);
  if (!ir) return null;
  if (x < ir.left || y < ir.top || x > ir.left + ir.w || y > ir.top + ir.h) return null;
  const uImg = ((x - ir.left) / ir.w) * ir.nw;
  const vImg = ((y - ir.top) / ir.h) * ir.nh;
  const sx = sceneData.width / ir.nw;
  const sy = sceneData.height / ir.nh;
  return { u: uImg * sx, v: vImg * sy };
}

function drawViewOverlay(v) {
  if (!sceneData || !v.overlay) return;
  const fr = sceneData.frames[v.frameIndex];
  const ctx = v.overlay.getContext("2d");
  ctx.clearRect(0, 0, v.overlay.width, v.overlay.height);
  const ir = imageRectOnStage(v);
  if (!ir) return;

  if (v.maskTint) {
    ctx.save();
    ctx.globalAlpha = 0.4;
    ctx.drawImage(v.maskTint, ir.left, ir.top, ir.w, ir.h);
    ctx.restore();
  }

  const toCanvas = (uSfM, vSfM) => [
    ir.left + (uSfM / sceneData.width) * ir.w,
    ir.top + (vSfM / sceneData.height) * ir.h,
  ];

  for (const s of v.fgSeeds) {
    const [x, y] = toCanvas(s.u, s.v);
    ctx.fillStyle = "#22c55e";
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  for (const s of v.bgSeeds) {
    const [x, y] = toCanvas(s.u, s.v);
    ctx.fillStyle = "#ef4444";
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
  }

  if (boxMesh) {
    const pts = boxCornersWorldFromMesh();
    const uv = pts.map((p) => projectPoint(sceneData.K, fr.w2c, p));
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 2;
    for (const [a, b] of BOX_EDGES) {
      const ua = uv[a + 1];
      const ub = uv[b + 1];
      if (!ua || !ub) continue;
      const [x0, y0] = toCanvas(ua[0], ua[1]);
      const [x1, y1] = toCanvas(ub[0], ub[1]);
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
    if (uv[0]) {
      const [cx, cy] = toCanvas(uv[0][0], uv[0][1]);
      ctx.fillStyle = "#22c55e";
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const nMasked = views.filter((x) => x.maskU8).length;
  v.hint.textContent = pickMode
    ? `点选 · 滚轮缩放 · 中键平移 · 已标注 ${nMasked}/${views.length} 视图`
    : `×${v.zoom.toFixed(2)} · 中键/Alt拖平移 · 滚轮缩放`;
}

function bindViewInteractions(v) {
  let panning = false;
  let panLast = null;

  v.stage.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const rect = v.stage.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const ir0 = imageRectOnStage(v);
      if (!ir0) return;
      // zoom toward cursor
      const prev = v.zoom;
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      v.zoom = Math.min(12, Math.max(0.4, v.zoom * factor));
      // adjust pan so point under cursor stays
      const cx = ir0.left + ir0.w / 2;
      const cy = ir0.top + ir0.h / 2;
      // before: point at (mx,my) relative to image center in stage
      const dx = mx - cx;
      const dy = my - cy;
      const ratio = v.zoom / prev;
      v.panX += dx - dx * ratio;
      v.panY += dy - dy * ratio;
      layoutView(v);
    },
    { passive: false }
  );

  v.stage.addEventListener("contextmenu", (e) => {
    if (pickMode) e.preventDefault();
  });

  v.stage.addEventListener("pointerdown", async (e) => {
    setActiveView(v.id);
    // Pan: middle button, Alt+left, or left when not in pick mode
    const wantPan =
      e.button === 1 || (e.button === 0 && e.altKey) || (e.button === 0 && !pickMode);
    if (wantPan) {
      panning = true;
      panLast = { x: e.clientX, y: e.clientY };
      v.stage.classList.add("panning");
      v.stage.setPointerCapture(e.pointerId);
      e.preventDefault();
      return;
    }
    if (!pickMode) return;
    if (e.button !== 0 && e.button !== 2) return;
    e.preventDefault();
    const uv = stageToImageUV(v, e.clientX, e.clientY);
    if (!uv) return;
    if (e.button === 2) v.bgSeeds.push({ u: uv.u, v: uv.v });
    else v.fgSeeds.push({ u: uv.u, v: uv.v });
    drawViewOverlay(v);
    await runSegmentForView(v);
  });

  v.stage.addEventListener("pointermove", (e) => {
    if (!panning || !panLast) return;
    const dx = e.clientX - panLast.x;
    const dy = e.clientY - panLast.y;
    panLast = { x: e.clientX, y: e.clientY };
    v.panX += dx;
    v.panY += dy;
    layoutView(v);
  });

  const endPan = (e) => {
    if (!panning) return;
    panning = false;
    panLast = null;
    v.stage.classList.remove("panning");
    try {
      v.stage.releasePointerCapture(e.pointerId);
    } catch (_) {}
  };
  v.stage.addEventListener("pointerup", endPan);
  v.stage.addEventListener("pointercancel", endPan);
}

/* ---------- Segmentation + multi-view intersection ---------- */

function setFitButtonsEnabled(on) {
  $("btnAABB").disabled = !on;
  $("btnOBB").disabled = !on;
}

function rebuildPointsGeometry(positions, colors) {
  if (!pointsCloud) return;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  pointsCloud.geometry.dispose();
  pointsCloud.geometry = geo;
  const n = positions.length / 3;
  const base = Math.max(sceneData.extent / 500, 1e-4);
  // Slightly larger when showing filtered FG subset so points stay visible
  pointsCloud.material.size = n < nAll * 0.5 ? base * 1.6 : base;
}

function restoreFullCloud() {
  if (!allPositions) return;
  rebuildPointsGeometry(allPositions, allColors);
  fgPositions = null;
  setFitButtonsEnabled(false);
  setStatus(`已恢复全部点云 (${nAll})`, true);
}

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (hi - i) + sorted[hi] * (i - lo);
}

/**
 * Classify each point vs one view mask:
 * 1 = FG, 0 = BG (in image, not FG), -1 = invisible (behind / out of frame)
 */
function classifyPointsForView(v) {
  const labels = new Int8Array(nAll);
  labels.fill(-1);
  if (!v.maskU8) return labels;
  const fr = sceneData.frames[v.frameIndex];
  const K = sceneData.K;
  const w2c = fr.w2c;
  const mw = v.maskW;
  const mh = v.maskH;
  for (let i = 0; i < nAll; i++) {
    const X = [allPositions[i * 3], allPositions[i * 3 + 1], allPositions[i * 3 + 2]];
    const uvz = projectPoint(K, w2c, X);
    if (!uvz) continue;
    const u = Math.round((uvz[0] / sceneData.width) * mw);
    const vv = Math.round((uvz[1] / sceneData.height) * mh);
    if (u < 0 || vv < 0 || u >= mw || vv >= mh) continue;
    labels[i] = v.maskU8[vv * mw + u] >= 128 ? 1 : 0;
  }
  return labels;
}

/**
 * Multi-view carving (visual hull style):
 * - FG∩: in every masked view where the point is visible, it must be FG
 * - BG∩: in every masked view where visible, it is BG (reported; excluded from keep)
 */
function applyMultiViewFilter() {
  const masked = views.filter((v) => v.maskU8);
  if (!masked.length) {
    restoreFullCloud();
    return;
  }

  const labelSets = masked.map(classifyPointsForView);
  const keep = [];
  let nFgInter = 0;
  let nBgInter = 0;

  for (let i = 0; i < nAll; i++) {
    let visibleCount = 0;
    let fgCount = 0;
    let bgCount = 0;
    for (const lab of labelSets) {
      const L = lab[i];
      if (L < 0) continue;
      visibleCount++;
      if (L === 1) fgCount++;
      else bgCount++;
    }
    if (visibleCount === 0) continue;
    const isFgInter = fgCount === visibleCount; // all visible views say FG
    const isBgInter = bgCount === visibleCount; // all visible views say BG
    if (isBgInter) nBgInter++;
    if (isFgInter) {
      nFgInter++;
      keep.push(i);
    }
  }

  if (keep.length < 20) {
    setStatus(
      `多视图 FG∩ 点太少 (${keep.length})；已标注 ${masked.length} 视图。请换互补视角或少用遮挡严重的帧`,
      false
    );
    return;
  }

  // Spatial outlier trim
  let cx = 0,
    cy = 0,
    cz = 0;
  for (const i of keep) {
    cx += allPositions[i * 3];
    cy += allPositions[i * 3 + 1];
    cz += allPositions[i * 3 + 2];
  }
  cx /= keep.length;
  cy /= keep.length;
  cz /= keep.length;
  const dists = keep.map((i) => {
    const dx = allPositions[i * 3] - cx;
    const dy = allPositions[i * 3 + 1] - cy;
    const dz = allPositions[i * 3 + 2] - cz;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  });
  const ds = dists.slice().sort((a, b) => a - b);
  const dCut = Math.max(percentile(ds, 0.5) * 2.8, percentile(ds, 0.9));
  const keep2 = keep.filter((_, k) => dists[k] <= dCut);

  const m = keep2.length;
  const pos = new Float32Array(m * 3);
  const col = new Float32Array(m * 3);
  for (let k = 0; k < m; k++) {
    const i = keep2[k];
    pos[k * 3] = allPositions[i * 3];
    pos[k * 3 + 1] = allPositions[i * 3 + 1];
    pos[k * 3 + 2] = allPositions[i * 3 + 2];
    col[k * 3] = allColors[i * 3];
    col[k * 3 + 1] = allColors[i * 3 + 1];
    col[k * 3 + 2] = allColors[i * 3 + 2];
  }
  fgPositions = pos;
  rebuildPointsGeometry(pos, col);
  setFitButtonsEnabled(true);
  focusOnPoint(new THREE.Vector3(cx, cy, cz), 0.18);
  setStatus(
    `多视图 FG∩ ${m} 点（标注 ${masked.length} 视图）· BG∩ ${nBgInter} · 可 OBB/AABB`,
    true
  );
  for (const v of views) drawViewOverlay(v);
}

async function decodeMaskPngB64(b64) {
  const url = "data:image/png;base64," + b64;
  const img = new Image();
  await new Promise((res, rej) => {
    img.onload = res;
    img.onerror = rej;
    img.src = url;
  });
  const tint = document.createElement("canvas");
  tint.width = img.width;
  tint.height = img.height;
  const tctx = tint.getContext("2d", { willReadFrequently: true });
  tctx.drawImage(img, 0, 0);
  const id = tctx.getImageData(0, 0, tint.width, tint.height);
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i] > 128) {
      d[i] = 34;
      d[i + 1] = 197;
      d[i + 2] = 94;
      d[i + 3] = 220;
    } else {
      d[i + 3] = 0;
    }
  }
  tctx.putImageData(id, 0, 0);

  const raw = document.createElement("canvas");
  raw.width = img.width;
  raw.height = img.height;
  const rctx = raw.getContext("2d", { willReadFrequently: true });
  rctx.drawImage(img, 0, 0);
  const rawData = rctx.getImageData(0, 0, raw.width, raw.height).data;
  const maskU8 = new Uint8Array(raw.width * raw.height);
  for (let i = 0; i < maskU8.length; i++) maskU8[i] = rawData[i * 4];
  return { maskU8, tint, w: img.width, h: img.height };
}

async function runSegmentForView(v) {
  if (!v.fgSeeds.length || segmentBusy) return;
  segmentBusy = true;
  setStatus(`视图#${v.id} SAM2… FG${v.fgSeeds.length}/BG${v.bgSeeds.length}`);
  try {
    const res = await fetch("/api/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame_index: v.frameIndex,
        fg: v.fgSeeds.map((s) => [s.u, s.v]),
        bg: v.bgSeeds.map((s) => [s.u, s.v]),
      }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "segment failed");
    const decoded = await decodeMaskPngB64(data.mask_png_b64);
    v.maskU8 = decoded.maskU8;
    v.maskTint = decoded.tint;
    v.maskW = decoded.w;
    v.maskH = decoded.h;
    drawViewOverlay(v);
    applyMultiViewFilter();
    const ms = data.infer_s != null ? ` · ${data.infer_s}s` : "";
    const np = data.n_fg != null ? ` · FG${data.n_fg}/BG${data.n_bg}` : "";
    const dumped = data.mask_path ? ` · 已存 ${data.mask_path}` : "";
    setStatus(`SAM2 完成${ms}${np}${dumped} · ${data.fg_pixels ?? "?"} px · FG∩ 已更新`, true);
  } catch (e) {
    setStatus("SAM2 分割失败: " + e.message);
  } finally {
    segmentBusy = false;
  }
}

function setPickMode(on) {
  pickMode = on;
  if (on && scalePickMode) {
    scalePickMode = null;
    updateScalePickButtons();
  }
  $("btnPickFg").classList.toggle("pick-on", on);
  $("btnPickFg").textContent = on ? "退出点选模式" : "提取前景（点选模式）";
  for (const v of views) {
    v.stage.classList.toggle("pick", on);
    drawViewOverlay(v);
  }
  if (on) {
    setStatus("点选中：在各 2D 视图左键=前景、右键=背景；多视图自动做 FG∩", true);
  }
}

/* ---------- Box / AABB / OBB ---------- */

function makeBox(frame) {
  if (boxMesh) {
    scene.remove(boxMesh);
    if (transform) transform.detach();
  }
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x3b82f6,
    wireframe: true,
    transparent: true,
    opacity: 0.95,
  });
  boxMesh = new THREE.Mesh(geo, mat);
  boxMesh.position.set(...frame.center);
  if (frame.quaternion_wxyz && frame.quaternion_wxyz.length === 4) {
    const [qw, qx, qy, qz] = frame.quaternion_wxyz;
    boxMesh.quaternion.set(qx, qy, qz, qw);
    boxMesh.rotation.setFromQuaternion(boxMesh.quaternion, "XYZ");
  } else {
    boxMesh.rotation.set(
      THREE.MathUtils.degToRad(frame.euler_deg[0]),
      THREE.MathUtils.degToRad(frame.euler_deg[1]),
      THREE.MathUtils.degToRad(frame.euler_deg[2]),
      "XYZ"
    );
    boxMesh.quaternion.setFromEuler(boxMesh.rotation);
  }
  boxMesh.scale.set(...frame.size);
  // RGB axes = local XYZ (aligned with box faces)
  const axes = new THREE.AxesHelper(0.55);
  axes.renderOrder = 2;
  boxMesh.add(axes);
  scene.add(boxMesh);

  if (!transform) {
    transform = new TransformControls(camera, canvas);
    transform.setSize(1.2);
    transform.setSpace("local");
    scene.add(transform.getHelper());
    transform.addEventListener("dragging-changed", (e) => {
      controls.enabled = !e.value;
    });
    transform.addEventListener("objectChange", () => {
      poseEditSource = "box";
      writeUIFromBox();
    });
    transform.addEventListener("change", () => {
      if (transform.dragging) {
        poseEditSource = "box";
        writeUIFromBox();
      }
    });
  }
  transform.attach(boxMesh);
  transform.setSpace("local");
  setGizmoMode("translate");
  writeUIFromBox();
  updateAlignPlaneUI();
}

function applyBoxPose(center, eulerDeg, size, quaternion_wxyz = null) {
  if (!boxMesh) return;
  boxMesh.position.set(center[0], center[1], center[2]);
  if (quaternion_wxyz && quaternion_wxyz.length === 4) {
    const [qw, qx, qy, qz] = quaternion_wxyz;
    boxMesh.quaternion.set(qx, qy, qz, qw);
    boxMesh.rotation.setFromQuaternion(boxMesh.quaternion, "XYZ");
  } else {
    boxMesh.rotation.set(
      THREE.MathUtils.degToRad(eulerDeg[0]),
      THREE.MathUtils.degToRad(eulerDeg[1]),
      THREE.MathUtils.degToRad(eulerDeg[2]),
      "XYZ"
    );
    boxMesh.quaternion.setFromEuler(boxMesh.rotation);
  }
  boxMesh.scale.set(size[0], size[1], size[2]);
  boxMesh.updateMatrixWorld(true);
  writeUIFromBox();
  focusOnPoint(boxMesh.position.clone(), 0.2);
}

function snapshotBoxPose() {
  // Capture current 3D box (don't pull stale UI over it)
  writeUIFromBox();
  const f = readFrameFromBox();
  return {
    center: [...f.center],
    euler_deg: [...f.euler_deg],
    size: [...f.size],
    quaternion_wxyz: f.quaternion_wxyz ? [...f.quaternion_wxyz] : null,
  };
}

function updateAlignPlaneUI() {
  const hasPlane = !!(planeCentroid && planeNormal);
  const hasBox = !!boxMesh;
  const btn = $("btnAlignPlane");
  const undo = $("btnUndoAlign");
  const flip = $("btnFlipZ");
  const yaw = $("btnApplyYaw");
  const hint = $("alignPlaneHint");
  if (btn) btn.disabled = !(hasPlane && hasBox);
  if (undo) undo.disabled = !alignPoseBackup;
  if (flip) flip.disabled = !hasBox;
  if (yaw) yaw.disabled = !hasBox;
  if (hint) {
    hint.textContent = hasPlane
      ? "贴齐：+Z∥桌面法向，−Z 面贴桌；可翻转 Z / 绕 Z 旋转（可撤销贴齐）"
      : "需先拟合桌面平面（或从上次尺度标定恢复）";
  }
}

/** Local axis direction in world (robust to non-uniform scale). */
function boxLocalAxisWorld(axisIndex) {
  const v = new THREE.Vector3(
    axisIndex === 0 ? 1 : 0,
    axisIndex === 1 ? 1 : 0,
    axisIndex === 2 ? 1 : 0
  );
  return v.applyQuaternion(boxMesh.quaternion).normalize();
}

/** Snap −Z face onto plane (assumes local +Z already // ±n). */
function snapBoxBottomToPlane() {
  if (!boxMesh || !planeCentroid || !planeNormal) return false;
  const n = planeNormal.clone().normalize();
  const C = planePoint();
  if (!C) return false;
  const half = Math.abs(boxMesh.scale.z) * 0.5;
  const t = boxMesh.position.clone();
  // Face whose outward normal is −n sits on the table
  const gap = n.dot(new THREE.Vector3().subVectors(t, C)) - half;
  t.addScaledVector(n, -gap);
  boxMesh.position.copy(t);
  boxMesh.updateMatrixWorld(true);
  return true;
}

/**
 * Align box so local +Z // plane normal, −Z face on plane.
 * Preserves twist around Z as much as setFromUnitVectors allows.
 */
function alignBoxToPlane() {
  if (!boxMesh || !planeCentroid || !planeNormal) {
    return setStatus("请先拟合桌面平面");
  }
  // Detach gizmo so TransformControls cannot overwrite pose mid-update
  const wasAttached = transform && transform.object === boxMesh;
  if (wasAttached) transform.detach();

  const n = planeNormal.clone().normalize();
  boxMesh.updateMatrixWorld(true);
  const zWorld = boxLocalAxisWorld(2);
  let from = zWorld.clone();
  let to = n.clone();
  if (from.dot(to) < -0.999) {
    const qFlip = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
    boxMesh.quaternion.multiply(qFlip);
    from = boxLocalAxisWorld(2);
  }
  const qAlign = new THREE.Quaternion().setFromUnitVectors(from, to);
  const angBefore = THREE.MathUtils.radToDeg(
    Math.acos(Math.min(1, Math.max(-1, zWorld.dot(n))))
  );

  alignPoseBackup = snapshotBoxPose();
  updateAlignPlaneUI();

  boxMesh.quaternion.premultiply(qAlign);
  boxMesh.rotation.setFromQuaternion(boxMesh.quaternion, "XYZ");
  boxMesh.updateMatrixWorld(true);
  snapBoxBottomToPlane();
  writeUIFromBox();
  if (wasAttached) transform.attach(boxMesh);
  focusOnPoint(boxMesh.position.clone(), 0.2);

  const zAfter = boxLocalAxisWorld(2);
  const angAfter = THREE.MathUtils.radToDeg(
    Math.acos(Math.min(1, Math.max(-1, Math.abs(zAfter.dot(n)))))
  );
  if (angAfter > 2) {
    setStatus(
      `贴齐异常：对齐后 +Z 与法向仍差 ${angAfter.toFixed(1)}°（请检查平面）`,
      false
    );
    return;
  }
  const f = readFrameFromBox();
  setStatus(
    `已贴齐平面 · +Z∥法向 · 原夹角 ${angBefore.toFixed(1)}° → ${angAfter.toFixed(1)}°` +
      ` · euler≈[${f.euler_deg.map((v) => v.toFixed(1)).join(", ")}]`,
    true
  );
  persistPlane();
}

/** Degrees between local +Z and plane normal; null if no plane/box. */
function planeAlignErrorDeg() {
  if (!boxMesh || !planeNormal) return null;
  const n = planeNormal.clone().normalize();
  const z = boxLocalAxisWorld(2);
  return THREE.MathUtils.radToDeg(
    Math.acos(Math.min(1, Math.max(-1, Math.abs(z.dot(n)))))
  );
}

/** Flip local +Z ↔ −Z (180° about local X), then re-snap to plane if available. */
function flipBoxZ() {
  if (!boxMesh) return setStatus("尚无包围盒");
  if (!alignPoseBackup) alignPoseBackup = snapshotBoxPose();
  const q = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(1, 0, 0),
    Math.PI
  );
  boxMesh.quaternion.multiply(q);
  boxMesh.rotation.setFromQuaternion(boxMesh.quaternion, "XYZ");
  boxMesh.updateMatrixWorld(true);
  if (planeNormal) snapBoxBottomToPlane();
  writeUIFromBox();
  updateAlignPlaneUI();
  setStatus("已翻转 Z（局部 +Z 反向）" + (planeNormal ? " · 已重新贴桌" : ""), true);
}

/** Apply relative yaw about local Z (degrees). */
function applyLocalYaw(deg) {
  if (!boxMesh) return setStatus("尚无包围盒");
  const d = Number(deg);
  if (!Number.isFinite(d) || d === 0) return setStatus("请输入非零角度");
  const wasAttached = transform && transform.object === boxMesh;
  if (wasAttached) transform.detach();
  const q = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 0, 1),
    THREE.MathUtils.degToRad(d)
  );
  boxMesh.quaternion.multiply(q);
  boxMesh.rotation.setFromQuaternion(boxMesh.quaternion, "XYZ");
  boxMesh.updateMatrixWorld(true);
  writeUIFromBox();
  if (wasAttached) transform.attach(boxMesh);
  const alignErr = planeAlignErrorDeg();
  const tip =
    alignErr != null
      ? ` · 当前|Z∠法向|=${alignErr.toFixed(1)}°（保存后看 yolo6d/preview_6d.mp4 · preview_mask.mp4）`
      : " · 保存后看 yolo6d/preview_6d.mp4 · preview_mask.mp4";
  setStatus(`已绕局部 Z 旋转 ${d}°${tip}`, true);
}

function toggleGizmoSpace() {
  if (!transform) return;
  const next = transform.space === "local" ? "world" : "local";
  transform.setSpace(next);
  const btn = $("btnGizmoLocal");
  if (btn) {
    btn.textContent = next === "local" ? "Gizmo: 局部" : "Gizmo: 世界";
    btn.classList.toggle("active", next === "local");
  }
  setStatus(`Gizmo 空间 → ${next === "local" ? "局部（与框面法向对齐）" : "世界"}`, true);
}

function undoAlignBoxToPlane() {
  if (!alignPoseBackup) return setStatus("没有可撤销的贴齐");
  const b = alignPoseBackup;
  alignPoseBackup = null;
  applyBoxPose(b.center, b.euler_deg, b.size, b.quaternion_wxyz);
  updateAlignPlaneUI();
  setStatus("已撤销贴齐", true);
}

function persistPlane() {
  if (!planeCentroid || !planeNormal) return;
  const payload = {
    centroid: planeCentroid.toArray(),
    normal: planeNormal.toArray(),
    offset: planeOffset,
    samples: planeSamples.map((p) => p.toArray()),
  };
  fetch("/api/save_plane", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).catch(() => {});
}

function setPlaneFromData(centroid, normal, offset = 0, samples = []) {
  planeCentroid = new THREE.Vector3(centroid[0], centroid[1], centroid[2]);
  planeNormal = new THREE.Vector3(normal[0], normal[1], normal[2]).normalize();
  planeOffset = Number(offset) || 0;
  if ($("planeOffset")) $("planeOffset").value = String(planeOffset);
  ensureScaleHelpers();
  if (planeSampleGroup) {
    while (planeSampleGroup.children.length) {
      const c = planeSampleGroup.children.pop();
      c.geometry?.dispose();
      c.material?.dispose();
    }
  }
  planeSamples = [];
  for (const s of samples || []) {
    const p = new THREE.Vector3(s[0], s[1], s[2]);
    planeSamples.push(p);
    if (planeSampleGroup) {
      const m = new THREE.Mesh(
        new THREE.SphereGeometry(markerRadius(0.003), 12, 12),
        new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
      );
      m.position.copy(p);
      planeSampleGroup.add(m);
    }
  }
  $("btnFitPlane").disabled = planeSamples.length < 3;
  $("btnPickTarget").disabled = false;
  updatePlaneHelper();
  refreshScalePreview();
  updateAlignPlaneUI();
}

async function tryRestorePlane() {
  // Prefer plane.json (current world units); fallback scale.json meta
  try {
    const pl = await fetch("/plane.json").then((r) => (r.ok ? r.json() : null));
    if (pl?.centroid && pl?.normal) {
      setPlaneFromData(pl.centroid, pl.normal, pl.offset || 0, pl.samples || []);
      setStatus("已恢复上次标定平面", true);
      return;
    }
  } catch (_) {}
  try {
    const sc = await fetch("/scale.json").then((r) => (r.ok ? r.json() : null));
    const meta = sc?.meta;
    if (!meta?.plane_centroid || !meta?.plane_normal) return;
    let c = meta.plane_centroid.map(Number);
    let off = Number(meta.plane_offset || 0);
    const n = meta.plane_normal.map(Number);
    const ms = Number(sc.metric_scale_cumulative || sceneData?.metric_scale || 1);
    if (ms && ms !== 1 && sceneData?.center) {
      const scenter = sceneData.center;
      const ext = sceneData.extent || 1;
      const d0 = Math.hypot(c[0] - scenter[0], c[1] - scenter[1], c[2] - scenter[2]);
      const c1 = c.map((v) => v * ms);
      const d1 = Math.hypot(c1[0] - scenter[0], c1[1] - scenter[1], c1[2] - scenter[2]);
      if (d1 < d0 && d1 < ext * 2) {
        c = c1;
        off *= ms;
      }
    }
    setPlaneFromData(c, n, off, []);
    persistPlane();
    setStatus("已从尺度标定恢复桌面平面", true);
  } catch (_) {}
}

function fitAABB() {
  const pos = fgPositions || allPositions;
  if (!pos || pos.length < 9) return setStatus("没有可拟合的点");
  const n = pos.length / 3;
  const xs = new Float32Array(n);
  const ys = new Float32Array(n);
  const zs = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = pos[i * 3];
    ys[i] = pos[i * 3 + 1];
    zs[i] = pos[i * 3 + 2];
  }
  xs.sort();
  ys.sort();
  zs.sort();
  const p = 0.02;
  const minX = percentile([...xs], p);
  const maxX = percentile([...xs], 1 - p);
  const minY = percentile([...ys], p);
  const maxY = percentile([...ys], 1 - p);
  const minZ = percentile([...zs], p);
  const maxZ = percentile([...zs], 1 - p);
  const pad = 1.02;
  const size = [(maxX - minX) * pad, (maxY - minY) * pad, (maxZ - minZ) * pad];
  const center = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
  applyBoxPose(center, [0, 0, 0], size);
  setStatus(`AABB 已拟合 · size=${size.map((v) => v.toFixed(3)).join(", ")}`, true);
}

function fitOBB() {
  const pos = fgPositions || allPositions;
  if (!pos || pos.length < 30) return setStatus("前景点太少，无法 OBB");
  const n = pos.length / 3;
  let mx = 0,
    my = 0,
    mz = 0;
  for (let i = 0; i < n; i++) {
    mx += pos[i * 3];
    my += pos[i * 3 + 1];
    mz += pos[i * 3 + 2];
  }
  mx /= n;
  my /= n;
  mz /= n;

  let cxx = 0,
    cxy = 0,
    cxz = 0,
    cyy = 0,
    cyz = 0,
    czz = 0;
  for (let i = 0; i < n; i++) {
    const dx = pos[i * 3] - mx;
    const dy = pos[i * 3 + 1] - my;
    const dz = pos[i * 3 + 2] - mz;
    cxx += dx * dx;
    cxy += dx * dy;
    cxz += dx * dz;
    cyy += dy * dy;
    cyz += dy * dz;
    czz += dz * dz;
  }
  const inv = 1 / Math.max(n - 1, 1);
  const cov = [
    [cxx * inv, cxy * inv, cxz * inv],
    [cxy * inv, cyy * inv, cyz * inv],
    [cxz * inv, cyz * inv, czz * inv],
  ];
  const { values, vectors } = eigenSymmetric3(cov);
  const order = [0, 1, 2].sort((a, b) => values[b] - values[a]);
  let axes = orthonormalizeRH(order.map((i) => vectors[i]));

  const locals = new Float32Array(n * 3);
  const minL = [Infinity, Infinity, Infinity];
  const maxL = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    const dx = pos[i * 3] - mx;
    const dy = pos[i * 3 + 1] - my;
    const dz = pos[i * 3 + 2] - mz;
    for (let a = 0; a < 3; a++) {
      const t = dx * axes[a][0] + dy * axes[a][1] + dz * axes[a][2];
      locals[i * 3 + a] = t;
      if (t < minL[a]) minL[a] = t;
      if (t > maxL[a]) maxL[a] = t;
    }
  }
  for (let a = 0; a < 3; a++) {
    const col = new Float32Array(n);
    for (let i = 0; i < n; i++) col[i] = locals[i * 3 + a];
    col.sort();
    minL[a] = percentile([...col], 0.02);
    maxL[a] = percentile([...col], 0.98);
  }
  const size = [
    Math.max(1e-4, (maxL[0] - minL[0]) * 1.02),
    Math.max(1e-4, (maxL[1] - minL[1]) * 1.02),
    Math.max(1e-4, (maxL[2] - minL[2]) * 1.02),
  ];
  const mid = [(minL[0] + maxL[0]) / 2, (minL[1] + maxL[1]) / 2, (minL[2] + maxL[2]) / 2];
  const center = [
    mx + axes[0][0] * mid[0] + axes[1][0] * mid[1] + axes[2][0] * mid[2],
    my + axes[0][1] * mid[0] + axes[1][1] * mid[1] + axes[2][1] * mid[2],
    mz + axes[0][2] * mid[0] + axes[1][2] * mid[1] + axes[2][2] * mid[2],
  ];
  const m = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(...axes[0]),
    new THREE.Vector3(...axes[1]),
    new THREE.Vector3(...axes[2])
  );
  const e = new THREE.Euler().setFromRotationMatrix(m, "XYZ");
  const eulerDeg = [
    THREE.MathUtils.radToDeg(e.x),
    THREE.MathUtils.radToDeg(e.y),
    THREE.MathUtils.radToDeg(e.z),
  ];
  applyBoxPose(center, eulerDeg, size);
  setStatus(`OBB 已拟合 · size=${size.map((v) => v.toFixed(3)).join(", ")}`, true);
}

function orthonormalizeRH(axes) {
  const a0 = new THREE.Vector3(...axes[0]).normalize();
  let a1 = new THREE.Vector3(...axes[1]);
  a1.addScaledVector(a0, -a1.dot(a0)).normalize();
  const a2 = new THREE.Vector3().crossVectors(a0, a1).normalize();
  a1 = new THREE.Vector3().crossVectors(a2, a0).normalize();
  return [
    [a0.x, a0.y, a0.z],
    [a1.x, a1.y, a1.z],
    [a2.x, a2.y, a2.z],
  ];
}

function eigenSymmetric3(A) {
  const vectors = [];
  const values = [];
  let M = A.map((r) => r.slice());
  const seeds = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let k = 0; k < 3; k++) {
    let v = seeds[k].slice();
    let norm = Math.hypot(v[0], v[1], v[2]) || 1;
    v = v.map((x) => x / norm);
    let val = 0;
    for (let it = 0; it < 40; it++) {
      let w = [
        M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
        M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
        M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
      ];
      for (const u of vectors) {
        const dot = w[0] * u[0] + w[1] * u[1] + w[2] * u[2];
        w[0] -= dot * u[0];
        w[1] -= dot * u[1];
        w[2] -= dot * u[2];
      }
      norm = Math.hypot(w[0], w[1], w[2]) || 1e-12;
      v = w.map((x) => x / norm);
      val =
        v[0] * (M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2]) +
        v[1] * (M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2]) +
        v[2] * (M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]);
    }
    vectors.push(v);
    values.push(val);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) M[i][j] -= val * v[i] * v[j];
  }
  return { values, vectors };
}

async function loadPointsBin(url) {
  const buf = await fetch(url).then((r) => r.arrayBuffer());
  const view = new DataView(buf);
  const n = view.getUint32(0, true);
  const xyz = new Float32Array(buf, 4, n * 3);
  const rgb = new Uint8Array(buf, 4 + n * 12, n * 3);
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    positions[i * 3] = xyz[i * 3];
    positions[i * 3 + 1] = xyz[i * 3 + 1];
    positions[i * 3 + 2] = xyz[i * 3 + 2];
    colors[i * 3] = rgb[i * 3] / 255;
    colors[i * 3 + 1] = rgb[i * 3 + 1] / 255;
    colors[i * 3 + 2] = rgb[i * 3 + 2] / 255;
  }
  allPositions = positions;
  allColors = colors;
  nAll = n;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions.slice(), 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors.slice(), 3));
  const mat = new THREE.PointsMaterial({ size: 0.015, vertexColors: true, sizeAttenuation: true });
  return new THREE.Points(geo, mat);
}

async function loadScene() {
  setStatus("加载 scene.json …");
  sceneData = await fetch("/scene.json").then((r) => r.json());

  let frame = sceneData.object_frame_default;
  try {
    const saved = await fetch("/object_frame.json").then((r) => (r.ok ? r.json() : null));
    if (saved && saved.center) frame = saved;
  } catch (_) {}

  $("classId").value = frame.class_id ?? 0;
  $("className").value = frame.class_name ?? "object";

  setStatus(`加载点云 (${sceneData.n_points} pts)…`);
  pointsCloud = await loadPointsBin("/" + sceneData.points_url);
  // World-space point diameter ~ extent/500; no absolute floor (breaks after metric scale)
  pointsCloud.material.size = Math.max(sceneData.extent / 500, 1e-4);
  scene.add(pointsCloud);

  const c = sceneData.center;
  const ext = sceneData.extent;
  pivotMarker.scale.setScalar(Math.max(ext * 0.008, 1e-4));
  camera.position.set(c[0] + ext * 0.6, c[1] + ext * 0.4, c[2] + ext * 0.6);
  setPivot(c[0], c[1], c[2], false);
  raycaster.params.Points.threshold = Math.max(ext * 0.004, 1e-4);

  makeBox(frame);
  focusOnPoint(boxMesh.position.clone(), 0.25);

  // Start with 2 complementary views (first + mid frame)
  const mid = Math.floor(sceneData.frames.length / 2);
  createView(0);
  createView(mid);
  resize();
  ensureScaleHelpers();
  // default lock: on if already metric, else off for first conversion
  if (typeof sceneData.metric_locked === "boolean") {
    metricLocked = sceneData.metric_locked;
  } else {
    metricLocked = !!(sceneData.metric_scale && sceneData.metric_scale !== 1);
  }
  updateScaleLockUI();
  updateAlignPlaneUI();
  await tryRestorePlane();
  const ms = sceneData.metric_scale;
  const msTip =
    ms && ms !== 1
      ? ` · metric_scale=${Number(ms).toFixed(4)} · 锁=${metricLocked ? "开" : "关"}`
      : " · 未米制化（建议解锁后首次定标）";
  const planeTip = planeNormal ? " · 平面已恢复" : "";
  setStatus(
    `就绪 · ${sceneData.n_points} 点 · ${sceneData.frames.length} 帧 · 已开 2 个 2D 视图${msTip}${planeTip}`,
    true
  );
}

function setGizmoMode(mode) {
  if (!transform) return setStatus("Gizmo 尚未就绪");
  transform.setMode(mode);
  const map = { translate: "btnTranslate", rotate: "btnRotate", scale: "btnScale" };
  for (const [m, id] of Object.entries(map)) $(id)?.classList.toggle("active", m === mode);
  const tip = { translate: "拖箭头", rotate: "拖圆环", scale: "拖方块" }[mode];
  setStatus(`Gizmo: ${mode} — ${tip}`, true);
}

$("btnTranslate").addEventListener("click", () => setGizmoMode("translate"));
$("btnRotate").addEventListener("click", () => setGizmoMode("rotate"));
$("btnScale").addEventListener("click", () => setGizmoMode("scale"));
$("btnPivotBox").addEventListener("click", () => {
  if (boxMesh) focusOnPoint(boxMesh.position.clone(), 0.25);
});
$("btnPivotScene").addEventListener("click", () => {
  if (!sceneData) return;
  const c = sceneData.center;
  focusOnPoint(new THREE.Vector3(c[0], c[1], c[2]), 0.55);
});

/* ---------- Metric scale: plane + known height ---------- */

function ensureScaleHelpers() {
  if (!planeSampleGroup) {
    planeSampleGroup = new THREE.Group();
    scene.add(planeSampleGroup);
  }
  if (!targetMarkerMesh) {
    targetMarkerMesh = new THREE.Mesh(
      new THREE.SphereGeometry(1, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0x22c55e })
    );
    targetMarkerMesh.visible = false;
    scene.add(targetMarkerMesh);
  }
}

function clearPlaneSamples() {
  planeSamples = [];
  planeCentroid = null;
  planeNormal = null;
  planeOffset = 0;
  lastScalePreview = null;
  if ($("planeOffset")) $("planeOffset").value = "0";
  if (planeSampleGroup) {
    while (planeSampleGroup.children.length) {
      const c = planeSampleGroup.children.pop();
      c.geometry?.dispose();
      c.material?.dispose();
    }
  }
  if (planeHelper) {
    scene.remove(planeHelper);
    planeHelper.geometry?.dispose();
    planeHelper.material?.dispose();
    planeHelper = null;
  }
  $("btnFitPlane").disabled = true;
  $("btnPickTarget").disabled = true;
  $("btnApplyScale").disabled = true;
  $("scalePreview").textContent = "先选 ≥3 个桌面点拟合平面";
  updateScalePickButtons();
  updateAlignPlaneUI();
  fetch("/api/save_plane", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clear: true }),
  }).catch(() => {});
}

function setScalePickMode(mode) {
  // exit FG pick when entering scale pick
  if (mode && pickMode) setPickMode(false);
  scalePickMode = scalePickMode === mode ? null : mode;
  updateScalePickButtons();
  if (scalePickMode === "plane") {
    setStatus("平面选点中：单击吸附到点云已有顶点（桌面 ≥3 点）", true);
  } else if (scalePickMode === "target") {
    setStatus("目标点选点中：单击吸附到点云已有顶点（测高点）", true);
  } else {
    setStatus("已退出尺度选点", true);
  }
}

function updateScaleLockUI() {
  const btn = $("btnScaleLock");
  const apply = $("btnApplyScale");
  const hint = $("scaleLockHint");
  if (!btn) return;
  btn.classList.toggle("pick-on", metricLocked);
  btn.textContent = metricLocked ? "尺度锁定：开（仅微调）" : "尺度锁定：关（可改单位）";
  if (hint) {
    hint.textContent = metricLocked
      ? "锁定中：单位固定为米；应用 = 按真实距离做数值校正（× real/d_sfm）"
      : "已解锁：应用 = 改单位/整场景缩放（首次 SfM→米，或你明确要重定标）";
  }
  if (apply) {
    apply.textContent = metricLocked
      ? "米制微调并导出 YOLO6D（新建）"
      : "应用尺度（改单位）并导出 YOLO6D（新建）";
  }
  refreshScalePreview();
}

async function setScaleLock(locked) {
  metricLocked = !!locked;
  updateScaleLockUI();
  try {
    const res = await fetch("/api/set_scale_lock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ locked: metricLocked }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "lock failed");
    if (sceneData) {
      sceneData.metric_locked = data.metric_locked;
      sceneData.metric_scale = data.metric_scale;
    }
    setStatus(
      metricLocked
        ? "尺度已锁定：单位=米，再次应用只微调数值"
        : "尺度已解锁：再次应用将改单位/整场景缩放（请确认）",
      true
    );
  } catch (e) {
    setStatus("更新尺度锁失败: " + e.message);
  }
}

function updateScalePickButtons() {
  const bp = $("btnPickPlane");
  const bt = $("btnPickTarget");
  if (bp) {
    bp.classList.toggle("pick-on", scalePickMode === "plane");
    bp.textContent = scalePickMode === "plane" ? "退出选平面点" : "选平面点";
  }
  if (bt) {
    bt.classList.toggle("pick-on", scalePickMode === "target");
    bt.textContent = scalePickMode === "target" ? "退出选目标点" : "选目标点";
  }
}

function markerRadius(frac = 0.004, minAbs = 1e-4) {
  const ext = sceneData?.extent || 1;
  return Math.max(ext * frac, minAbs);
}

function addPlaneSample(p) {
  ensureScaleHelpers();
  planeSamples.push(p.clone());
  const r = markerRadius(0.003);
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(r, 12, 12),
    new THREE.MeshBasicMaterial({ color: 0xf59e0b })
  );
  m.position.copy(p);
  planeSampleGroup.add(m);
  $("btnFitPlane").disabled = planeSamples.length < 3;
  setStatus(
    `平面样本 ${planeSamples.length} · 已吸附点云顶点 (${p.x.toFixed(4)}, ${p.y.toFixed(4)}, ${p.z.toFixed(4)})`,
    true
  );
  if (planeSamples.length >= 3 && planeNormal) {
    // re-fit live if already fitted
    fitPlaneFromSamples(false);
  }
}

function setTargetFromHit(p) {
  ensureScaleHelpers();
  targetPoint = p.clone();
  targetMarkerMesh.scale.setScalar(markerRadius(0.004));
  targetMarkerMesh.position.copy(p);
  targetMarkerMesh.visible = true;
  refreshScalePreview();
  setStatus(
    `目标点 → (${p.x.toFixed(4)}, ${p.y.toFixed(4)}, ${p.z.toFixed(4)})`,
    true
  );
}

function fitPlaneFromSamples(announce = true) {
  if (planeSamples.length < 3) return setStatus("至少需要 3 个平面点");
  let mx = 0,
    my = 0,
    mz = 0;
  const n = planeSamples.length;
  for (const p of planeSamples) {
    mx += p.x;
    my += p.y;
    mz += p.z;
  }
  mx /= n;
  my /= n;
  mz /= n;
  let cxx = 0,
    cxy = 0,
    cxz = 0,
    cyy = 0,
    cyz = 0,
    czz = 0;
  for (const p of planeSamples) {
    const dx = p.x - mx,
      dy = p.y - my,
      dz = p.z - mz;
    cxx += dx * dx;
    cxy += dx * dy;
    cxz += dx * dz;
    cyy += dy * dy;
    cyz += dy * dz;
    czz += dz * dz;
  }
  const inv = 1 / Math.max(n - 1, 1);
  const cov = [
    [cxx * inv, cxy * inv, cxz * inv],
    [cxy * inv, cyy * inv, cyz * inv],
    [cxz * inv, cyz * inv, czz * inv],
  ];
  const { values, vectors } = eigenSymmetric3(cov);
  let imin = 0;
  if (values[1] < values[imin]) imin = 1;
  if (values[2] < values[imin]) imin = 2;
  planeNormal = new THREE.Vector3(...vectors[imin]).normalize();
  planeCentroid = new THREE.Vector3(mx, my, mz);
  // Prefer normal pointing toward scene/camera-ish up from centroid→box
  if (boxMesh) {
    const toBox = boxMesh.position.clone().sub(planeCentroid);
    if (toBox.dot(planeNormal) < 0) planeNormal.negate();
  }
  $("btnPickTarget").disabled = false;
  updatePlaneHelper();
  refreshScalePreview();
  updateAlignPlaneUI();
  persistPlane();
  if (announce) {
    setStatus(
      `平面已拟合 · n=(${planeNormal.x.toFixed(3)}, ${planeNormal.y.toFixed(3)}, ${planeNormal.z.toFixed(3)}) · 可调 offset / 贴齐包围盒 / 选目标点`,
      true
    );
  }
}

function planePoint() {
  if (!planeCentroid || !planeNormal) return null;
  return planeCentroid.clone().addScaledVector(planeNormal, planeOffset);
}

function updatePlaneHelper() {
  if (!planeCentroid || !planeNormal) return;
  const c = planePoint();
  const ext = sceneData?.extent || 5;
  const size = Math.max(ext * 0.35, 0.5);
  if (planeHelper) {
    scene.remove(planeHelper);
    planeHelper.geometry.dispose();
    planeHelper.material.dispose();
  }
  const geo = new THREE.PlaneGeometry(size, size, 1, 1);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.35,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  planeHelper = new THREE.Mesh(geo, mat);
  // orient +Z of plane geometry → normal
  const quat = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 0, 1),
    planeNormal.clone().normalize()
  );
  planeHelper.quaternion.copy(quat);
  planeHelper.position.copy(c);
  scene.add(planeHelper);

  // grid helper edges
  const edges = new THREE.EdgesGeometry(geo);
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color: 0x7dd3fc })
  );
  planeHelper.add(line);
}

function signedDistToPlane(p) {
  const c = planePoint();
  if (!c || !planeNormal) return null;
  return planeNormal.dot(new THREE.Vector3().subVectors(p, c));
}

function refreshScalePreview() {
  const el = $("scalePreview");
  if (!planeNormal || !planeCentroid) {
    el.textContent = `平面样本 ${planeSamples.length}/3+`;
    $("btnApplyScale").disabled = true;
    return;
  }
  const offset = +($("planeOffset")?.value || 0);
  planeOffset = offset;
  updatePlaneHelper();

  let text = `平面点 ${planeSamples.length} · offset=${offset.toFixed(4)} (SfM)`;
  if (!targetPoint) {
    el.textContent = text + " · 请选目标点并输入真实距离";
    $("btnApplyScale").disabled = true;
    lastScalePreview = null;
    return;
  }
  const dSfm = Math.abs(signedDistToPlane(targetPoint));
  const real = +($("realDist")?.value || 0);
  text += `\n目标垂距 d_sfm=${dSfm.toFixed(6)}`;
  if (!(real > 0) || !(dSfm > 1e-9)) {
    el.textContent = text + " · 输入真实距离 (m)";
    $("btnApplyScale").disabled = true;
    lastScalePreview = { d_sfm: dSfm, scale: null, real: real };
    return;
  }
  const scale = real / dSfm;
  lastScalePreview = { d_sfm: dSfm, scale, real };
  if (metricLocked) {
    const pct = ((scale - 1) * 100).toFixed(2);
    el.textContent =
      text +
      `\n真实距离=${real} m → 校正因子 ×${scale.toFixed(6)} (${pct}%)` +
      `\n模式：米制微调（单位不变）` +
      (Math.abs(scale - 1) < 1e-6 ? "\n已一致，应用将跳过几何缩放并仅导出" : "");
  } else {
    el.textContent =
      text +
      `\n真实距离=${real} m → scale=${scale.toFixed(6)}` +
      `\n模式：改单位/整场景缩放（完成后会自动锁定）`;
  }
  $("btnApplyScale").disabled = false;
}

function raycastCloud(ev) {
  if (!pointsCloud) return null;
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(pointsCloud, false);
  if (!hits.length) return null;
  // Among hits in threshold, pick the vertex closest to the ray (not merely
  // nearest along the ray). hit.point itself lies ON the ray — never use it.
  const posAttr = pointsCloud.geometry?.getAttribute("position");
  if (!posAttr) return null;
  let best = null;
  let bestD = Infinity;
  for (const h of hits) {
    if (h.index == null || h.index < 0 || h.index >= posAttr.count) continue;
    const d = h.distanceToRay != null ? h.distanceToRay : h.distance;
    if (d < bestD) {
      bestD = d;
      best = h.index;
    }
  }
  if (best == null) return null;
  return new THREE.Vector3()
    .fromBufferAttribute(posAttr, best)
    .applyMatrix4(pointsCloud.matrixWorld);
}

async function applyScaleAndExport() {
  refreshScalePreview();
  if (!lastScalePreview?.scale) {
    setStatus("尺度未就绪：需要平面、目标点和真实距离");
    return;
  }
  const scale = lastScalePreview.scale;
  const mode = metricLocked ? "refine" : "convert";
  if (!metricLocked && sceneData?.metric_scale && sceneData.metric_scale !== 1) {
    const ok = confirm(
      `当前已有 metric_scale=${sceneData.metric_scale}，解锁状态下将再次整场景缩放（可能叠乘）。确定继续？`
    );
    if (!ok) return;
  }
  if (metricLocked && Math.abs(scale - 1) > 0.25) {
    const ok = confirm(
      `米制微调校正量较大（×${scale.toFixed(4)}）。确认真实距离与选点无误？`
    );
    if (!ok) return;
  }
  const frame = commitPoseForSave();
  setStatus(
    mode === "refine"
      ? `米制微调 ×${scale.toFixed(6)}（annotator+yolo6d 新建）…`
      : `应用尺度（改单位）×${scale.toFixed(6)}（annotator+yolo6d 新建）…`
  );
  const res = await fetch("/api/apply_scale", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scale,
      mode,
      locked: true, // always lock after a successful apply
      object_frame: frame,
      export: true,
      meta: {
        d_sfm: lastScalePreview.d_sfm,
        real_m: lastScalePreview.real,
        plane_centroid: planeCentroid?.toArray(),
        plane_normal: planeNormal?.toArray(),
        plane_offset: planeOffset,
        target: targetPoint?.toArray(),
        n_plane_samples: planeSamples.length,
        mode,
      },
    }),
  });
  const data = await res.json();
  if (!data.ok) {
    setStatus("应用尺度失败: " + (data.error || "unknown"));
    return;
  }
  const skip = data.skipped_noop ? "（几何未变，已一致）" : "";
  const snaps = [];
  if (data.annotator_snap) {
    snaps.push(`annotator→${String(data.annotator_snap).split(/[/\\]/).pop()}`);
  }
  if (data.snap_path) {
    snaps.push(`yolo6d→${String(data.snap_path).split(/[/\\]/).pop()}`);
  }
  setStatus(
    `${mode === "refine" ? "米制微调" : "尺度转换"}完成 ${skip} ×${data.scale_applied}` +
      ` · 累计=${data.metric_scale_cumulative?.toFixed?.(6) ?? data.metric_scale_cumulative}` +
      ` · 已锁定` +
      (data.exported ? ` · 导出 ${data.labels} 标签` : "") +
      (snaps.length ? ` · 快照 ${snaps.join(", ")}` : "") +
      " · 正在重载…",
    true
  );
  setTimeout(() => location.reload(), 600);
}

$("btnAddView").addEventListener("click", () => {
  if (!sceneData) return;
  const used = new Set(views.map((v) => v.frameIndex));
  let idx = 0;
  for (let i = 0; i < sceneData.frames.length; i++) {
    if (!used.has(i)) {
      idx = i;
      break;
    }
  }
  createView(idx);
  resize();
  setStatus(`已添加 2D 视图（共 ${views.length}）· 选不同视角点选前景`, true);
});
$("btnRemoveView").addEventListener("click", () => removeActiveView());
$("btnPickFg").addEventListener("click", () => setPickMode(!pickMode));
$("btnClearSeeds").addEventListener("click", () => {
  const v = getActiveView();
  if (!v) return;
  v.fgSeeds = [];
  v.bgSeeds = [];
  v.maskU8 = null;
  v.maskTint = null;
  drawViewOverlay(v);
  applyMultiViewFilter();
  setStatus(`已清除视图#${v.id} 的种子/掩膜`, true);
});
$("btnRestoreCloud").addEventListener("click", () => restoreFullCloud());
$("btnAABB").addEventListener("click", () => fitAABB());
$("btnOBB").addEventListener("click", () => fitOBB());
$("btnAlignPlane").addEventListener("click", () => alignBoxToPlane());
$("btnUndoAlign").addEventListener("click", () => undoAlignBoxToPlane());
$("btnFlipZ").addEventListener("click", () => flipBoxZ());
$("btnGizmoLocal").addEventListener("click", () => toggleGizmoSpace());
$("btnApplyYaw").addEventListener("click", () => applyLocalYaw($("rzYaw")?.value));
$("btnYawZero").addEventListener("click", () => {
  if ($("rzYaw")) $("rzYaw").value = "0";
});
$("rzYaw")?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") applyLocalYaw($("rzYaw").value);
});

$("btnPickPlane").addEventListener("click", () => setScalePickMode("plane"));
$("btnScaleLock").addEventListener("click", () => setScaleLock(!metricLocked));
$("btnClearPlane").addEventListener("click", () => {
  clearPlaneSamples();
  targetPoint = null;
  if (targetMarkerMesh) targetMarkerMesh.visible = false;
  setStatus("已清除平面/目标点", true);
});
$("btnFitPlane").addEventListener("click", () => fitPlaneFromSamples(true));
$("btnPickTarget").addEventListener("click", () => {
  if (!planeNormal) return setStatus("请先拟合平面");
  setScalePickMode("target");
});
$("planeOffset").addEventListener("input", () => refreshScalePreview());
$("planeOffset").addEventListener("change", () => {
  refreshScalePreview();
  if (planeNormal) persistPlane();
});
$("realDist").addEventListener("input", () => refreshScalePreview());
$("realDist").addEventListener("change", () => refreshScalePreview());
$("btnApplyScale").addEventListener("click", () => applyScaleAndExport());

["cx", "cy", "cz", "rx", "ry", "rz", "sx", "sy", "sz"].forEach((id) => {
  $(id).addEventListener("change", applyUIToBox);
  $(id).addEventListener("input", applyUIToBox);
});

let _ptrDown = null;
canvas.addEventListener("pointerdown", (ev) => {
  canvas.focus({ preventScroll: true });
  _ptrDown = { x: ev.clientX, y: ev.clientY, t: performance.now() };
});
canvas.addEventListener("pointerup", (ev) => {
  if (!_ptrDown) return;
  const dx = ev.clientX - _ptrDown.x;
  const dy = ev.clientY - _ptrDown.y;
  const dt = performance.now() - _ptrDown.t;
  _ptrDown = null;
  if (!scalePickMode) return;
  if (dx * dx + dy * dy > 25 || dt > 500) return; // ignore drag / long press
  if (ev.button !== 0) return;
  const hit = raycastCloud(ev);
  if (!hit) {
    setStatus("未点到点云，再试一次或放大点尺寸");
    return;
  }
  if (scalePickMode === "plane") addPlaneSample(hit);
  else if (scalePickMode === "target") setTargetFromHit(hit);
});

canvas.addEventListener("dblclick", (ev) => {
  if (scalePickMode) return; // scale pick uses single click
  if (!pointsCloud || !controls.enabled) return;
  const hit = raycastCloud(ev);
  if (hit) {
    focusOnPoint(hit, 0.2);
    setStatus(
      `旋转中心 → (${hit.x.toFixed(3)}, ${hit.y.toFixed(3)}, ${hit.z.toFixed(3)})`,
      true
    );
  }
});

function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

document.addEventListener(
  "keydown",
  (e) => {
    if (!transform) return;
    if (isTypingTarget(e.target) && e.target !== canvas) return;
    const code = e.code;
    let handled = true;
    if (code === "Digit1" || code === "Numpad1" || code === "KeyT") setGizmoMode("translate");
    else if (code === "Digit2" || code === "Numpad2" || code === "KeyR") setGizmoMode("rotate");
    else if (code === "Digit3" || code === "Numpad3" || code === "KeyS") setGizmoMode("scale");
    else if (code === "KeyF" && boxMesh) focusOnPoint(boxMesh.position.clone(), 0.25);
    else if (code === "KeyP") setPickMode(!pickMode);
    else handled = false;
    if (handled) {
      e.preventDefault();
      e.stopPropagation();
    }
  },
  true
);

$("btnSave").addEventListener("click", async () => {
  const frame = commitPoseForSave();
  const alignErr = planeAlignErrorDeg();
  if (alignErr != null && alignErr > 5) {
    const ok = confirm(
      `当前 3D 框 +Z 与桌面法向夹角约 ${alignErr.toFixed(1)}°（未贴齐）。\n` +
        `继续保存会导出歪的姿态。建议先点「贴齐平面」，再用「绕局部 Z」微调后保存。\n\n仍要保存吗？`
    );
    if (!ok) {
      setStatus(`已取消保存 · +Z 与法向差 ${alignErr.toFixed(1)}°`);
      return;
    }
  }
  setStatus(
    `保存并导出 · euler=[${frame.euler_deg.map((v) => v.toFixed(1)).join(", ")}]` +
      (alignErr != null ? ` · |Z∠n|=${alignErr.toFixed(1)}°` : "") +
      " …"
  );
  const res = await fetch("/api/save_frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(frame),
  });
  const data = await res.json();
  if (data.export_error || data.ok === false) {
    setStatus(`保存/导出失败: ${data.export_error || "unknown"}`);
  } else {
    const parts = [
      `新结果在 yolo6d/preview_6d.mp4 · preview_mask.mp4 · ${data.labels} 标签` +
        (alignErr != null ? ` · |Z∠n|=${alignErr.toFixed(1)}°` : ""),
    ];
    if (data.annotator_snap) {
      parts.push(`旧 annotator→${String(data.annotator_snap).split(/[/\\]/).pop()}`);
    }
    if (data.snap_path) {
      parts.push(`旧 yolo6d→${String(data.snap_path).split(/[/\\]/).pop()}`);
    }
    setStatus(parts.join(" · "), true);
  }
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
loadScene().catch((e) => setStatus("加载失败: " + e.message));
