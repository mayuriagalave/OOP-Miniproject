/* Handwritten Digit Recognition - browser side.
 *
 * 1. The user draws on the <canvas> pad.
 * 2. The stroke is normalised the same way MNIST was (crop -> scale to 20x20
 *    -> centre of mass at the middle of a 28x28 frame).
 * 3. The pixels are pushed through the trained network exported by train.py.
 */

const CANVAS_SIZE = 280;
const INPUT_SIZE = 28;
const NORMALISED = 20; // MNIST digits live in a 20x20 box inside 28x28

const pad = document.getElementById("pad");
const ctx = pad.getContext("2d", { willReadFrequently: true });
const clearBtn = document.getElementById("clear");
const statusEl = document.getElementById("status");
const barsEl = document.getElementById("bars");
const footnoteEl = document.getElementById("footnote");

let model = null;
let drawing = false;

/* ------------------------------------------------------------------ model */

async function loadModel() {
  const res = await fetch("model.json");
  if (!res.ok) throw new Error(`could not load model.json (${res.status})`);
  return res.json();
}

/** One forward pass: returns the 10 class probabilities. */
function forward(net, input) {
  let a = input;
  for (let li = 0; li < net.layers.length; li++) {
    const { W, b } = net.layers[li];
    const out = new Float64Array(b.length);
    out.set(b);
    for (let r = 0; r < a.length; r++) {
      const av = a[r];
      if (av === 0) continue;
      const row = W[r];
      for (let c = 0; c < row.length; c++) out[c] += av * row[c];
    }
    if (li < net.layers.length - 1) {
      for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0; // ReLU
    } else {
      // softmax
      let max = -Infinity;
      for (const v of out) if (v > max) max = v;
      let sum = 0;
      for (let i = 0; i < out.length; i++) {
        out[i] = Math.exp(out[i] - max);
        sum += out[i];
      }
      for (let i = 0; i < out.length; i++) out[i] /= sum;
    }
    a = out;
  }
  return a;
}

/* --------------------------------------------------------- preprocessing */

/** Converts the drawing canvas into a 784-length Float32Array in [0, 1]. */
function preprocess() {
  const { data } = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // 1. bounding box of the drawn strokes
  let minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = -1, maxY = -1;
  for (let y = 0; y < CANVAS_SIZE; y++) {
    for (let x = 0; x < CANVAS_SIZE; x++) {
      if (data[(y * CANVAS_SIZE + x) * 4] > 25) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null; // blank canvas

  const cropW = maxX - minX + 1;
  const cropH = maxY - minY + 1;

  // 2. scale the crop into a 20x20 box, keeping the aspect ratio
  const scale = NORMALISED / Math.max(cropW, cropH);
  const nw = Math.max(1, Math.round(cropW * scale));
  const nh = Math.max(1, Math.round(cropH * scale));

  const tmp = document.createElement("canvas");
  tmp.width = nw;
  tmp.height = nh;
  const tctx = tmp.getContext("2d", { willReadFrequently: true });
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = "high";
  tctx.drawImage(pad, minX, minY, cropW, cropH, 0, 0, nw, nh);
  const scaled = tctx.getImageData(0, 0, nw, nh).data;

  // 3. centre of mass of the scaled digit
  let sum = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const v = scaled[(y * nw + x) * 4] / 255;
      sum += v;
      sumX += v * x;
      sumY += v * y;
    }
  }
  if (sum < 1e-6) return null;
  const cx = sumX / sum;
  const cy = sumY / sum;

  // 4. paste into the 28x28 frame with the centre of mass in the middle
  const input = new Float32Array(INPUT_SIZE * INPUT_SIZE);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const v = scaled[(y * nw + x) * 4] / 255;
      const tx = Math.round(INPUT_SIZE / 2 - cx + x);
      const ty = Math.round(INPUT_SIZE / 2 - cy + y);
      if (tx < 0 || tx >= INPUT_SIZE || ty < 0 || ty >= INPUT_SIZE) continue;
      const idx = ty * INPUT_SIZE + tx;
      if (v > input[idx]) input[idx] = v;
    }
  }
  return input;
}

/* ------------------------------------------------------------- rendering */

function renderPrediction(probs) {
  // forward() returns a Float64Array; TypedArray#map() would coerce the HTML
  // strings below back to numbers (NaN), so use a plain array first.
  probs = Array.from(probs);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;

  const result = document.getElementById("result");
  const pct = (probs[best] * 100).toFixed(1);
  result.innerHTML = `<div class="big-digit">${best}</div>
                      <div class="confidence">${pct}% confidence</div>`;

  barsEl.hidden = false;
  barsEl.innerHTML = probs
    .map((p, d) => {
      const width = (p * 100).toFixed(1);
      return `<div class="bar-row${d === best ? " top" : ""}">
                <span class="digit">${d}</span>
                <span class="track"><span class="fill" style="width:${width}%"></span></span>
                <span class="pct">${width}%</span>
              </div>`;
    })
    .join("");

  footnoteEl.hidden = false;
  footnoteEl.textContent = "Recognised as soon as you lift the pen.";
}

function resetPrediction(message) {
  document.getElementById("result").innerHTML =
    `<div class="big-dash">&ndash;</div><div class="status">${message}</div>`;
  barsEl.hidden = true;
  barsEl.innerHTML = "";
  footnoteEl.hidden = true;
}

function predictNow() {
  if (!model) return;
  const input = preprocess();
  if (!input) {
    resetPrediction("Draw a digit to get started.");
    return;
  }
  renderPrediction(forward(model, input));
}

/* -------------------------------------------------------------- drawing */

function position(e) {
  const rect = pad.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) / rect.width) * CANVAS_SIZE,
    y: ((e.clientY - rect.top) / rect.height) * CANVAS_SIZE,
  };
}

function startStroke(e) {
  e.preventDefault();
  drawing = true;
  if (pad.setPointerCapture) pad.setPointerCapture(e.pointerId);
  const { x, y } = position(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function moveStroke(e) {
  if (!drawing) return;
  e.preventDefault();
  const { x, y } = position(e);
  ctx.lineTo(x, y);
  ctx.stroke();
}

function endStroke() {
  if (!drawing) return;
  drawing = false;
  ctx.closePath();
  predictNow();
}

function clearPad() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  resetPrediction(model ? "Draw a digit to get started." : "Loading model…");
}

/* ------------------------------------------------------------------ init */

function initCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  pad.addEventListener("pointerdown", startStroke);
  pad.addEventListener("pointermove", moveStroke);
  pad.addEventListener("pointerup", endStroke);
  pad.addEventListener("pointerleave", endStroke);
  pad.addEventListener("pointercancel", endStroke);
  clearBtn.addEventListener("click", clearPad);
}

initCanvas();

loadModel()
  .then((net) => {
    model = net;
    statusEl.textContent = `Model ready \u00b7 ${(net.testAccuracy * 100).toFixed(1)}% MNIST test accuracy`;
    resetPrediction("Draw a digit to get started.");
  })
  .catch((err) => {
    statusEl.textContent = `Failed to load model: ${err.message}`;
    console.error(err);
  });
