# Handwritten Digit Recognition

A small OOP mini-project: draw a digit (0–9) on a canvas in your browser and a
neural network recognises it instantly.

```
┌─────────────┐   crop → scale 20×20 → centre-of-mass   ┌──────────────────┐
│ drawing pad │ ───────────────────────────────────────▶ │ 784→128→64→10 MLP│ ─▶ 7
└─────────────┘          28 × 28 input                  └──────────────────┘    + bars
```

## How it works

1. **Training (`train.py`)** — a fully connected network trained on the MNIST
   dataset of 60,000 handwritten digits. Written with four classes:
   - `MNISTDataset` — reads the MNIST IDX files, downloads them if missing, and
     serves shuffled mini-batches.
   - `NeuralNetwork` — forward pass (ReLU hidden layers, softmax output),
     backpropagation, Adam updates and evaluation.
   - `Trainer` — the epoch/loss/accuracy training loop.
   - `ModelExporter` — writes the trained weights to `web/model.json`.
2. **Inference (`web/app.js`)** — the browser normalises your drawing exactly
   the way MNIST was normalised (bounding box → fit into a 20×20 box keeping the
   aspect ratio → centre of mass at the centre of a 28×28 frame), then runs the
   network in plain JavaScript. No server, no libraries — everything happens
   locally in the browser.

## Run it

```bash
# 1. train (downloads MNIST automatically, ~10 s) and export web/model.json
python3 train.py

# 2. open the app in the browser
cd web && python3 -m http.server 3000
# → http://localhost:3000
```

## Results

| Metric | Value |
| --- | --- |
| Architecture | 784 → 128 → 64 → 10 (ReLU, softmax) |
| Optimiser | Adam, learning rate 1e-3, batch 128, 8 epochs |
| MNIST test accuracy | **97.46 %** (9,746 / 10,000) |
| Accuracy on digits drawn at random positions/sizes on the pad | 97.4 % (974 / 1,000) |

## Files

```
train.py         training pipeline (OOP)
data/            MNIST images (downloaded automatically, not committed)
web/index.html   the app page
web/app.js       canvas drawing, preprocessing and JS inference
web/style.css    styling
web/model.json   trained weights exported by train.py
```

## Ideas to extend

- Swap the MLP for a convolutional network (or import a pretrained one).
- Add a "did you mean" list of the top-3 predictions and a few user corrections.
- Let users test the model on their own 28×28 image files.
