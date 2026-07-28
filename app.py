"""Gradio web UI: draw a digit, get a prediction.

Loads the trained weights from ``models/nn.npz`` and applies the exact same
standardization used at train time. Drawn digits are put through canonical
MNIST normalization (crop -> fit 20x20 box -> center by center of mass) so any
size/position maps onto the training distribution.

    python app.py        # open http://127.0.0.1:7860
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import nn

WEIGHTS = ROOT / "models" / "nn.npz"
assert WEIGHTS.exists(), f"Run `python train.py` first (weights missing at {WEIGHTS})."

print("Loading model ...")
model = nn.load_weights(WEIGHTS)
mean = getattr(model, "mean", None)
std = getattr(model, "std", None)


def _mnist_normalize(ink, box=20, field=28):
    """Crop, fit into 20x20 (aspect preserved), center by center of mass."""
    mask = ink > ink.max() * 0.2
    if not mask.any():
        return np.zeros((field, field))
    ys, xs = np.where(mask)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = box / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = np.asarray(
        Image.fromarray(crop.astype(np.float32)).resize((nw, nh), Image.LANCZOS),
        dtype=np.float64)
    canvas = np.zeros((field, field))
    oy, ox = (field - nh) // 2, (field - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    total = canvas.sum()
    if total > 0:
        yy, xx = np.indices(canvas.shape)
        cy, cx = (canvas * yy).sum() / total, (canvas * xx).sum() / total
        canvas = np.roll(canvas, (int(round(field / 2 - cy)), int(round(field / 2 - cx))),
                         axis=(0, 1))
    return canvas


def preprocess(drawing):
    if drawing is None or drawing.get("composite") is None:
        return np.zeros(784)
    img = Image.fromarray(drawing["composite"]).convert("L")
    arr = 255.0 - np.asarray(img, dtype=np.float64)   # invert -> bright ink on black
    arr = _mnist_normalize(arr)
    flat = arr.reshape(1, -1)
    if mean is not None and std is not None:
        flat = (flat - mean) / std
    return flat.reshape(784)


def predict(drawing):
    x = preprocess(drawing)
    probs = model.predict_proba(x.reshape(1, -1))[0]
    pred = int(np.argmax(probs))
    bar = {str(i): float(probs[i]) for i in range(10)}
    return pred, bar, x.reshape(28, 28)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(label="Draw a digit (0-9)", type="numpy"),
    outputs=[
        gr.Label(num_top_classes=1, label="Predicted digit"),
        gr.Label(label="Class probabilities"),
        gr.Image(label="28x28 input seen by the network", height=200),
    ],
    title="MNIST Neural Network — from scratch (NumPy only)",
    description=(
        "Draw a digit. A 784-128-64-10 MLP implemented entirely in NumPy "
        "classifies it. The right panel shows the preprocessed 28x28 input "
        "(centered, standardized like MNIST)."),
)


if __name__ == "__main__":
    demo.launch()
