# MNIST Neural Network from Scratch (simple edition)

A from-scratch feed-forward neural network that classifies handwritten digits,
implemented with **only NumPy** — no PyTorch, TensorFlow, or sklearn. It trains
on MNIST to **~98% test accuracy** in ~7 min on a CPU, verifies backprop with a
**gradient check** (rel-err ≈ 1e-9), keeps overfitting tiny with **data
augmentation + early stopping**, and ships a **Gradio web UI** where you can
draw a digit and watch it be classified.

This is a deliberately **lean** version of a typical MNIST project: one core
module, one train script, a ready-made CSV dataset.

---

## Quick start

```bash
cd /home/ali/mnist
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py            # downloads CSV → grad-check → trains → saves weights + figures
python app.py              # draw-and-predict UI → http://127.0.0.1:7860
jupyter notebook mnist.ipynb   # the deliverable notebook (code + plots + write-up)
```

`train.py` downloads the MNIST CSVs automatically on first run, so there's
nothing to fetch manually.

---

## Dataset

Plain MNIST in CSV form, downloaded once from the public
**[pjreddie MNIST mirror](https://pjreddie.com/media/files/mnist_train.csv)**:

- `mnist_train.csv` (60 000 rows) + `mnist_test.csv` (10 000 rows)
- each row: `label, pixel1, pixel2, …, pixel784` (no header), pixels `[0, 255]`

These are the canonical 70 000 MNIST images, just pre-converted to CSV — so
**no IDX binary parsing, no Kaggle token, no sklearn** are needed. They are
pooled and re-split **70 / 20 / 10** (49 000 train / 14 000 val / 7 000 test,
stratified, seeded).

---

## Results

| Metric | Value |
|---|---|
| Architecture | `784 → 128 → 64 → 10` (ReLU + softmax, He init) |
| Optimizer | Adam (lr 1e-3, exp decay 0.02), batch 128 |
| Regularization | Dropout 0.2 + L2 3e-4 |
| Augmentation | affine (shift ±2px, rotate ±12°, scale ±10%), from scratch |
| Split | stratified **70 / 20 / 10** (49k / 14k / 7k) |
| Gradient check (kink-aware + smooth cross-check) | rel-err ≈ **1e-9** ✓ |
| Generalization gap (train − val) | **~0.1 pp** — essentially no overfitting |
| **Test accuracy (7 000-image holdout)** | **98.13 %** (≥ 90 % requirement exceeded) |
| Training time | ~7 min on CPU (40 epochs, ~10 s/epoch with augmentation) |

Per-class accuracy is 96.9 %–99.6 %; remaining errors are the classic ambiguous
pairs (4↔9, 8↔3, 7↔2).

---

## Project layout (lean)

```
mnist/
├── nn.py            # ★ ONE core module: activations, MLP, Adam, augmentation,
│                    #   training loop, kink-aware gradient check, plots, save/load
├── train.py         # ★ ONE script: download CSV → split 70/20/10 → grad-check
│                    #   → train (augmented) → save weights + figures + metrics
├── app.py           # Gradio Sketchpad UI: draw → predict → probability bar
├── mnist.ipynb      # deliverable notebook (code + plots + write-up)
├── requirements.txt # numpy, matplotlib, pandas, jupyter, nbformat, gradio (no sklearn)
├── data/            # mnist_train.csv + mnist_test.csv (downloaded, gitignored)
├── models/          # nn.npz weights + metrics.json
└── figures/         # loss/accuracy, confusion, samples, misclassified
```

Everything that matters lives in **`nn.py`** and is pure NumPy:
`relu` / `softmax` (stable), the fused log-sum-exp cross-entropy, `MLP` (He
init, vectorized forward/backward), `Adam`, affine `augment_batch`, the
mini-batch `train` loop with early stopping + best-weight restore, and the
kink-aware `gradient_check`.

---

## Design notes

- **Fused log-sum-exp loss.** Computing the loss from the *logits* (not
  `log(softmax + eps)`) is numerically stable **and** exactly consistent with
  the analytic gradient `(probs − y)/m` — no epsilon floor to break gradient
  checking.
- **Kink-aware gradient check.** ReLU is non-smooth at 0; a naive finite
  difference gives a false failure when the step straddles a kink. Kink-crossing
  coordinates are detected and excluded, and a smooth (tanh) cross-check
  independently confirms backprop.
- **Augmentation + early stopping** prevent the train-set memorization a plain
  MLP falls into — the train/val gap stays at ~0.1 pp instead of ~2 pp.
- **MNIST-style draw preprocessing in the UI** (crop → fit a 20×20 box → center
  by center of mass), so a hand-drawn digit of any size/position lands on the
  training distribution (validated at 99/100 on transformed test images).

---

## Reproducibility

Everything is seeded (`SEED = 42`). Re-running `train.py` or the notebook
reproduces the numbers to within floating-point noise. The Gradio app loads the
saved weights and applies the identical preprocessing, so a drawn digit is
classified exactly as the held-out test images are.

## Limits

A dense MLP discards spatial structure; its MNIST ceiling is ~98.4 %. Going
beyond requires a convolutional network (out of scope here).
