"""Train the from-scratch MNIST MLP end-to-end.

Pipeline
  1. Download the MNIST CSVs (pjreddie mirror) if missing.
  2. Load + pool all 70 000 images, stratified 70/20/10 split
     (49 000 train / 14 000 val / 7 000 test).
  3. Standardize with the TRAIN statistics.
  4. Gradient-check gate (kink-aware; must pass).
  5. Train 784-128-64-10 with Adam + affine augmentation + early stopping.
  6. Evaluate on the held-out 7 000-image test set, save weights + figures.

    python train.py [--epochs 40] [--no-augment]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import nn

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGS_DIR = ROOT / "figures"

# pjreddie public MNIST CSV mirror (label, pixel1..pixel784, no header).
CSV_URLS = {
    "mnist_train.csv": "https://pjreddie.com/media/files/mnist_train.csv",
    "mnist_test.csv":  "https://pjreddie.com/media/files/mnist_test.csv",
}
# Approx byte sizes for a sanity check on download.
CSV_BYTES = {"mnist_train.csv": 100_000_000, "mnist_test.csv": 15_000_000}


def download_data():
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in CSV_URLS.items():
        dst = DATA_DIR / name
        if dst.exists() and dst.stat().st_size > CSV_BYTES[name] * 0.9:
            print(f"  OK  {name} already present ({dst.stat().st_size//1024} KB)")
            continue
        print(f"  downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "mnist-nn/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
        print(f"       saved {dst.stat().st_size//1024} KB")


def load_data():
    """Pool both CSVs into one (X, y) of 70 000 samples."""
    tr = pd.read_csv(DATA_DIR / "mnist_train.csv", header=None)
    te = pd.read_csv(DATA_DIR / "mnist_test.csv", header=None)
    all_df = pd.concat([tr, te], ignore_index=True)
    y = all_df.iloc[:, 0].to_numpy().astype(np.int64)
    X = all_df.iloc[:, 1:].to_numpy().astype(np.float64)   # [0, 255]
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay", type=float, default=0.02)
    ap.add_argument("--l2", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=nn.SEED)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()
    use_augment = not args.no_augment

    nn.seed_all(args.seed)
    MODELS_DIR.mkdir(exist_ok=True); FIGS_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------- 1. data
    print("Ensuring MNIST CSVs are present ...")
    download_data()
    print("\nLoading + pooling CSVs ...")
    X, y = load_data()
    print(f"  pooled: {X.shape}  (will split 70/20/10)")

    Xtr, ytr, Xva, yva, Xte, yte = nn.stratified_split_70_20_10(X, y, seed=args.seed)
    print(f"  train={Xtr.shape[0]}  val={Xva.shape[0]}  test={Xte.shape[0]}")

    # Standardize with TRAIN stats.
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True); std_safe = std.copy(); std_safe[std_safe < 1e-8] = 1.0
    Xte_s = (Xte - mean) / std_safe

    # ---------------------------------------------- 2. gradient check
    print("\nGradient check (small model, kink-aware) ...")
    small = nn.MLP([784, 32, 16, 10], seed=1, l2=1e-3, dropout=0.0)
    Xchk = ((Xtr[:64] - mean) / std_safe)
    relu_rel = nn.gradient_check(small, Xchk, nn.one_hot(ytr[:64]), num_samples=300, eps=1e-6)
    smooth_rel = nn.gradient_check_smooth(small, Xchk, nn.one_hot(ytr[:64]), num_samples=300, eps=1e-6)
    print(f"  smooth (tanh) cross-check rel-err = {smooth_rel:.3e}")
    if relu_rel >= 1e-5 or smooth_rel >= 1e-5:
        print("!! Gradient check FAILED. Aborting.")
        return 1
    print("  => backprop VERIFIED.")

    # ---------------------------------------------------- 3. train
    tag = "WITH augmentation + early stopping" if use_augment else "(no augmentation)"
    print(f"\nTraining MLP {[784,128,64,10]} for up to {args.epochs} epochs  {tag}")
    print(f"  dropout={args.dropout}  L2={args.l2}  patience={args.patience}")
    model = nn.MLP([784, 128, 64, 10], seed=args.seed, l2=args.l2, dropout=args.dropout)
    aug_fn = nn.make_augment() if use_augment else None
    t0 = time.time()
    hist = nn.train(model, Xtr, ytr, Xva, yva,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    lr_decay=args.lr_decay, seed=args.seed, verbose=True,
                    augment_fn=aug_fn, mean=mean, std=std_safe,
                    early_stopping_patience=args.patience, restore_best=True)
    dt = time.time() - t0
    n_run = len(hist["train_loss"])
    print(f"\nTraining time: {dt:.1f}s  ({dt/max(n_run,1):.2f}s/epoch, "
          f"ran {n_run} epochs{' [early-stopped]' if hist.get('stopped_early') else ''})")

    # ---------------------------------------------------- 4. evaluate
    test_acc = nn.accuracy(model, Xte_s, yte)
    be = hist.get("best_epoch", n_run - 1) or (n_run - 1)
    gap = hist["train_acc"][min(be, n_run - 1)] - hist["val_acc"][min(be, n_run - 1)]
    print(f"\n*** TEST ACCURACY (7 000-image holdout): {test_acc*100:.2f}% ***")
    print(f"    best-epoch train={hist['train_acc'][min(be,n_run-1)]*100:.2f}%  "
          f"val={hist['val_acc'][min(be,n_run-1)]*100:.2f}%  gap={gap*100:+.2f}pp")

    y_pred = model.predict(Xte_s)
    cm = nn.confusion_matrix(yte, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)

    # ---------------------------------------------------- 5. save
    nn.save_weights(model, MODELS_DIR / "nn.npz", mean=mean, std=std_safe)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"test_acc": test_acc, "per_class_acc": per_class.tolist(),
                   "history": hist, "config": vars(args),
                   "train_seconds": dt, "augmented": use_augment,
                   "split": "70/20/10"}, f, indent=2)

    nn.plot_curves(hist, FIGS_DIR / "loss_accuracy_curves.png")
    nn.plot_confusion(cm, FIGS_DIR / "confusion_matrix.png")
    nn.plot_samples(Xte, yte, y_pred, n=16, savepath=FIGS_DIR / "sample_predictions.png",
                    title="Sample predictions (green=correct, red=wrong)")
    nn.plot_samples(Xte, yte, y_pred, n=16, savepath=FIGS_DIR / "misclassified.png",
                    title="Misclassified samples")

    print("\nPer-class accuracy:")
    for d in range(10):
        print(f"  digit {d}: {per_class[d]*100:.2f}%")
    print("\nSaved weights -> models/nn.npz, metrics.json, figures/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
