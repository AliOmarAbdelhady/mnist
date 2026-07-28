"""Neural Network from Scratch — single-file core module (pure NumPy).

Contains everything needed to train a from-scratch MLP on MNIST:
  * activations (ReLU, stable softmax)
  * fused log-sum-exp cross-entropy loss (exactly consistent with its gradient)
  * MLP with He init, vectorized forward + backward
  * Adam optimizer
  * from-scratch affine data augmentation (shift/rotate/scale, bilinear)
  * mini-batch training loop with augmentation, early stopping, best-weight restore
  * kink-aware gradient check + smooth cross-check
  * accuracy, standardization, 70/20/10 stratified split, plotting helpers
  * weight (de)serialization

No PyTorch / TensorFlow / sklearn.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Headless matplotlib so scripts/CI never need a display; the notebook sets
# %matplotlib inline itself.
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42


# =====================================================================
#  reproducibility
# =====================================================================
def seed_all(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =====================================================================
#  activations + loss
# =====================================================================
def relu(z):               return np.maximum(0.0, z)
def relu_deriv(z):         return (z > 0.0).astype(np.float64)


def softmax(z, axis=1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def one_hot(y, num_classes=10):
    o = np.zeros((y.shape[0], num_classes))
    o[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return o


# =====================================================================
#  model
# =====================================================================
class MLP:
    """Fully-connected net: input -> h1 -> ... -> 10  (ReLU + softmax).

    He initialization, ReLU hidden layers, fused softmax+CE at the output.
    Forward/backward are fully vectorized (no per-example loops).
    """

    def __init__(self, layer_sizes: Sequence[int], seed: int = SEED,
                 l2: float = 1e-4, dropout: float = 0.0):
        self.layer_sizes = list(layer_sizes)
        self.seed = seed
        self.l2 = float(l2)
        self.dropout = float(dropout)
        self.rng = np.random.default_rng(seed)
        self.params: List[dict] = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            std = np.sqrt(2.0 / fan_in)                      # He init
            W = self.rng.standard_normal((fan_in, fan_out)) * std
            b = np.zeros((1, fan_out))
            self.params.append({"W": W, "b": b})

    @property
    def n_layers(self):
        return len(self.params)

    # ---- inference ----
    def forward(self, X, train=False):
        """Returns (probs, cache). cache[i] = (z_i, a_prev_i, mask_i)."""
        cache = []
        a = X
        L = self.n_layers
        for i, p in enumerate(self.params):
            z = a @ p["W"] + p["b"]
            a_prev = a
            if i == L - 1:                                   # output layer
                probs = softmax(z, axis=1)
                cache.append((z, a_prev, None))
                a = probs
            else:                                            # hidden layer
                h = relu(z)
                if train and self.dropout > 0:
                    keep = 1.0 - self.dropout
                    mask = (self.rng.random(h.shape) < keep) / keep   # inverted dropout
                    h = h * mask
                else:
                    mask = None
                cache.append((z, a_prev, mask))
                a = h
        return a, cache

    def predict_proba(self, X):
        return self.forward(X, train=False)[0]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ---- loss (stable, from logits; consistent with the gradient) ----
    def compute_loss(self, logits, y_onehot):
        m = logits.shape[0]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_denom = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_denom                       # log softmax
        data_loss = float(-np.sum(y_onehot * log_probs) / m)
        reg_loss = sum(self.l2 * 0.5 * float(np.sum(p["W"] * p["W"]))
                       for p in self.params)
        return data_loss + reg_loss

    def loss_and_grads(self, X, y_onehot, train=False):
        probs, cache = self.forward(X, train=train)
        loss = self.compute_loss(cache[-1][0], y_onehot)
        grads = self.backward(cache, probs, y_onehot)
        return loss, grads, probs

    # ---- backprop ----
    def backward(self, cache, probs, y_onehot):
        m = y_onehot.shape[0]
        grads = [{"dW": None, "db": None} for _ in self.params]
        delta = (probs - y_onehot) / m                        # fused softmax+CE grad
        L = self.n_layers
        for i in reversed(range(L)):
            z_i, a_prev, _ = cache[i]
            grads[i]["dW"] = a_prev.T @ delta + self.l2 * self.params[i]["W"]
            grads[i]["db"] = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                da_prev = delta @ self.params[i]["W"].T
                z_prev, _, mask_prev = cache[i - 1]
                if mask_prev is not None:
                    da_prev = da_prev * mask_prev
                delta = da_prev * relu_deriv(z_prev)
        return grads

    # ---- flat-param helpers (for gradient checking) ----
    def num_parameters(self):
        return sum(p["W"].size + p["b"].size for p in self.params)

    def get_flat_params(self):
        return np.concatenate([np.concatenate([p["W"].ravel(), p["b"].ravel()])
                               for p in self.params])

    def set_flat_params(self, theta):
        idx = 0
        for p in self.params:
            nw, nb = p["W"].size, p["b"].size
            p["W"] = theta[idx:idx + nw].reshape(p["W"].shape).copy(); idx += nw
            p["b"] = theta[idx:idx + nb].reshape(p["b"].shape).copy(); idx += nb


# =====================================================================
#  optimizer (Adam, from scratch)
# =====================================================================
class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [{"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])} for p in params]
            self.v = [{"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])} for p in params]
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            for key, gkey in (("W", "dW"), ("b", "db")):
                gi = g[gkey]
                self.m[i][key] = self.b1 * self.m[i][key] + (1 - self.b1) * gi
                self.v[i][key] = self.b2 * self.v[i][key] + (1 - self.b2) * gi * gi
                m_hat = self.m[i][key] / bc1
                v_hat = self.v[i][key] / bc2
                p[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =====================================================================
#  data augmentation (affine, from scratch)
# =====================================================================
_IMG = 28
_C = (_IMG - 1) / 2.0
_GX, _GY = np.meshgrid(np.linspace(-_C, _C, _IMG), np.linspace(-_C, _C, _IMG))
_GX = _GX.astype(np.float64)
_GY = _GY.astype(np.float64)


def _bilinear_sample(imgs, xs, ys):
    """Sample imgs (B,28,28) at float coords xs/ys (B,28,28); border clamped."""
    B, H, W = imgs.shape
    x0 = np.floor(xs).astype(np.int64); y0 = np.floor(ys).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = xs - x0, ys - y0
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    b = np.arange(B)[:, None, None]
    Ia = imgs[b, y0c, x0c]; Ib = imgs[b, y0c, x1c]
    Ic = imgs[b, y1c, x0c]; Id = imgs[b, y1c, x1c]
    return (Ia * (1 - dx) * (1 - dy) + Ib * dx * (1 - dy)
            + Ic * (1 - dx) * dy + Id * dx * dy)


def augment_batch(X, rng, max_shift=2.0, max_rot=12.0, max_scale=0.10):
    """Random affine augmentation of a raw [0,255] batch (B,784)."""
    X = np.asarray(X, dtype=np.float64)
    B = X.shape[0]
    imgs = X.reshape(B, _IMG, _IMG)
    angles = rng.uniform(-max_rot, max_rot, B) * np.pi / 180.0
    scales = rng.uniform(1.0 - max_scale, 1.0 + max_scale, B)
    tx = rng.uniform(-max_shift, max_shift, B)
    ty = rng.uniform(-max_shift, max_shift, B)
    cos, sin = np.cos(angles), np.sin(angles)
    # inverse affine: src = R(-theta) * (out - t) / s
    Dx = _GX[None] - tx[:, None, None]
    Dy = _GY[None] - ty[:, None, None]
    sx = (cos[:, None, None] * Dx + sin[:, None, None] * Dy) / scales[:, None, None]
    sy = (-sin[:, None, None] * Dx + cos[:, None, None] * Dy) / scales[:, None, None]
    xs, ys = sx + _C, sy + _C
    out = _bilinear_sample(imgs, xs, ys)
    inside = (xs >= -0.5) & (xs <= _IMG - 0.5) & (ys >= -0.5) & (ys <= _IMG - 0.5)
    return (out * inside).reshape(B, _IMG * _IMG)


def make_augment(max_shift=2.0, max_rot=12.0, max_scale=0.10):
    def aug(X, rng):
        return augment_batch(X, rng, max_shift, max_rot, max_scale)
    return aug


# =====================================================================
#  training loop (mini-batch Adam, augmentation, early stopping)
# =====================================================================
def _std_safe(std):
    s = np.asarray(std).copy()
    s[s < 1e-8] = 1.0
    return s


def accuracy(model, X, y, batch=8192):
    preds = []
    for s in range(0, X.shape[0], batch):
        preds.append(model.predict(X[s:s + batch]))
    return float(np.mean(np.concatenate(preds) == y))


def train(model, X_train, y_train, X_val=None, y_val=None,
          epochs=40, batch_size=128, lr=1e-3, lr_decay=0.02,
          seed=SEED, verbose=True, augment_fn=None, mean=None, std=None,
          early_stopping_patience=None, restore_best=True):
    """Mini-batch Adam training.

    If mean/std are given the inputs are RAW and standardized internally
    (val once, train per-batch after augmentation). Otherwise inputs must be
    pre-standardized.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(lr)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    raw_mode = mean is not None and std is not None
    if raw_mode:
        _std, _mean = _std_safe(std), np.asarray(mean)
        Xv_eval = (X_val - _mean) / _std if X_val is not None else None
        Xt_eval = (X_train - _mean) / _std
    else:
        Xv_eval, Xt_eval = X_val, X_train
    Y_val = one_hot(y_val) if Xv_eval is not None else None

    n = X_train.shape[0]
    best_val, best_params, best_epoch, no_improve = np.inf, None, -1, 0
    stopped_early = False

    for epoch in range(epochs):
        cur_lr = lr * np.exp(-lr_decay * epoch)
        opt.lr = cur_lr
        perm = rng.permutation(n)
        Xs, ys = X_train[perm], y_train[perm]
        rl, steps = 0.0, 0
        for s in range(0, n, batch_size):
            xb = Xs[s:s + batch_size]
            yb = ys[s:s + batch_size]
            if augment_fn is not None:
                xb = augment_fn(xb, rng)
            if raw_mode:
                xb = (xb - _mean) / _std
            loss, grads, _ = model.loss_and_grads(xb, one_hot(yb), train=True)
            opt.step(model.params, grads)
            rl += loss; steps += 1

        tl = rl / steps
        ta = accuracy(model, Xt_eval, y_train)
        hist["train_loss"].append(tl); hist["train_acc"].append(ta); hist["lr"].append(cur_lr)

        if Xv_eval is not None:
            vl, _, _ = model.loss_and_grads(Xv_eval, Y_val, train=False)
            va = accuracy(model, Xv_eval, y_val)
            hist["val_loss"].append(vl); hist["val_acc"].append(va)
            if verbose:
                mark = "  *" if vl < best_val - 1e-6 else ""
                print(f"epoch {epoch+1:3d}/{epochs}  lr={cur_lr:.4g}  "
                      f"loss={tl:.4f}  acc={ta:.4f}  val_loss={vl:.4f}  val_acc={va:.4f}{mark}")
            if vl < best_val - 1e-6:
                best_val, best_epoch, no_improve = vl, epoch, 0
                if restore_best:
                    best_params = [{"W": p["W"].copy(), "b": p["b"].copy()} for p in model.params]
            else:
                no_improve += 1
                if early_stopping_patience and no_improve >= early_stopping_patience:
                    stopped_early = True
                    if verbose:
                        print(f"  >> early stopping (best at epoch {best_epoch+1})")
                    break
        elif verbose:
            print(f"epoch {epoch+1:3d}/{epochs}  lr={cur_lr:.4g}  loss={tl:.4f}  acc={ta:.4f}")

    if restore_best and best_params is not None:
        for p, bp in zip(model.params, best_params):
            p["W"], p["b"] = bp["W"], bp["b"]
    hist["best_epoch"] = best_epoch
    hist["stopped_early"] = stopped_early
    return hist


# =====================================================================
#  gradient check (kink-aware + smooth cross-check)
# =====================================================================
def _relerr(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)


def _hidden_masks(cache):
    return [cache[i][0] > 0.0 for i in range(len(cache) - 1)]


def gradient_check(model, X, y_onehot, num_samples=400, eps=1e-6, seed=0, verbose=True):
    """Kink-aware analytic-vs-numerical gradient check (ReLU)."""
    orig_dropout = model.dropout
    model.dropout = 0.0
    try:
        _, grads, _ = model.loss_and_grads(X, y_onehot, train=False)
        g_flat = np.concatenate([np.concatenate([g["dW"].ravel(), g["db"].ravel()])
                                 for g in grads])
        theta = model.get_flat_params(); P = theta.size
        rng = np.random.default_rng(seed)
        idxs = rng.choice(P, min(num_samples, P), replace=False)
        g_num = np.zeros(P); kinked = np.zeros(P, bool)
        for j in idxs:
            o = theta[j]
            theta[j] = o + eps; model.set_flat_params(theta)
            lp, _, cp = model.loss_and_grads(X, y_onehot, train=False)
            theta[j] = o - eps; model.set_flat_params(theta)
            lm, _, cm = model.loss_and_grads(X, y_onehot, train=False)
            theta[j] = o
            g_num[j] = (lp - lm) / (2 * eps)
            kinked[j] = any((a != b).any() for a, b in zip(_hidden_masks(cp), _hidden_masks(cm)))
        model.set_flat_params(theta)
        sm = ~kinked
        si = idxs[~kinked[idxs]]
        rel = _relerr(g_flat[si], g_num[si]) if si.size else float("nan")
        if verbose:
            print(f"  kink-crossing coords: {int(kinked[idxs].sum())}/{len(idxs)} excluded")
            print(f"  smooth relative error: {rel:.3e}  ->  "
                  f"{'PASS' if rel < 1e-6 else ('ok' if rel < 1e-4 else 'FAIL')}")
        return rel
    finally:
        model.dropout = orig_dropout


def gradient_check_smooth(model, X, y_onehot, num_samples=400, eps=1e-6, seed=0):
    """Independent confirmation: swap ReLU->tanh (smooth), re-check."""
    import nn as _self
    orig_relu, orig_deriv = _self.relu, _self.relu_deriv
    _self.relu = np.tanh
    _self.relu_deriv = lambda z: 1.0 - np.tanh(z) ** 2
    try:
        orig_dropout = model.dropout
        model.dropout = 0.0
        _, grads, _ = model.loss_and_grads(X, y_onehot, train=False)
        g_flat = np.concatenate([np.concatenate([g["dW"].ravel(), g["db"].ravel()])
                                 for g in grads])
        theta = model.get_flat_params(); P = theta.size
        rng = np.random.default_rng(seed)
        idxs = rng.choice(P, min(num_samples, P), replace=False)
        g_num = np.zeros(P)
        for j in idxs:
            o = theta[j]
            theta[j] = o + eps; model.set_flat_params(theta)
            lp, _, _ = model.loss_and_grads(X, y_onehot, train=False)
            theta[j] = o - eps; model.set_flat_params(theta)
            lm, _, _ = model.loss_and_grads(X, y_onehot, train=False)
            theta[j] = o
            g_num[j] = (lp - lm) / (2 * eps)
        model.set_flat_params(theta)
        model.dropout = orig_dropout
        return _relerr(g_flat[idxs], g_num[idxs])
    finally:
        _self.relu, _self.relu_deriv = orig_relu, orig_deriv


# =====================================================================
#  data utilities + split
# =====================================================================
def standardize(X_train, *others, eps=1e-8):
    """Standardize with train stats. Returns (X_train_s, *others_s, mean, std_safe)."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std_safe = std.copy(); std_safe[std_safe < eps] = 1.0
    out = [(X_train - mean) / std_safe] + [(o - mean) / std_safe for o in others]
    out += [mean, std_safe]
    return tuple(out)


def stratified_split_70_20_10(X, y, seed=SEED):
    """Stratified 70/20/10 split. Returns (Xtr,ytr,Xva,yva,Xte,yte)."""
    rng = np.random.default_rng(seed)
    tr_idx, va_idx, te_idx = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_va = int(round(n * 0.20))
        n_te = int(round(n * 0.10))
        n_tr = n - n_va - n_te
        tr_idx.append(idx[:n_tr])
        va_idx.append(idx[n_tr:n_tr + n_va])
        te_idx.append(idx[n_tr + n_va:])
    tr_idx = np.concatenate(tr_idx); va_idx = np.concatenate(va_idx); te_idx = np.concatenate(te_idx)
    rng.shuffle(tr_idx); rng.shuffle(va_idx); rng.shuffle(te_idx)
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], X[te_idx], y[te_idx]


# =====================================================================
#  weight persistence
# =====================================================================
def save_weights(model, path, mean=None, std=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = {f"W{i}": p["W"] for i, p in enumerate(model.params)}
    data.update({f"b{i}": p["b"] for i, p in enumerate(model.params)})
    data["layer_sizes"] = np.array(model.layer_sizes)
    if mean is not None: data["mean"] = np.asarray(mean)
    if std is not None:  data["std"] = np.asarray(std)
    np.savez(path, **data)


def load_weights(path):
    d = np.load(path, allow_pickle=False)
    model = MLP(d["layer_sizes"].tolist(), seed=0, l2=0.0, dropout=0.0)
    for i in range(len(model.params)):
        model.params[i]["W"] = d[f"W{i}"]
        model.params[i]["b"] = d[f"b{i}"]
    model.mean = d["mean"] if "mean" in d else None
    model.std = d["std"] if "std" in d else None
    return model


# =====================================================================
#  plotting helpers
# =====================================================================
def plot_curves(hist, savepath=None, baseline=None):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    if baseline is not None:
        ax[0].plot(baseline["val_loss"], "--", color="C1", alpha=.4, label="val (no-aug ref)")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss curve"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    if baseline is not None:
        ax[1].plot(baseline["val_acc"], "--", color="C1", alpha=.4, label="val (no-aug ref)")
    ax[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy curve"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=120, bbox_inches="tight")


def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def plot_confusion(cm, savepath=None):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues"); thr = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=7)
    ax.set(xticks=range(10), yticks=range(10), xlabel="predicted", ylabel="true",
           title="Confusion matrix")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=120, bbox_inches="tight")


def plot_samples(X, y_true, y_pred, shape=(28, 28), n=16, ncols=8, savepath=None, title=""):
    idx = np.arange(min(n, len(y_true)))
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.4, nrows * 1.6))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, idx):
        ax.imshow(X[i].reshape(shape), cmap="gray")
        col = "green" if y_pred[i] == y_true[i] else "red"
        ax.set_title(f"p{y_pred[i]}/t{y_true[i]}", color=col, fontsize=9); ax.set_axis_off()
    for ax in axes[len(idx):]:
        ax.set_axis_off()
    if title: fig.suptitle(title)
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=120, bbox_inches="tight")


def show_digits(X, y, shape=(28, 28), n=16, ncols=8, savepath=None):
    idx = np.arange(min(n, len(y)))
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.3, nrows * 1.5))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, idx):
        ax.imshow(X[i].reshape(shape), cmap="gray")
        ax.set_title(str(int(y[i])), fontsize=9); ax.set_axis_off()
    for ax in axes[len(idx):]:
        ax.set_axis_off()
    fig.suptitle("Sample MNIST digits"); fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=120, bbox_inches="tight")
