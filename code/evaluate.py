#!/usr/bin/env python3
"""
Evaluate super-resolution models on the LGG MRI dataset.

Reports PSNR and SSIM on a held-out, patient-level test split, alongside a
bicubic-interpolation baseline, and can emit the rows directly as LaTeX for
the results table in the paper.

The preprocessing here mirrors trainSRNET.ipynb / trainSRGAN.ipynb exactly:
Gaussian blur (radius 2) -> bicubic downsample -> float32 in [0, 1].

IMPORTANT: this script splits by *patient*, whereas the training notebooks
split by *slice*. Numbers produced here are therefore not directly comparable
to the validation losses in those notebooks, and should be expected to be
worse. That is the point -- see TODO.md.

Usage
-----
    # bicubic baseline only, no model needed
    python evaluate.py --data-root ../mri_dataset/kaggle_3m --baseline-only

    # evaluate a trained checkpoint
    python evaluate.py --data-root ../mri_dataset/kaggle_3m \
        --model SRResNet/weights_epoch_06.h --scale 2

    # emit a LaTeX table row
    python evaluate.py ... --latex --label "SRResNet $\\times 2$"
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Configuration mirroring the training notebooks
# ---------------------------------------------------------------------------

HR_SIZE = 256
BLUR_RADIUS = 2
SEED = 42

# Fractions are applied at the patient level, not the slice level.
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_root, scale, cache=None, limit_patients=None):
    """Load LR/HR pairs plus a per-image patient group array.

    Returns (X_lr, Y_hr, groups) with images as float32 in [0, 1].
    """
    if cache and os.path.exists(cache):
        print(f"[data] loading cached arrays from {cache}")
        blob = np.load(cache, allow_pickle=True)
        return blob["X"], blob["Y"], blob["groups"]

    from PIL import Image, ImageFilter

    lr_size = HR_SIZE // scale
    if HR_SIZE % scale:
        raise ValueError(f"scale {scale} does not divide HR size {HR_SIZE}")

    if not os.path.isdir(data_root):
        raise SystemExit(
            f"[data] not found: {data_root}\n"
            "        Download 'mateuszbuda/lgg-mri-segmentation' from Kaggle\n"
            "        and point --data-root at the kaggle_3m directory."
        )

    patients = sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )
    if limit_patients:
        patients = patients[:limit_patients]
    if not patients:
        raise SystemExit(f"[data] no patient directories under {data_root}")

    print(f"[data] {len(patients)} patients, building x{scale} pairs "
          f"({lr_size}^2 -> {HR_SIZE}^2)")

    X, Y, groups = [], [], []
    for pi, patient in enumerate(patients):
        pdir = os.path.join(data_root, patient)
        for fname in sorted(os.listdir(pdir)):
            # The dataset ships a segmentation mask alongside every slice.
            # Masks are not imagery and must not be treated as samples.
            if not fname.endswith(".tif") or fname.endswith("_mask.tif"):
                continue

            hr_img = Image.open(os.path.join(pdir, fname)).convert("RGB")
            if hr_img.size != (HR_SIZE, HR_SIZE):
                hr_img = hr_img.resize((HR_SIZE, HR_SIZE), Image.BICUBIC)

            lr_img = hr_img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
            lr_img = lr_img.resize((lr_size, lr_size), Image.BICUBIC)

            X.append(np.asarray(lr_img, dtype=np.uint8))
            Y.append(np.asarray(hr_img, dtype=np.uint8))
            groups.append(patient)

        if (pi + 1) % 20 == 0:
            print(f"[data]   {pi + 1}/{len(patients)} patients, {len(X)} slices")

    X = np.asarray(X, dtype=np.float32) / 255.0
    Y = np.asarray(Y, dtype=np.float32) / 255.0
    groups = np.asarray(groups)
    print(f"[data] {len(X)} slices total")

    if cache:
        print(f"[data] caching to {cache}")
        np.savez_compressed(cache, X=X, Y=Y, groups=groups)

    return X, Y, groups


def split_by_patient(groups, seed=SEED):
    """Three-way patient-level split. Returns (train_idx, val_idx, test_idx).

    No patient appears in more than one partition, so correlated adjacent
    slices cannot leak across the boundary.
    """
    from sklearn.model_selection import GroupShuffleSplit

    idx = np.arange(len(groups))

    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                              random_state=seed)
    rest_idx, test_idx = next(outer.split(idx, groups=groups))

    # Re-normalize the validation fraction against the remaining pool.
    inner_frac = VAL_FRACTION / (1.0 - TEST_FRACTION)
    inner = GroupShuffleSplit(n_splits=1, test_size=inner_frac,
                              random_state=seed)
    tr_rel, val_rel = next(
        inner.split(rest_idx, groups=groups[rest_idx])
    )
    train_idx, val_idx = rest_idx[tr_rel], rest_idx[val_rel]

    for name, part in (("train", train_idx), ("val", val_idx),
                       ("test", test_idx)):
        print(f"[split] {name:5s} {len(part):5d} slices, "
              f"{len(set(groups[part])):3d} patients")

    overlap = (set(groups[train_idx]) & set(groups[test_idx])) | \
              (set(groups[val_idx]) & set(groups[test_idx]))
    assert not overlap, f"patient leak across split: {overlap}"

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _psnr_numpy(sr, hr):
    """Per-image PSNR in dB for [0,1] arrays. Matches tf.image.psnr."""
    mse = np.mean((sr - hr) ** 2, axis=(1, 2, 3))
    # Identical images would divide by zero; clamp to a finite ceiling.
    mse = np.maximum(mse, 1e-12)
    return 10.0 * np.log10(1.0 / mse)


def _ssim_skimage(sr, hr):
    """Per-image SSIM via scikit-image, configured to match tf.image.ssim
    (11x11 Gaussian window, sigma 1.5, K1=0.01, K2=0.03)."""
    from skimage.metrics import structural_similarity

    return np.array([
        structural_similarity(
            h, s, data_range=1.0, channel_axis=-1,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        )
        for s, h in zip(sr, hr)
    ])


def _ssim_tensorflow(sr, hr, batch=64):
    import tensorflow as tf

    out = []
    for i in range(0, len(sr), batch):
        a = tf.convert_to_tensor(sr[i:i + batch])
        b = tf.convert_to_tensor(hr[i:i + batch])
        out.append(tf.image.ssim(a, b, max_val=1.0).numpy())
    return np.concatenate(out)


def compute_metrics(sr, hr):
    """Per-image PSNR (dB) and SSIM for [0,1] float arrays.

    PSNR is computed in numpy. SSIM prefers scikit-image and falls back to
    TensorFlow, so that --baseline-only runs without TensorFlow installed.
    """
    psnr = _psnr_numpy(sr, hr)

    try:
        ssim = _ssim_skimage(sr, hr)
    except ImportError:
        try:
            ssim = _ssim_tensorflow(sr, hr)
        except ImportError:
            raise SystemExit(
                "[eval] SSIM needs either scikit-image or tensorflow:\n"
                "        pip install scikit-image")

    return psnr, ssim



def summarize(name, psnr, ssim):
    print(f"\n  {name}")
    print(f"    PSNR  {psnr.mean():6.2f} +/- {psnr.std():.2f} dB")
    print(f"    SSIM  {ssim.mean():6.4f} +/- {ssim.std():.4f}")
    print(f"    n     {len(psnr)}")
    return psnr.mean(), psnr.std(), ssim.mean(), ssim.std()


def latex_row(label, psnr, ssim):
    return (f"{label} & ${psnr.mean():.2f} \\pm {psnr.std():.2f}$ "
            f"& ${ssim.mean():.4f} \\pm {ssim.std():.4f}$ \\\\")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def bicubic_upsample(X_lr):
    """Classical baseline: bicubic back up to HR resolution."""
    from PIL import Image

    out = np.empty((len(X_lr), HR_SIZE, HR_SIZE, 3), dtype=np.float32)
    for i, lr in enumerate(X_lr):
        img = Image.fromarray((lr * 255).astype(np.uint8))
        img = img.resize((HR_SIZE, HR_SIZE), Image.BICUBIC)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0
    return out


def model_predict(model_path, X_lr, batch=16):
    """Run a saved Keras generator over the LR inputs."""
    import tensorflow as tf

    print(f"[model] loading {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    preds = []
    for i in range(0, len(X_lr), batch):
        sr = model.predict(X_lr[i:i + batch], verbose=0)
        # The generator's final activation is tanh, so its range is [-1, 1]
        # while the targets live in [0, 1]. The SRGAN training loop clips
        # identically; see TODO.md.
        preds.append(np.clip(sr, 0.0, 1.0))
    return np.concatenate(preds).astype(np.float32)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="PSNR/SSIM evaluation on a patient-level test split.")
    ap.add_argument("--data-root", default="../mri_dataset/kaggle_3m",
                    help="path to the kaggle_3m directory")
    ap.add_argument("--model", help="saved Keras model / checkpoint to evaluate")
    ap.add_argument("--scale", type=int, default=2, choices=(2, 4, 8),
                    help="upscaling factor the model was trained for")
    ap.add_argument("--baseline-only", action="store_true",
                    help="report bicubic only, skip model inference")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"),
                    help="partition to evaluate on (default: test)")
    ap.add_argument("--cache", help="npz path to cache the built arrays")
    ap.add_argument("--limit-patients", type=int,
                    help="load only the first N patients (smoke test)")
    ap.add_argument("--latex", action="store_true",
                    help="also print LaTeX table rows")
    ap.add_argument("--label", help="row label for --latex")
    args = ap.parse_args()

    if not args.model and not args.baseline_only:
        ap.error("pass --model, or --baseline-only for the bicubic baseline")

    X, Y, groups = load_dataset(args.data_root, args.scale,
                                cache=args.cache,
                                limit_patients=args.limit_patients)

    train_idx, val_idx, test_idx = split_by_patient(groups)
    idx = {"train": train_idx, "val": val_idx, "test": test_idx}[args.split]
    X_eval, Y_eval = X[idx], Y[idx]

    print(f"\n[eval] {args.split} split, x{args.scale}, {len(X_eval)} slices")
    rows = []

    print("[eval] bicubic baseline")
    bic = bicubic_upsample(X_eval)
    b_psnr, b_ssim = compute_metrics(bic, Y_eval)
    summarize("Bicubic", b_psnr, b_ssim)
    rows.append(latex_row(f"Bicubic $\\times {args.scale}$", b_psnr, b_ssim))

    if not args.baseline_only:
        sr = model_predict(args.model, X_eval)
        if sr.shape[1:3] != (HR_SIZE, HR_SIZE):
            raise SystemExit(
                f"[eval] model produced {sr.shape[1:3]}, expected "
                f"{(HR_SIZE, HR_SIZE)} -- is --scale correct?")

        label = args.label or os.path.basename(args.model)
        m_psnr, m_ssim = compute_metrics(sr, Y_eval)
        summarize(label, m_psnr, m_ssim)
        rows.append(latex_row(label, m_psnr, m_ssim))

        d_psnr = m_psnr.mean() - b_psnr.mean()
        d_ssim = m_ssim.mean() - b_ssim.mean()
        print(f"\n  vs bicubic: {d_psnr:+.2f} dB PSNR, {d_ssim:+.4f} SSIM")
        if d_psnr <= 0:
            print("  NOTE: the model does not beat bicubic interpolation on "
                  "PSNR.\n        This needs to be stated plainly in the paper "
                  "if it holds.")

    if args.latex:
        print("\n% --- paste into srgan.tex ---")
        for r in rows:
            print(r)

    return 0


if __name__ == "__main__":
    sys.exit(main())
