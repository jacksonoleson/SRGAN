# SRGAN paper — remaining work

Tracking the gap between the current write-up (`srgan.tex`) and a defensible
submission. The paper builds clean and the layout is done; what remains is
almost entirely about **quantitative evaluation**.

Code lives in the sibling repo `../SRGAN` (`code/trainSRNET.ipynb`,
`code/trainSRGAN.ipynb`). The evaluation harness is `../SRGAN/code/evaluate.py`.

---

## Blockers

Nothing downstream works until these are resolved.

- [ ] **Download the dataset.** Buda et al. LGG Segmentation, Kaggle
      `mateuszbuda/lgg-mri-segmentation` (~1.5 GB). Both notebooks expect it at
      `../mri_dataset/kaggle_3m/` relative to `code/`, i.e. a sibling of the
      repo. `evaluate.py --data-root` can point anywhere.
- [ ] **Confirm the trained weights are gone.** Check HPC scratch and home for
      `code/SRResNet/weights_epoch_*.h` or `super_resolution_model.keras`.
      If any survive, skip the SRResNet retrain and go straight to evaluation —
      this removes most of the remaining effort.

## Core evaluation

This is what actually fills the results table.

- [ ] **Switch to a patient-level split.** Both notebooks call
      `train_test_split(X, y, test_size=..., random_state=42)`, which splits
      per-slice. Adjacent slices from one MRI session are near-duplicates, so
      the current validation numbers are optimistic. Needs `GroupShuffleSplit`
      grouped on patient directory — which means retaining the patient ID when
      loading, currently discarded in cell 2 of both notebooks.
      `evaluate.py` already does this correctly; port the same logic into
      training.
- [ ] **Carve out a held-out test set.** 70/15/15 by patient. Model selection
      (e.g. the epoch-6 overfitting call) stays on validation; the paper table
      reports test only.
- [ ] **Retrain SRResNet x2.** 10 epochs, batch 32, Adam 1e-3, MSE. Cheap and
      low-risk — this is the win to secure first.
- [ ] **Retrain SRResNet x4.** Same, `scale=4`, 64x64 inputs.
- [ ] **Run `evaluate.py`** against each checkpoint to get mean +/- std PSNR
      and SSIM on the test split.
- [ ] **Bicubic baseline.** `evaluate.py --baseline-only` produces it. Without
      this the table cannot support any claim that the network beats classical
      interpolation — and that claim is currently implicit throughout the paper.

## Paper updates

Once numbers exist.

- [ ] **Add the results table** to `srgan.tex`. Rows: Bicubic, SRResNet x2,
      SRResNet x4 (, SRGAN). Columns: PSNR (dB), SSIM.
      `evaluate.py --latex` emits the rows directly.
- [ ] **Rewrite the Results prose** (Sec. V) to cite numbers rather than
      "not completely satisfactory" / "much less satisfactory".
- [ ] **Trim the hedging.** Sec. IV-A (Evaluation Metrics) and Sec. VI-C
      (Evaluation and Experimental Design) were written on the assumption that
      no metrics exist. Once the table lands, the PSNR/SSIM definitions can
      stay but the apologetics should go.
- [ ] **Re-check the epoch-6 overfitting claim** against the new patient-level
      validation curve. It may not survive the stricter split.
- [ ] **Regenerate loss figures** if retraining — `SRResNet_losses.png`,
      `SRResNet_64_losses.png` currently correspond to the old split.

## Optional

- [ ] **Retrain SRGAN.** Expensive, unstable, ~3700 steps at batch size 1, and
      the outcome is already known to be non-convergent. Not required: a
      qualitative negative result is legitimate as long as the working model is
      quantified. PSNR also structurally disfavours adversarial output, so the
      number would be easy to misread — see the argument now in Sec. IV-A.
- [ ] **Fix the PSNR logging bug.** `trainSRGAN.ipynb` cell 13 declares
      `psnr_values = []` inside the validation block and never appends to it;
      `tf.image.psnr` is imported but never called, and `srgan_checkpoint.psnr`
      stays 0.0 for the whole run. One-line fix, worth doing before any retrain.
- [ ] **Extend Fig. 6 with `sr_3300.png` / `sr_3700.png`.** Both are in
      `images/` and unused. The figure currently stops at step 3000, which
      under-illustrates the "destabilizes as training proceeds" claim using
      images you already have. Would make it a 2x4 grid.
- [ ] **Consider the `tanh` output range.** The generator's final layer is
      `activation='tanh'` (range [-1, 1]) while targets are normalized to
      [0, 1]; the original Ledig formulation denormalizes from [-1, 1], and
      that `Lambda(denormalize_m11)` call is commented out in
      `build_srresnet`. Not fatal — [0,1] is inside the tanh range and the
      SRGAN loop clips — but half the output range is unused, and switching to
      `sigmoid` (or restoring the denormalization) is worth a quick ablation.
- [ ] **Vendored TeX files.** `IEEEtran.cls` and `IEEEtran.bst` are committed
      for portability. Alternative: drop them and `apt install
      texlive-publishers`.
- [ ] **PDF size.** 13 MB, because the panel PNGs are embedded at full
      3000x1500. Downsample if there is a submission cap.

---

## Fixed already

For reference, corrected during the layout pass:

- Fig. 5 was captioned as SRResNet losses; it is the SRGAN loss plot.
- Fig. 4 was captioned "from epoch 6"; the image is `..._epoch9`.
- Text claimed "batches of 16 images"; the notebook uses `batch_size=32`.
- Perceptual loss equation summed over `i` in both indices; now `x` and `y`.
- Two figures shared `\label{fig:enter-label}`.
- All floats were `[H]`, which was the cause of the large column gaps.

## Build

```
latexmk -pdf srgan.tex
```
