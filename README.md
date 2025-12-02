# AI Security Project – Augmentation Consistency Score (ACS)
**Course:** AI and Cybersecurity - University of Luxembourg (2025/2026)

**Name:** Kim Meyers (021105204B)

This repository contains my work for the AI Security course project.
The goal is to implement a custom metric and integrate it into the A4S evaluation framework.

---

## 1. Metric: Augmentation Consistency Score (ACS)

**Location:**
`a4s-eval/a4s_eval/metrics/model_metrics/acs_tabular_monai.py`

**Registry decorator:**
`@model_metric`

**Idea:**
ACS measures how stable a model’s predictions are when the input is slightly perturbed.

The model predicts on:

* the original input data
* an augmented/perturbed version created with MONAI (Gaussian noise + smoothing)

**Formula:**

```bash
ACS = % of predictions that remain the same after augmentation
```

**The metric returns:**

* baseline accuracy (`acc_base`)
* augmented accuracy (`acc_aug`)
* ACS value (`acs`)
* accuracy drop (`acc_drop`)

**Applies to:**
Tabular classification models in A4S (`TabularClassificationModel`).

---

## 2. MONAI Usage

Although MONAI is mainly a medical imaging library, in this project it was used as an augmentation toolkit to introduce small, realistic perturbations to both tabular and image data.

**Transforms used:**

* `RandGaussianNoise` – adds small random Gaussian noise
* `RandGaussianSmooth` – smooths out features slightly
* `RandAffine` *(image tests only)* – small rotations, translations, and scaling

These augmentations simulate small input changes to test how robust and consistent models remain in their predictions.

---

## 3. Tabular Data Metric (A4S Integration)

A4S automatically discovers and runs the metric using:

```bash
tests/metrics/model_metrics/test_execute.py
```

Additional import validation:

```bash
tests/metrics/model_metrics/test_acs_tabular_monai.py
```

**Run tests inside `a4s-eval/`:**

```bash
uv sync
uv run pytest -s tests/metrics/model_metrics
```

**Expected output example:**

```bash
[ACS_TABULAR_MONAI] acs=0.9400, acc_base=0.7310, acc_noisy=0.7290, acc_drop=0.0020
```

**Results saved to:**

```bash
tests/data/measures/acs_tabular_monai.csv
```

**Notebook preview:**
`acs_results.ipynb` displays a before-and-after view of the tabular dataset.

This confirms the metric works and all tests pass.

---

## 4. CIFAR-100 ACS Evaluation (Image Extension)

To visually and quantitatively compare how ACS behaves on image models, an additional test script was created.

**File:**
`tests/metrics/model_metrics/test_cifar100_acs_models.py`

**Models tested:**

* TinyCNN100
* DeeperCNN100
* ResNet18_100

Each model is trained briefly on a subset of CIFAR-100, then evaluated using MONAI augmentations to measure ACS.

**Run command:**

```bash
uv run pytest -s tests/metrics/model_metrics/test_cifar100_acs_models.py
```

**Example output:**

```bash
Using device: cuda

Training TinyCNN100
TinyCNN100 - train loss: 4.6034
Evaluating ACS for TinyCNN100 on CIFAR-100...
TinyCNN100 - acc_base=0.0200, acc_aug=0.0230, acs=0.4830, acc_drop=-0.0030
```

**Results saved to:**

```bash
tests/data/measures/acs_cifar100_models.csv
```

**Visual results:**
Images where the model’s prediction flips after augmentation are automatically saved to:

```bash
image_acs_flips_cifar100/<model_name>/
```

Each image shows `[original | augmented]` side-by-side where the prediction changed.

> **Note:** The CIFAR-100 dataset is not stored in this repository due to GitHub size limits.
> It is automatically downloaded using `torchvision.datasets.CIFAR100(download=True)` the first time you run the test.

---

## 5. Notebook

**File:**
`acs_results.ipynb`

The notebook visualizes ACS results for both tabular and image data, including:

* Bar plots comparing baseline accuracy, augmented accuracy, and ACS for each CNN model
* Before/after tables showing how Gaussian noise modifies tabular features
* CIFAR-100 examples where predictions change between original and augmented images

This notebook provides the main visuals for the final presentation.

---

## 6. Experimental Findings

* ACS reveals how robust models are to small input changes.
* **Higher ACS values** indicate **more stable and consistent models**.
* On CIFAR-100, deeper models like ResNet18 show higher robustness but still experience a small accuracy drop.
* On tabular data, models are more stable overall due to lower feature variability.
* Visual examples highlight prediction flips caused by minor perturbations, emphasizing the importance of robustness testing.

---

## 7. How to Reproduce

All tests and metrics can be run directly:

```bash
uv run pytest -s
```

CIFAR datasets are downloaded automatically if missing.

**Results saved to:**

```bash
tests/data/measures/
```

**Flip images (qualitative analysis):**

```bash
image_acs_flips_cifar100/
```

**Tabular before/after noise comparison:**
Included in `acs_results.ipynb`.

---