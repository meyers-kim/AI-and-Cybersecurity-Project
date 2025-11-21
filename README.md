# AI Security Project – Augmentation Consistency Score (ACS)
**Course:** AI and Cybersecurity  
**University:** University of Luxembourg (2025/2026)  
**Name:** Kim Meyers  
**Student ID:** 021105204B

This repository contains my work for the AI Security course project.
The goal is to implement a custom metric and integrate it into the A4S evaluation framework.

---

## 1. Metric: Augmentation Consistency Score (ACS)

Location:  
a4s-eval/a4s_eval/metrics/model_metrics/acs_tabular_monai.py

Registry decorator:  
@model_metric

Idea:  
ACS measures how stable a model’s predictions are when the input is slightly perturbed.

The model predicts on:
- the original input data
- an augmented/perturbed version created with MONAI (Gaussian noise + smoothing)

ACS = % of predictions that remain the same after augmentation

The metric returns:
- baseline accuracy
- augmented accuracy
- ACS value
- accuracy drop

Applies to:
Tabular classification models in A4S (TabularClassificationModel).


## 2. MONAI Usage

Although MONAI is mainly used for medical imaging, here it is used purely as an augmentation library for mild perturbations on tabular feature arrays.

Transforms used:
- RandGaussianNoise
- RandGaussianSmooth

---

## 3. Tests

A4S automatically discovers and runs the metric using:

tests/metrics/model_metrics/test_execute.py

Additional import validation:

tests/metrics/model_metrics/test_acs_tabular_monai.py

Run tests inside a4s-eval/:

uv sync
uv run pytest -s tests/metrics/model_metrics

Expected output example:

[ACS_TABULAR_MONAI] acs=0.9400, acc_base=0.7310, acc_noisy=0.7290, acc_drop=0.0020

This confirms the metric works and all tests pass.



## 4. Notebook (for presentation)

A Jupyter notebook is included to visually demonstrate the ACS idea using CIFAR-10 images.  
This was allowed and approved in class since tabular data is not visually interpretable very well.

The notebook shows:
- original vs augmented images
- prediction flips
- accuracy and ACS bar charts

The notebook is not required by A4S but is helpful for my final presentation.