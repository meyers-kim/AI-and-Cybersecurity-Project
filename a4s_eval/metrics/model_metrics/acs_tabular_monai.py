from datetime import datetime
from typing import List
import numpy as np
import pandas as pd

from monai.transforms import RandGaussianNoise

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import TabularClassificationModel


@model_metric(name="acs_tabular_monai")
def acs_tabular_monai(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: TabularClassificationModel,
) -> List[Measure]:
    """
    ACS metric using MONAI GaussianNoise transform applied to tabular data.
    """

    df: pd.DataFrame = dataset.data

    target_col = datashape.target.name
    feature_cols = [f.name for f in datashape.features]

    y_true = df[target_col].to_numpy()
    X = df[feature_cols]

    # baseline
    X_np = X.to_numpy()
    y_pred_base_raw = functional_model.predict_class(X_np)
    y_pred_base = np.asarray(y_pred_base_raw)
    if y_pred_base.ndim > 1:
        y_pred_base = y_pred_base.argmax(axis=1)

    # MONAI transform, it works directly on Numpy arrays
    noise = RandGaussianNoise(prob=1.0, std=0.05)
    X_noisy_np = noise(X_np)

    y_pred_noisy_raw = functional_model.predict_class(X_noisy_np)
    y_pred_noisy = np.asarray(y_pred_noisy_raw)
    if y_pred_noisy.ndim > 1:
        y_pred_noisy = y_pred_noisy.argmax(axis=1)

    y_true = np.asarray(y_true)

    acc_base = (y_pred_base == y_true).mean()
    acc_noisy = (y_pred_noisy == y_true).mean()
    acs = (y_pred_base == y_pred_noisy).mean()
    acc_drop = acc_base - acc_noisy

    print(
        f"[ACS_TABULAR_MONAI] acs={acs:.4f}, "
        f"acc_base={acc_base:.4f}, acc_noisy={acc_noisy:.4f}, acc_drop={acc_drop:.4f}"
    )


    now = datetime.now()
    return [
        Measure(name="acs_monai", score=float(acs), time=now),
        Measure(name="accuracy_baseline", score=float(acc_base), time=now),
        Measure(name="accuracy_noisy", score=float(acc_noisy), time=now),
        Measure(name="acc_drop", score=float(acc_drop), time=now),
    ]