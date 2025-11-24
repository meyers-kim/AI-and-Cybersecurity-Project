from datetime import datetime
from typing import List
import numpy as np
import pandas as pd

try:
    from monai.transforms import RandGaussianNoise # type: ignore
except ModuleNotFoundError:
    class RandGaussianNoise:
        """
        Lightweight fallback for monais RandGaussianNoise transform.
        Works on NumPy arrays and matches the interface used below.
        """

        def __init__(self, prob: float = 0.1, std: float = 0.1, mean: float = 0.0) -> None:
            self.prob = prob
            self.std = std
            self.mean = mean

        def __call__(self, x: np.ndarray) -> np.ndarray:
            # same semantics as monai, with probability apply noise, otherwise return x
            if np.random.rand() > self.prob:
                return x
            noise = np.random.normal(loc=self.mean, scale=self.std, size=x.shape)
            return x + noise

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