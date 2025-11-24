from datetime import datetime
import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import TabularClassificationModel


@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: TabularClassificationModel,
) -> list[Measure]:
    # 1. full dataframe
    df: pd.DataFrame = dataset.data

    # we can add a limit to avoid heavy computation if needed
    # df = df.head(10_000)

    # 2. target and feature columns from datashape
    target_col = datashape.target.name
    feature_cols = [f.name for f in datashape.features]

    y_true = df[target_col].to_numpy()
    X = df[feature_cols]

    # 3. useing the functional model to get class predictions:
    # TabularClassificationModel defines predict_class: Array -> Array
    # Array is typically a numpy array.
    X_np = X.to_numpy()
    y_pred_raw = functional_model.predict_class(X_np)

    y_pred = np.asarray(y_pred_raw)

    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)

    # 4. accuracy
    acc_value = float((y_true == y_pred).mean())

    current_time = datetime.now()
    return [Measure(name="accuracy", score=acc_value, time=current_time)]
