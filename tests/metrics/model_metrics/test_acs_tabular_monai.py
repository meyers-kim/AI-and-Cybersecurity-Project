from a4s_eval.metrics.model_metrics.acs_tabular_monai import acs_tabular_monai

def test_acs_tabular_monai_import():
    assert callable(acs_tabular_monai)