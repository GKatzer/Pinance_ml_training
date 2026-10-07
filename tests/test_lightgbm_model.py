import lightgbm as lgb
import numpy as np
import pandas as pd

from pinance_ml.metrics import coverage, mae, pinball_loss
from pinance_ml.models.lightgbm_model import (
    predict_horizons,
    predict_quantile_horizons,
    train_horizon_models,
    train_quantile_models,
)


def _signal_dataset(n=2000, seed=0):
    """x1 carries a real (noisy) linear signal for r_1; x2 is pure noise."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.01, size=n)
    r_1 = 0.05 * x1 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "r_1": r_1, "r_2": r_1 * 2})


def test_train_horizon_models_returns_one_independent_model_per_horizon():
    df = _signal_dataset()
    models = train_horizon_models(df, feature_cols=["x1", "x2"], horizons=[1, 2])

    assert set(models.keys()) == {1, 2}
    assert all(isinstance(m, lgb.LGBMRegressor) for m in models.values())
    assert models[1] is not models[2]


def test_predict_horizons_output_shape_and_columns():
    df = _signal_dataset()
    models = train_horizon_models(df, feature_cols=["x1", "x2"], horizons=[1])
    test = df.iloc[:10]

    preds = predict_horizons(models, test, feature_cols=["x1", "x2"])

    assert list(preds.columns) == ["r_1_pred"]
    assert len(preds) == 10


def test_model_recovers_a_real_signal_and_beats_predicting_zero():
    train = _signal_dataset(n=4000, seed=1)
    test = _signal_dataset(n=1000, seed=2)

    models = train_horizon_models(train, feature_cols=["x1", "x2"], horizons=[1])
    preds = predict_horizons(models, test, feature_cols=["x1", "x2"])

    model_mae = mae(test["r_1"].to_numpy(), preds["r_1_pred"].to_numpy())
    naive_mae = mae(test["r_1"].to_numpy(), np.zeros(len(test)))

    assert model_mae < naive_mae


def test_training_rows_with_nan_target_are_excluded_not_errored():
    df = _signal_dataset(n=200)
    df.loc[:9, "r_1"] = np.nan  # simulate warmup/purged rows
    models = train_horizon_models(df, feature_cols=["x1", "x2"], horizons=[1])
    assert 1 in models


def test_train_quantile_models_returns_one_independent_model_per_horizon_quantile_pair():
    df = _signal_dataset()
    models = train_quantile_models(df, feature_cols=["x1", "x2"], horizons=[1, 2], quantiles=(0.1, 0.5, 0.9))

    assert set(models.keys()) == {(1, 0.1), (1, 0.5), (1, 0.9), (2, 0.1), (2, 0.5), (2, 0.9)}
    assert all(isinstance(m, lgb.LGBMRegressor) for m in models.values())
    assert models[(1, 0.1)] is not models[(1, 0.9)]


def test_predict_quantile_horizons_output_shape_and_columns():
    df = _signal_dataset()
    models = train_quantile_models(df, feature_cols=["x1", "x2"], horizons=[1], quantiles=(0.1, 0.9))
    test = df.iloc[:10]

    preds = predict_quantile_horizons(models, test, feature_cols=["x1", "x2"])

    assert set(preds.columns) == {"r_1_q0.1_pred", "r_1_q0.9_pred"}
    assert len(preds) == 10


def test_quantile_models_are_monotonic_on_a_real_signal():
    # not guaranteed in general (quantile crossing is possible for
    # independently-fit models), but on this easy a signal with plenty of
    # data the 0.1/0.5/0.9 predictions should still come out ordered.
    train = _signal_dataset(n=4000, seed=1)
    test = _signal_dataset(n=1000, seed=2)

    models = train_quantile_models(train, feature_cols=["x1", "x2"], horizons=[1], quantiles=(0.1, 0.5, 0.9))
    preds = predict_quantile_horizons(models, test, feature_cols=["x1", "x2"])

    assert (preds["r_1_q0.1_pred"] <= preds["r_1_q0.5_pred"]).mean() > 0.95
    assert (preds["r_1_q0.5_pred"] <= preds["r_1_q0.9_pred"]).mean() > 0.95


def test_quantile_model_beats_naive_pinball_loss_on_a_real_signal():
    train = _signal_dataset(n=4000, seed=1)
    test = _signal_dataset(n=1000, seed=2)

    models = train_quantile_models(train, feature_cols=["x1", "x2"], horizons=[1], quantiles=(0.9,))
    preds = predict_quantile_horizons(models, test, feature_cols=["x1", "x2"])

    model_loss = pinball_loss(test["r_1"].to_numpy(), preds["r_1_q0.9_pred"].to_numpy(), quantile=0.9)
    naive_loss = pinball_loss(test["r_1"].to_numpy(), np.zeros(len(test)), quantile=0.9)
    assert model_loss < naive_loss


def test_quantile_model_training_rows_with_nan_target_are_excluded_not_errored():
    df = _signal_dataset(n=200)
    df.loc[:9, "r_1"] = np.nan
    models = train_quantile_models(df, feature_cols=["x1", "x2"], horizons=[1], quantiles=(0.5,))
    assert (1, 0.5) in models


def test_quantile_model_coverage_roughly_matches_alpha_on_a_real_signal():
    train = _signal_dataset(n=8000, seed=3)
    test = _signal_dataset(n=2000, seed=4)

    models = train_quantile_models(train, feature_cols=["x1", "x2"], horizons=[1], quantiles=(0.1, 0.9))
    preds = predict_quantile_horizons(models, test, feature_cols=["x1", "x2"])

    cov_low = coverage(test["r_1"].to_numpy(), preds["r_1_q0.1_pred"].to_numpy())
    cov_high = coverage(test["r_1"].to_numpy(), preds["r_1_q0.9_pred"].to_numpy())
    assert abs(cov_low - 0.1) < 0.05
    assert abs(cov_high - 0.9) < 0.05
