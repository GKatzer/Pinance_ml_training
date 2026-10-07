import lightgbm as lgb
import pandas as pd

from pinance_ml.config import HORIZONS

# objective='regression_l1' trains on MAE directly, matching the metric
# we evaluate with (README's chosen metric), rather than the L2 default.
# deterministic + force_row_wise trade a bit of speed for reproducible
# splits across runs/threads, per LightGBM's own reproducibility guidance.
DEFAULT_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "random_state": 42,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": -1,
    "verbosity": -1,
}


def train_horizon_models(
    train: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int] = HORIZONS,
    params: dict | None = None,
    sample_weight: pd.Series | None = None,
) -> dict[int, lgb.LGBMRegressor]:
    """Direct multi-horizon, per README: one independent LGBMRegressor per
    horizon, all sharing the same feature matrix but fit to a different
    target column (r_h) — no model sees another horizon's target, and
    predicting horizon h+1 never runs through horizon h's output.

    LightGBM routes NaN feature values to a learned branch at each split
    (its native missing-value handling), so rows in the warm-up period
    (short on lag/rolling history) don't need to be dropped or imputed —
    only rows with a NaN *target* are unusable for training that horizon.

    sample_weight, if given, must share `train`'s index (e.g.
    splits.recency_sample_weight's output) -- sliced by the same per-horizon
    `valid` mask as X/y so weights stay aligned with the rows LightGBM
    actually sees for that horizon, not `train`'s full row count.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    models = {}
    for h in horizons:
        y_col = f"r_{h}"
        valid = train[y_col].notna()
        model = lgb.LGBMRegressor(**params)
        weights = sample_weight.loc[valid] if sample_weight is not None else None
        model.fit(train.loc[valid, feature_cols], train.loc[valid, y_col], sample_weight=weights)
        models[h] = model
    return models


def predict_horizons(
    models: dict[int, lgb.LGBMRegressor], test: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        {f"r_{h}_pred": model.predict(test[feature_cols]) for h, model in models.items()},
        index=test.index,
    )


# Same defaults as DEFAULT_PARAMS except the objective: quantile regression
# optimizes pinball loss at a given `alpha`, not L1/L2 around a single point
# estimate -- everything else (tree size, learning rate, determinism knobs)
# is kept identical on purpose, so a quantile-vs-point comparison isn't
# confounded by also changing the base model's capacity.
DEFAULT_QUANTILE_PARAMS = {
    "objective": "quantile",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "random_state": 42,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": -1,
    "verbosity": -1,
}

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


def train_quantile_models(
    train: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int] = HORIZONS,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    params: dict | None = None,
    sample_weight: pd.Series | None = None,
) -> dict[tuple[int, float], lgb.LGBMRegressor]:
    """One independent LGBMRegressor per (horizon, quantile) pair -- the
    confidence-corridor counterpart of train_horizon_models. Kept as a
    separate function rather than a branch inside train_horizon_models: the
    output shape differs (3x models per horizon, keyed by (horizon,
    quantile) instead of horizon alone) and the two are never meant to
    be trained together in one call.

    Each (horizon, quantile) model is fit independently -- LightGBM's
    `objective="quantile"` with `alpha=quantile` trains toward that one
    quantile in isolation, so nothing here enforces the alpha=0.1 model's
    predictions to stay below the alpha=0.9 model's (quantile crossing is a
    known possibility with independently-fit quantile models); checking for
    it is an evaluation-time concern (coverage in metrics.py), not something
    this function needs to prevent.
    """
    params = {**DEFAULT_QUANTILE_PARAMS, **(params or {})}
    models = {}
    for h in horizons:
        y_col = f"r_{h}"
        valid = train[y_col].notna()
        X = train.loc[valid, feature_cols]
        y = train.loc[valid, y_col]
        weights = sample_weight.loc[valid] if sample_weight is not None else None
        for q in quantiles:
            model = lgb.LGBMRegressor(**{**params, "alpha": q})
            model.fit(X, y, sample_weight=weights)
            models[(h, q)] = model
    return models


def predict_quantile_horizons(
    models: dict[tuple[int, float], lgb.LGBMRegressor], test: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        {f"r_{h}_q{q}_pred": model.predict(test[feature_cols]) for (h, q), model in models.items()},
        index=test.index,
    )
