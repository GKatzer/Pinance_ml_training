import numpy as np
import pandas as pd

from pinance_ml.evaluation import QUANTILE_GROUP_COLS, QUANTILE_METRIC_COLS, pool_fold_metrics


def _point_fold_results():
    # two folds, one symbol/horizon: fold 0 has 3x the test rows of fold 1,
    # so a naive unweighted mean would land on a different number than the
    # weighted-by-n_test one this function is supposed to compute.
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "horizon": [1, 1],
            "fold": [0, 1],
            "mae": [0.1, 0.2],
            "directional_accuracy": [0.6, 0.4],
            "n_test": [300, 100],
        }
    )


def test_pool_fold_metrics_weights_by_n_test_not_plain_mean():
    pooled = pool_fold_metrics(_point_fold_results())
    row = pooled.iloc[0]

    # weighted: (0.1*300 + 0.2*100) / 400 = 0.125, not the plain mean 0.15
    assert np.isclose(row["mae"], 0.125)
    assert np.isclose(row["directional_accuracy"], (0.6 * 300 + 0.4 * 100) / 400)
    assert row["n_folds"] == 2
    assert row["n_test_total"] == 400


def test_pool_fold_metrics_drops_zero_n_test_folds():
    df = _point_fold_results()
    df.loc[len(df)] = ["BTCUSDT", 1, 2, 999.0, 999.0, 0]  # should be excluded entirely

    pooled = pool_fold_metrics(df)
    assert pooled.iloc[0]["n_folds"] == 2  # not 3


def test_pool_fold_metrics_groups_independently_per_key():
    df = pd.concat(
        [_point_fold_results(), _point_fold_results().assign(horizon=2)],
        ignore_index=True,
    )
    pooled = pool_fold_metrics(df)
    assert set(pooled["horizon"]) == {1, 2}
    assert len(pooled) == 2


def test_pool_fold_metrics_quantile_group_and_metric_cols():
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 4,
            "horizon": [1, 1, 1, 1],
            "quantile": [0.1, 0.1, 0.9, 0.9],
            "fold": [0, 1, 0, 1],
            "pinball_loss": [0.01, 0.02, 0.03, 0.04],
            "coverage": [0.09, 0.11, 0.88, 0.92],
            "n_test": [300, 100, 300, 100],
        }
    )

    pooled = pool_fold_metrics(df, metric_cols=QUANTILE_METRIC_COLS, group_cols=QUANTILE_GROUP_COLS)

    assert set(pooled.columns) >= {"symbol", "horizon", "quantile", "pinball_loss", "coverage"}
    assert len(pooled) == 2  # one row per quantile level, not collapsed together
    low = pooled[pooled["quantile"] == 0.1].iloc[0]
    assert np.isclose(low["pinball_loss"], (0.01 * 300 + 0.02 * 100) / 400)
    assert np.isclose(low["coverage"], (0.09 * 300 + 0.11 * 100) / 400)


def test_pool_fold_metrics_default_args_unchanged_for_existing_callers():
    # every existing script calls pool_fold_metrics(df) with no other args
    # -- generalizing it must not change that call's behavior.
    df = _point_fold_results()
    default_call = pool_fold_metrics(df)
    explicit_call = pool_fold_metrics(df, metric_cols=["mae", "directional_accuracy"], group_cols=["symbol", "horizon"])
    pd.testing.assert_frame_equal(default_call, explicit_call)
