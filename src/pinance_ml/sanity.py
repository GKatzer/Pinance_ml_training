"""Sanity gate for candidates whose feature schema changed.

On a schema change auto_retrain*.py can't compare the candidate against
production (different feature sets), so the push gate used to have no
quality bar on that path at all. The only always-available reference is
the naive baseline (`baselines.naive`), scored on the same held-out
window as the candidate. Everything here is pure (no DB/MinIO/model
training) so it's unit-testable; the scripts feed it numbers they
already computed.
"""

import numpy as np

from pinance_ml.baselines.naive import naive_directional_accuracy, naive_mae
from pinance_ml.metrics import pinball_loss


def naive_point_baseline(train_data, eval_data, horizons: list[int]) -> tuple[float, float]:
    """(avg naive MAE, avg naive majority-class directional accuracy)
    across `horizons` on `eval_data`, the same two averages the candidate
    is scored by. The majority class is fit on `train_data` (the candidate's
    own training window) so the baseline sees no more than the model did."""
    maes, accs = [], []
    for h in horizons:
        y_train = train_data[f"r_{h}"].to_numpy()
        y_eval = eval_data[f"r_{h}"].to_numpy()
        if np.isnan(y_eval).all():
            continue
        maes.append(naive_mae(y_eval))
        accs.append(naive_directional_accuracy(y_train, y_eval))
    return float(np.mean(maes)), float(np.mean(accs))


def naive_quantile_pinball(train_data, eval_data, horizons: list[int], quantiles: tuple[float, ...]) -> float:
    """Avg pinball loss across every (horizon, quantile) pair of the
    constant baseline "predict the training window's empirical q-quantile
    of r_h everywhere" -- the quantile counterpart of the naive point
    forecast, and what a corridor that learned nothing converges to."""
    losses = []
    for h in horizons:
        y_train = train_data[f"r_{h}"].to_numpy()
        y_eval = eval_data[f"r_{h}"].to_numpy()
        valid_eval = y_eval[~np.isnan(y_eval)]
        y_train = y_train[~np.isnan(y_train)]
        if len(valid_eval) == 0 or len(y_train) == 0:
            continue
        for q in quantiles:
            const = float(np.quantile(y_train, q))
            losses.append(pinball_loss(valid_eval, np.full_like(valid_eval, const), quantile=q))
    return float(np.mean(losses))


def point_sanity_check(
    candidate_mae: float,
    candidate_dir_acc: float,
    naive_avg_mae: float,
    naive_avg_dir_acc: float,
    max_mae_ratio: float,
    min_dir_acc_edge: float,
) -> tuple[bool, str]:
    """Does a new-scheme point candidate at least match the naive
    baseline? Returns (passed, reason) -- reason is "ok" when passed,
    else which bar failed (goes into the audit log / eval_metrics)."""
    if any(np.isnan(v) for v in (candidate_mae, candidate_dir_acc, naive_avg_mae, naive_avg_dir_acc)):
        return False, "nan_metric"
    if candidate_mae > naive_avg_mae * max_mae_ratio:
        return False, f"mae_worse_than_naive ({candidate_mae:.6f} > {naive_avg_mae:.6f} * {max_mae_ratio})"
    if candidate_dir_acc < naive_avg_dir_acc + min_dir_acc_edge:
        return False, f"dir_acc_below_naive ({candidate_dir_acc:.4f} < {naive_avg_dir_acc:.4f} + {min_dir_acc_edge})"
    return True, "ok"


def quantile_sanity_check(
    candidate_pinball: float,
    naive_pinball: float,
    coverage_ok: bool,
    max_pinball_ratio: float,
) -> tuple[bool, str]:
    """Corridor counterpart of point_sanity_check: pinball loss no worse
    than the constant-quantile baseline (within `max_pinball_ratio`) and
    calibrated coverage."""
    if np.isnan(candidate_pinball) or np.isnan(naive_pinball):
        return False, "nan_metric"
    if not coverage_ok:
        return False, "coverage_not_calibrated"
    if candidate_pinball > naive_pinball * max_pinball_ratio:
        return False, f"pinball_worse_than_naive ({candidate_pinball:.6f} > {naive_pinball:.6f} * {max_pinball_ratio})"
    return True, "ok"
