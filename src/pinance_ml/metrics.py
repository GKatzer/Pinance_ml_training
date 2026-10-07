import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.nanmean(np.abs(actual - predicted)))


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of rows where the predicted return has the same sign as the
    actual one. Unlike `baselines.naive.naive_directional_accuracy` (which
    fits a majority-class constant, since r_h=0 has no sign of its own),
    a real model's predictions already carry a sign — no fitting step
    needed, just compare directly."""
    actual_sign = np.sign(np.asarray(actual, dtype=float))
    predicted_sign = np.sign(np.asarray(predicted, dtype=float))
    valid = ~(np.isnan(actual_sign) | np.isnan(predicted_sign))
    return float(np.mean(actual_sign[valid] == predicted_sign[valid]))


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    """Quantile ("pinball") loss: what `objective="quantile"` actually
    optimizes, and so the fair metric to score a quantile model by (MAE
    is only the right loss for the median, alpha=0.5).

    loss = mean(max(q*(y-yhat), (q-1)*(y-yhat))) -- asymmetric around 0,
    penalizing under-prediction q times as hard as over-prediction for
    q > 0.5 (and the reverse for q < 0.5), so a model that actually shades
    its output toward the target quantile scores better than one that
    just predicts the conditional mean/median everywhere.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    diff = actual - predicted
    loss = np.maximum(quantile * diff, (quantile - 1) * diff)
    return float(np.nanmean(loss))


def coverage(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of actual values at or below `predicted` -- the calibration
    check for a quantile model, the pinball-loss analog of what
    `directional_accuracy` is for a point regressor: a low pinball loss
    doesn't by itself say whether the model is honest about uncertainty,
    only a coverage check does. For an alpha-quantile model evaluated
    out-of-sample, this should be close to alpha (e.g. ~0.1 for the
    alpha=0.1 model, ~0.9 for alpha=0.9)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    return float(np.mean(actual[valid] <= predicted[valid]))
