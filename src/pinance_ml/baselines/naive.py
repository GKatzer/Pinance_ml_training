import numpy as np


def naive_mae(y_test: np.ndarray) -> float:
    """MAE of the naive r_h=0 forecast: mean(|actual|), since predicted is always 0."""
    y_test = np.asarray(y_test, dtype=float)
    return float(np.nanmean(np.abs(y_test)))


def naive_directional_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Majority-class directional accuracy baseline.

    r_h=0 has no sign of its own, so the direction baseline is a separate
    naive classifier: predict whichever sign (up/down/flat) was most common
    in y_train, apply that single constant prediction across y_test, and
    score how often it matches. This is the base-rate baseline implied by
    the efficient-market framing in the README (~50% expected).
    """
    train_sign = np.sign(np.asarray(y_train, dtype=float))
    train_sign = train_sign[~np.isnan(train_sign)]
    values, counts = np.unique(train_sign, return_counts=True)
    majority_class = values[np.argmax(counts)]

    test_sign = np.sign(np.asarray(y_test, dtype=float))
    valid = ~np.isnan(test_sign)
    return float(np.mean(test_sign[valid] == majority_class))
