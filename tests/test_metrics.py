import numpy as np

from pinance_ml.metrics import coverage, directional_accuracy, mae, pinball_loss


def test_mae_is_mean_abs_error_ignoring_nan():
    actual = np.array([1.0, -2.0, 3.0, np.nan])
    predicted = np.array([0.5, -2.5, 3.0, 1.0])
    # |1-0.5|=0.5, |-2-(-2.5)|=0.5, |3-3|=0, nan row dropped -> mean(0.5,0.5,0)
    assert np.isclose(mae(actual, predicted), 1.0 / 3)


def test_directional_accuracy_counts_matching_signs():
    actual = np.array([0.1, -0.1, 0.2, -0.3])
    predicted = np.array([0.05, -0.2, -0.1, -0.05])  # matches, matches, wrong, matches
    assert np.isclose(directional_accuracy(actual, predicted), 0.75)


def test_directional_accuracy_ignores_nan_on_either_side():
    actual = np.array([0.1, np.nan, 0.2])
    predicted = np.array([0.1, 0.1, np.nan])
    assert directional_accuracy(actual, predicted) == 1.0  # only row 0 is scoreable


def test_pinball_loss_at_median_is_half_mae():
    # q=0.5: max(0.5*d, -0.5*d) == 0.5*|d| for every row, so pinball == mae/2.
    actual = np.array([1.0, -2.0, 3.0])
    predicted = np.array([0.5, -2.5, 3.0])
    assert np.isclose(pinball_loss(actual, predicted, 0.5), mae(actual, predicted) / 2)


def test_pinball_loss_penalizes_underprediction_more_at_high_quantile():
    # same |error| either side of the target, but q=0.9 should score the
    # under-prediction (actual > predicted) worse than the over-prediction.
    actual = np.array([1.0])
    under = pinball_loss(actual, np.array([0.5]), quantile=0.9)  # d=+0.5 -> 0.9*0.5=0.45
    over = pinball_loss(actual, np.array([1.5]), quantile=0.9)  # d=-0.5 -> -0.1*-0.5=0.05
    assert under > over
    assert np.isclose(under, 0.45)
    assert np.isclose(over, 0.05)


def test_pinball_loss_zero_for_perfect_predictions():
    actual = np.array([1.0, -2.0, 0.3])
    assert pinball_loss(actual, actual.copy(), quantile=0.1) == 0.0


def test_pinball_loss_ignores_nan():
    actual = np.array([1.0, np.nan])
    predicted = np.array([1.0, 5.0])
    assert pinball_loss(actual, predicted, quantile=0.5) == 0.0


def test_coverage_fraction_at_or_below_predicted():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([2.0, 2.0, 2.0, 2.0])  # actual <= predicted for rows 0,1
    assert np.isclose(coverage(actual, predicted), 0.5)


def test_coverage_is_near_alpha_for_well_calibrated_quantile():
    rng = np.random.default_rng(0)
    actual = rng.normal(size=5000)
    # the true 0.1-quantile of a standard normal
    from scipy.stats import norm

    predicted = np.full_like(actual, norm.ppf(0.1))
    assert abs(coverage(actual, predicted) - 0.1) < 0.02


def test_coverage_ignores_nan_on_either_side():
    actual = np.array([1.0, np.nan, 3.0])
    predicted = np.array([1.0, 1.0, np.nan])
    assert coverage(actual, predicted) == 1.0  # only row 0 is scoreable
