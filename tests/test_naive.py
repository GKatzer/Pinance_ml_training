import numpy as np

from pinance_ml.baselines.naive import naive_directional_accuracy, naive_mae


def test_naive_mae_is_mean_abs_actual_ignoring_nan():
    y_test = np.array([1.0, -2.0, 3.0, np.nan])
    assert naive_mae(y_test) == 2.0


def test_naive_directional_accuracy_picks_majority_class_from_train():
    y_train = np.array([0.1, 0.2, -0.1, 0.3])  # 3 up, 1 down -> majority = up (+1)
    y_test = np.array([0.05, -0.05, 0.2, -0.2])  # 2 up, 2 down

    acc = naive_directional_accuracy(y_train, y_test)
    assert acc == 0.5


def test_naive_directional_accuracy_perfect_when_test_all_matches_majority():
    y_train = np.array([-0.1, -0.2, 0.05])  # majority = down (-1)
    y_test = np.array([-0.3, -0.1])

    assert naive_directional_accuracy(y_train, y_test) == 1.0


def test_naive_directional_accuracy_ignores_nan_test_rows():
    y_train = np.array([0.1, 0.2, -0.1])
    y_test = np.array([0.1, np.nan, 0.1])

    assert naive_directional_accuracy(y_train, y_test) == 1.0
