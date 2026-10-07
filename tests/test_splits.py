import numpy as np
import pandas as pd

from pinance_ml.splits import recency_sample_weight, walk_forward_folds


def _daily_candles(n_days: int, start: str = "2026-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=n_days, freq="1D", tz="UTC")
    return pd.DataFrame({"ts": ts, "value": range(n_days)})


def test_first_fold_test_window_starts_after_min_train_days():
    df = _daily_candles(40)
    folds = list(walk_forward_folds(df, min_train_days=10, test_days=5, purge_rows=0))

    assert folds[0].test_start == df["ts"].iloc[0] + pd.Timedelta(days=10)
    assert folds[0].test_end == folds[0].test_start + pd.Timedelta(days=5)


def test_folds_tile_forward_without_gaps_or_overlap():
    df = _daily_candles(40)
    folds = list(walk_forward_folds(df, min_train_days=10, test_days=5, purge_rows=0))

    for a, b in zip(folds, folds[1:]):
        assert a.test_end == b.test_start


def test_train_window_expands_across_folds():
    df = _daily_candles(40)
    folds = list(walk_forward_folds(df, min_train_days=10, test_days=5, purge_rows=0))

    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_no_overlap_between_train_and_test_within_a_fold():
    df = _daily_candles(40)
    for fold in walk_forward_folds(df, min_train_days=10, test_days=5, purge_rows=0):
        assert fold.train["ts"].max() < fold.test["ts"].min()


def test_purge_removes_rows_immediately_before_test_boundary():
    df = _daily_candles(40)  # 1 row/day, so purge_rows directly maps to days removed
    unpurged = list(walk_forward_folds(df, min_train_days=10, test_days=5, purge_rows=0))
    purged = list(walk_forward_folds(df, min_train_days=10, test_days=5, purge_rows=3))

    for a, b in zip(unpurged, purged):
        assert len(b.train) == len(a.train) - 3
        assert b.train["ts"].max() == a.train["ts"].max() - pd.Timedelta(days=3)


def test_train_window_bounded_by_max_train_days():
    df = _daily_candles(40)
    folds = list(
        walk_forward_folds(df, min_train_days=10, test_days=5, purge_rows=0, max_train_days=7)
    )

    for fold in folds:
        assert len(fold.train) <= 7
        assert fold.train["ts"].min() >= fold.test_start - pd.Timedelta(days=7)


def test_max_train_days_none_keeps_expanding_behavior():
    df = _daily_candles(40)
    bounded_none = list(
        walk_forward_folds(df, min_train_days=10, test_days=5, purge_rows=0, max_train_days=None)
    )
    default = list(walk_forward_folds(df, min_train_days=10, test_days=5, purge_rows=0))

    assert [len(f.train) for f in bounded_none] == [len(f.train) for f in default]


def test_no_folds_when_history_shorter_than_min_train_days():
    df = _daily_candles(5)
    folds = list(walk_forward_folds(df, min_train_days=10, test_days=5, purge_rows=0))
    assert folds == []


def test_recency_weight_is_one_at_as_of():
    df = _daily_candles(10)
    as_of = df["ts"].iloc[-1]
    weights = recency_sample_weight(df["ts"], as_of, half_life_days=5)
    assert np.isclose(weights.iloc[-1], 1.0)


def test_recency_weight_halves_after_one_half_life():
    df = _daily_candles(10)
    as_of = df["ts"].iloc[-1]
    weights = recency_sample_weight(df["ts"], as_of, half_life_days=3)
    one_half_life_back = df["ts"].iloc[-1] - pd.Timedelta(days=3)
    row = weights[df["ts"] == one_half_life_back]
    assert np.isclose(row.iloc[0], 0.5)


def test_recency_weight_monotonically_decreases_going_back():
    df = _daily_candles(10)
    as_of = df["ts"].iloc[-1]
    weights = recency_sample_weight(df["ts"], as_of, half_life_days=5)
    assert list(weights) == sorted(weights)


def test_recency_weight_keeps_input_index():
    df = _daily_candles(10).iloc[2:]  # non-default index, like a sliced fold.train
    weights = recency_sample_weight(df["ts"], df["ts"].iloc[-1], half_life_days=5)
    assert list(weights.index) == list(df.index)


def test_trailing_partial_window_still_yielded_if_nonempty():
    # 40 days of data, test_days=7 doesn't evenly divide the 30 days left
    # after min_train_days=10, so the last fold's test window is clipped
    # short by however much data is left, rather than being dropped.
    df = _daily_candles(40)
    folds = list(walk_forward_folds(df, min_train_days=10, test_days=7, purge_rows=0))

    last = folds[-1]
    assert len(last.test) > 0
    assert last.test["ts"].max() < last.test_end
