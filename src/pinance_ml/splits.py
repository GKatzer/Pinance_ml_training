from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


def recency_sample_weight(ts: pd.Series, as_of: pd.Timestamp, half_life_days: float) -> pd.Series:
    """exp(-ln(2)/half_life_days * age_days) sample weight, indexed like
    `ts` so it lines up with train_horizon_models' per-horizon `valid`
    mask (train.loc[valid, ...]) without separate positional bookkeeping.

    as_of should be the fold's test_start, not e.g. each row's own
    "latest available" time -- every horizon's weighting for a given fold
    then shares one reference point, regardless of how far back that
    horizon's NaN-target rows get dropped before fitting.
    """
    age_days = (as_of - ts) / pd.Timedelta(days=1)
    return np.exp(-np.log(2) / half_life_days * age_days)


@dataclass
class Fold:
    index: int
    train: pd.DataFrame
    test: pd.DataFrame
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(
    df: pd.DataFrame,
    min_train_days: float,
    test_days: float,
    purge_rows: int,
    max_train_days: float | None = None,
) -> Iterator[Fold]:
    """Walk-forward folds with purging; expanding by default, sliding if
    max_train_days is set.

    Per README: train on [0, T], test on [T, T+test_days], slide T forward
    by test_days, repeat until the data runs out. `df` must be sorted by a
    `ts` column (tz-aware). Each fold's test window is a fresh,
    non-overlapping `test_days`-wide block; its train window is, by
    default, *every* row strictly before that window (expanding, not
    rolling) minus the last `purge_rows` rows, whose target windows would
    otherwise reach into the test period the fold is about to score
    against.

    max_train_days bounds that train window to the most recent
    max_train_days before the test period instead of everything since the
    start of history. Expanding-forever stops being a meaningful "keep the
    model fresh" mechanism once history spans years: on an 8-year history,
    one more day of daily retraining is ~0.03% of the training set, its
    marginal influence on the fit is negligible regardless of retrain
    cadence. A bounded window makes old regime data actually drop out
    instead of being diluted to irrelevance by sheer row count.

    Folds with an empty train or test window (not enough history yet, or a
    trailing partial window with no rows) are skipped rather than yielded.

    Boundaries are found via `searchsorted` (O(log n), relies on `ts` being
    sorted) and rows are pulled out with positional `.iloc` slices rather
    than `df[ts < ...]` boolean masks: on a several-hundred-thousand-row
    history with hundreds of folds, re-scanning and boolean-gathering the
    full frame every fold turns an O(n) per-fold cost into O(folds * n)
    overall — a few hundred folds over ~1M rows was empirically ~30s that
    way. Contiguous `.iloc` slices copy the same rows without the gather.
    """
    ts = df["ts"]
    end = ts.iloc[-1]

    test_start = ts.iloc[0] + pd.Timedelta(days=min_train_days)
    step = pd.Timedelta(days=test_days)

    fold_index = 0
    while test_start < end:
        test_end = test_start + step

        boundary_pos = ts.searchsorted(test_start, side="left")
        train_end_pos = max(boundary_pos - purge_rows, 0)

        if max_train_days is None:
            train_start_pos = 0
        else:
            window_start = test_start - pd.Timedelta(days=max_train_days)
            train_start_pos = ts.searchsorted(window_start, side="left")

        test_end_pos = ts.searchsorted(test_end, side="left")

        train = df.iloc[train_start_pos:train_end_pos]
        test = df.iloc[boundary_pos:test_end_pos]

        if len(train) > 0 and len(test) > 0:
            yield Fold(fold_index, train, test, test_start, test_end)
            fold_index += 1

        test_start = test_end
