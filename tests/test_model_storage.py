import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pinance_ml.model_storage import (
    _is_point_model_file,
    _is_quantile_model_file,
    _split_metadata_by_pipeline,
    merge_metadata,
)


def test_merge_metadata_with_no_existing_slot_returns_updates_only():
    assert merge_metadata(None, {"model_version": "v1"}) == {"model_version": "v1"}


def test_merge_metadata_preserves_quantile_fields_on_a_point_only_push():
    # The exact bug this guards against: a point-only retrain must not
    # wipe an already-promoted corridor's quantile_* fields just because
    # it writes a fresh metadata.json.
    existing = {"model_version": "v1", "quantile_levels": [0.1, 0.9], "quantile_model_version": "qv1"}
    updates = {"model_version": "v2", "horizons": [1, 2, 3]}

    result = merge_metadata(existing, updates)

    assert result["model_version"] == "v2"
    assert result["horizons"] == [1, 2, 3]
    assert result["quantile_levels"] == [0.1, 0.9]
    assert result["quantile_model_version"] == "qv1"


def test_merge_metadata_preserves_point_fields_on_a_quantile_only_push():
    existing = {"model_version": "v1", "horizons": [1, 2, 3]}
    updates = {"quantile_levels": [0.1, 0.9], "quantile_model_version": "qv1"}

    result = merge_metadata(existing, updates)

    assert result["model_version"] == "v1"
    assert result["horizons"] == [1, 2, 3]
    assert result["quantile_levels"] == [0.1, 0.9]
    assert result["quantile_model_version"] == "qv1"


# --- _is_point_model_file / _is_quantile_model_file: the file-classifier
# primitives promote_candidate_point/promote_candidate_quantiles use to
# pick which candidate files a partial promotion may touch ---


def test_is_point_model_file_matches_plain_horizon_files():
    assert _is_point_model_file("h1.txt") is True
    assert _is_point_model_file("h12.txt") is True


def test_is_point_model_file_rejects_quantile_and_metadata_files():
    assert _is_point_model_file("h1_q0.1.txt") is False
    assert _is_point_model_file("h1_q0.9.txt") is False
    assert _is_point_model_file("metadata.json") is False


def test_is_quantile_model_file_matches_quantile_tail_files():
    assert _is_quantile_model_file("h1_q0.1.txt") is True
    assert _is_quantile_model_file("h12_q0.9.txt") is True


def test_is_quantile_model_file_rejects_point_and_metadata_files():
    assert _is_quantile_model_file("h1.txt") is False
    assert _is_quantile_model_file("metadata.json") is False


# --- _split_metadata_by_pipeline: the metadata-key-classifier primitive
# promote_candidate_point/promote_candidate_quantiles use to decide which
# keys to merge onto production -- same disjoint-keys contract
# merge_metadata already relies on. ---


def test_split_metadata_separates_prefixed_from_unprefixed_keys():
    metadata = {
        "symbol": "BTCUSDT",
        "model_version": "v1",
        "horizons": [1, 2, 3],
        "quantile_model_version": "qv1",
        "quantile_levels": [0.1, 0.9],
    }

    point_keys, quantile_keys = _split_metadata_by_pipeline(metadata)

    assert point_keys == {"symbol": "BTCUSDT", "model_version": "v1", "horizons": [1, 2, 3]}
    assert quantile_keys == {"quantile_model_version": "qv1", "quantile_levels": [0.1, 0.9]}


def test_split_metadata_partitions_completely_with_no_overlap_or_loss():
    metadata = {"a": 1, "quantile_b": 2, "c": 3, "quantile_d": 4}

    point_keys, quantile_keys = _split_metadata_by_pipeline(metadata)

    assert point_keys.keys() | quantile_keys.keys() == metadata.keys()
    assert point_keys.keys() & quantile_keys.keys() == set()


# --- slot_consistency_problems / schema hash pin ---

from pinance_ml.model_storage import _expected_schema_version, slot_consistency_problems


def _good_point_metadata(cols=("a", "b")):
    return {
        "feature_columns": list(cols),
        "schema_version": _expected_schema_version(list(cols)),
        "horizons": [1, 2],
        "model_version": "v1",
    }


def test_schema_hash_matches_export_models():
    # model_storage can't import scripts/export_models.py; this pins the
    # duplicated one-liner to the real thing.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from export_models import _schema_version

    assert _expected_schema_version(["x", "y", "z"]) == _schema_version(["x", "y", "z"])


def test_consistent_point_slot_has_no_problems():
    assert slot_consistency_problems(_good_point_metadata(), {"h1.txt", "h2.txt", "metadata.json"}) == []


def test_missing_metadata_is_a_problem():
    assert slot_consistency_problems(None, set()) == ["metadata.json missing"]


def test_missing_horizon_file_is_reported():
    problems = slot_consistency_problems(_good_point_metadata(), {"h1.txt"})
    assert problems == ["point: h2.txt missing"]


def test_schema_version_not_matching_feature_columns_is_reported():
    metadata = {**_good_point_metadata(), "schema_version": "deadbeef0000"}
    problems = slot_consistency_problems(metadata, {"h1.txt", "h2.txt"})
    assert len(problems) == 1 and problems[0].startswith("point: schema_version")


def test_quantile_files_checked_from_levels_and_horizons():
    metadata = {
        **_good_point_metadata(),
        "quantile_levels": [0.1, 0.9], "quantile_horizons": [1],
        "quantile_model_version": "q1", "quantile_schema_version": _good_point_metadata()["schema_version"],
    }
    problems = slot_consistency_problems(metadata, {"h1.txt", "h2.txt", "h1_q0.1.txt"})
    assert problems == ["quantile: h1_q0.9.txt missing"]


def test_mixed_schema_between_point_and_corridor_is_reported():
    metadata = {
        **_good_point_metadata(),
        "quantile_levels": [0.1], "quantile_horizons": [1],
        "quantile_model_version": "q1", "quantile_schema_version": "oldschema000",
    }
    problems = slot_consistency_problems(metadata, {"h1.txt", "h2.txt", "h1_q0.1.txt"})
    assert len(problems) == 1 and problems[0].startswith("mixed schema")


def test_half_selection_ignores_the_other_pipelines_problems():
    # A half-written corridor must not block a point-only promotion.
    metadata = {**_good_point_metadata(), "quantile_levels": [0.1], "quantile_horizons": [1]}
    assert slot_consistency_problems(metadata, {"h1.txt", "h2.txt"}, quantiles=False) == []
    assert slot_consistency_problems(metadata, set(), point=False, quantiles=True) != []
