import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from promote_new_scheme import _half_inputs, decide_new_scheme_promotion

_DEFAULTS = dict(
    has_candidate=True,
    has_production=True,
    label=True,
    candidate_schema="new",
    production_schema="old",
    sanity_passed=True,
    candidate_problems=[],
    skip_sanity_check=False,
)


def _decide(**overrides):
    return decide_new_scheme_promotion(**{**_DEFAULTS, **overrides})


def test_all_guards_satisfied_promotes():
    assert _decide() == ("promote_new_scheme", True)


def test_no_candidate_or_production_never_promotes():
    assert _decide(has_candidate=False) == ("no_candidate", False)
    assert _decide(has_production=False) == ("no_production", False)


def test_missing_label_is_never_inferred():
    assert _decide(label=None) == ("label_missing", False)


def test_explicit_false_label_is_not_a_new_scheme():
    assert _decide(label=False) == ("not_new_scheme", False)


def test_label_true_but_schema_actually_unchanged_is_refused():
    assert _decide(candidate_schema="old") == ("schema_unchanged", False)
    assert _decide(candidate_schema=None) == ("schema_unchanged", False)


def test_inconsistent_candidate_is_refused():
    assert _decide(candidate_problems=["point: h2.txt missing"]) == ("candidate_inconsistent", False)


def test_missing_sanity_verdict_blocks_unless_explicitly_skipped():
    assert _decide(sanity_passed=None) == ("sanity_missing", False)
    assert _decide(sanity_passed=None, skip_sanity_check=True) == ("promote_new_scheme", True)


def test_recorded_sanity_failure_cannot_be_overridden():
    assert _decide(sanity_passed=False) == ("sanity_failed", False)
    assert _decide(sanity_passed=False, skip_sanity_check=True) == ("sanity_failed", False)


def test_half_inputs_point_reads_unprefixed_keys():
    cand = {"model_version": "v2", "new_scheme": True, "schema_version": "new", "eval_metrics": {"sanity": {"passed": True}}}
    prod = {"schema_version": "old"}
    inputs = _half_inputs(cand, prod, quantile=False)
    assert inputs == dict(
        has_candidate=True, label=True, candidate_schema="new", production_schema="old", sanity_passed=True
    )


def test_half_inputs_quantile_compares_against_point_schema_when_production_has_no_corridor():
    cand = {
        "quantile_levels": [0.1, 0.9], "quantile_new_scheme": True, "quantile_schema_version": "new",
        "quantile_eval_metrics": {"sanity": {"passed": True}},
    }
    prod = {"schema_version": "old"}  # point-only production
    inputs = _half_inputs(cand, prod, quantile=True)
    assert inputs["production_schema"] == "old" and inputs["has_candidate"] and inputs["sanity_passed"] is True


def test_half_inputs_without_a_corridor_is_no_candidate():
    assert _half_inputs({"model_version": "v2"}, {"schema_version": "old"}, quantile=True)["has_candidate"] is False


def test_unlabeled_candidate_with_unchanged_schema_is_not_a_labeling_problem():
    assert _decide(label=None, candidate_schema="old") == ("schema_unchanged", False)
