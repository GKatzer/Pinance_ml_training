import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from auto_retrain import CANDIDATE_SLOT, PRODUCTION_SLOT, decide_target_slot


def test_bootstrap_when_no_production_exists():
    decision, slot = decide_target_slot(
        has_production=False, schema_matches=False, candidate_mae=0.01, production_mae=None, min_improvement=0.005
    )
    assert decision == "bootstrap"
    assert slot == PRODUCTION_SLOT


def test_schema_drift_pushes_as_candidate_without_comparing():
    decision, slot = decide_target_slot(
        has_production=True, schema_matches=False, candidate_mae=0.01, production_mae=None, min_improvement=0.005
    )
    assert decision == "schema_drift_push_as_candidate"
    assert slot == CANDIDATE_SLOT


def test_improvement_above_threshold_pushes_as_candidate_not_production():
    # production_mae=0.010, candidate_mae=0.0095 -> 5% relative improvement
    decision, slot = decide_target_slot(
        has_production=True, schema_matches=True, candidate_mae=0.0095, production_mae=0.010, min_improvement=0.005
    )
    assert decision == "pushed_to_candidate"
    assert slot == CANDIDATE_SLOT  # never straight to production once one already exists


def test_improvement_below_threshold_is_rejected():
    # production_mae=0.010, candidate_mae=0.00998 -> 0.2% relative improvement, below the 0.5% bar
    decision, slot = decide_target_slot(
        has_production=True, schema_matches=True, candidate_mae=0.00998, production_mae=0.010, min_improvement=0.005
    )
    assert decision == "rejected"
    assert slot is None


def test_worse_candidate_is_rejected():
    decision, slot = decide_target_slot(
        has_production=True, schema_matches=True, candidate_mae=0.011, production_mae=0.010, min_improvement=0.005
    )
    assert decision == "rejected"
    assert slot is None


def test_improvement_exactly_at_threshold_is_accepted():
    # production_mae=1.0, candidate_mae=0.995 -> exactly 0.5% relative improvement:
    # the gate is ">=", not ">", so sitting right on the bar still passes.
    decision, slot = decide_target_slot(
        has_production=True, schema_matches=True, candidate_mae=0.995, production_mae=1.0, min_improvement=0.005
    )
    assert decision == "pushed_to_candidate"
    assert slot == CANDIDATE_SLOT


def test_schema_drift_with_failed_sanity_pushes_nothing():
    decision, slot = decide_target_slot(
        has_production=True, schema_matches=False, candidate_mae=0.01, production_mae=None,
        min_improvement=0.005, sanity_passed=False,
    )
    assert decision == "schema_drift_rejected_sanity"
    assert slot is None


def test_schema_drift_with_passed_sanity_pushes_as_candidate():
    decision, slot = decide_target_slot(
        has_production=True, schema_matches=False, candidate_mae=0.01, production_mae=None,
        min_improvement=0.005, sanity_passed=True,
    )
    assert decision == "schema_drift_push_as_candidate"
    assert slot == CANDIDATE_SLOT


def test_sanity_verdict_is_ignored_when_schema_matches():
    # Sanity only gates the no-comparison path; with a comparable production
    # the normal improvement gate decides.
    decision, slot = decide_target_slot(
        has_production=True, schema_matches=True, candidate_mae=0.0095, production_mae=0.010,
        min_improvement=0.005, sanity_passed=False,
    )
    assert decision == "pushed_to_candidate"
    assert slot == CANDIDATE_SLOT
