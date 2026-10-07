import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from auto_retrain_quantiles import CANDIDATE_SLOT, PRODUCTION_SLOT, decide_quantile_target_slot


def test_bootstrap_when_no_production_exists():
    decision, slot = decide_quantile_target_slot(
        has_production=False,
        quantile_schema_matches=False,
        has_production_quantiles=False,
        candidate_pinball=0.001,
        production_pinball=None,
        candidate_coverage_ok=True,
        min_improvement=0.005,
    )
    assert decision == "bootstrap"
    assert slot == PRODUCTION_SLOT


def test_no_production_quantiles_pushes_as_candidate_without_comparing():
    # Production has a point regressor already (or a corridor trained
    # under a different CORRIDOR_QUANTILES set) but no matching corridor
    # yet -- nothing to compare pinball loss against, so it pushes
    # unconditionally, same as bootstrap/schema-drift, rather than being
    # treated as a regression.
    decision, slot = decide_quantile_target_slot(
        has_production=True,
        quantile_schema_matches=False,
        has_production_quantiles=False,
        candidate_pinball=0.001,
        production_pinball=None,
        candidate_coverage_ok=True,
        min_improvement=0.005,
    )
    assert decision == "no_production_quantiles_push_as_candidate"
    assert slot == CANDIDATE_SLOT


def test_schema_drift_pushes_as_candidate_without_comparing():
    # A matching quantile_levels set exists in production, but the
    # feature schema it was trained on doesn't match this repo's current
    # pipeline -- distinct from "no corridor at all yet".
    decision, slot = decide_quantile_target_slot(
        has_production=True,
        quantile_schema_matches=False,
        has_production_quantiles=True,
        candidate_pinball=0.001,
        production_pinball=0.0012,
        candidate_coverage_ok=True,
        min_improvement=0.005,
    )
    assert decision == "schema_drift_push_as_candidate"
    assert slot == CANDIDATE_SLOT


def test_miscalibrated_candidate_is_rejected_even_if_pinball_improves():
    # candidate_pinball beats production_pinball by 10% -- would clear the
    # bar on pinball loss alone, but coverage_ok=False must veto it: a
    # corridor that lies about its own uncertainty is worse, not better.
    decision, slot = decide_quantile_target_slot(
        has_production=True,
        quantile_schema_matches=True,
        has_production_quantiles=True,
        candidate_pinball=0.0009,
        production_pinball=0.0010,
        candidate_coverage_ok=False,
        min_improvement=0.005,
    )
    assert decision == "rejected_miscalibrated"
    assert slot is None


def test_improvement_above_threshold_pushes_as_candidate():
    # production_pinball=0.0010, candidate_pinball=0.00095 -> 5% relative improvement
    decision, slot = decide_quantile_target_slot(
        has_production=True,
        quantile_schema_matches=True,
        has_production_quantiles=True,
        candidate_pinball=0.00095,
        production_pinball=0.0010,
        candidate_coverage_ok=True,
        min_improvement=0.005,
    )
    assert decision == "pushed_to_candidate"
    assert slot == CANDIDATE_SLOT


def test_improvement_below_threshold_is_rejected():
    # production_pinball=0.0010, candidate_pinball=0.000998 -> 0.2% relative improvement, below the 0.5% bar
    decision, slot = decide_quantile_target_slot(
        has_production=True,
        quantile_schema_matches=True,
        has_production_quantiles=True,
        candidate_pinball=0.000998,
        production_pinball=0.0010,
        candidate_coverage_ok=True,
        min_improvement=0.005,
    )
    assert decision == "rejected"
    assert slot is None


def test_worse_candidate_is_rejected():
    decision, slot = decide_quantile_target_slot(
        has_production=True,
        quantile_schema_matches=True,
        has_production_quantiles=True,
        candidate_pinball=0.0011,
        production_pinball=0.0010,
        candidate_coverage_ok=True,
        min_improvement=0.005,
    )
    assert decision == "rejected"
    assert slot is None


def _drift_kwargs(**overrides):
    return dict(
        has_production=True, quantile_schema_matches=False, has_production_quantiles=True,
        candidate_pinball=0.001, production_pinball=None, candidate_coverage_ok=True,
        min_improvement=0.005, **overrides,
    )


def test_schema_drift_failed_sanity_pushes_nothing():
    decision, slot = decide_quantile_target_slot(**_drift_kwargs(sanity_passed=False))
    assert decision == "schema_drift_rejected_sanity"
    assert slot is None


def test_schema_drift_passed_sanity_pushes_as_candidate():
    decision, slot = decide_quantile_target_slot(**_drift_kwargs(sanity_passed=True))
    assert decision == "schema_drift_push_as_candidate"
    assert slot == CANDIDATE_SLOT


def test_new_corridor_on_changed_point_schema_is_also_sanity_gated():
    kwargs = _drift_kwargs(sanity_passed=False)
    kwargs.update(has_production_quantiles=False)
    decision, slot = decide_quantile_target_slot(**kwargs)
    assert decision == "schema_drift_rejected_sanity"
    assert slot is None


def test_sanity_none_keeps_unconditional_push():
    kwargs = _drift_kwargs()
    kwargs.update(has_production_quantiles=False)
    decision, slot = decide_quantile_target_slot(**kwargs)
    assert decision == "no_production_quantiles_push_as_candidate"
    assert slot == CANDIDATE_SLOT
