import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from promote_if_better import _corridor_verified, decide_point_promotion, decide_quantile_promotion

# --- decide_point_promotion ---

_POINT_DEFAULTS = dict(
    has_candidate=True,
    has_production=True,
    candidate_n=200,
    min_samples=100,
    candidate_accuracy=53.0,
    production_accuracy=51.0,
    candidate_point_rejected=False,
    min_accuracy_improvement_pp=1.0,
)


def _decide_point(**overrides):
    return decide_point_promotion(**{**_POINT_DEFAULTS, **overrides})


def test_point_no_candidate_never_promotes():
    decision, promote = _decide_point(has_candidate=False)
    assert decision == "no_candidate"
    assert promote is False


def test_point_no_production_never_promotes():
    # auto_retrain.py's own bootstrap case handles this by pushing straight
    # to PRODUCTION_SLOT itself -- out of scope here.
    decision, promote = _decide_point(has_production=False)
    assert decision == "no_production"
    assert promote is False


def test_point_below_min_samples_is_rejected_even_if_accuracy_looks_great():
    decision, promote = _decide_point(candidate_n=50, min_samples=100, candidate_accuracy=90.0, production_accuracy=10.0)
    assert decision == "insufficient_samples"
    assert promote is False


def test_point_missing_production_accuracy_is_not_comparable():
    # e.g. production's current model_version has no matured rows in this
    # window yet (freshly promoted itself) -- nothing to diff against.
    decision, promote = _decide_point(production_accuracy=None)
    assert decision == "no_comparable_metrics"
    assert promote is False


def test_point_accuracy_improvement_above_threshold_promotes():
    decision, promote = _decide_point(candidate_accuracy=52.5, production_accuracy=51.0, min_accuracy_improvement_pp=1.0)
    assert decision == "accuracy_improved"
    assert promote is True


def test_point_accuracy_improvement_exactly_at_threshold_promotes():
    # gate is ">=", not ">" -- same convention as auto_retrain.py's decide_target_slot.
    decision, promote = _decide_point(candidate_accuracy=52.0, production_accuracy=51.0, min_accuracy_improvement_pp=1.0)
    assert decision == "accuracy_improved"
    assert promote is True


def test_point_accuracy_improvement_below_threshold_is_rejected():
    decision, promote = _decide_point(candidate_accuracy=51.9, production_accuracy=51.0, min_accuracy_improvement_pp=1.0)
    assert decision == "not_better"
    assert promote is False


def test_point_worse_candidate_is_rejected():
    decision, promote = _decide_point(candidate_accuracy=48.0, production_accuracy=51.0)
    assert decision == "not_better"
    assert promote is False


def test_point_rejected_relabels_an_already_failing_comparison():
    # candidate_point_rejected only changes the LABEL of a comparison that
    # was already going to fail (delta below threshold) -- from generic
    # "not_better" to the more specific "point_model_rejected", for a
    # clearer audit trail. It does not change the outcome (still False).
    decision, promote = _decide_point(
        candidate_accuracy=51.5, production_accuracy=51.0, candidate_point_rejected=True, min_accuracy_improvement_pp=1.0
    )
    assert decision == "point_model_rejected"
    assert promote is False


def test_point_rejected_does_not_block_genuine_live_improvement():
    # A single offline holdout snapshot isn't infallible (that's the whole
    # reason this second, live-traffic gate exists at all -- see module
    # docstring). A full pp of live outperformance over min_samples real
    # resolved predictions is independent, stronger evidence, and must
    # still promote -- otherwise a symbol whose candidate and production
    # have converged to the same once-rejected artifact would be stuck
    # reporting point_model_rejected forever, with no path back to a
    # genuinely better model.
    decision, promote = _decide_point(
        candidate_accuracy=55.0, production_accuracy=51.0, candidate_point_rejected=True, min_accuracy_improvement_pp=1.0
    )
    assert decision == "accuracy_improved"
    assert promote is True


def test_point_rejected_checked_after_comparable_metrics_gate():
    # If there's nothing to compare against at all, that's still reported
    # as no_comparable_metrics, not point_model_rejected -- the rejection
    # flag only matters once there's live data to weigh against it.
    decision, promote = _decide_point(production_accuracy=None, candidate_point_rejected=True)
    assert decision == "no_comparable_metrics"
    assert promote is False


# --- decide_quantile_promotion ---

_QUANTILE_DEFAULTS = dict(
    has_candidate=True,
    has_production=True,
    candidate_n=200,
    min_samples=100,
    candidate_has_new_quantiles=True,
    corridor_verified=True,
)


def _decide_quantile(**overrides):
    return decide_quantile_promotion(**{**_QUANTILE_DEFAULTS, **overrides})


def test_quantile_no_candidate_never_promotes():
    decision, promote = _decide_quantile(has_candidate=False)
    assert decision == "no_candidate"
    assert promote is False


def test_quantile_no_production_never_promotes():
    decision, promote = _decide_quantile(has_production=False)
    assert decision == "no_production"
    assert promote is False


def test_quantile_no_new_quantiles_is_a_no_op():
    # candidate's quantile_model_version matches production's (or candidate
    # has none) -- nothing new to evaluate, not an error.
    decision, promote = _decide_quantile(candidate_has_new_quantiles=False)
    assert decision == "no_new_quantiles"
    assert promote is False


def test_quantile_no_new_quantiles_checked_before_sample_floor():
    # A steady-state symbol whose corridor hasn't changed shouldn't report
    # insufficient_samples every run -- there's nothing new being judged.
    decision, promote = _decide_quantile(candidate_has_new_quantiles=False, candidate_n=0, min_samples=100)
    assert decision == "no_new_quantiles"
    assert promote is False


def test_quantile_below_min_samples_is_rejected():
    decision, promote = _decide_quantile(candidate_n=50, min_samples=100)
    assert decision == "insufficient_samples"
    assert promote is False


def test_quantile_gain_promotes_when_corridor_is_verified():
    decision, promote = _decide_quantile(candidate_has_new_quantiles=True, corridor_verified=True)
    assert decision == "quantile_gain"
    assert promote is True


def test_quantile_gain_blocked_when_corridor_not_verified():
    # "it exists" is not the same claim as "it's honest" -- must NOT
    # promote just because a new quantile_model_version is present.
    decision, promote = _decide_quantile(candidate_has_new_quantiles=True, corridor_verified=False)
    assert decision == "quantile_unverified"
    assert promote is False


def test_quantile_promotion_is_independent_of_point_accuracy():
    # No accuracy/regression inputs at all in decide_quantile_promotion's
    # signature -- promote_candidate_quantiles only ever moves quantile
    # files/keys, so the point model's live accuracy is structurally
    # irrelevant here now (unlike the old coupled decide_promotion).
    decision, promote = _decide_quantile(candidate_has_new_quantiles=True, corridor_verified=True)
    assert decision == "quantile_gain"
    assert promote is True


# --- _corridor_verified: pure helper, the live-data counterpart of
# auto_retrain_quantiles.py's offline _avg_pinball_and_coverage_ok ---


def test_corridor_verified_true_when_both_tails_within_tolerance():
    assert _corridor_verified(0.11, 0.89, (0.1, 0.9), 0.03) is True


def test_corridor_verified_false_when_low_tail_outside_tolerance():
    assert _corridor_verified(0.20, 0.89, (0.1, 0.9), 0.03) is False


def test_corridor_verified_false_when_high_tail_outside_tolerance():
    assert _corridor_verified(0.11, 0.70, (0.1, 0.9), 0.03) is False


def test_corridor_verified_false_when_coverage_missing():
    # predictor-backend hasn't shipped live coverage for this row -- must
    # degrade to "not verified", never accidentally read as "verified".
    assert _corridor_verified(None, None, (0.1, 0.9), 0.03) is False
    assert _corridor_verified(0.11, None, (0.1, 0.9), 0.03) is False
    assert _corridor_verified(None, 0.89, (0.1, 0.9), 0.03) is False


# --- new-scheme: surfaced, never auto-promoted ---

from promote_if_better import NEW_SCHEME_MANUAL


def test_point_new_scheme_is_never_auto_promoted_even_with_great_live_numbers():
    assert _decide_point(needs_manual_new_scheme=True, candidate_accuracy=70.0) == (NEW_SCHEME_MANUAL, False)


def test_point_new_scheme_does_not_mask_missing_candidate_or_production():
    assert _decide_point(has_candidate=False, needs_manual_new_scheme=True) == ("no_candidate", False)
    assert _decide_point(has_production=False, needs_manual_new_scheme=True) == ("no_production", False)


def test_quantile_new_scheme_is_never_auto_promoted_even_when_verified():
    assert decide_quantile_promotion(True, True, 200, 100, True, True, needs_manual_new_scheme=True) == (
        NEW_SCHEME_MANUAL,
        False,
    )


def test_quantile_new_scheme_flag_irrelevant_when_no_new_quantiles():
    assert decide_quantile_promotion(True, True, 200, 100, False, True, needs_manual_new_scheme=True) == (
        "no_new_quantiles",
        False,
    )
