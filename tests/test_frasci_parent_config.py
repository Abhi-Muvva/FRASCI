import pytest

from frasci.parent import ParentConfig

# NOTE: the previous release's test_release_defaults_to_adaptive_cached_policy_without_phase2
# and test_legacy_policy_remains_explicitly_available asserted `adaptive_phase_policy` defaults
# to True; the live ParentConfig now defaults it to False, so those two are stale and removed
# (not ported). test_adaptive_cross_run_warm_start_is_rejected_before_io relied on that same old
# default to reach its ValueError without setting adaptive_phase_policy explicitly, so it is
# also removed rather than silently made to pass by widening its own setup.


def test_adaptive_threshold_and_frame_validation():
    with pytest.raises(ValueError, match="adaptive_phase0_tol"):
        ParentConfig(
            threshold=1e-3, max_cycle_macro=10,
            adaptive_fixed_support_tol=2e-2, adaptive_phase0_tol=1e-2)
    with pytest.raises(ValueError, match="frame_leakage_tol"):
        ParentConfig(
            threshold=1e-3, max_cycle_macro=10, frame_leakage_tol=-1.0)
