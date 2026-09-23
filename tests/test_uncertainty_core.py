"""Uncertainty, the pure parts: the ensemble mechanics, conformal calibration, the interval
methods, the uncertainty metrics and the column contract.

The ensemble mechanics: seeds, the bootstrap decision, and the variance decomposition.
"""

import logging
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from tests.support.builders import eval_frame
from yg_eo_soilnet.metrics import METRIC_DIRECTION, METRIC_SPACE, ORIGINAL_UNITS
from yg_eo_soilnet.plot_utils import _resolve_prediction_column
from yg_eo_soilnet.uncertainty import attach_uncertainty_columns
from yg_eo_soilnet.uncertainty.columns import (
    column_name,
    interval_columns,
    is_prediction_column,
    sigma_column,
)
from yg_eo_soilnet.uncertainty.conformal import (
    MIN_RELIABLE_CALIBRATION_ROWS,
    ConformalCalibrator,
    fit_conformal,
)
from yg_eo_soilnet.uncertainty.ensemble import (
    aggregate,
    bootstrap_indices,
    member_seeds,
    should_bootstrap,
)
from yg_eo_soilnet.uncertainty.intervals import (
    GaussianInterval,
    SigmaInterval,
    build_interval_estimators,
    describe,
    effective_alpha,
    needs_calibration_set,
    normalize_method,
)
from yg_eo_soilnet.uncertainty.metrics import UNCERTAINTY_METRIC_STEMS, uncertainty_metrics


def test_member_seeds_are_spaced_by_the_stride_not_by_one():
    assert member_seeds(42, 3, stride=1000) == [42, 1042, 2042]


def test_member_seeds_are_reproducible_for_the_same_base_seed():
    assert member_seeds(7, 5) == member_seeds(7, 5)


def test_member_seeds_rejects_an_empty_ensemble():
    with pytest.raises(ValueError, match="at least 1"):
        member_seeds(42, 0)


def test_auto_bootstraps_every_sklearn_estimator():
    assert should_bootstrap(Ridge()) is True
    assert should_bootstrap(PLSRegression()) is True
    assert should_bootstrap(GradientBoostingRegressor()) is True


def test_exposing_a_random_state_does_not_make_an_estimator_stochastic():
    # The reason `auto` cannot use the cheap "does it expose random_state?" test, pinned here so
    # nobody reintroduces it. Ridge REPORTS a random_state but only its sag/saga solvers consult
    # one; under the default solver the fit is a closed-form solve and the seed is inert. Two
    # differently-seeded Ridges are therefore byte-identical, and a seed-only ensemble of them has
    # a standard deviation of exactly zero.
    first = Ridge().set_params(random_state=1).fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    second = Ridge().set_params(random_state=999).fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    assert "random_state" in Ridge().get_params(deep=False)
    assert np.array_equal(first.coef_, second.coef_)


def test_never_is_the_way_to_opt_out_of_bootstrapping():
    # The Lightning family: a different weight initialization already produces a different model,
    # so resampling would shrink the training set for no additional spread.
    assert should_bootstrap(GradientBoostingRegressor(), mode="never") is False
    assert should_bootstrap(Ridge(), mode="always") is True


def test_an_unknown_bootstrap_mode_is_refused_rather_than_treated_as_auto():
    with pytest.raises(ValueError, match="bootstrap must be one of"):
        should_bootstrap(Ridge(), mode="sometimes")


def test_a_non_estimator_is_bootstrapped_rather_than_inspected():
    assert should_bootstrap(object()) is True


def test_bootstrap_indices_are_reproducible_and_drawn_with_replacement():
    first = bootstrap_indices(100, seed=42)
    assert np.array_equal(first, bootstrap_indices(100, seed=42))
    assert first.shape == (100,)
    assert first.max() < 100
    # With replacement over 100 draws, seeing every row exactly once has probability ~1e-42.
    assert len(np.unique(first)) < 100


def test_bootstrap_indices_differ_between_seeds():
    assert not np.array_equal(bootstrap_indices(100, seed=1), bootstrap_indices(100, seed=2))


def test_aggregate_mean_is_the_mean_of_the_members():
    members = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    result = aggregate(members)
    assert np.allclose(result.mean.reshape(-1), [3.0, 4.0])


def test_aggregate_epistemic_std_is_the_spread_across_members():
    members = [np.array([1.0]), np.array([3.0])]
    result = aggregate(members)
    assert np.allclose(result.epistemic_std.reshape(-1), [1.0])


def test_aggregate_without_sigmas_reports_no_aleatoric_component():
    result = aggregate([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    assert np.allclose(result.aleatoric_std, 0.0)
    assert np.allclose(result.total_std, result.epistemic_std)


def test_aggregate_averages_variances_not_standard_deviations():
    # Members claiming sigma 3 and 4 average to sqrt((9+16)/2) = 3.5355, not to 3.5. Averaging the
    # standard deviations understates a mixture whose members disagree about the noise level.
    result = aggregate([np.array([0.0]), np.array([0.0])], [np.array([3.0]), np.array([4.0])])
    assert result.aleatoric_std.reshape(-1)[0] == pytest.approx(np.sqrt(12.5))


def test_total_std_adds_variances_under_the_root():
    # epistemic 3, aleatoric 4 -> total 5, not 7.
    result = aggregate([np.array([-3.0]), np.array([3.0])], [np.array([4.0]), np.array([4.0])])
    assert result.epistemic_std.reshape(-1)[0] == pytest.approx(3.0)
    assert result.aleatoric_std.reshape(-1)[0] == pytest.approx(4.0)
    assert result.total_std.reshape(-1)[0] == pytest.approx(5.0)


def test_the_two_components_sum_to_the_total_variance():
    rng = np.random.default_rng(0)
    members = [rng.normal(size=(20, 3)) for _ in range(5)]
    sigmas = [np.abs(rng.normal(size=(20, 3))) for _ in range(5)]
    result = aggregate(members, sigmas)
    assert np.allclose(
        result.total_std**2, result.epistemic_std**2 + result.aleatoric_std**2
    )


def test_aggregate_widens_a_single_target_member_to_two_dimensions():
    result = aggregate([np.array([1.0, 2.0, 3.0])])
    assert result.mean.shape == (3, 1)


def test_aggregate_preserves_the_target_axis_of_a_multi_target_member():
    result = aggregate([np.zeros((4, 3)), np.ones((4, 3))])
    assert result.mean.shape == (4, 3)


def test_aggregate_refuses_a_sigma_list_that_does_not_match_the_members():
    with pytest.raises(ValueError, match="correspond one to one"):
        aggregate([np.array([1.0]), np.array([2.0])], [np.array([1.0])])


def test_aggregate_refuses_an_empty_ensemble():
    with pytest.raises(ValueError, match="at least one member"):
        aggregate([])


# --- conformal calibration --------------------------------------------------------------------
# Split-conformal calibration: the order statistic, the guarantee, and the degenerate cases.


def _gaussian_calibration(n_rows: int, sigma: float = 1.0, seed: int = 0):
    """`n_rows` points whose residuals really are Normal(0, sigma), with sigma known."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=n_rows) * 10.0
    observed = mean + rng.normal(scale=sigma, size=n_rows)
    return observed, mean, np.full(n_rows, sigma)


def test_the_quantile_is_the_finite_sample_corrected_order_statistic():
    # 19 scores, alpha=0.05: ceil(20 * 0.95) = 19, which is rank 19 of 19 -> saturates at the max.
    # 99 scores: ceil(100 * 0.95) = 95, i.e. the 95th of 99 rather than np.quantile's interpolation.
    scores = np.arange(1.0, 100.0)
    observed, mean = scores, np.zeros_like(scores)
    calibrator = fit_conformal(observed, mean, np.ones_like(scores), alpha=0.05)
    rank = math.ceil((99 + 1) * 0.95)
    assert calibrator.q == pytest.approx(float(np.sort(np.abs(scores))[rank - 1]))


def test_the_corrected_quantile_is_at_least_the_plain_one():
    # The correction can only widen the interval. A version that under-covers would be strictly
    # narrower here, and always in the same direction, which is the failure it exists to prevent.
    observed, mean, sigma = _gaussian_calibration(200, seed=3)
    calibrator = fit_conformal(observed, mean, sigma, alpha=0.05)
    scores = np.abs(observed - mean) / sigma
    assert calibrator.q >= float(np.quantile(scores, 0.95))


def test_coverage_on_held_out_data_reaches_the_nominal_level():
    calibration = _gaussian_calibration(2000, seed=1)
    calibrator = fit_conformal(*calibration, alpha=0.05)

    observed, mean, sigma = _gaussian_calibration(4000, seed=2)
    lower, upper = calibrator.intervals(mean, sigma)
    coverage = float(np.mean((observed >= lower) & (observed <= upper)))
    assert coverage == pytest.approx(0.95, abs=0.02)


def test_coverage_holds_even_when_the_model_sigma_is_badly_wrong():
    # The point of conformal: the guarantee needs no assumption that sigma is right. Here sigma is
    # understated tenfold, which would make a naive mu +- 1.96*sigma interval cover ~15%.
    rng = np.random.default_rng(5)
    mean = rng.normal(size=3000) * 10.0
    observed = mean + rng.normal(scale=1.0, size=3000)
    claimed_sigma = np.full(3000, 0.1)

    calibrator = fit_conformal(observed[:1500], mean[:1500], claimed_sigma[:1500], alpha=0.1)
    lower, upper = calibrator.intervals(mean[1500:], claimed_sigma[1500:])
    coverage = float(np.mean((observed[1500:] >= lower) & (observed[1500:] <= upper)))
    assert coverage == pytest.approx(0.90, abs=0.03)


def test_a_heteroscedastic_sigma_produces_intervals_that_vary_per_point():
    rng = np.random.default_rng(7)
    sigma = np.abs(rng.normal(size=1000)) + 0.1
    mean = np.zeros(1000)
    observed = rng.normal(scale=sigma)

    calibrator = fit_conformal(observed, mean, sigma, alpha=0.05)
    lower, upper = calibrator.intervals(mean, sigma)
    widths = upper - lower
    assert calibrator.normalized is True
    assert widths.std() > 0.0


def test_a_degenerate_sigma_falls_back_to_a_constant_width_band():
    # A deterministic ensemble reports sigma == 0 everywhere. Dividing by it would be meaningless,
    # so q becomes an absolute residual quantile and the flag says so.
    rng = np.random.default_rng(11)
    observed = rng.normal(size=500)
    mean = np.zeros(500)

    calibrator = fit_conformal(observed, mean, np.zeros(500), alpha=0.05)
    assert calibrator.normalized is False

    lower, upper = calibrator.intervals(mean, np.zeros(500))
    widths = upper - lower
    assert np.allclose(widths, widths[0])
    assert widths[0] > 0.0


def test_the_constant_width_fallback_still_covers():
    rng = np.random.default_rng(13)
    observed = rng.normal(size=4000)
    mean = np.zeros(4000)

    calibrator = fit_conformal(observed[:2000], mean[:2000], None, alpha=0.05)
    lower, upper = calibrator.intervals(mean[2000:], np.zeros(2000))
    coverage = float(np.mean((observed[2000:] >= lower) & (observed[2000:] <= upper)))
    assert coverage == pytest.approx(0.95, abs=0.02)


def test_omitting_sigma_entirely_is_the_same_as_passing_zeros():
    observed = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    mean = np.zeros(5)
    assert fit_conformal(observed, mean, None).q == fit_conformal(observed, mean, np.zeros(5)).q


def test_non_finite_calibration_rows_are_dropped_not_propagated():
    observed = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    mean = np.zeros(5)
    calibrator = fit_conformal(observed, mean, np.ones(5), alpha=0.2)
    assert calibrator.n_calib == 4
    assert np.isfinite(calibrator.q)


def test_a_small_calibration_set_warns_rather_than_failing_silently(caplog):
    observed, mean, sigma = _gaussian_calibration(MIN_RELIABLE_CALIBRATION_ROWS - 10, seed=17)
    logger = logging.getLogger("conformal-test")
    with caplog.at_level(logging.WARNING, logger="conformal-test"):
        calibrator = fit_conformal(observed, mean, sigma, alpha=0.05, logger=logger)
    assert calibrator.n_calib == MIN_RELIABLE_CALIBRATION_ROWS - 10
    assert "coverage guarantee" in caplog.text


def test_too_few_calibration_rows_is_an_error_not_a_silent_wrong_answer():
    with pytest.raises(ValueError, match="at least 2 finite calibration rows"):
        fit_conformal(np.array([1.0]), np.array([0.0]), np.array([1.0]))


def test_mismatched_calibration_lengths_are_refused():
    with pytest.raises(ValueError, match="pairs them row for row"):
        fit_conformal(np.zeros(5), np.zeros(4), np.ones(5))


def test_an_alpha_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="alpha must be in"):
        fit_conformal(np.zeros(10), np.zeros(10), np.ones(10), alpha=1.0)


def test_the_calibrator_describes_itself_in_loggable_scalars():
    payload = ConformalCalibrator(q=1.5, alpha=0.05, n_calib=100).to_dict()
    assert payload == {
        "conformal_q": 1.5,
        "conformal_alpha": 0.05,
        "conformal_n_calib": 100,
        "conformal_normalized": True,
    }
    assert all(isinstance(value, (int, float, bool)) for value in payload.values())


def test_a_lower_alpha_gives_a_wider_interval():
    observed, mean, sigma = _gaussian_calibration(1000, seed=19)
    tight = fit_conformal(observed, mean, sigma, alpha=0.20)
    wide = fit_conformal(observed, mean, sigma, alpha=0.01)
    assert wide.q > tight.q


def test_the_calibrator_round_trips_through_its_own_dict():
    """replot.py rebuilds a finished run's calibrator from uncertainty_summary.json.

    Without this the regenerated reliability curve grades Gaussian z-multiples of the raw sigma
    instead of the conformal procedure the run actually used - a different claim, and a visibly
    different line beside an unchanged PICP.
    """
    original = ConformalCalibrator(q=1.5, alpha=0.05, n_calib=100, normalized=False)
    assert ConformalCalibrator.from_dict(original.to_dict()) == original


def test_a_summary_that_is_not_conformal_yields_no_calibrator():
    # A run whose interval came from SigmaInterval writes no conformal_* keys at all. The caller's
    # fallback is a calibrator-less curve, so this returns None rather than raising.
    assert ConformalCalibrator.from_dict({"target": "clay_pct", "mean_sigma": 2.0}) is None
    assert ConformalCalibrator.from_dict(None) is None
    assert ConformalCalibrator.from_dict({"conformal_q": "not a number"}) is None


# --- interval methods -------------------------------------------------------------------------
# How sigma becomes an interval, and the three answers that are not interchangeable.
#
# The sigma columns are the same whichever method is chosen; what differs is the claim the
# lower/upper pair makes. These tests pin that each method makes the claim it says it does, and that
# the two things which silently depend on the choice - whether the sklearn family gives up its val
# split, and what nominal level the metrics grade against - follow it.


def test_a_gaussian_interval_is_the_normal_multiple_of_sigma():
    lower, upper = GaussianInterval(alpha=0.05).intervals(np.zeros(3), np.ones(3))
    assert np.allclose(upper, 1.959964, atol=1e-5)
    assert np.allclose(lower, -1.959964, atol=1e-5)


def test_a_sigma_interval_is_k_standard_deviations():
    lower, upper = SigmaInterval(k=2.0).intervals(np.array([10.0]), np.array([0.5]))
    assert lower[0] == pytest.approx(9.0) and upper[0] == pytest.approx(11.0)


def test_a_conformal_interval_scales_sigma_by_its_fitted_q():
    lower, upper = ConformalCalibrator(q=3.0, alpha=0.05, n_calib=100).intervals(
        np.array([10.0]), np.array([0.5])
    )
    assert lower[0] == pytest.approx(8.5) and upper[0] == pytest.approx(11.5)


def test_the_three_methods_disagree_on_the_same_sigma():
    """The point of the whole feature: same sigma, three different bands."""
    mean, sigma = np.array([0.0]), np.array([1.0])
    conformal = ConformalCalibrator(q=50.3, alpha=0.05, n_calib=800).intervals(mean, sigma)[1][0]
    gaussian = GaussianInterval(alpha=0.05).intervals(mean, sigma)[1][0]
    one_sigma = SigmaInterval(k=1.0).intervals(mean, sigma)[1][0]
    assert one_sigma < gaussian < conformal


# --- nominal coverage ------------------------------------------------------


def test_conformal_and_gaussian_claim_one_minus_alpha():
    assert ConformalCalibrator(q=1.0, alpha=0.05, n_calib=10).nominal_coverage == pytest.approx(0.95)
    assert GaussianInterval(alpha=0.05).nominal_coverage == pytest.approx(0.95)
    assert GaussianInterval(alpha=0.20).nominal_coverage == pytest.approx(0.80)


def test_a_sigma_band_claims_what_k_implies_not_one_minus_alpha():
    # This is why nominal_coverage is a property rather than a config lookup.
    assert SigmaInterval(k=1.0).nominal_coverage == pytest.approx(0.6827, abs=1e-4)
    assert SigmaInterval(k=2.0).nominal_coverage == pytest.approx(0.9545, abs=1e-4)


def test_the_effective_alpha_follows_the_method():
    assert effective_alpha("conformal", alpha=0.05, k=1.0) == pytest.approx(0.05)
    assert effective_alpha("gaussian", alpha=0.10, k=1.0) == pytest.approx(0.10)
    # A +-1 sigma band misses ~31.7% of the time, not 5%.
    assert effective_alpha("sigma", alpha=0.05, k=1.0) == pytest.approx(0.3173, abs=1e-4)


def test_grading_a_sigma_band_against_alpha_would_have_been_wrong():
    """The concrete failure the effective alpha prevents: a perfect ±1σ band, mis-graded.

    Its coverage_error against 1-alpha would read -0.27, which looks like a badly broken model.
    Against what k actually claims it reads ~0.
    """
    from yg_eo_soilnet.uncertainty.metrics import uncertainty_metrics

    rng = np.random.default_rng(0)
    observed = rng.normal(size=20000)
    predicted, sigma = np.zeros(20000), np.ones(20000)
    lower, upper = SigmaInterval(k=1.0).intervals(predicted, sigma)

    naive = uncertainty_metrics(observed, predicted, sigma, lower, upper, alpha=0.05)
    correct = uncertainty_metrics(
        observed, predicted, sigma, lower, upper, alpha=effective_alpha("sigma", k=1.0)
    )
    assert naive["coverage_error_test"] == pytest.approx(-0.27, abs=0.02)
    assert correct["coverage_error_test"] == pytest.approx(0.0, abs=0.02)


# --- method resolution -----------------------------------------------------


def test_the_legacy_split_conformal_spelling_still_resolves():
    assert normalize_method("split_conformal") == "conformal"
    assert normalize_method("CONFORMAL") == "conformal"
    assert normalize_method(None) == "conformal"


def test_an_unknown_method_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown uncertainty.interval.method"):
        normalize_method("bootstrap")


def test_only_conformal_needs_held_out_rows():
    assert needs_calibration_set("conformal") is True
    assert needs_calibration_set("split_conformal") is True
    assert needs_calibration_set("gaussian") is False
    assert needs_calibration_set("sigma") is False
    assert needs_calibration_set("none") is False


# --- the dispatch ----------------------------------------------------------


def test_gaussian_and_sigma_are_built_without_any_data():
    for method, kind in (("gaussian", GaussianInterval), ("sigma", SigmaInterval)):
        built = build_interval_estimators(method, ["clay_pct", "sand_pct"], alpha=0.05, k=1.0)
        assert set(built) == {"clay_pct", "sand_pct"}
        assert all(isinstance(v, kind) for v in built.values())


def test_none_produces_no_estimators_and_therefore_no_interval_columns():
    assert build_interval_estimators("none", ["clay_pct"]) == {}


def test_conformal_without_a_calibration_set_says_what_to_do_instead():
    with pytest.raises(ValueError, match="gaussian|sigma"):
        build_interval_estimators("conformal", ["clay_pct"])


def test_the_bar_label_names_the_claim_being_made():
    assert describe(SigmaInterval(k=1.0)) == "±1σ"
    assert describe(SigmaInterval(k=2.5)) == "±2.5σ"
    assert describe(GaussianInterval(alpha=0.05)) == "gaussian 95%"
    assert describe(ConformalCalibrator(q=2.0, alpha=0.05, n_calib=10)) == "conformal 95%"
    assert describe(None) == ""


# --- uncertainty metrics ----------------------------------------------------------------------
# The interval and distributional metrics, and their contract with yg_eo_soilnet.metrics.


def test_picp_is_the_fraction_of_observations_inside_the_interval():
    observed = np.array([0.0, 1.0, 2.0, 30.0])
    predicted = np.zeros(4)
    metrics = uncertainty_metrics(
        observed, predicted, np.ones(4), lower=np.full(4, -5.0), upper=np.full(4, 5.0)
    )
    assert metrics["picp_test"] == pytest.approx(0.75)


def test_mpiw_is_the_mean_interval_width():
    metrics = uncertainty_metrics(
        np.zeros(4), np.zeros(4), np.ones(4),
        lower=np.array([-1.0, -2.0, -3.0, -4.0]),
        upper=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    assert metrics["mpiw_test"] == pytest.approx(5.0)


def test_coverage_error_is_signed_so_over_and_under_confidence_are_distinguishable():
    # 100% coverage at a nominal 95% is over-cautious: positive.
    over = uncertainty_metrics(
        np.zeros(4), np.zeros(4), np.ones(4), lower=np.full(4, -9.0), upper=np.full(4, 9.0),
        alpha=0.05,
    )
    assert over["coverage_error_test"] == pytest.approx(0.05)

    # 0% coverage is over-confident: negative.
    under = uncertainty_metrics(
        np.ones(4), np.zeros(4), np.ones(4), lower=np.full(4, -0.1), upper=np.full(4, 0.1),
        alpha=0.05,
    )
    assert under["coverage_error_test"] == pytest.approx(-0.95)


def test_the_interval_score_charges_for_width_and_for_misses():
    # One point inside a width-2 band: score is the width alone.
    inside = uncertainty_metrics(
        np.zeros(2), np.zeros(2), np.ones(2), lower=np.full(2, -1.0), upper=np.full(2, 1.0),
        alpha=0.1,
    )
    assert inside["interval_score_test"] == pytest.approx(2.0)

    # Same band, observations 1.0 outside it: width 2 plus (2/alpha) * 1.0 = 2 + 20.
    outside = uncertainty_metrics(
        np.full(2, 2.0), np.zeros(2), np.ones(2), lower=np.full(2, -1.0), upper=np.full(2, 1.0),
        alpha=0.1,
    )
    assert outside["interval_score_test"] == pytest.approx(22.0)


def test_the_interval_score_prefers_a_narrow_covering_band_over_a_vacuous_one():
    # The property that makes it the one rankable number: coverage alone would tie these at 100%.
    narrow = uncertainty_metrics(
        np.zeros(10), np.zeros(10), np.ones(10),
        lower=np.full(10, -1.0), upper=np.full(10, 1.0),
    )
    vacuous = uncertainty_metrics(
        np.zeros(10), np.zeros(10), np.ones(10),
        lower=np.full(10, -1000.0), upper=np.full(10, 1000.0),
    )
    assert narrow["picp_test"] == vacuous["picp_test"] == 1.0
    assert narrow["interval_score_test"] < vacuous["interval_score_test"]


def test_nll_matches_the_gaussian_log_density():
    residual, sigma = 1.5, 2.0
    observed = np.array([residual, residual])
    metrics = uncertainty_metrics(observed, np.zeros(2), np.full(2, sigma))
    expected = -float(np.mean(stats.norm.logpdf(observed, loc=0.0, scale=sigma)))
    assert metrics["nll_test"] == pytest.approx(expected)


def test_crps_of_a_perfect_prediction_matches_the_closed_form():
    # residual 0, sigma 1: CRPS = 2*phi(0) - 1/sqrt(pi) = 0.7979 - 0.5642.
    metrics = uncertainty_metrics(np.zeros(2), np.zeros(2), np.ones(2))
    expected = 2.0 * stats.norm.pdf(0.0) - 1.0 / np.sqrt(np.pi)
    assert metrics["crps_test"] == pytest.approx(expected)


def test_crps_scales_with_sigma_when_the_prediction_is_exact():
    tight = uncertainty_metrics(np.zeros(2), np.zeros(2), np.ones(2))["crps_test"]
    loose = uncertainty_metrics(np.zeros(2), np.zeros(2), np.full(2, 2.0))["crps_test"]
    assert loose == pytest.approx(2.0 * tight)


def test_ence_is_near_zero_when_sigma_matches_the_realised_error():
    rng = np.random.default_rng(0)
    sigma = rng.uniform(0.5, 3.0, size=4000)
    observed = rng.normal(scale=sigma)
    metrics = uncertainty_metrics(observed, np.zeros(4000), sigma)
    assert metrics["ence_test"] < 0.15


def test_ence_is_large_when_sigma_is_uniformly_wrong():
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.5, 3.0, size=4000)
    observed = rng.normal(scale=sigma)
    # Claim a tenth of the true noise everywhere.
    metrics = uncertainty_metrics(observed, np.zeros(4000), sigma * 0.1)
    assert metrics["ence_test"] > 1.0


def test_sigma_error_corr_detects_uncertainty_that_tracks_the_error():
    rng = np.random.default_rng(2)
    sigma = rng.uniform(0.1, 5.0, size=2000)
    observed = rng.normal(scale=sigma)
    metrics = uncertainty_metrics(observed, np.zeros(2000), sigma)
    assert metrics["sigma_error_corr_test"] > 0.4


def test_sigma_error_corr_is_near_zero_when_sigma_ranks_points_at_random():
    # The failure global coverage cannot see: perfectly calibrated on average, no per-point
    # information at all.
    rng = np.random.default_rng(3)
    observed = rng.normal(size=2000)
    sigma = rng.uniform(0.5, 1.5, size=2000)
    metrics = uncertainty_metrics(observed, np.zeros(2000), sigma)
    assert abs(metrics["sigma_error_corr_test"]) < 0.1


def test_a_constant_sigma_omits_the_rank_correlation_rather_than_reporting_zero():
    metrics = uncertainty_metrics(np.arange(100.0), np.zeros(100), np.ones(100))
    assert "sigma_error_corr_test" not in metrics


def test_omitting_the_interval_skips_the_interval_metrics_only():
    metrics = uncertainty_metrics(np.arange(100.0), np.zeros(100), np.ones(100))
    assert "picp_test" not in metrics
    assert "mpiw_test" not in metrics
    assert "nll_test" in metrics
    assert "crps_test" in metrics


def test_a_degenerate_sigma_omits_the_distributional_metrics_only():
    metrics = uncertainty_metrics(
        np.arange(100.0), np.zeros(100), np.zeros(100),
        lower=np.full(100, -200.0), upper=np.full(100, 200.0),
    )
    assert metrics["picp_test"] == 1.0
    assert "nll_test" not in metrics
    assert "crps_test" not in metrics


def test_suffixed_keys_are_produced_for_a_multi_target_run():
    metrics = uncertainty_metrics(
        np.zeros(10), np.zeros(10), np.ones(10),
        lower=np.full(10, -1.0), upper=np.full(10, 1.0),
        suffix="_clay_pct",
    )
    assert "picp_test_clay_pct" in metrics
    assert "picp_test" not in metrics


def test_fewer_than_two_finite_rows_returns_an_empty_dict_not_nans():
    # Same policy as regression_metrics: NaN metrics would poison the leaderboard.
    assert uncertainty_metrics([1.0], [0.0], [1.0]) == {}
    assert uncertainty_metrics([np.nan, np.nan], [0.0, 0.0], [1.0, 1.0]) == {}


def test_rows_non_finite_in_any_column_are_dropped_together():
    observed = np.array([0.0, 0.0, 0.0, 0.0])
    sigma = np.array([1.0, np.nan, 1.0, 1.0])
    metrics = uncertainty_metrics(
        observed, np.zeros(4), sigma,
        lower=np.full(4, -1.0), upper=np.full(4, 1.0),
    )
    assert metrics["picp_test"] == 1.0
    assert metrics["mean_sigma_test"] == pytest.approx(1.0)


def test_a_negative_sigma_is_dropped_rather_than_taken_as_its_absolute_value():
    # A negative sigma means a variance was handed over where a standard deviation was expected.
    # Dropping keeps the mistake visible instead of quietly halving the reported uncertainty.
    metrics = uncertainty_metrics(
        np.zeros(4), np.zeros(4), np.array([1.0, -1.0, 1.0, 1.0]),
    )
    assert metrics["mean_sigma_test"] == pytest.approx(1.0)


def test_a_multi_column_input_is_refused_the_way_regression_metrics_refuses_one():
    with pytest.raises(ValueError, match="scores ONE target at a time"):
        uncertainty_metrics(np.zeros((10, 2)), np.zeros((10, 2)), np.ones((10, 2)))


def test_mismatched_input_lengths_are_refused():
    with pytest.raises(ValueError, match="mismatched lengths"):
        uncertainty_metrics(np.zeros(10), np.zeros(9), np.ones(10))


def test_every_uncertainty_stem_is_registered_in_the_shared_direction_table():
    # The rule the metrics module exists to enforce: a metric name means one thing, and which way is
    # better is recorded in one place rather than inferred from a sign.
    for stem in UNCERTAINTY_METRIC_STEMS:
        assert stem in METRIC_DIRECTION


def test_every_uncertainty_metric_is_declared_to_be_in_original_units():
    for stem in UNCERTAINTY_METRIC_STEMS:
        assert METRIC_SPACE[f"{stem}_test"] == ORIGINAL_UNITS


def test_picp_has_no_ranking_direction_because_maximising_it_is_wrong():
    # A band spanning the whole target range covers 100%. `coverage_error` is the rankable form.
    assert METRIC_DIRECTION["picp"] is None
    assert METRIC_DIRECTION["coverage_error"] == "zero"


# --- the column contract ----------------------------------------------------------------------
# The uncertainty column contract: how the columns are named, written, and read back.
#
# The logger's collectors are tested here too, because the bugs they had came from the naming:
# `prediction_*` also matches the sigma and interval columns.


def test_a_single_target_run_writes_unsuffixed_columns():
    assert column_name("prediction_std", "clay_pct", multi_target=False) == "prediction_std"


def test_a_multi_target_run_suffixes_every_column_with_its_target():
    assert column_name("prediction_std", "clay_pct", multi_target=True) == "prediction_std_clay_pct"


def test_an_uncertainty_column_is_not_mistaken_for_a_prediction_column():
    assert is_prediction_column("prediction") is True
    assert is_prediction_column("prediction_clay_pct") is True
    assert is_prediction_column("prediction_std") is False
    assert is_prediction_column("prediction_std_clay_pct") is False
    assert is_prediction_column("prediction_lower_clay_pct") is False
    assert is_prediction_column("prediction_epistemic_std_clay_pct") is False
    assert is_prediction_column("clay_pct") is False


def test_the_prediction_column_fallback_never_selects_a_standard_deviation():
    # The specific trap: the positional fallback in _resolve_prediction_column takes the first
    # column starting with "prediction_", and on a joint uncertainty frame that can be a sigma.
    # Plotting it would put standard deviations on the predicted axis and look entirely plausible.
    frame = pd.DataFrame(
        {
            "prediction_epistemic_std_clay_pct": [9.0],
            "prediction_std_clay_pct": [9.0],
            "prediction_clay_pct": [1.0],
        }
    )
    assert _resolve_prediction_column(frame, target_name=None, target_index=0) == "prediction_clay_pct"


def test_interval_columns_prefers_the_suffixed_pair_on_a_joint_frame():
    frame = pd.DataFrame(
        {
            "prediction_lower_clay_pct": [1.0],
            "prediction_upper_clay_pct": [2.0],
            "prediction_lower_sand_pct": [3.0],
            "prediction_upper_sand_pct": [4.0],
        }
    )
    lower, upper = interval_columns(frame, "sand_pct")
    assert lower.iloc[0] == 3.0 and upper.iloc[0] == 4.0


def test_interval_columns_returns_none_for_a_frame_from_a_run_without_uncertainty():
    assert interval_columns(eval_frame(with_uncertainty=False)) is None
    assert sigma_column(eval_frame(with_uncertainty=False)) is None


def test_a_half_written_interval_is_treated_as_absent_rather_than_half_used():
    frame = eval_frame()
    frame = frame.drop(columns=["prediction_upper"])
    assert interval_columns(frame) is None


def test_attach_writes_unsuffixed_columns_for_one_target():
    frame = pd.DataFrame({"target": np.zeros(5), "prediction": np.zeros(5)})
    prediction = aggregate([np.zeros(5), np.ones(5)])
    calibrator = ConformalCalibrator(q=2.0, alpha=0.05, n_calib=100)

    attach_uncertainty_columns(frame, prediction, ["clay_pct"], {"clay_pct": calibrator})

    assert "prediction_std" in frame.columns
    assert "prediction_std_clay_pct" not in frame.columns
    # Members 0 and 1: ensemble mean 0.5, epistemic std 0.5, no aleatoric part, so the interval is
    # 0.5 -+ q * 0.5 with q = 2.
    assert frame["prediction_epistemic_std"].iloc[0] == pytest.approx(0.5)
    assert frame["prediction_aleatoric_std"].iloc[0] == pytest.approx(0.0)
    assert frame["prediction_lower"].iloc[0] == pytest.approx(-0.5)
    assert frame["prediction_upper"].iloc[0] == pytest.approx(1.5)


def test_attach_suffixes_every_column_for_a_joint_group():
    frame = pd.DataFrame({"a": np.zeros(5), "b": np.zeros(5)})
    prediction = aggregate([np.zeros((5, 2)), np.ones((5, 2))])

    attach_uncertainty_columns(frame, prediction, ["a", "b"])

    for stem in ("prediction_std", "prediction_epistemic_std", "prediction_aleatoric_std"):
        assert f"{stem}_a" in frame.columns
        assert f"{stem}_b" in frame.columns
        assert stem not in frame.columns


def test_attach_writes_no_interval_when_there_is_no_calibrator():
    frame = pd.DataFrame({"target": np.zeros(5)})
    attach_uncertainty_columns(frame, aggregate([np.zeros(5), np.ones(5)]), ["clay_pct"])
    assert "prediction_std" in frame.columns
    assert "prediction_lower" not in frame.columns


def test_a_single_target_frame_with_uncertainty_still_reads_as_single_target():
    """The uncertainty columns must not make a one-target frame look like a joint one.

    _iter_target_eval_frames decided that by scanning for `prediction_*`, which the sigma and
    interval columns now also match. A single-target uncertainty frame then found no
    prediction_<target>, yielded nothing, and the run logged neither rmse_test nor picp_test - with
    no error anywhere, because an empty metric dict is indistinguishable from a metric-free run.
    """
    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger

    frame = eval_frame()
    frame["clay_pct"] = frame["target"]

    frames = list(
        ChildRunLogger()._iter_target_eval_frames(frame, target="clay_pct", model_name="Ridge")
    )
    assert len(frames) == 1
    _yielded, target_name, prediction_column = frames[0]
    assert target_name == "clay_pct"
    assert prediction_column == "prediction"


def test_the_parent_collector_never_plots_a_standard_deviation_as_a_prediction(tmp_path):
    """`_collect_eval_dfs` had the same `prediction_*` bug as `_iter_target_eval_frames`.

    A single-target uncertainty frame took the multi-target branch, and the fallback there picked
    the first `prediction_*` column - which is `prediction_std`, since it sorts right after
    `prediction`. That value was then assigned to `prediction`, so the parent's pred_error_plot.png
    plotted standard deviations on the predicted axis and looked entirely plausible.
    """
    import mlflow

    from yg_eo_soilnet.logger.mlflow_loggers import ParentRunLogger

    frame = eval_frame(40)
    frame["clay_pct"] = frame["target"]
    # Sigma is deliberately far from the prediction, so picking the wrong column is unmissable.
    frame["prediction_std"] = 999.0
    frame["target_names"] = "clay_pct"

    with mlflow.start_run() as parent:
        parent_id = parent.info.run_id
        with mlflow.start_run(nested=True) as child:
            mlflow.set_tags({"target": "clay_pct", "model_name": "Ridge"})
            path = tmp_path / "eval_results.csv"
            frame.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="eval_results")
            child.info.run_id

    collected = ParentRunLogger()._collect_eval_dfs(parent_id)
    assert len(collected) == 1
    assert not (collected[0]["prediction"] == 999.0).any()
    assert np.allclose(collected[0]["prediction"], frame["prediction"])


def test_a_joint_frame_with_uncertainty_still_fans_out_per_target():
    from yg_eo_soilnet.logger.mlflow_loggers import ChildRunLogger

    frame = pd.DataFrame(
        {
            "clay_pct": [1.0, 2.0],
            "sand_pct": [3.0, 4.0],
            "prediction_clay_pct": [1.1, 2.1],
            "prediction_sand_pct": [3.1, 4.1],
            "prediction_std_clay_pct": [0.5, 0.5],
            "prediction_std_sand_pct": [0.5, 0.5],
            "target_names": ["clay_pct__sand_pct"] * 2,
        }
    )
    frames = list(
        ChildRunLogger()._iter_target_eval_frames(
            frame, target="clay_pct__sand_pct", model_name="Ridge"
        )
    )
    assert [name for _f, name, _c in frames] == ["clay_pct", "sand_pct"]
    assert [col for _f, _n, col in frames] == ["prediction_clay_pct", "prediction_sand_pct"]
