import json
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sensor_design import GaussianDesign
from sensor_design.experiment import basis, prior_predictive_audit, synthetic_problem


def test_shared_noise_construction_and_matched_marginals():
    x, groups, models = synthetic_problem()
    a, b = models["independent"], models["correlated"]
    assert_allclose(np.diag(a.noise), np.diag(b.noise))
    assert a.independent_noise and not b.independent_noise
    assert_allclose(b.features, basis(x))
    for i in range(12):
        for j in range(i):
            assert b.noise[i, j] == (.25 if groups[i] == groups[j] else 0)


def test_prior_predictive_matches_analytic_mse_and_coverage():
    _, _, models = synthetic_problem()
    model = models["correlated"]
    cases = {"aware": ((0, 2, 5, 9), model), "misspecified": ((0, 2, 5, 9), models["independent"])}
    result = prior_predictive_audit(model, cases, basis(np.linspace(-1, 1, 21)), 15000, 42)
    for row in result.values():
        mse = row["empirical_mean_squared_error"]
        assert abs(mse["estimate"] - row["analytic_mean_squared_error"]) < 6 * mse["mcse"]
    correct = result["aware"]
    assert_allclose(correct["analytic_mean_squared_error"], correct["reported_mean_field_variance"], rtol=1e-12)
    coverage = correct["marginal_90_percent_coverage"]
    assert abs(coverage["estimate"] - .9) < 6 * coverage["mcse"]
    wrong = result["misspecified"]
    assert not np.isclose(wrong["analytic_mean_squared_error"], wrong["reported_mean_field_variance"])


def test_seed_reproducibility():
    _, _, models = synthetic_problem()
    model = models["correlated"]
    cases = {"one": ((0, 1), model)}
    grid = basis([-1, 0, 1])
    assert prior_predictive_audit(model, cases, grid, 100, 7) == prior_predictive_audit(model, cases, grid, 100, 7)


@pytest.mark.parametrize("replicates", [0, 1, 1.5, True])
def test_invalid_simulation_count(replicates):
    _, _, models = synthetic_problem()
    with pytest.raises(ValueError, match="replicates"):
        prior_predictive_audit(models["correlated"], {}, basis([0]), replicates, 7)


def test_mismatched_inference_problem_rejected():
    _, _, models = synthetic_problem()
    true = models["correlated"]
    wrong = GaussianDesign(true.features * 2, true.prior, true.noise)
    with pytest.raises(ValueError, match="noise assumptions only"):
        prior_predictive_audit(true, {"wrong": ((0, 1), wrong)}, basis([0]), 2, 1)


def test_cli_outputs_are_finite_and_reproducible(tmp_path):
    directories = [tmp_path / "a", tmp_path / "b"]
    for path in directories:
        subprocess.run([
            sys.executable, "-W", "error", "-m", "sensor_design.experiment",
            "--output", str(path), "--replicates", "100", "--seed", "17",
        ], check=True, capture_output=True, text=True)
        for name in ("design_information.png", "noise_and_uncertainty.png"):
            assert (path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (directories[0] / "summary.json").read_bytes() == (directories[1] / "summary.json").read_bytes()
    data = json.loads((directories[0] / "summary.json").read_text())
    assert len(data["design_results"]) == 18
    assert not data["correlated_noise_counterexample"]["diminishing_returns_holds"]
    for row in data["design_results"]:
        assert row["exact_optimal_gap_nats"] >= 0
        assert len(set(row["sensors"])) == row["budget"]
        assert row["mean_latent_field_variance"] > 0
        if row["scenario"] == "correlated":
            assert row["independent_noise_optimum_upper_bound_nats"] is None
