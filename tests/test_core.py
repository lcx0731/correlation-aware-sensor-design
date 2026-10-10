from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sensor_design import GaussianDesign, exact_select, greedy_select


def small_model(correlated=False):
    h = np.array([[1, -1], [1, -.3], [1, 0], [1, .6], [1, 1]])
    r = np.diag([.2, .3, .2, .4, .3])
    if correlated:
        r += .4 * np.ones((5, 5))
    return GaussianDesign(h, [[1, .2], [.2, .6]], r)


def test_empty_selection():
    model = small_model()
    post = model.posterior([])
    assert_allclose(post.covariance, model.prior)
    assert post.information == 0
    assert post.gain.shape == (2, 0)
    assert_allclose(post.mean(np.empty((3, 0))), np.zeros((3, 2)))
    assert greedy_select(model, 0).sensors == exact_select(model, 0).sensors == ()


def test_scalar_analytic_solution():
    post = GaussianDesign([[1]], [[2]], [[.5]]).posterior([0])
    assert_allclose(post.covariance, [[.4]], rtol=1e-13)
    assert_allclose(post.gain, [[.8]], rtol=1e-13)
    assert_allclose(post.mean([3]), [2.4], rtol=1e-13)
    assert_allclose(post.information, .5 * np.log(5), rtol=1e-13)


@pytest.mark.parametrize("correlated", [False, True])
def test_matches_joint_gaussian_schur_complement(correlated):
    model = small_model(correlated)
    indices = [4, 1, 2]
    h = model.features[indices]
    r = model.noise[np.ix_(indices, indices)]
    total = h @ model.prior @ h.T + r
    reference_gain = np.linalg.solve(total, h @ model.prior).T
    reference_covariance = model.prior - reference_gain @ h @ model.prior
    post = model.posterior(indices)
    assert_allclose(post.gain, reference_gain, atol=1e-13)
    assert_allclose(post.covariance, reference_covariance, atol=1e-13)
    reference_information = .5 * (np.linalg.slogdet(total)[1] - np.linalg.slogdet(r)[1])
    assert_allclose(post.information, reference_information, atol=1e-13)


def test_independent_marginal_determinant_lemma():
    model = small_model()
    for selected in ([], [0], [2, 4]):
        for j in range(model.n_sensors):
            if j not in selected:
                h = model.features[j]
                variance = h @ model.posterior(selected).covariance @ h
                expected = .5 * np.log1p(variance / model.noise[j, j])
                assert_allclose(model.marginal(selected, j), expected, atol=1e-13)


def test_independent_diminishing_returns_and_certificates():
    model = small_model()
    optima = {k: exact_select(model, k).information for k in range(6)}
    for size in range(6):
        for selected in combinations(range(5), size):
            for budget, optimum in optima.items():
                assert model.independent_upper_bound(selected, budget) >= optimum - 1e-12
            for a, b in combinations([j for j in range(5) if j not in selected], 2):
                assert model.marginal(selected, b) >= model.marginal((*selected, a), b) - 1e-12


def test_correlated_counterexample_and_no_certificate():
    model = GaussianDesign([[1], [-1]], [[1]], [[1, .9], [.9, 1]])
    assert_allclose(model.information([0]), .5 * np.log(2), atol=1e-13)
    assert_allclose(model.information([0, 1]), .5 * np.log(21), atol=1e-13)
    assert model.marginal([0], 1) > model.marginal([], 1)
    with pytest.raises(ValueError, match="independent noise"):
        model.independent_upper_bound([0], 1)


def test_tiny_correlations_still_disable_certificate():
    model = GaussianDesign([[1], [1]], [[1]], [[1, 1e-14], [1e-14, 1]])
    assert not model.independent_noise
    with pytest.raises(ValueError, match="independent noise"):
        model.independent_upper_bound([], 1)


@pytest.mark.parametrize("correlated", [False, True])
def test_monotone_information_and_covariance_reduction(correlated):
    model = small_model(correlated)
    for k in range(1, 6):
        previous = model.posterior(range(k - 1))
        current = model.posterior(range(k))
        assert current.information >= previous.information - 1e-13
        assert np.linalg.eigvalsh(current.covariance).min() > 0
        assert np.linalg.eigvalsh(previous.covariance - current.covariance).min() >= -1e-13


def test_exact_known_solution_and_tie_breaking():
    model = GaussianDesign([[1, 0], [0, 1], [1, 0]], np.eye(2), np.eye(3))
    oracle = exact_select(model, 2)
    assert oracle.sensors == (0, 1)
    assert_allclose(oracle.information, np.log(2), atol=1e-13)
    assert greedy_select(model, 2).sensors == (0, 1)


@pytest.mark.parametrize("correlated", [False, True])
def test_greedy_path_is_reproducible_and_dominated_by_oracle(correlated):
    model = small_model(correlated)
    result = greedy_select(model, 4)
    assert result == greedy_select(model, 4)
    assert len(set(result.sensors)) == 4
    assert len(result.history) == 5
    assert result.history[0] == 0
    assert np.min(np.diff(result.history)) >= -1e-13
    assert result.information <= exact_select(model, 4).information + 1e-12


def test_change_of_parameter_coordinates():
    model = small_model(True)
    transform = np.array([[2, .3], [-.2, .7]])
    # theta = transform @ eta, computed without an explicit inverse.
    left = np.linalg.solve(transform, model.prior)
    prior_eta = np.linalg.solve(transform, left.T).T
    reparameterized = GaussianDesign(model.features @ transform, prior_eta, model.noise)
    a, b = model.posterior([0, 2, 4]), reparameterized.posterior([0, 2, 4])
    assert_allclose(a.information, b.information, atol=1e-13)
    assert_allclose(a.covariance, transform @ b.covariance @ transform.T, atol=1e-13)


def test_subset_permutation_preserves_inference():
    model = small_model(True)
    a, b = model.posterior([0, 2, 4]), model.posterior([4, 0, 2])
    assert_allclose(a.information, b.information, atol=1e-13)
    assert_allclose(a.covariance, b.covariance, atol=1e-13)
    assert_allclose(a.mean([1, 2, 3]), b.mean([3, 1, 2]), atol=1e-13)


def test_model_copies_inputs():
    h, prior, noise = np.ones((2, 1)), np.eye(1), np.eye(2)
    model = GaussianDesign(h, prior, noise)
    h[:] = 42
    prior[:] = 3
    noise[:] = 5
    assert_allclose(model.features, np.ones((2, 1)))
    assert_allclose(model.prior, np.eye(1))
    assert_allclose(model.noise, np.eye(2))


@pytest.mark.parametrize("budget", [-1, 6, 1.2, True])
def test_invalid_budgets(budget):
    for select in (greedy_select, exact_select):
        with pytest.raises(ValueError):
            select(small_model(), budget)


@pytest.mark.parametrize("indices", [[0, 0], [-1], [5], [1.0], [True]])
def test_invalid_subsets(indices):
    with pytest.raises(ValueError):
        small_model().posterior(indices)


@pytest.mark.parametrize("h,p,r", [
    ([], [[1]], [[1]]),
    ([[np.nan]], [[1]], [[1]]),
    ([[1]], [[0]], [[1]]),
    ([[1]], [[1]], [[-1]]),
    ([[1]], [[1]], [[np.inf]]),
    ([[1]], np.eye(2), [[1]]),
    ([[1], [2]], [[1]], [[1, .8], [.2, 1]]),
    ([[1], [2]], [[1]], [[1, 2], [2, 1]]),
])
def test_invalid_models(h, p, r):
    with pytest.raises(ValueError):
        GaussianDesign(h, p, r)


@pytest.mark.parametrize("observations", [[1, 2], [np.nan], 2])
def test_invalid_observations(observations):
    with pytest.raises(ValueError):
        small_model().posterior([0]).mean(observations)


def test_exhaustive_size_guard():
    with pytest.raises(ValueError, match="max_subsets"):
        exact_select(small_model(), 2, max_subsets=9)
    with pytest.raises(ValueError, match="max_subsets"):
        exact_select(small_model(), 0, max_subsets=0)
