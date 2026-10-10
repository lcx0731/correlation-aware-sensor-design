"""Small-instance Bayesian D-optimal design without explicit matrix inverses."""

from dataclasses import dataclass
from itertools import combinations
from math import comb
from operator import index

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        return index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _spd(value, size, name):
    array = np.array(value, dtype=float, copy=True)
    if array.shape != (size, size) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {size} by {size} matrix")
    if not np.allclose(array, array.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    array = (array + array.T) / 2
    try:
        cholesky(array, lower=True)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class Posterior:
    covariance: np.ndarray
    gain: np.ndarray
    information: float

    def mean(self, observations):
        """Accept selected readings in subset order, shape (..., n_selected)."""
        values = np.asarray(observations, dtype=float)
        if values.ndim < 1 or values.shape[-1] != self.gain.shape[1]:
            raise ValueError("last observation axis must match the selected subset")
        if not np.all(np.isfinite(values)):
            raise ValueError("observations must be finite")
        return values @ self.gain.T


class GaussianDesign:
    """Zero-mean theta ~ N(0, prior), readings y = H theta + noise.

    All candidate noise covariances are specified in advance. For a subset,
    use the corresponding marginal noise covariance, NOT a noise conditional
    on unobserved readings. Costs are one unit per sensor.
    """

    def __init__(self, features, prior, noise):
        h = np.array(features, dtype=float, copy=True)
        if h.ndim != 2 or min(h.shape) < 1 or not np.all(np.isfinite(h)):
            raise ValueError("features must be a finite, nonempty 2D matrix")
        h.flags.writeable = False
        self.features = h
        self.n_sensors, self.dimension = h.shape
        self.prior = _spd(prior, self.dimension, "prior")
        self.noise = _spd(noise, self.n_sensors, "noise")
        self._prior_chol = cholesky(self.prior, lower=True)

    def subset(self, sensors):
        indices = tuple(_integer(i, "sensor index") for i in sensors)
        if len(indices) != len(set(indices)):
            raise ValueError("sensor indices must be unique")
        if any(i < 0 or i >= self.n_sensors for i in indices):
            raise ValueError("sensor index is out of range")
        return indices

    def budget(self, budget):
        budget = _integer(budget, "budget")
        if not 0 <= budget <= self.n_sensors:
            raise ValueError("budget must be between zero and the sensor count")
        return budget

    @property
    def independent_noise(self):
        # No approximate threshold: any specified off-diagonal entry excludes
        # the independent-noise certificate, even if that entry is tiny.
        return bool(np.array_equal(self.noise, np.diag(np.diag(self.noise))))

    def posterior(self, sensors):
        selected = self.subset(sensors)
        if not selected:
            return Posterior(self.prior.copy(), np.zeros((self.dimension, 0)), 0.0)
        h = self.features[list(selected)]
        r = self.noise[np.ix_(selected, selected)]
        lr = cholesky(r, lower=True)
        whitened = solve_triangular(lr, h @ self._prior_chol, lower=True)
        precision_chol = cholesky(
            np.eye(self.dimension) + whitened.T @ whitened, lower=True
        )
        covariance = self._prior_chol @ cho_solve(
            (precision_chol, True), self._prior_chol.T
        )
        covariance = (covariance + covariance.T) / 2
        gain = cho_solve((lr, True), h @ covariance).T
        information = float(np.log(np.diag(precision_chol)).sum())
        return Posterior(covariance, gain, information)

    def information(self, sensors):
        return self.posterior(sensors).information

    def marginal(self, selected, candidate):
        selected = self.subset(selected)
        candidate = _integer(candidate, "candidate")
        return self.information((*selected, candidate)) - self.information(selected)

    def independent_upper_bound(self, selected, budget):
        """Upper bound on the best size-budget objective, only for diagonal R.

        Submodularity implies f(O) <= f(S) + sum over O minus S of Delta(j | S).
        We upper-bound that sum by the budget largest remaining marginals.
        This also applies when len(S) != budget. It is NOT valid for general R.
        """
        if not self.independent_noise:
            raise ValueError("submodular certificate requires independent noise")
        selected = self.subset(selected)
        budget = self.budget(budget)
        current = self.information(selected)
        gains = sorted(
            (max(0.0, self.marginal(selected, i))
             for i in range(self.n_sensors) if i not in selected),
            reverse=True,
        )
        return min(self.information(range(self.n_sensors)), current + sum(gains[:budget]))


@dataclass(frozen=True)
class Selection:
    sensors: tuple[int, ...]
    information: float
    # Greedy history includes f(empty)=0. Oracle has no greedy path.
    history: tuple[float, ...] = ()


def greedy_select(model, budget):
    """Deterministic greedy addition; absolute 1e-12 ties favor lower indices."""
    budget = model.budget(budget)
    selected = []
    history = [0.0]
    for _ in range(budget):
        best_index, best_value = None, -np.inf
        for candidate in range(model.n_sensors):
            if candidate not in selected:
                value = model.information((*selected, candidate))
                if value > best_value + 1e-12:
                    best_index, best_value = candidate, value
        selected.append(best_index)
        history.append(best_value)
    return Selection(tuple(selected), history[-1], tuple(history))


def exact_select(model, budget, max_subsets=100_000):
    """Enumerate every size-budget subset; reject combinatorial blow-ups."""
    budget = model.budget(budget)
    max_subsets = _integer(max_subsets, "max_subsets")
    if max_subsets < 1 or comb(model.n_sensors, budget) > max_subsets:
        raise ValueError("exhaustive search exceeds max_subsets")
    best_subset, best_value = (), -np.inf
    for selected in combinations(range(model.n_sensors), budget):
        value = model.information(selected)
        if value > best_value + 1e-12:
            best_subset, best_value = selected, value
    return Selection(best_subset, best_value)
