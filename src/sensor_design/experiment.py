"""Deterministic synthetic design audit with seeded prior-predictive checks."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm

from .core import GaussianDesign, exact_select, greedy_select


def basis(locations):
    """Constant, linear and quadratic Legendre basis; not a fitted regression."""
    x = np.asarray(locations, dtype=float)
    return np.column_stack((np.ones_like(x), x, 0.5 * (3 * x**2 - 1)))


def synthetic_problem():
    x = np.array([-1, -.85, -.65, -.45, -.25, -.05, .05, .25, .45, .65, .85, 1])
    groups = np.arange(len(x)) % 3
    shared = np.eye(3)[groups]
    individual_sd = 0.10 + 0.02 * (np.arange(len(x)) % 4)
    correlated = np.diag(individual_sd**2) + 0.5**2 * shared @ shared.T
    prior = np.diag([1.0, 0.5, 0.25])
    models = {
        "independent": GaussianDesign(basis(x), prior, np.diag(np.diag(correlated))),
        "correlated": GaussianDesign(basis(x), prior, correlated),
    }
    return x, groups, models


def counterexample():
    model = GaussianDesign([[1], [-1]], [[1]], [[1, .9], [.9, 1]])
    first = model.information([0])
    second_alone = model.information([1])
    together = model.information([0, 1])
    return {
        "first_information": first,
        "second_information_alone": second_alone,
        "joint_information": together,
        "second_marginal_after_first": together - first,
        "diminishing_returns_holds": together - first <= second_alone,
    }


def field_variances(grid_features, covariance):
    return np.einsum("ij,jk,ik->i", grid_features, covariance, grid_features)


def _estimate(values):
    return {
        "estimate": float(np.mean(values)),
        "mcse": float(np.std(values, ddof=1) / np.sqrt(len(values))),
    }


def prior_predictive_audit(true_model, cases, grid_features, replicates, seed):
    """Independent simulations are the MCSE units, not dependent grid points.

    cases maps a label to (selected indices, assumed model). All cases use
    the same simulated latent parameters and full candidate readings.
    """
    if not isinstance(replicates, (int, np.integer)) or isinstance(replicates, bool):
        raise ValueError("replicates must be an integer >= 2")
    if replicates < 2:
        raise ValueError("replicates must be >= 2")
    grid = np.asarray(grid_features, dtype=float)
    if (grid.ndim != 2 or grid.shape[0] < 1
            or grid.shape[1] != true_model.dimension or not np.all(np.isfinite(grid))):
        raise ValueError("grid features must be finite and match parameter dimension")
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((replicates, true_model.dimension)) @ cholesky(
        true_model.prior, lower=True
    ).T
    noise = rng.standard_normal((replicates, true_model.n_sensors)) @ cholesky(
        true_model.noise, lower=True
    ).T
    readings = theta @ true_model.features.T + noise
    truth = theta @ grid.T
    z = norm.ppf(.95)
    results = {}
    for name, (selected, assumed) in cases.items():
        selected = true_model.subset(selected)
        if (assumed.features.shape != true_model.features.shape
                or not np.array_equal(assumed.features, true_model.features)
                or not np.array_equal(assumed.prior, true_model.prior)):
            raise ValueError("audit cases may change noise assumptions only")
        post = assumed.posterior(selected)
        error = post.mean(readings[:, list(selected)]) @ grid.T - truth
        predicted = field_variances(grid, post.covariance)
        half_width = z * np.sqrt(predicted)
        coverage_per_simulation = np.mean(np.abs(error) <= half_width, axis=1)
        mse_per_simulation = np.mean(error**2, axis=1)
        residual = np.eye(true_model.dimension) - post.gain @ true_model.features[list(selected)]
        actual_covariance = (
            residual @ true_model.prior @ residual.T
            + post.gain @ true_model.noise[np.ix_(selected, selected)] @ post.gain.T
        )
        results[name] = {
            "sensors": list(selected),
            "assumed_noise": "diagonal" if assumed.independent_noise else "correlated",
            "reported_mean_field_variance": float(np.mean(predicted)),
            "analytic_mean_squared_error": float(np.mean(field_variances(grid, actual_covariance))),
            "empirical_mean_squared_error": _estimate(mse_per_simulation),
            "marginal_90_percent_coverage": _estimate(coverage_per_simulation),
        }
    return results


def run_audit(replicates=5000, seed=20261010):
    x, groups, models = synthetic_problem()
    grid_x = np.linspace(-1, 1, 101)
    grid = basis(grid_x)
    budgets = (2, 4, 6)
    rows = []
    selected_at_four = {}
    for scenario, model in models.items():
        diagonal = GaussianDesign(model.features, model.prior, np.diag(np.diag(model.noise)))
        for budget in budgets:
            oracle = exact_select(model, budget)
            greedy = greedy_select(model, budget)
            diagonal_greedy = greedy_select(diagonal, budget)
            uniform_average = float(np.mean([
                model.information(s) for s in combinations(range(model.n_sensors), budget)
            ]))
            for method, result in (
                ("exact", oracle), ("aware_greedy", greedy), ("diagonal_greedy", diagonal_greedy)
            ):
                true_information = model.information(result.sensors)
                variances = field_variances(grid, model.posterior(result.sensors).covariance)
                rows.append({
                    "scenario": scenario, "budget": budget, "method": method,
                    "sensors": list(result.sensors),
                    "locations": x[list(result.sensors)].tolist(),
                    "noise_groups": groups[list(result.sensors)].tolist(),
                    "information_nats_under_true_noise": true_information,
                    "exact_optimal_gap_nats": max(0.0, oracle.information - true_information),
                    "mean_latent_field_variance": float(np.mean(variances)),
                    "max_latent_field_variance": float(np.max(variances)),
                    "uniform_subset_mean_information_nats": uniform_average,
                    "independent_noise_optimum_upper_bound_nats": (
                        model.independent_upper_bound(result.sensors, budget)
                        if model.independent_noise else None
                    ),
                })
                if budget == 4 and scenario == "correlated":
                    selected_at_four[method] = result.sensors
    true = models["correlated"]
    cases = {
        "exact_aware": (selected_at_four["exact"], true),
        "greedy_aware": (selected_at_four["aware_greedy"], true),
        "diagonal_design_aware_inference": (selected_at_four["diagonal_greedy"], true),
        "diagonal_design_diagonal_inference": (
            selected_at_four["diagonal_greedy"], models["independent"]
        ),
    }
    return {
        "data": "Fully synthetic, specified in source; no observations from real sensors.",
        "seed": seed, "replicates": replicates, "budgets": list(budgets),
        "candidate_locations": x.tolist(), "noise_groups": groups.tolist(),
        "features": true.features.tolist(), "prior_covariance": true.prior.tolist(),
        "correlated_noise_covariance": true.noise.tolist(),
        "grid_points": len(grid_x), "design_results": rows,
        "correlated_noise_counterexample": counterexample(),
        "prior_predictive_budget_4": prior_predictive_audit(true, cases, grid, replicates, seed),
    }


def make_figures(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"exact": "#23395b", "aware_greedy": "#148878", "diagonal_greedy": "#d47938"}
    labels = {"exact": "Exhaustive optimum", "aware_greedy": "Noise-aware greedy", "diagonal_greedy": "Diagonal-noise greedy"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout="constrained")
    for scenario, ax in zip(("independent", "correlated"), axes):
        for method in labels:
            rows = [r for r in summary["design_results"] if r["scenario"] == scenario and r["method"] == method]
            ax.plot([r["budget"] for r in rows], [r["information_nats_under_true_noise"] for r in rows],
                    marker={"exact": "o", "aware_greedy": "x", "diagonal_greedy": "s"}[method],
                    linestyle="--" if method == "diagonal_greedy" else "-",
                    color=colors[method], label=labels[method])
        ax.plot([r["budget"] for r in rows], [r["uniform_subset_mean_information_nats"] for r in rows],
                ":", color="#808080", label="Uniform subset (exact mean)")
        ax.set(title=f"{scenario.capitalize()} measurement noise", xlabel="Sensor budget", xticks=summary["budgets"])
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Information about latent parameters (nats)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Same marginal noise levels; different joint noise structure")
    fig.savefig(output / "design_information.png", dpi=160)
    plt.close(fig)

    x, _, models = synthetic_problem()
    grid_x = np.linspace(-1, 1, 101)
    grid = basis(grid_x)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    im = axes[0, 0].imshow(models["correlated"].noise, cmap="viridis", vmin=0)
    fig.colorbar(im, ax=axes[0, 0], label="Covariance")
    axes[0, 0].set(title="Known noise covariance (12 candidates)", xlabel="Sensor index", ylabel="Sensor index")
    for method in labels:
        row = next(r for r in summary["design_results"] if r["scenario"] == "correlated" and r["budget"] == 4 and r["method"] == method)
        curve = field_variances(grid, models["correlated"].posterior(row["sensors"]).covariance)
        axes[0, 1].plot(grid_x, curve, color=colors[method], label=labels[method],
                        linestyle="--" if method == "diagonal_greedy" else "-")
    axes[0, 1].set(title="Budget 4: true posterior field variance", xlabel="Field location", ylabel="Latent variance (no observation noise)")
    axes[0, 1].legend(fontsize=8)
    audits = list(summary["prior_predictive_budget_4"].values())
    positions = np.arange(len(audits))
    case_labels = ["Exact / aware", "Greedy / aware", "Diag. / aware", "Diag. / diag."]
    mse = [a["empirical_mean_squared_error"]["estimate"] for a in audits]
    mse_se = [a["empirical_mean_squared_error"]["mcse"] for a in audits]
    axes[1, 0].bar(positions - .16, mse, width=.32, yerr=np.array(mse_se) * 2,
                   color="#148878", label="Empirical MSE ± 2 MCSE", capsize=3)
    axes[1, 0].bar(positions + .16, [a["reported_mean_field_variance"] for a in audits], width=.32,
                   color="#d47938", label="Model-reported variance")
    axes[1, 0].set(title="Prior-predictive check: mean over 101 locations", ylabel="Mean squared error / variance")
    axes[1, 0].legend(fontsize=8)
    coverage = [a["marginal_90_percent_coverage"]["estimate"] for a in audits]
    coverage_se = [a["marginal_90_percent_coverage"]["mcse"] for a in audits]
    axes[1, 1].bar(positions, coverage, yerr=np.array(coverage_se) * 2, color="#23395b", capsize=3)
    axes[1, 1].axhline(.9, linestyle="--", color="#d47938", label="Nominal 90%")
    axes[1, 1].set(title="Average marginal coverage (± 2 MCSE)", ylabel="Coverage", ylim=(0, 1.04))
    axes[1, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xticks(positions, case_labels, rotation=13)
        ax.set_xlabel("Selection rule / inference noise model")
    fig.suptitle(f"Correlated-noise audit · {summary['replicates']:,} independent prior-predictive simulations")
    fig.savefig(output / "noise_and_uncertainty.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("--replicates must be at least 2")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    result = run_audit(args.replicates, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    make_figures(result, args.output)
    print(f"Wrote summary.json and two figures to {args.output}")


if __name__ == "__main__":
    main()
