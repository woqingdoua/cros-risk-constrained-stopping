#!/usr/bin/env python3
"""Exact synthetic proof-of-value study for episode-level CROS-MIX.

The environment is a finite-horizon diagnostic process with a hidden binary
disease state and two ordered tests. A deterministic stopping policy maps the
complete observed history to diagnose, continue, or defer.  We enumerate the
entire deterministic policy class, solve the exact episode-level mixture LP,
and run independent finite-sample certification simulations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.stats import beta


A_DIAGNOSE = 0
A_CONTINUE = 1
A_DEFER = 2
ACTION_NAMES = {A_DIAGNOSE: "diagnose", A_CONTINUE: "continue", A_DEFER: "defer"}
CATEGORY_NAMES = [
    "auto_correct_nonsevere",
    "auto_error_nonsevere",
    "auto_correct_severe",
    "auto_error_severe",
    "defer_nonsevere",
    "defer_severe",
]


@dataclass(frozen=True)
class Scenario:
    name: str
    prevalence: float
    obs_y0: tuple[tuple[float, float, float], tuple[float, float, float]]
    obs_y1: tuple[tuple[float, float, float], tuple[float, float, float]]
    test_costs: tuple[float, float]
    alpha_any: float
    alpha_severe: float
    min_coverage: float
    min_severe_coverage: float


SCENARIOS = [
    Scenario(
        "balanced_moderate_cost",
        0.30,
        ((0.62, 0.25, 0.13), (0.72, 0.20, 0.08)),
        ((0.14, 0.28, 0.58), (0.07, 0.20, 0.73)),
        (0.10, 0.14),
        0.16,
        0.12,
        0.55,
        0.50,
    ),
    Scenario(
        "weak_first_test",
        0.30,
        ((0.50, 0.32, 0.18), (0.76, 0.17, 0.07)),
        ((0.22, 0.34, 0.44), (0.08, 0.20, 0.72)),
        (0.08, 0.16),
        0.17,
        0.13,
        0.50,
        0.45,
    ),
    Scenario(
        "expensive_second_test",
        0.28,
        ((0.65, 0.24, 0.11), (0.78, 0.16, 0.06)),
        ((0.13, 0.28, 0.59), (0.06, 0.17, 0.77)),
        (0.09, 0.28),
        0.15,
        0.11,
        0.52,
        0.45,
    ),
    Scenario(
        "rare_severe",
        0.14,
        ((0.70, 0.22, 0.08), (0.80, 0.15, 0.05)),
        ((0.16, 0.30, 0.54), (0.08, 0.23, 0.69)),
        (0.08, 0.13),
        0.12,
        0.12,
        0.55,
        0.45,
    ),
    Scenario(
        "asymmetric_miss_harm",
        0.36,
        ((0.61, 0.27, 0.12), (0.73, 0.20, 0.07)),
        ((0.16, 0.30, 0.54), (0.08, 0.22, 0.70)),
        (0.11, 0.17),
        0.18,
        0.08,
        0.50,
        0.55,
    ),
    Scenario(
        "low_information_shift",
        0.34,
        ((0.52, 0.30, 0.18), (0.61, 0.27, 0.12)),
        ((0.21, 0.34, 0.45), (0.15, 0.30, 0.55)),
        (0.07, 0.12),
        0.21,
        0.18,
        0.45,
        0.40,
    ),
    Scenario(
        "slack_constraints_negative_control",
        0.30,
        ((0.80, 0.15, 0.05), (0.90, 0.08, 0.02)),
        ((0.05, 0.15, 0.80), (0.02, 0.08, 0.90)),
        (0.01, 0.02),
        0.40,
        1.00,
        0.20,
        0.20,
    ),
]


def all_states() -> list[tuple[int, ...]]:
    return [()] + [(x,) for x in range(3)] + list(itertools.product(range(3), repeat=2))


STATES = all_states()
STATE_INDEX = {state: index for index, state in enumerate(STATES)}
NONTERMINAL = [index for index, state in enumerate(STATES) if len(state) < 2]
TERMINAL = [index for index, state in enumerate(STATES) if len(state) == 2]


def enumerate_policies() -> np.ndarray:
    policies = []
    for early in itertools.product((A_DIAGNOSE, A_CONTINUE, A_DEFER), repeat=len(NONTERMINAL)):
        for terminal in itertools.product((A_DIAGNOSE, A_DEFER), repeat=len(TERMINAL)):
            actions = np.empty(len(STATES), dtype=np.int8)
            actions[NONTERMINAL] = early
            actions[TERMINAL] = terminal
            policies.append(actions)
    return np.stack(policies)


def state_joint_probability(scenario: Scenario, history: tuple[int, ...], y: int) -> float:
    probability = scenario.prevalence if y else 1.0 - scenario.prevalence
    observations = scenario.obs_y1 if y else scenario.obs_y0
    for step, observation in enumerate(history):
        probability *= observations[step][observation]
    return probability


def posterior_and_prediction(scenario: Scenario, history: tuple[int, ...]) -> tuple[float, int]:
    p0 = state_joint_probability(scenario, history, 0)
    p1 = state_joint_probability(scenario, history, 1)
    posterior = p1 / (p0 + p1)
    return posterior, int(posterior >= 0.5)


def evaluate_policy(scenario: Scenario, actions: np.ndarray) -> dict[str, float | np.ndarray]:
    masses: dict[tuple[int, ...], np.ndarray] = {(): np.array([1.0 - scenario.prevalence, scenario.prevalence])}
    categories = np.zeros(6, dtype=float)
    expected_cost = 0.0

    for state in STATES:
        mass = masses.get(state)
        if mass is None:
            continue
        action = int(actions[STATE_INDEX[state]])
        if action == A_CONTINUE and len(state) < 2:
            expected_cost += float(mass.sum()) * scenario.test_costs[len(state)]
            for observation in range(3):
                child = state + (observation,)
                child_mass = np.array(
                    [
                        mass[0] * scenario.obs_y0[len(state)][observation],
                        mass[1] * scenario.obs_y1[len(state)][observation],
                    ]
                )
                masses[child] = masses.get(child, np.zeros(2)) + child_mass
        elif action == A_DIAGNOSE:
            _, prediction = posterior_and_prediction(scenario, state)
            if prediction == 0:
                categories[0] += mass[0]
                categories[3] += mass[1]
            else:
                categories[1] += mass[0]
                categories[2] += mass[1]
        else:
            categories[4] += mass[0]
            categories[5] += mass[1]

    coverage = float(categories[:4].sum())
    errors = float(categories[1] + categories[3])
    severe_coverage_mass = float(categories[2] + categories[3])
    severe_errors = float(categories[3])
    correct_auto = float(categories[0] + categories[2])
    utility = correct_auto - expected_cost
    return {
        "utility": utility,
        "coverage": coverage,
        "errors": errors,
        "severe_coverage_mass": severe_coverage_mass,
        "severe_errors": severe_errors,
        "expected_cost": expected_cost,
        "defer_rate": float(categories[4] + categories[5]),
        "risk_any": errors / coverage if coverage > 0 else math.nan,
        "risk_severe": severe_errors / severe_coverage_mass if severe_coverage_mass > 0 else math.nan,
        "severe_coverage": severe_coverage_mass / scenario.prevalence,
        "categories": categories,
    }


def evaluate_all(scenario: Scenario, policies: np.ndarray) -> dict[str, np.ndarray]:
    rows = [evaluate_policy(scenario, policy) for policy in policies]
    keys = [
        "utility",
        "coverage",
        "errors",
        "severe_coverage_mass",
        "severe_errors",
        "expected_cost",
        "defer_rate",
        "risk_any",
        "risk_severe",
        "severe_coverage",
    ]
    output = {key: np.array([row[key] for row in rows], dtype=float) for key in keys}
    output["categories"] = np.stack([row["categories"] for row in rows])
    if not np.allclose(output["categories"].sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError(f"terminal probability is not conserved in {scenario.name}")
    if not np.allclose(output["coverage"] + output["defer_rate"], 1.0, atol=1e-10):
        raise AssertionError(f"coverage/defer identity failed in {scenario.name}")
    if np.any(output["errors"] > output["coverage"] + 1e-12):
        raise AssertionError(f"errors exceed accepted mass in {scenario.name}")
    if np.any(output["severe_errors"] > output["severe_coverage_mass"] + 1e-12):
        raise AssertionError(f"severe errors exceed severe accepted mass in {scenario.name}")
    return output


def feasible_mask(metrics: dict[str, np.ndarray], scenario: Scenario) -> np.ndarray:
    return (
        (metrics["errors"] <= scenario.alpha_any * metrics["coverage"] + 1e-12)
        & (
            metrics["severe_errors"]
            <= scenario.alpha_severe * metrics["severe_coverage_mass"] + 1e-12
        )
        & (metrics["coverage"] >= scenario.min_coverage - 1e-12)
        & (
            metrics["severe_coverage_mass"]
            >= scenario.min_severe_coverage * scenario.prevalence - 1e-12
        )
    )


def solve_mixture(
    metrics: dict[str, np.ndarray],
    scenario: Scenario,
    risk_factor: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    n = len(metrics["utility"])
    a_ub = np.vstack(
        [
            metrics["errors"] - risk_factor * scenario.alpha_any * metrics["coverage"],
            metrics["severe_errors"]
            - risk_factor * scenario.alpha_severe * metrics["severe_coverage_mass"],
            -metrics["coverage"],
            -metrics["severe_coverage_mass"],
        ]
    )
    b_ub = np.array(
        [
            0.0,
            0.0,
            -scenario.min_coverage,
            -scenario.min_severe_coverage * scenario.prevalence,
        ]
    )
    result = linprog(
        -metrics["utility"],
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, n)),
        b_eq=np.ones(1),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"mixture LP failed for {scenario.name}: {result.message}")
    weights = np.asarray(result.x)
    summary = summarize_weights(metrics, scenario, weights)
    if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-8):
        raise AssertionError(f"mixture weights do not sum to one in {scenario.name}")
    if summary["support_size"] > 5:
        raise AssertionError(f"mixture support exceeds R+1 in {scenario.name}")
    if summary["risk_any"] > risk_factor * scenario.alpha_any + 1e-8:
        raise AssertionError(f"overall risk constraint violated in {scenario.name}")
    if summary["risk_severe"] > risk_factor * scenario.alpha_severe + 1e-8:
        raise AssertionError(f"severe risk constraint violated in {scenario.name}")
    if summary["coverage"] < scenario.min_coverage - 1e-8:
        raise AssertionError(f"coverage constraint violated in {scenario.name}")
    if summary["severe_coverage"] < scenario.min_severe_coverage - 1e-8:
        raise AssertionError(f"severe coverage constraint violated in {scenario.name}")
    summary["solver_status"] = result.message
    return weights, summary


def summarize_weights(
    metrics: dict[str, np.ndarray], scenario: Scenario, weights: np.ndarray
) -> dict[str, float]:
    def dot(key: str) -> float:
        return float(weights @ metrics[key])

    coverage = dot("coverage")
    errors = dot("errors")
    severe_mass = dot("severe_coverage_mass")
    severe_errors = dot("severe_errors")
    return {
        "utility": dot("utility"),
        "coverage": coverage,
        "errors": errors,
        "risk_any": errors / coverage if coverage > 0 else math.nan,
        "severe_coverage": severe_mass / scenario.prevalence,
        "severe_errors": severe_errors,
        "risk_severe": severe_errors / severe_mass if severe_mass > 0 else math.nan,
        "expected_cost": dot("expected_cost"),
        "defer_rate": dot("defer_rate"),
        "support_size": int(np.sum(weights > 1e-8)),
    }


def policy_label(actions: np.ndarray) -> str:
    entries = []
    for state, action in zip(STATES, actions):
        state_label = "root" if not state else "".join(str(value) for value in state)
        entries.append(f"{state_label}:{ACTION_NAMES[int(action)][0]}")
    return "|".join(entries)


def named_baselines(scenario: Scenario, policies: np.ndarray) -> dict[str, int]:
    lookup = {tuple(policy.tolist()): index for index, policy in enumerate(policies)}
    named: dict[str, int] = {}

    def add(name: str, rule) -> None:
        actions = []
        for state in STATES:
            if len(state) == 2:
                action = rule(state, True)
                if action == A_CONTINUE:
                    action = A_DEFER
            else:
                action = rule(state, False)
            actions.append(action)
        named[name] = lookup[tuple(actions)]

    add("always_defer", lambda state, terminal: A_DEFER)
    add("initial_only", lambda state, terminal: A_DIAGNOSE if len(state) == 0 else A_DEFER)
    add(
        "fixed_horizon_1",
        lambda state, terminal: A_CONTINUE if len(state) == 0 else A_DIAGNOSE,
    )
    add(
        "full_information",
        lambda state, terminal: A_CONTINUE if len(state) < 2 else A_DIAGNOSE,
    )
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.25):
        def threshold_rule(state, terminal, threshold=threshold):
            posterior, _ = posterior_and_prediction(scenario, state)
            risk = min(posterior, 1.0 - posterior)
            if risk <= threshold:
                return A_DIAGNOSE
            return A_DEFER if terminal else A_CONTINUE

        add(f"posterior_threshold_{threshold:.2f}", threshold_rule)
    return named


def one_sided_cp_upper(successes: int, total: int, delta: float) -> float:
    if total <= 0:
        return 1.0
    if successes >= total:
        return 1.0
    return float(beta.ppf(1.0 - delta, successes + 1, total - successes))


def one_sided_cp_lower(successes: int, total: int, delta: float) -> float:
    if total <= 0 or successes <= 0:
        return 0.0
    return float(beta.ppf(delta, successes, total - successes + 1))


def certification_experiment(
    scenario: Scenario,
    metrics: dict[str, np.ndarray],
    candidates: list[dict],
    sample_sizes: list[int],
    repeats: int,
    delta: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    candidate_count = len(candidates)
    per_bound_delta = delta / (4 * candidate_count)
    candidate_rows = []
    for rank, candidate in enumerate(candidates):
        weights = candidate["weights"]
        categories = weights @ metrics["categories"]
        summary = summarize_weights(metrics, scenario, weights)
        candidate_rows.append(
            {
                "scenario": scenario.name,
                "candidate_rank": rank,
                "risk_factor": candidate["risk_factor"],
                **summary,
                **{f"prob_{name}": categories[index] for index, name in enumerate(CATEGORY_NAMES)},
                "true_constraint_violation": int(
                    summary["risk_any"] > scenario.alpha_any + 1e-10
                    or summary["risk_severe"] > scenario.alpha_severe + 1e-10
                    or summary["coverage"] < scenario.min_coverage - 1e-10
                    or summary["severe_coverage"] < scenario.min_severe_coverage - 1e-10
                ),
            }
        )

    rows = []
    ranking = sorted(range(candidate_count), key=lambda index: candidates[index]["summary"]["utility"], reverse=True)
    for sample_size in sample_sizes:
        false_certifications = 0
        selected = 0
        no_policy = 0
        selected_utilities = []
        for _ in range(repeats):
            passes = []
            for candidate_index, candidate in enumerate(candidates):
                category_probabilities = candidate["weights"] @ metrics["categories"]
                counts = rng.multinomial(sample_size, category_probabilities)
                accepted = int(counts[:4].sum())
                errors = int(counts[1] + counts[3])
                severe_accepted = int(counts[2] + counts[3])
                severe_errors = int(counts[3])
                severe_total = int(counts[2] + counts[3] + counts[5])
                risk_upper = one_sided_cp_upper(errors, accepted, per_bound_delta)
                severe_risk_upper = one_sided_cp_upper(severe_errors, severe_accepted, per_bound_delta)
                coverage_lower = one_sided_cp_lower(accepted, sample_size, per_bound_delta)
                severe_coverage_lower = one_sided_cp_lower(severe_accepted, severe_total, per_bound_delta)
                passed = (
                    accepted >= 30
                    and severe_accepted >= 10
                    and risk_upper <= scenario.alpha_any
                    and severe_risk_upper <= scenario.alpha_severe
                    and coverage_lower >= scenario.min_coverage
                    and severe_coverage_lower >= scenario.min_severe_coverage
                )
                passes.append(passed)
            chosen = next((index for index in ranking if passes[index]), None)
            if chosen is None:
                no_policy += 1
            else:
                selected += 1
                selected_utilities.append(candidates[chosen]["summary"]["utility"])
                if candidate_rows[chosen]["true_constraint_violation"]:
                    false_certifications += 1
        rows.append(
            {
                "scenario": scenario.name,
                "calibration_n": sample_size,
                "repeats": repeats,
                "family_delta": delta,
                "per_bound_delta": per_bound_delta,
                "certified_selection_rate": selected / repeats,
                "no_policy_certified_rate": no_policy / repeats,
                "false_certification_rate": false_certifications / repeats,
                "mean_selected_utility": float(np.mean(selected_utilities)) if selected_utilities else math.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_report(
    output_dir: Path,
    scenario_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    component_df: pd.DataFrame,
    certification_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    metadata: dict,
) -> None:
    positive_mix = scenario_df[scenario_df["mixture_utility_gain"] > 1e-6]
    material_mix = scenario_df[scenario_df["relative_mixture_gain"] >= 0.01]
    max_false = float(certification_df["false_certification_rate"].max())
    large_n = certification_df[certification_df["calibration_n"] == certification_df["calibration_n"].max()]
    large_n_pass = float(large_n["certified_selection_rate"].mean())
    verdict = (
        "CONTINUE"
        if len(material_mix) >= 1 and max_false <= 0.05 and large_n_pass >= 0.50
        else "CONDITIONAL_CONTINUE"
        if len(positive_mix) >= 1 and max_false <= 0.05
        else "STOP_OR_REDESIGN"
    )

    lines = [
        "# CROS-MIX synthetic proof-of-value report",
        "",
        f"**Verdict: {verdict}**",
        "",
        "This is an exact finite synthetic CMDP validation, not evidence of clinical effectiveness.",
        "The full deterministic full-history stopping-policy class was enumerated, and the",
        "episode-level randomized mixture was solved over its convex hull.",
        "",
        "## Decision criteria",
        "",
        f"- Scenarios with positive exact mixture gain: {len(positive_mix)}/{len(scenario_df)}.",
        f"- Scenarios with at least 1% relative mixture gain: {len(material_mix)}/{len(scenario_df)}.",
        f"- Maximum observed false-certification rate: {max_false:.4f}.",
        f"- Mean certification rate at largest calibration size: {large_n_pass:.4f}.",
        "",
        "## Exact oracle results",
        "",
        scenario_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Baselines",
        "",
        baseline_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Mixture support",
        "",
        component_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Independent certification simulation",
        "",
        certification_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Candidate details, including deliberately unsafe high-utility candidates, are in",
        "`certification_candidates.csv`.",
        "",
        "## Interpretation",
        "",
        "- A positive mixture gain establishes that deterministic stopping can be strictly",
        "  suboptimal under simultaneous selective-risk and coverage constraints.",
        "- The result does not establish that fitted CROS-MIX learns the oracle from finite",
        "  observational clinical data; that requires the CDM/ED experiment.",
        "- Certification pass rate is a power result, not a validity result. False certification",
        "  is the validity diagnostic; no-policy-certified is the correct safe fallback.",
        "- If gains appear only in one hand-designed scenario or are below 1%, the randomized",
        "  component should be framed as a theoretical completion rather than the main empirical claim.",
        "",
        "## Reproducibility metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2, sort_keys=True),
        "```",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="cros_mix_validation/results")
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = enumerate_policies()
    if policies.shape != (41472, len(STATES)):
        raise AssertionError(f"unexpected policy class shape: {policies.shape}")
    if len({tuple(policy.tolist()) for policy in policies}) != len(policies):
        raise AssertionError("deterministic policy enumeration contains duplicates")
    scenario_rows = []
    baseline_rows = []
    component_rows = []
    certification_frames = []
    candidate_frames = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        metrics = evaluate_all(scenario, policies)
        feasible = feasible_mask(metrics, scenario)
        feasible_indices = np.flatnonzero(feasible)
        if len(feasible_indices) == 0:
            raise RuntimeError(f"no deterministic feasible policy in {scenario.name}")
        deterministic_index = int(feasible_indices[np.argmax(metrics["utility"][feasible_indices])])
        deterministic_weights = np.zeros(len(policies))
        deterministic_weights[deterministic_index] = 1.0
        deterministic_summary = summarize_weights(metrics, scenario, deterministic_weights)

        mixture_weights, mixture_summary = solve_mixture(metrics, scenario, risk_factor=1.0)
        mixture_gain = mixture_summary["utility"] - deterministic_summary["utility"]
        relative_gain = mixture_gain / max(abs(deterministic_summary["utility"]), 1e-12)
        scenario_rows.append(
            {
                "scenario": scenario.name,
                "n_policies": len(policies),
                "n_feasible_deterministic": len(feasible_indices),
                "deterministic_utility": deterministic_summary["utility"],
                "mixture_utility": mixture_summary["utility"],
                "mixture_utility_gain": mixture_gain,
                "relative_mixture_gain": relative_gain,
                "mixture_support_size": mixture_summary["support_size"],
                "mixture_risk_any": mixture_summary["risk_any"],
                "target_risk_any": scenario.alpha_any,
                "mixture_risk_severe": mixture_summary["risk_severe"],
                "target_risk_severe": scenario.alpha_severe,
                "mixture_coverage": mixture_summary["coverage"],
                "min_coverage": scenario.min_coverage,
                "mixture_severe_coverage": mixture_summary["severe_coverage"],
                "min_severe_coverage": scenario.min_severe_coverage,
            }
        )

        for policy_index, weight in zip(np.flatnonzero(mixture_weights > 1e-8), mixture_weights[mixture_weights > 1e-8]):
            component_rows.append(
                {
                    "scenario": scenario.name,
                    "policy_index": int(policy_index),
                    "weight": float(weight),
                    "policy": policy_label(policies[policy_index]),
                    "component_utility": metrics["utility"][policy_index],
                    "component_risk_any": metrics["risk_any"][policy_index],
                    "component_risk_severe": metrics["risk_severe"][policy_index],
                    "component_coverage": metrics["coverage"][policy_index],
                    "component_severe_coverage": metrics["severe_coverage"][policy_index],
                }
            )

        named = named_baselines(scenario, policies)
        named["best_feasible_deterministic_oracle"] = deterministic_index
        for name, policy_index in named.items():
            weights = np.zeros(len(policies))
            weights[policy_index] = 1.0
            summary = summarize_weights(metrics, scenario, weights)
            baseline_rows.append(
                {
                    "scenario": scenario.name,
                    "method": name,
                    **summary,
                    "feasible": bool(feasible[policy_index]),
                }
            )
        baseline_rows.append(
            {
                "scenario": scenario.name,
                "method": "CROS_MIX_exact",
                **mixture_summary,
                "feasible": True,
            }
        )

        candidates = []
        for risk_factor in (0.60, 0.75, 0.90, 1.15):
            weights, summary = solve_mixture(metrics, scenario, risk_factor=risk_factor)
            candidates.append({"risk_factor": risk_factor, "weights": weights, "summary": summary})
        certification, candidate_details = certification_experiment(
            scenario,
            metrics,
            candidates,
            sample_sizes=[100, 250, 500, 1000, 2000],
            repeats=args.repeats,
            delta=0.05,
            seed=args.seed + scenario_index,
        )
        certification_frames.append(certification)
        candidate_frames.append(candidate_details)

    scenario_df = pd.DataFrame(scenario_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    component_df = pd.DataFrame(component_rows)
    certification_df = pd.concat(certification_frames, ignore_index=True)
    candidate_df = pd.concat(candidate_frames, ignore_index=True)

    scenario_df.to_csv(output_dir / "scenario_summary.csv", index=False)
    baseline_df.to_csv(output_dir / "baseline_summary.csv", index=False)
    component_df.to_csv(output_dir / "mixture_components.csv", index=False)
    certification_df.to_csv(output_dir / "certification_summary.csv", index=False)
    candidate_df.to_csv(output_dir / "certification_candidates.csv", index=False)
    (output_dir / "scenarios.json").write_text(
        json.dumps([asdict(scenario) for scenario in SCENARIOS], indent=2), encoding="utf-8"
    )

    script_path = Path(__file__).resolve()
    metadata = {
        "seed": args.seed,
        "certification_repeats": args.repeats,
        "policy_count": int(len(policies)),
        "scenario_count": len(SCENARIOS),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "script_sha256": hash_file(script_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(
        output_dir,
        scenario_df,
        baseline_df,
        component_df,
        certification_df,
        candidate_df,
        metadata,
    )
    print(scenario_df.to_string(index=False))
    print()
    print(certification_df.to_string(index=False))
    print(f"wrote results to {output_dir}")


if __name__ == "__main__":
    main()
