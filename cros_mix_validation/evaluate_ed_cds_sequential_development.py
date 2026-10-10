#!/usr/bin/env python3
"""Development-only evaluation of realistic ED/Ext-CDS sequential policies.

All diagnostic predictions are out of fold.  Each repeated trial separates
policy/gate fitting (selection), risk certification (calibration), and reporting
(evaluation).  The locked benchmark calibration and test files are never opened.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import beta, binom
from sklearn.linear_model import Ridge


ALPHA = 0.25
DELTA = 0.05
ROTATIONS = 3
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
VOI_PENALTIES = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


@dataclass
class Policy:
    name: str
    family: str
    errors: np.ndarray
    costs: np.ndarray
    latencies: np.ndarray
    tests: np.ndarray


def exact_null_pvalue(errors: int, n: int, alpha: float) -> float:
    return float(binom.cdf(errors, n, alpha))


def exact_upper_bound(errors: int, n: int, delta: float) -> float:
    if errors >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


def metrics(policy: Policy, indices: np.ndarray) -> dict[str, float]:
    return {
        "risk": float(policy.errors[indices].mean()),
        "cost": float(policy.costs[indices].mean()),
        "latency": float(policy.latencies[indices].mean()),
        "tests": float(policy.tests[indices].mean()),
        "support_size": 1,
    }


def absent() -> dict[str, object]:
    return {
        "available": False,
        "risk": None,
        "cost": None,
        "latency": None,
        "tests": None,
        "support_size": 0,
        "selected_policy": None,
    }


def empirical_choice(
    policies: list[Policy], candidates: list[int], selection: np.ndarray
) -> int | None:
    feasible = [
        index for index in candidates if policies[index].errors[selection].mean() <= ALPHA
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda index: (
            policies[index].costs[selection].mean(),
            policies[index].errors[selection].mean(),
            policies[index].name,
        ),
    )


def cheapest_choice(
    policies: list[Policy], candidates: list[int], selection: np.ndarray
) -> int | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda index: (
            policies[index].costs[selection].mean(),
            policies[index].errors[selection].mean(),
            policies[index].name,
        ),
    )


def screen_candidates(
    policies: list[Policy], selection: np.ndarray, cap: int = 32
) -> list[int]:
    risks = np.asarray([policy.errors[selection].mean() for policy in policies])
    costs = np.asarray([policy.costs[selection].mean() for policy in policies])
    selected: set[int] = set()
    for index in range(len(policies)):
        dominated = np.any(
            (risks <= risks[index] + 1e-12)
            & (costs <= costs[index] + 1e-12)
            & ((risks < risks[index] - 1e-12) | (costs < costs[index] - 1e-12))
        )
        if not dominated:
            selected.add(index)
    for target in np.linspace(0.12, ALPHA, 10):
        feasible = np.flatnonzero(risks <= target)
        if len(feasible):
            selected.add(int(feasible[np.argmin(costs[feasible])]))
    selected.add(int(np.argmin(risks)))
    selected.add(int(np.argmin(costs)))
    ordered = sorted(selected, key=lambda i: (costs[i], risks[i], policies[i].name))
    if len(ordered) <= cap:
        return ordered
    positions = np.unique(np.linspace(0, len(ordered) - 1, cap).round().astype(int))
    return [ordered[position] for position in positions]


def posterior_features(probabilities: np.ndarray) -> np.ndarray:
    confidence = probabilities.max(axis=1, keepdims=True)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-8, 1.0)), axis=1, keepdims=True
    )
    top = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1], dtype=float)[top]
    return np.hstack((probabilities, confidence, entropy, one_hot))


def make_myopic_policies(
    labels: np.ndarray,
    state_probabilities: dict[int, np.ndarray],
    hpi_state: int,
    singleton_states: dict[str, int],
    action_specs: dict[str, dict[str, float]],
    selection: np.ndarray,
) -> list[Policy]:
    """Fit a selection-only gate and allow at most one diagnostic action."""
    hpi_prob = state_probabilities[hpi_state]
    hpi_prediction = hpi_prob.argmax(axis=1)
    hpi_error = (hpi_prediction != labels).astype(float)
    features = posterior_features(hpi_prob)
    estimated_benefits = []
    actions = list(singleton_states)
    for action in actions:
        after_prediction = state_probabilities[singleton_states[action]].argmax(axis=1)
        target = hpi_error - (after_prediction != labels).astype(float)
        regressor = Ridge(alpha=2.0)
        regressor.fit(features[selection], target[selection])
        estimated_benefits.append(regressor.predict(features))
    benefit = np.asarray(estimated_benefits).T

    policies = []
    row = np.arange(len(labels))
    action_costs = np.asarray(
        [float(action_specs[action]["resource_cost"]) for action in actions]
    )
    action_latencies = np.asarray(
        [float(action_specs[action]["latency_minutes"]) for action in actions]
    )
    for penalty in VOI_PENALTIES:
        utility = benefit - penalty * action_costs[None, :]
        selected_action = utility.argmax(axis=1)
        execute = utility[row, selected_action] > 0.0
        predictions = hpi_prediction.copy()
        costs = np.zeros(len(labels), dtype=float)
        latencies = np.zeros(len(labels), dtype=float)
        tests = np.zeros(len(labels), dtype=float)
        for action_index, action in enumerate(actions):
            mask = execute & (selected_action == action_index)
            predictions[mask] = state_probabilities[singleton_states[action]][mask].argmax(
                axis=1
            )
            costs[mask] = action_costs[action_index]
            latencies[mask] = action_latencies[action_index]
            tests[mask] = 1.0
        policies.append(
            Policy(
                name=f"myopic_voi:lambda={penalty:g}",
                family="myopic_voi",
                errors=(predictions != labels).astype(float),
                costs=costs,
                latencies=latencies,
                tests=tests,
            )
        )
    return policies


def build_core_policies(
    labels: np.ndarray,
    state_probabilities: dict[int, np.ndarray],
    state_manifest: dict[str, object],
    action_specs: dict[str, dict[str, float]],
) -> list[Policy]:
    policies: list[Policy] = []
    actions = state_manifest["actions"]
    for sequence_name, sequence in state_manifest["sequences"].items():
        state_ids = state_manifest["prefix_state_ids"][sequence_name]
        cumulative_cost = np.zeros(len(state_ids), dtype=float)
        cumulative_latency = np.zeros(len(state_ids), dtype=float)
        for horizon in range(1, len(state_ids)):
            action = sequence[horizon - 1]
            cumulative_cost[horizon] = cumulative_cost[horizon - 1] + float(
                action_specs[action]["resource_cost"]
            )
            cumulative_latency[horizon] = cumulative_latency[horizon - 1] + float(
                action_specs[action]["latency_minutes"]
            )
        prefix_prob = np.stack(
            [state_probabilities[int(state_id)] for state_id in state_ids], axis=0
        )
        for horizon in range(len(state_ids)):
            if horizon in {0, len(actions)} and sequence_name != "clinical":
                continue
            name = f"fixed:{sequence_name}:k={horizon}"
            if sequence_name == "clinical" and horizon == 0:
                name = "hpi_only"
            if sequence_name == "clinical" and horizon == len(actions):
                name = "full_workup"
            prediction = prefix_prob[horizon].argmax(axis=1)
            policies.append(
                Policy(
                    name=name,
                    family="fixed_stage",
                    errors=(prediction != labels).astype(float),
                    costs=np.full(len(labels), cumulative_cost[horizon]),
                    latencies=np.full(len(labels), cumulative_latency[horizon]),
                    tests=np.full(len(labels), horizon, dtype=float),
                )
            )
        for threshold in THRESHOLDS:
            chosen = np.full(len(labels), len(actions), dtype=int)
            for horizon in range(len(actions)):
                stop = (chosen == len(actions)) & (
                    prefix_prob[horizon].max(axis=1) >= threshold
                )
                chosen[stop] = horizon
            prediction = prefix_prob[chosen, np.arange(len(labels))].argmax(axis=1)
            policies.append(
                Policy(
                    name=f"confidence:{sequence_name}:tau={threshold:g}",
                    family="confidence_threshold",
                    errors=(prediction != labels).astype(float),
                    costs=cumulative_cost[chosen],
                    latencies=cumulative_latency[chosen],
                    tests=chosen.astype(float),
                )
            )
    return policies


def split_three_way(labels: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = [[], [], []]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for group, part in zip(groups, np.array_split(indices, 3)):
            group.extend(part.tolist())
    return [np.asarray(sorted(group), dtype=int) for group in groups]


def evaluate_trial(
    policies: list[Policy],
    selection: np.ndarray,
    calibration: np.ndarray,
    evaluation: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_named(method: str, index: int | None) -> None:
        if index is None:
            rows.append({"method": method, **absent()})
        else:
            rows.append(
                {
                    "method": method,
                    "available": True,
                    **metrics(policies[index], evaluation),
                    "selected_policy": policies[index].name,
                }
            )

    hpi = next(i for i, policy in enumerate(policies) if policy.name == "hpi_only")
    full = next(i for i, policy in enumerate(policies) if policy.name == "full_workup")
    add_named("hpi_only", hpi)
    add_named("full_workup", full)
    for method, family in (
        ("best_fixed_stage", "fixed_stage"),
        ("confidence_threshold", "confidence_threshold"),
        ("myopic_voi", "myopic_voi"),
    ):
        candidates = [i for i, policy in enumerate(policies) if policy.family == family]
        add_named(method, empirical_choice(policies, candidates, selection))
    add_named("empirical_erm", empirical_choice(policies, list(range(len(policies))), selection))

    screened = screen_candidates(policies, selection)
    n = len(calibration)
    errors = np.asarray([int(policies[i].errors[calibration].sum()) for i in screened])
    pvalues = np.asarray([exact_null_pvalue(int(error), n, ALPHA) for error in errors])
    trial_delta = DELTA / ROTATIONS

    fixed_order = sorted(
        range(len(screened)),
        key=lambda j: (
            policies[screened[j]].errors[selection].mean(),
            -policies[screened[j]].costs[selection].mean(),
            policies[screened[j]].name,
        ),
    )
    rejected = []
    for position in fixed_order:
        if pvalues[position] <= trial_delta:
            rejected.append(screened[position])
        else:
            break
    add_named("ltt_fixed_sequence", cheapest_choice(policies, rejected, selection))

    order = np.argsort(pvalues)
    holm_positions = []
    for rank, position in enumerate(order):
        if pvalues[position] <= trial_delta / (len(screened) - rank):
            holm_positions.append(int(position))
        else:
            break
    add_named(
        "holm_stepdown",
        cheapest_choice(
            policies, [screened[position] for position in holm_positions], selection
        ),
    )
    bonf_positions = np.flatnonzero(pvalues <= trial_delta / len(screened)).tolist()
    add_named(
        "bonferroni",
        cheapest_choice(
            policies, [screened[position] for position in bonf_positions], selection
        ),
    )

    simultaneous_delta = trial_delta / len(screened)
    upper = np.asarray(
        [exact_upper_bound(int(error), n, simultaneous_delta) for error in errors]
    )
    certified = [screened[position] for position in np.flatnonzero(upper <= ALPHA)]
    add_named("cros_det", cheapest_choice(policies, certified, selection))

    selection_costs = np.asarray(
        [policies[index].costs[selection].mean() for index in screened]
    )
    solution = linprog(
        c=selection_costs,
        A_ub=upper[None, :],
        b_ub=np.asarray([ALPHA]),
        A_eq=np.ones((1, len(screened))),
        b_eq=np.ones(1),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not solution.success:
        rows.append({"method": "cros_mix", **absent()})
    else:
        weights = np.where(solution.x < 1e-9, 0.0, solution.x)
        weights /= weights.sum()
        rows.append(
            {
                "method": "cros_mix",
                "available": True,
                "risk": float(
                    sum(
                        weight * policies[index].errors[evaluation].mean()
                        for weight, index in zip(weights, screened)
                    )
                ),
                "cost": float(
                    sum(
                        weight * policies[index].costs[evaluation].mean()
                        for weight, index in zip(weights, screened)
                    )
                ),
                "latency": float(
                    sum(
                        weight * policies[index].latencies[evaluation].mean()
                        for weight, index in zip(weights, screened)
                    )
                ),
                "tests": float(
                    sum(
                        weight * policies[index].tests[evaluation].mean()
                        for weight, index in zip(weights, screened)
                    )
                ),
                "support_size": int(np.sum(weights > 1e-9)),
                "selected_policy": ";".join(
                    f"{policies[index].name}@{weight:.6f}"
                    for weight, index in zip(weights, screened)
                    if weight > 1e-9
                ),
            }
        )
    for row in rows:
        row["screened_candidate_n"] = len(screened)
        row["transfer"] = bool(row["available"] and row["risk"] <= ALPHA)
        row["violation"] = (
            max(0.0, float(row["risk"]) - ALPHA) if row["available"] else None
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-manifest",
        default="cros_mix_validation/results/ed_cds_sequential_v1/state_manifest.json",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="cros_mix_validation/data/derived/ed_cds_action_v1/manifest.json",
    )
    parser.add_argument(
        "--output-root", default="cros_mix_validation/results/ed_cds_sequential_v1"
    )
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=20261001)
    args = parser.parse_args()

    root = Path(args.output_root)
    state_manifest = json.loads(Path(args.state_manifest).read_text(encoding="utf-8"))
    benchmark = json.loads(Path(args.benchmark_manifest).read_text(encoding="utf-8"))
    state_probabilities: dict[int, np.ndarray] = {}
    labels = None
    state_rows = []
    for state in state_manifest["states"]:
        state_id = int(state["state_id"])
        data = np.load(root / "states" / f"state_{state_id:03d}.npz")
        if labels is None:
            labels = data["labels"].astype(int)
        elif not np.array_equal(labels, data["labels"]):
            raise AssertionError("state label order mismatch")
        probability = data["probabilities"].astype(float)
        state_probabilities[state_id] = probability
        state_rows.append(
            {
                "state_id": state_id,
                "acquired": "+".join(state["acquired"]) or "hpi_only",
                "test_count": state["test_count"],
                "resource_cost": state["resource_cost"],
                "risk": float(np.mean(probability.argmax(axis=1) != labels)),
            }
        )
    assert labels is not None
    pd.DataFrame(state_rows).sort_values(["risk", "resource_cost"]).to_csv(
        root / "state_summary.csv", index=False
    )
    core = build_core_policies(
        labels, state_probabilities, state_manifest, benchmark["actions"]
    )

    result_rows = []
    for seed_offset in range(args.num_seeds):
        seed = args.seed_start + seed_offset
        groups = split_three_way(labels, seed)
        for rotation in range(ROTATIONS):
            selection = groups[rotation]
            calibration = groups[(rotation + 1) % ROTATIONS]
            evaluation = groups[(rotation + 2) % ROTATIONS]
            myopic = make_myopic_policies(
                labels,
                state_probabilities,
                int(state_manifest["prefix_state_ids"]["clinical"][0]),
                {key: int(value) for key, value in state_manifest["singleton_state_ids"].items()},
                benchmark["actions"],
                selection,
            )
            rows = evaluate_trial(core + myopic, selection, calibration, evaluation)
            for row in rows:
                row.update(
                    {
                        "seed": seed,
                        "rotation": rotation,
                        "selection_n": len(selection),
                        "calibration_n": len(calibration),
                        "evaluation_n": len(evaluation),
                    }
                )
                result_rows.append(row)
    results = pd.DataFrame(result_rows)
    results.to_csv(root / "development_trials.csv", index=False)
    available = results[results["available"]].copy()
    availability = (
        results.groupby("method", as_index=False)
        .agg(available_trials=("available", "sum"))
    )
    metric_summary = (
        available.groupby("method", as_index=False)
        .agg(
            mean_risk=("risk", "mean"),
            sd_risk=("risk", "std"),
            transfer_rate=("transfer", "mean"),
            mean_cost=("cost", "mean"),
            mean_latency=("latency", "mean"),
            mean_tests=("tests", "mean"),
            mean_support_size=("support_size", "mean"),
        )
    )
    summary = availability.merge(metric_summary, on="method", how="left")
    total_trials = args.num_seeds * ROTATIONS
    summary["availability_rate"] = summary["available_trials"] / total_trials
    summary = summary.sort_values(["mean_risk", "mean_cost"])
    summary.to_csv(root / "development_summary.csv", index=False)
    report = [
        "# ED/Ext-CDS realistic-action sequential diagnosis pilot",
        "",
        f"Development-only repeated three-way evaluation; alpha={ALPHA}, delta={DELTA}.",
        "All diagnostic state predictions are five-fold out of fold. The myopic",
        "one-step VOI gate is fit only on each trial's selection split. Locked",
        "benchmark calibration and test episodes were not opened.",
        "",
        summary.to_markdown(index=False),
        "",
        "The environment contains 12 named actions plus STOP. Ext-CDS test",
        "snippets are untimestamped retrospective observations, so the task measures",
        "logged information acquisition and must not be described as a causal",
        "counterfactual test-ordering simulator.",
        "",
    ]
    (root / "development_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "status": "ED_CDS_SEQUENTIAL_DEVELOPMENT_COMPLETE",
        "alpha": ALPHA,
        "delta": DELTA,
        "seeds": args.num_seeds,
        "rotations_per_seed": ROTATIONS,
        "state_n": int(state_manifest["state_n"]),
        "core_policy_n": len(core),
        "locked_calibration_or_test_loaded": False,
    }
    (root / "development_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    for path in root.iterdir():
        if path.is_file():
            os.chmod(path, 0o600)
    print(summary.to_string(index=False))
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
