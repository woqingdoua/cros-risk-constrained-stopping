#!/usr/bin/env python3
"""Certified selective sequential diagnosis on the ED/Ext-CDS development set.

Risk is the diagnostic error conditional on an autonomous diagnosis.  Deferred
episodes are explicitly unresolved and never counted as correct.  Certification
requires both selective risk <= alpha and autonomous coverage >= gamma.
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

from evaluate_ed_cds_sequential_development import split_three_way


ALPHA = 0.25
GAMMA = 0.70
DELTA = 0.05
ROTATIONS = 3
DEFER_COST = 10.0
FROZEN_TARGET_RISK = float(os.environ.get("ED_CDS_MIX_TARGET_RISK", "0.20"))
FROZEN_TARGET_COVERAGE = float(os.environ.get("ED_CDS_MIX_TARGET_COVERAGE", "0.75"))
THRESHOLDS = (0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


@dataclass
class SelectivePolicy:
    name: str
    family: str
    errors: np.ndarray
    autonomous: np.ndarray
    test_costs: np.ndarray
    total_costs: np.ndarray
    latencies: np.ndarray
    tests: np.ndarray
    support_size: int = 1


def upper_cp(events: int, n: int, delta: float) -> float:
    if events >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, events + 1, n - events))


def lower_cp(events: int, n: int, delta: float) -> float:
    if events <= 0:
        return 0.0
    return float(beta.ppf(delta, events, n - events + 1))


def selective_stats(policy: SelectivePolicy, indices: np.ndarray) -> dict[str, float]:
    autonomous = policy.autonomous[indices]
    count = int(autonomous.sum())
    error_count = int(np.sum(policy.errors[indices] & autonomous))
    coverage = count / len(indices)
    return {
        "selective_risk": error_count / count if count else 1.0,
        "autonomous_coverage": coverage,
        "defer_rate": 1.0 - coverage,
        "autonomous_error_mass": error_count / len(indices),
        "mean_test_cost": float(policy.test_costs[indices].mean()),
        "mean_total_cost": float(policy.total_costs[indices].mean()),
        "mean_latency": float(policy.latencies[indices].mean()),
        "mean_tests": float(policy.tests[indices].mean()),
        "support_size": policy.support_size,
    }


def empirically_feasible(policy: SelectivePolicy, indices: np.ndarray) -> bool:
    metrics = selective_stats(policy, indices)
    return bool(
        metrics["selective_risk"] <= ALPHA
        and metrics["autonomous_coverage"] >= GAMMA
    )


def cheapest(
    policies: list[SelectivePolicy], candidates: list[int], selection: np.ndarray
) -> int | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda index: (
            policies[index].total_costs[selection].mean(),
            -policies[index].autonomous[selection].mean(),
            policies[index].name,
        ),
    )


def empirical_choice(
    policies: list[SelectivePolicy], candidates: list[int], selection: np.ndarray
) -> int | None:
    return cheapest(
        policies,
        [index for index in candidates if empirically_feasible(policies[index], selection)],
        selection,
    )


def screen(
    policies: list[SelectivePolicy], selection: np.ndarray, cap: int = 8
) -> list[int]:
    risk = np.asarray([selective_stats(policy, selection)["selective_risk"] for policy in policies])
    coverage = np.asarray([policy.autonomous[selection].mean() for policy in policies])
    cost = np.asarray([policy.total_costs[selection].mean() for policy in policies])
    candidates: set[int] = {int(np.argmin(cost)), int(np.argmin(risk)), int(np.argmax(coverage))}
    for target in np.linspace(GAMMA, 0.95, 8):
        valid = np.flatnonzero((coverage >= target) & (risk <= ALPHA))
        if len(valid):
            candidates.add(int(valid[np.argmin(cost[valid])]))
    for index in range(len(policies)):
        dominated = np.any(
            (risk <= risk[index] + 1e-12)
            & (coverage >= coverage[index] - 1e-12)
            & (cost <= cost[index] + 1e-12)
            & (
                (risk < risk[index] - 1e-12)
                | (coverage > coverage[index] + 1e-12)
                | (cost < cost[index] - 1e-12)
            )
        )
        if not dominated:
            candidates.add(index)
    ordered = sorted(candidates, key=lambda i: (cost[i], -coverage[i], risk[i], policies[i].name))
    if len(ordered) <= cap:
        return ordered
    positions = np.unique(np.linspace(0, len(ordered) - 1, cap).round().astype(int))
    return [ordered[position] for position in positions]


def candidate_pvalue(
    policy: SelectivePolicy, calibration: np.ndarray
) -> tuple[float, float, float]:
    autonomous = policy.autonomous[calibration]
    selected_n = int(autonomous.sum())
    errors = int(np.sum(policy.errors[calibration] & autonomous))
    risk_p = float(binom.cdf(errors, selected_n, ALPHA)) if selected_n else 1.0
    coverage_p = float(binom.sf(selected_n - 1, len(calibration), GAMMA))
    return max(risk_p, coverage_p), risk_p, coverage_p


def candidate_bounds(
    policy: SelectivePolicy, calibration: np.ndarray, endpoint_delta: float
) -> tuple[float, float, float]:
    autonomous = policy.autonomous[calibration]
    selected_n = int(autonomous.sum())
    errors = int(np.sum(policy.errors[calibration] & autonomous))
    # Error mass and coverage bounds make randomized mixtures linear and valid.
    error_mass_upper = upper_cp(errors, len(calibration), endpoint_delta)
    coverage_lower = lower_cp(selected_n, len(calibration), endpoint_delta)
    selective_upper = (
        min(1.0, error_mass_upper / coverage_lower) if coverage_lower > 0 else 1.0
    )
    return error_mass_upper, coverage_lower, selective_upper


def deterministic_bounds(
    policy: SelectivePolicy, calibration: np.ndarray, endpoint_delta: float
) -> tuple[float, float]:
    autonomous = policy.autonomous[calibration]
    selected_n = int(autonomous.sum())
    errors = int(np.sum(policy.errors[calibration] & autonomous))
    selective_upper = (
        upper_cp(errors, selected_n, endpoint_delta) if selected_n else 1.0
    )
    coverage_lower = lower_cp(selected_n, len(calibration), endpoint_delta)
    return selective_upper, coverage_lower


def frozen_selection_mixture(
    policies: list[SelectivePolicy], selection: np.ndarray, seed: int
) -> SelectivePolicy | None:
    """Fit one safety-margin mixture on selection and freeze its assignments."""
    selection_coverage = np.asarray(
        [policy.autonomous[selection].mean() for policy in policies]
    )
    selection_error_mass = np.asarray(
        [
            np.mean(policy.errors[selection] & policy.autonomous[selection])
            for policy in policies
        ]
    )
    selection_cost = np.asarray(
        [policy.total_costs[selection].mean() for policy in policies]
    )
    target_risk = FROZEN_TARGET_RISK
    target_coverage = FROZEN_TARGET_COVERAGE
    solution = linprog(
        c=selection_cost,
        A_ub=np.vstack(
            (
                selection_error_mass - target_risk * selection_coverage,
                -selection_coverage,
            )
        ),
        b_ub=np.asarray((0.0, -target_coverage)),
        A_eq=np.ones((1, len(policies))),
        b_eq=np.ones(1),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not solution.success:
        return None
    weights = np.where(solution.x < 1e-9, 0.0, solution.x)
    weights /= weights.sum()
    support = np.flatnonzero(weights > 0)
    support_weights = weights[support]
    support_weights /= support_weights.sum()
    rng = np.random.default_rng(seed)
    selected_position = np.searchsorted(
        np.cumsum(support_weights), rng.random(len(policies[0].errors)), side="right"
    )
    selected_position = np.minimum(selected_position, len(support) - 1)
    selected_policy = support[selected_position]
    episode = np.arange(len(selected_policy))

    def realized(attribute: str) -> np.ndarray:
        matrix = np.stack([getattr(policy, attribute) for policy in policies], axis=0)
        return matrix[selected_policy, episode]

    return SelectivePolicy(
        name=(
            "cros_mix_frozen_selection:"
            + ";".join(
                f"{policies[index].name}@{weights[index]:.6f}" for index in support
            )
        ),
        family="frozen_selection_mixture",
        errors=realized("errors").astype(bool),
        autonomous=realized("autonomous").astype(bool),
        test_costs=realized("test_costs").astype(float),
        total_costs=realized("total_costs").astype(float),
        latencies=realized("latencies").astype(float),
        tests=realized("tests").astype(float),
        support_size=len(support),
    )


def make_policies(
    labels: np.ndarray,
    probabilities: dict[int, np.ndarray],
    state_manifest: dict[str, object],
    action_specs: dict[str, dict[str, float]],
) -> list[SelectivePolicy]:
    policies = []
    n = len(labels)
    for sequence_name, sequence in state_manifest["sequences"].items():
        state_ids = [int(value) for value in state_manifest["prefix_state_ids"][sequence_name]]
        prefix = np.stack([probabilities[state_id] for state_id in state_ids], axis=0)
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
        for maximum_horizon in range(len(state_ids)):
            for threshold in THRESHOLDS:
                chosen = np.full(n, maximum_horizon, dtype=int)
                autonomous = np.zeros(n, dtype=bool)
                for horizon in range(maximum_horizon + 1):
                    stop = (~autonomous) & (prefix[horizon].max(axis=1) >= threshold)
                    chosen[stop] = horizon
                    autonomous[stop] = True
                prediction = prefix[chosen, np.arange(n)].argmax(axis=1)
                test_cost = cumulative_cost[chosen]
                total_cost = test_cost + (~autonomous).astype(float) * DEFER_COST
                policies.append(
                    SelectivePolicy(
                        name=(
                            f"selective:{sequence_name}:h={maximum_horizon}:"
                            f"tau={threshold:g}"
                        ),
                        family="selective_sequential",
                        errors=prediction != labels,
                        autonomous=autonomous,
                        test_costs=test_cost,
                        total_costs=total_cost,
                        latencies=cumulative_latency[chosen],
                        tests=chosen.astype(float),
                    )
                )
    clinical_ids = [
        int(value) for value in state_manifest["prefix_state_ids"]["clinical"]
    ]
    for name, state_id, test_count in (
        ("forced:hpi", clinical_ids[0], 0),
        ("forced:full_workup", clinical_ids[-1], len(state_manifest["actions"])),
    ):
        state = state_manifest["states"][state_id]
        acquired = list(state["acquired"])
        prediction = probabilities[state_id].argmax(axis=1)
        test_cost = float(
            sum(action_specs[action]["resource_cost"] for action in acquired)
        )
        latency = float(
            sum(action_specs[action]["latency_minutes"] for action in acquired)
        )
        policies.append(
            SelectivePolicy(
                name=name,
                family="forced_choice",
                errors=prediction != labels,
                autonomous=np.ones(n, dtype=bool),
                test_costs=np.full(n, test_cost),
                total_costs=np.full(n, test_cost),
                latencies=np.full(n, latency),
                tests=np.full(n, test_count, dtype=float),
            )
        )
    return policies


def evaluate_trial(
    policies: list[SelectivePolicy],
    selection: np.ndarray,
    calibration: np.ndarray,
    evaluation: np.ndarray,
    randomization_seed: int,
) -> list[dict[str, object]]:
    policies = list(policies)
    rows = []

    def add(method: str, index: int | None, certified: bool = False) -> None:
        if index is None:
            rows.append(
                {
                    "method": method,
                    "available": False,
                    "certified": certified,
                    "selected_policy": None,
                }
            )
        else:
            rows.append(
                {
                    "method": method,
                    "available": True,
                    "certified": certified,
                    "selected_policy": policies[index].name,
                    **selective_stats(policies[index], evaluation),
                }
            )

    hpi_forced = next(i for i, policy in enumerate(policies) if policy.name == "forced:hpi")
    full_forced = next(
        i for i, policy in enumerate(policies) if policy.name == "forced:full_workup"
    )
    add("hpi_forced", hpi_forced)
    add("full_workup_forced", full_forced)
    empirical = empirical_choice(policies, list(range(len(policies))), selection)
    add("empirical_selective_erm", empirical)
    deterministic_policy_n = len(policies)

    frozen_mix = frozen_selection_mixture(policies, selection, randomization_seed)
    if frozen_mix is None:
        add("cros_mix_frozen", None, certified=True)
    else:
        policies.append(frozen_mix)
        mix_index = len(policies) - 1
        joint_pvalue = candidate_pvalue(frozen_mix, calibration)[0]
        add(
            "cros_mix_frozen",
            mix_index if joint_pvalue <= DELTA else None,
            certified=True,
        )

    screened = screen(policies[:deterministic_policy_n], selection)
    pvalues = np.asarray(
        [candidate_pvalue(policies[index], calibration)[0] for index in screened]
    )
    # Each rotation is a separate hypothetical locked-calibration trial used to
    # estimate power, not three simultaneous claims in one deployment.
    trial_delta = DELTA
    fixed_order = sorted(
        range(len(screened)),
        key=lambda j: (
            max(
                selective_stats(policies[screened[j]], selection)["selective_risk"]
                / ALPHA,
                GAMMA
                / max(policies[screened[j]].autonomous[selection].mean(), 1e-12),
            ),
            -policies[screened[j]].total_costs[selection].mean(),
            policies[screened[j]].name,
        ),
    )
    rejected = []
    for position in fixed_order:
        if pvalues[position] <= trial_delta:
            rejected.append(screened[position])
        else:
            break
    add("ltt_joint", cheapest(policies, rejected, selection), certified=True)

    ordered = np.argsort(pvalues)
    holm_positions = []
    for rank, position in enumerate(ordered):
        if pvalues[position] <= trial_delta / (len(screened) - rank):
            holm_positions.append(int(position))
        else:
            break
    add(
        "holm_joint",
        cheapest(policies, [screened[pos] for pos in holm_positions], selection),
        certified=True,
    )
    bonf_positions = np.flatnonzero(pvalues <= trial_delta / len(screened)).tolist()
    add(
        "bonferroni_joint",
        cheapest(policies, [screened[pos] for pos in bonf_positions], selection),
        certified=True,
    )

    endpoint_delta = trial_delta / (2 * len(screened))
    bounds = np.asarray(
        [candidate_bounds(policies[index], calibration, endpoint_delta) for index in screened]
    )
    error_upper = bounds[:, 0]
    coverage_lower = bounds[:, 1]
    direct_bounds = np.asarray(
        [
            deterministic_bounds(policies[index], calibration, endpoint_delta)
            for index in screened
        ]
    )
    deterministic = [
        screened[position]
        for position in range(len(screened))
        if direct_bounds[position, 0] <= ALPHA
        and direct_bounds[position, 1] >= GAMMA
    ]
    add("cros_det_joint", cheapest(policies, deterministic, selection), certified=True)

    objective = np.asarray(
        [policies[index].total_costs[selection].mean() for index in screened]
    )
    solution = linprog(
        c=objective,
        A_ub=np.vstack((error_upper - ALPHA * coverage_lower, -coverage_lower)),
        b_ub=np.asarray((0.0, -GAMMA)),
        A_eq=np.ones((1, len(screened))),
        b_eq=np.ones(1),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not solution.success:
        add("cros_mix_joint", None, certified=True)
    else:
        weights = np.where(solution.x < 1e-9, 0.0, solution.x)
        weights /= weights.sum()
        eval_n = len(evaluation)
        error_mass = sum(
            weight
            * np.mean(
                policies[index].errors[evaluation]
                & policies[index].autonomous[evaluation]
            )
            for weight, index in zip(weights, screened)
        )
        coverage = sum(
            weight * policies[index].autonomous[evaluation].mean()
            for weight, index in zip(weights, screened)
        )
        rows.append(
            {
                "method": "cros_mix_joint",
                "available": True,
                "certified": True,
                "selected_policy": ";".join(
                    f"{policies[index].name}@{weight:.6f}"
                    for weight, index in zip(weights, screened)
                    if weight > 1e-9
                ),
                "selective_risk": error_mass / coverage if coverage > 0 else 1.0,
                "autonomous_coverage": coverage,
                "defer_rate": 1.0 - coverage,
                "autonomous_error_mass": error_mass,
                "mean_test_cost": sum(
                    weight * policies[index].test_costs[evaluation].mean()
                    for weight, index in zip(weights, screened)
                ),
                "mean_total_cost": sum(
                    weight * policies[index].total_costs[evaluation].mean()
                    for weight, index in zip(weights, screened)
                ),
                "mean_latency": sum(
                    weight * policies[index].latencies[evaluation].mean()
                    for weight, index in zip(weights, screened)
                ),
                "mean_tests": sum(
                    weight * policies[index].tests[evaluation].mean()
                    for weight, index in zip(weights, screened)
                ),
                "support_size": int(np.sum(weights > 1e-9)),
            }
        )
    for row in rows:
        row["screened_candidate_n"] = len(screened)
        row["joint_transfer"] = bool(
            row["available"]
            and row.get("selective_risk", 1.0) <= ALPHA
            and row.get("autonomous_coverage", 0.0) >= GAMMA
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="cros_mix_validation/results/ed_cds_sequential_v1"
    )
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=20261201)
    args = parser.parse_args()
    root = Path(args.root)
    state_manifest = json.loads((root / "state_manifest.json").read_text(encoding="utf-8"))
    benchmark = json.loads(
        Path("cros_mix_validation/data/derived/ed_cds_action_v1/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    probabilities = {}
    labels = None
    for state in state_manifest["states"]:
        state_id = int(state["state_id"])
        data = np.load(root / "states" / f"state_{state_id:03d}.npz")
        probabilities[state_id] = data["probabilities"].astype(float)
        if labels is None:
            labels = data["labels"].astype(int)
    assert labels is not None
    policies = make_policies(labels, probabilities, state_manifest, benchmark["actions"])
    trial_rows = []
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        groups = split_three_way(labels, seed)
        for rotation in range(ROTATIONS):
            selection = groups[rotation]
            calibration = groups[(rotation + 1) % ROTATIONS]
            evaluation = groups[(rotation + 2) % ROTATIONS]
            rows = evaluate_trial(
                policies,
                selection,
                calibration,
                evaluation,
                randomization_seed=seed * 10 + rotation,
            )
            for row in rows:
                row.update({"seed": seed, "rotation": rotation})
                trial_rows.append(row)
    results = pd.DataFrame(trial_rows)
    results.to_csv(root / "selective_trials.csv", index=False)
    availability = results.groupby("method", as_index=False).agg(
        available_trials=("available", "sum")
    )
    available = results[results["available"]]
    metrics = available.groupby("method", as_index=False).agg(
        mean_selective_risk=("selective_risk", "mean"),
        sd_selective_risk=("selective_risk", "std"),
        mean_coverage=("autonomous_coverage", "mean"),
        mean_defer_rate=("defer_rate", "mean"),
        joint_transfer_rate=("joint_transfer", "mean"),
        mean_test_cost=("mean_test_cost", "mean"),
        mean_total_cost=("mean_total_cost", "mean"),
        mean_latency=("mean_latency", "mean"),
        mean_tests=("mean_tests", "mean"),
        mean_support_size=("support_size", "mean"),
    )
    summary = availability.merge(metrics, on="method", how="left")
    summary["availability_rate"] = summary["available_trials"] / (
        args.num_seeds * ROTATIONS
    )
    summary = summary.sort_values(
        ["availability_rate", "mean_total_cost"],
        ascending=[False, True],
        na_position="last",
    )
    summary.to_csv(root / "selective_summary.csv", index=False)
    report = [
        "# ED/Ext-CDS selective sequential diagnosis pilot",
        "",
        f"Joint target: selective diagnostic risk <= {ALPHA} and autonomous coverage >= {GAMMA}.",
        f"A terminal clinician deferral has benchmark burden {DEFER_COST} relative units.",
        "Deferred cases are unresolved and are never counted as correctly diagnosed.",
        "All model predictions are development-only five-fold OOF; every repeated",
        "trial separates selection, calibration, and evaluation episodes.",
        "",
        summary.to_markdown(index=False),
        "",
        "CROS-MIX uses simultaneous upper bounds on autonomous error mass and",
        "lower bounds on autonomous coverage. Its selective-risk constraint is",
        "linearized as q_upper - alpha * coverage_lower <= 0.",
        "",
    ]
    (root / "selective_report.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "status": "ED_CDS_SELECTIVE_SEQUENTIAL_DEVELOPMENT_COMPLETE",
        "alpha": ALPHA,
        "minimum_autonomous_coverage": GAMMA,
        "delta": DELTA,
        "defer_cost": DEFER_COST,
        "frozen_mix_selection_target_risk": FROZEN_TARGET_RISK,
        "frozen_mix_selection_target_coverage": FROZEN_TARGET_COVERAGE,
        "policy_n": len(policies),
        "seeds": args.num_seeds,
        "rotations_per_seed": ROTATIONS,
        "locked_calibration_or_test_loaded": False,
        "deferred_cases_counted_correct": False,
    }
    (root / "selective_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    for path in root.iterdir():
        if path.is_file():
            os.chmod(path, 0o600)
    print(summary.to_string(index=False))
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
