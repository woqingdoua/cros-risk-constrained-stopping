#!/usr/bin/env python3
"""Development-only common-backbone stopping-controller comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_ed_cds_sequential_development import split_three_way
from run_ed_cds_selective_sequential_pilot import (
    ALPHA,
    DELTA,
    GAMMA,
    SelectivePolicy,
    candidate_pvalue,
    cheapest,
    deterministic_bounds,
    empirical_choice,
    frozen_selection_mixture,
    screen,
    selective_stats,
)


THRESHOLDS = (
    0.0,
    0.50,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
    0.9975,
    0.999,
    0.9995,
    0.9999,
)
DEFER_COST = 10.0


def load_shards(root: Path, expected_n: int) -> dict[str, np.ndarray]:
    paths = sorted(root.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError("common-path shards not found")
    shards = [np.load(path) for path in paths]
    values = {
        key: np.concatenate([shard[key] for shard in shards], axis=0)
        for key in shards[0].files
    }
    order = np.argsort(values["global_indices"])
    values = {key: value[order] for key, value in values.items()}
    if len(values["labels"]) != expected_n:
        raise AssertionError(f"expected {expected_n} rows, got {len(values['labels'])}")
    if not np.array_equal(values["global_indices"], np.arange(expected_n)):
        raise AssertionError("common-path rows are incomplete")
    return values


def policy_from_stage(
    name: str,
    family: str,
    data: dict[str, np.ndarray],
    stage: np.ndarray,
    autonomous: np.ndarray,
) -> SelectivePolicy:
    episode = np.arange(len(data["labels"]))
    probabilities = data["probabilities"][episode, stage]
    predictions = probabilities.argmax(axis=1)
    errors = predictions != data["labels"]
    test_costs = data["cumulative_costs"][episode, stage]
    total_costs = test_costs + (~autonomous).astype(float) * DEFER_COST
    return SelectivePolicy(
        name=name,
        family=family,
        errors=errors,
        autonomous=autonomous,
        test_costs=test_costs,
        total_costs=total_costs,
        latencies=data["cumulative_latencies"][episode, stage],
        tests=stage.astype(float),
    )


def make_policies(data: dict[str, np.ndarray]) -> list[SelectivePolicy]:
    n, stages = data["confidences"].shape
    episode = np.arange(n)
    policies: list[SelectivePolicy] = []
    native_stage = data["native_stop_stage"].astype(int)
    policies.append(
        policy_from_stage(
            "la_native_forced",
            "la_native",
            data,
            native_stage,
            np.ones(n, dtype=bool),
        )
    )
    for threshold in THRESHOLDS:
        native_auto = data["confidences"][episode, native_stage] >= threshold
        policies.append(
            policy_from_stage(
                f"la_native_defer:tau={threshold:g}",
                "la_native_defer",
                data,
                native_stage,
                native_auto,
            )
        )
    for horizon in range(stages):
        policies.append(
            policy_from_stage(
                f"forced_common_path:h={horizon}",
                "forced_horizon",
                data,
                np.full(n, horizon, dtype=int),
                np.ones(n, dtype=bool),
            )
        )
        for threshold in THRESHOLDS:
            chosen = np.full(n, horizon, dtype=int)
            autonomous = np.zeros(n, dtype=bool)
            for stage in range(horizon + 1):
                stop = (~autonomous) & (data["confidences"][:, stage] >= threshold)
                chosen[stop] = stage
                autonomous[stop] = True
            policies.append(
                policy_from_stage(
                    f"common_path_selective:h={horizon}:tau={threshold:g}",
                    "common_path_selective",
                    data,
                    chosen,
                    autonomous,
                )
            )
    return policies


def add_result(
    rows: list[dict[str, object]],
    method: str,
    policy: SelectivePolicy | None,
    evaluation: np.ndarray,
    certified: bool,
) -> None:
    if policy is None:
        rows.append(
            {
                "method": method,
                "available": False,
                "certified": certified,
                "selected_policy": None,
            }
        )
        return
    rows.append(
        {
            "method": method,
            "available": True,
            "certified": certified,
            "selected_policy": policy.name,
            **selective_stats(policy, evaluation),
        }
    )


def evaluate_trial(
    base_policies: list[SelectivePolicy],
    selection: np.ndarray,
    calibration: np.ndarray,
    evaluation: np.ndarray,
    randomization_seed: int,
) -> list[dict[str, object]]:
    policies = list(base_policies)
    rows: list[dict[str, object]] = []
    native = next(policy for policy in policies if policy.name == "la_native_forced")
    add_result(rows, "la_native_forced", native, evaluation, False)
    native_defer_indices = [
        index for index, policy in enumerate(policies) if policy.family == "la_native_defer"
    ]
    native_empirical_index = empirical_choice(policies, native_defer_indices, selection)
    add_result(
        rows,
        "la_native_empirical_defer",
        policies[native_empirical_index] if native_empirical_index is not None else None,
        evaluation,
        False,
    )
    empirical_index = empirical_choice(policies, list(range(len(policies))), selection)
    add_result(
        rows,
        "common_backbone_empirical_erm",
        policies[empirical_index] if empirical_index is not None else None,
        evaluation,
        False,
    )

    frozen_mix = frozen_selection_mixture(policies, selection, randomization_seed)
    if frozen_mix is not None and candidate_pvalue(frozen_mix, calibration)[0] <= DELTA:
        add_result(rows, "cros_mix_frozen", frozen_mix, evaluation, True)
    else:
        add_result(rows, "cros_mix_frozen", None, evaluation, True)

    screened = screen(policies, selection, cap=12)
    endpoint_delta = DELTA / (2 * len(screened))
    certified = []
    for index in screened:
        risk_upper, coverage_lower = deterministic_bounds(
            policies[index], calibration, endpoint_delta
        )
        if risk_upper <= ALPHA and coverage_lower >= GAMMA:
            certified.append(index)
    det_index = cheapest(policies, certified, selection)
    add_result(
        rows,
        "cros_det",
        policies[det_index] if det_index is not None else None,
        evaluation,
        True,
    )
    for row in rows:
        row["joint_transfer"] = bool(
            row.get("available", False)
            and row.get("selective_risk", 1.0) <= ALPHA
            and row.get("autonomous_coverage", 0.0) >= GAMMA
        )
        row["screened_candidate_n"] = len(screened)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="cros_mix_validation/results/la_cdm_common_backbone_dev_v1"
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="cros_mix_validation/data/derived/ed_cds_action_v1/manifest.json",
    )
    parser.add_argument("--expected-n", type=int)
    parser.add_argument(
        "--model-label",
        default="Qwen2.5-7B-Instruct zero-shot hypothesis and decision agents",
    )
    parser.add_argument("--split-description", default="development only")
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=20260828)
    args = parser.parse_args()

    root = Path(args.root)
    manifest = json.loads(Path(args.benchmark_manifest).read_text(encoding="utf-8"))
    expected_n = (
        args.expected_n
        if args.expected_n is not None
        else int(manifest["split_counts"]["development"])
    )
    data = load_shards(root / "shards", expected_n)
    policies = make_policies(data)
    labels = data["labels"]
    trial_rows: list[dict[str, object]] = []
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        groups = split_three_way(labels, seed)
        for rotation in range(3):
            selection = groups[rotation]
            calibration = groups[(rotation + 1) % 3]
            evaluation = groups[(rotation + 2) % 3]
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
    trials = pd.DataFrame(trial_rows)
    trials.to_csv(root / "common_backbone_trials.csv", index=False)
    availability = trials.groupby("method", as_index=False).agg(
        available_trials=("available", "sum")
    )
    available = trials[trials["available"]]
    summary = available.groupby("method", as_index=False).agg(
        mean_selective_risk=("selective_risk", "mean"),
        sd_selective_risk=("selective_risk", "std"),
        mean_coverage=("autonomous_coverage", "mean"),
        mean_test_cost=("mean_test_cost", "mean"),
        mean_total_cost=("mean_total_cost", "mean"),
        mean_latency=("mean_latency", "mean"),
        mean_tests=("mean_tests", "mean"),
        joint_transfer_rate=("joint_transfer", "mean"),
    )
    summary = availability.merge(summary, on="method", how="left")
    summary["availability_rate"] = summary["available_trials"] / (args.num_seeds * 3)
    summary = summary.sort_values(
        ["availability_rate", "mean_total_cost"],
        ascending=[False, True],
        na_position="last",
    )
    summary.to_csv(root / "common_backbone_summary.csv", index=False)
    native_stage = data["native_stop_stage"].astype(int)
    episode = np.arange(len(labels))
    native_predictions = data["probabilities"][episode, native_stage].argmax(axis=1)
    action_audit = {
        "unique_forced_action_paths": int(
            len({tuple(row.tolist()) for row in data["actions"]})
        ),
        "native_mean_stop_stage": float(native_stage.mean()),
        "native_forced_error": float(np.mean(native_predictions != labels)),
        "forced_path_full_workup_error": float(
            np.mean(data["probabilities"][:, -1].argmax(axis=1) != labels)
        ),
        "forced_path_mean_missing_fraction": float(
            np.mean(data["cumulative_missing"][:, -1] / (data["actions"].shape[1]))
        ),
    }
    metadata = {
        "status": "LA_CDM_COMMON_BACKBONE_DEVELOPMENT_COMPLETE",
        "confirmatory": False,
        "split": args.split_description,
        "locked_calibration_or_test_loaded": False,
        "model": args.model_label,
        "policy_n": len(policies),
        "thresholds": list(THRESHOLDS),
        "seeds": args.num_seeds,
        "rotations_per_seed": 3,
        "targets": {"selective_risk_max": ALPHA, "coverage_min": GAMMA},
        "action_audit": action_audit,
    }
    (root / "common_backbone_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    for path in root.glob("*.csv"):
        os.chmod(path, 0o600)
    os.chmod(root / "common_backbone_metadata.json", 0o600)
    print(summary.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
