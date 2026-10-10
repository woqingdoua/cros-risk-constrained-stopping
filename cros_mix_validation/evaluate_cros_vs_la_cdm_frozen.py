#!/usr/bin/env python3
"""Fair same-backbone comparison of frozen CROS v2 and LA-CDM controls.

All methods reuse the same trained adapter, diagnosis probabilities, proposed
action path, missing-result semantics, and relative resource-cost table.  The
previously opened 367-episode test split makes this an exploratory paired
comparison, not a new confirmatory certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluate_cros_power_v2 import (
    ALPHA,
    GAMMA,
    SEED,
    CandidateSpec,
    apply_mix,
    make_policies as make_cros_policies,
    predict_risk,
    selective_stats,
)
from evaluate_la_cdm_common_backbone_dev import (
    load_shards,
    make_policies as make_baseline_policies,
)
from run_ed_cds_selective_sequential_pilot import DELTA, candidate_pvalue


TIME_SENSITIVE_PROXY_LABELS = {
    "appendicitis",
    "biliary_disease",
    "bowel_obstruction",
    "diverticulitis",
    "pancreatitis",
    "renal_colic",
    "urinary_infection",
}
BOOTSTRAP_REPLICATES = 10_000
RANDOMIZATION_REPLICATES = 1_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_map(policies: list) -> dict[str, object]:
    return {policy.name: policy for policy in policies}


def empirical_feasible(policy, indices: np.ndarray) -> bool:
    stats = selective_stats(policy, indices)
    return bool(
        stats["selective_risk"] <= ALPHA
        and stats["autonomous_coverage"] >= GAMMA
    )


def select_feasible(policies: list, indices: np.ndarray):
    feasible = [policy for policy in policies if empirical_feasible(policy, indices)]
    if feasible:
        return min(
            feasible,
            key=lambda policy: (
                policy.total_costs[indices].mean(),
                -policy.autonomous[indices].mean(),
                policy.name,
            ),
        ), True
    return min(
        policies,
        key=lambda policy: (
            max(0.0, selective_stats(policy, indices)["selective_risk"] - ALPHA)
            + max(
                0.0,
                GAMMA - selective_stats(policy, indices)["autonomous_coverage"],
            ),
            policy.total_costs[indices].mean(),
            policy.name,
        ),
    ), False


def extended_stats(policy, labels: np.ndarray, label_names: list[str]) -> dict[str, object]:
    indices = np.arange(len(labels))
    stats = selective_stats(policy, indices)
    serious = np.isin(np.asarray(label_names, dtype=object)[labels], list(TIME_SENSITIVE_PROXY_LABELS))
    serious_n = int(serious.sum())
    return {
        **stats,
        "n": int(len(labels)),
        "autonomous_n": int(policy.autonomous.sum()),
        "autonomous_error_n": int((policy.autonomous & policy.errors).sum()),
        "time_sensitive_proxy_n": serious_n,
        "time_sensitive_wrong_autonomous_n": int(
            (serious & policy.autonomous & policy.errors).sum()
        ),
        "time_sensitive_proxy_miss_rate": float(
            np.mean(policy.autonomous[serious] & policy.errors[serious])
        ),
        "time_sensitive_proxy_defer_rate": float(np.mean(~policy.autonomous[serious])),
        "constraints_pass_descriptive": bool(
            stats["selective_risk"] <= ALPHA
            and stats["autonomous_coverage"] >= GAMMA
        ),
    }


def bootstrap_metrics(
    policy,
    labels: np.ndarray,
    label_names: list[str],
    draws: np.ndarray,
) -> dict[str, np.ndarray]:
    autonomous = policy.autonomous.astype(float)
    error_mass = (policy.autonomous & policy.errors).astype(float)
    serious = np.isin(
        np.asarray(label_names, dtype=object)[labels],
        list(TIME_SENSITIVE_PROXY_LABELS),
    ).astype(float)
    selected = autonomous[draws].sum(axis=1)
    errors = error_mass[draws].sum(axis=1)
    serious_denominator = serious[draws].sum(axis=1)
    serious_misses = (error_mass * serious)[draws].sum(axis=1)
    return {
        "selective_risk": errors / selected,
        "coverage": autonomous[draws].mean(axis=1),
        "error_mass": error_mass[draws].mean(axis=1),
        "total_cost": policy.total_costs[draws].mean(axis=1),
        "tests": policy.tests[draws].mean(axis=1),
        "time_sensitive_proxy_miss_rate": serious_misses / serious_denominator,
    }


def interval(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(values.mean()),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def expected_mixture_stats(policies: list, support: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    coverage = sum(weights[position] * policies[index].autonomous for position, index in enumerate(support))
    error_mass = sum(
        weights[position] * (policies[index].autonomous & policies[index].errors)
        for position, index in enumerate(support)
    )
    return {
        "selective_risk": float(error_mass.mean() / coverage.mean()),
        "autonomous_coverage": float(coverage.mean()),
        "autonomous_error_mass": float(error_mass.mean()),
        "mean_test_cost": float(
            sum(
                weights[position] * policies[index].test_costs.mean()
                for position, index in enumerate(support)
            )
        ),
        "mean_total_cost": float(
            sum(
                weights[position] * policies[index].total_costs.mean()
                for position, index in enumerate(support)
            )
        ),
        "mean_latency": float(
            sum(
                weights[position] * policies[index].latencies.mean()
                for position, index in enumerate(support)
            )
        ),
        "mean_tests": float(
            sum(
                weights[position] * policies[index].tests.mean()
                for position, index in enumerate(support)
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cros-root", default="cros_mix_validation/results/la_cdm_cros_power_v2"
    )
    parser.add_argument(
        "--selection-root",
        default=(
            "cros_mix_validation/results/"
            "la_cdm_trained_common_backbone_validation_v1/shards"
        ),
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="cros_mix_validation/data/derived/ed_cds_action_v1/manifest.json",
    )
    parser.add_argument(
        "--output-root",
        default="cros_mix_validation/results/cros_vs_la_cdm_frozen_v1",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--randomization-replicates", type=int, default=RANDOMIZATION_REPLICATES
    )
    args = parser.parse_args()

    cros_root = Path(args.cros_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = json.loads(Path(args.benchmark_manifest).read_text(encoding="utf-8"))
    label_names = list(manifest["labels"])
    selection = load_shards(Path(args.selection_root), 165)
    calibration = load_shards(cros_root / "locked_calibration" / "shards", 367)
    test = load_shards(cros_root / "locked_test" / "shards", 367)
    split_rows = [
        set(selection["row_indices"]),
        set(calibration["row_indices"]),
        set(test["row_indices"]),
    ]
    if any(
        split_rows[left] & split_rows[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise AssertionError("selection/calibration/test episode overlap")

    freeze_path = cros_root / "cros_power_v2_freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    ranker_path = cros_root / "cros_power_v2_ranker.joblib"
    specs_path = cros_root / "cros_power_v2_candidate_specs.json"
    result_path = cros_root / "cros_power_v2_results.csv"
    frozen_paths = {
        "ranker": ranker_path,
        "candidate_specs": specs_path,
        "fixed_order": cros_root / "cros_power_v2_order.json",
        "results": result_path,
        "metadata": cros_root / "cros_power_v2_metadata.json",
        "evaluator": Path("cros_mix_validation/evaluate_cros_power_v2.py"),
    }
    for name, path in frozen_paths.items():
        if sha256(path) != freeze["artifacts_sha256"][name]:
            raise AssertionError(f"frozen artifact changed: {name}")

    ranker = joblib.load(ranker_path)
    specs = [
        CandidateSpec(**payload)
        for payload in json.loads(specs_path.read_text(encoding="utf-8"))
    ]
    cros_selection = make_cros_policies(
        selection, predict_risk(ranker, selection), specs
    )
    cros_calibration = make_cros_policies(
        calibration, predict_risk(ranker, calibration), specs
    )
    cros_test = make_cros_policies(test, predict_risk(ranker, test), specs)
    cros_selection_by_name = policy_map(cros_selection)
    cros_calibration_by_name = policy_map(cros_calibration)
    cros_test_by_name = policy_map(cros_test)
    frozen_results = pd.read_csv(result_path)
    frozen_locked = frozen_results[
        frozen_results["scenario"]
        == "locked_calibration_367_to_locked_test_367_exploratory"
    ]

    selected: dict[str, tuple[object, object, object, str, bool, str]] = {}
    for method, display, procedure in (
        ("cros_risk_single", "CROS-SINGLE", "single_pre_frozen"),
        ("cros_risk_ltt", "CROS-LTT", "fixed_sequence_ltt"),
        ("cros_risk_bonferroni", "CROS-BONFERRONI", "bonferroni_12"),
    ):
        policy_name = frozen_locked.loc[
            frozen_locked["method"] == method, "selected_policy"
        ].iloc[0]
        selected[display] = (
            cros_selection_by_name[policy_name],
            cros_calibration_by_name[policy_name],
            cros_test_by_name[policy_name],
            "frozen_cros",
            True,
            procedure,
        )

    cros_names = [policy.name for policy in cros_selection]
    support = np.asarray(
        [cros_names.index(item["policy"]) for item in freeze["mix_support"]],
        dtype=int,
    )
    weights = np.asarray([item["weight"] for item in freeze["mix_support"]])
    mix_selection = apply_mix(cros_selection, support, weights, SEED + 1)
    mix_calibration = apply_mix(cros_calibration, support, weights, SEED + 2)
    mix_test = apply_mix(cros_test, support, weights, SEED + 3)
    selected["CROS-MIX"] = (
        mix_selection,
        mix_calibration,
        mix_test,
        "frozen_cros_randomized",
        True,
        "single_pre_frozen_randomized",
    )

    baseline_selection = make_baseline_policies(selection)
    baseline_calibration = make_baseline_policies(calibration)
    baseline_test = make_baseline_policies(test)
    baseline_calibration_by_name = policy_map(baseline_calibration)
    baseline_test_by_name = policy_map(baseline_test)
    baseline_selection_by_name = policy_map(baseline_selection)

    for display, name in (
        ("HPI-ONLY", "forced_common_path:h=0"),
        ("FULL-WORKUP", "forced_common_path:h=12"),
        ("LA-CDM-NATIVE", "la_native_forced"),
    ):
        selected[display] = (
            baseline_selection_by_name[name],
            baseline_calibration_by_name[name],
            baseline_test_by_name[name],
            "same_backbone_baseline",
            False,
            "single_pre_frozen_baseline",
        )

    fixed_candidates = [
        policy for policy in baseline_selection if policy.family == "forced_horizon"
    ]
    best_fixed = min(
        fixed_candidates,
        key=lambda policy: (
            selective_stats(policy, np.arange(165))["selective_risk"],
            policy.total_costs.mean(),
            policy.name,
        ),
    )
    selected["BEST-FIXED-STAGE"] = (
        best_fixed,
        baseline_calibration_by_name[best_fixed.name],
        baseline_test_by_name[best_fixed.name],
        "selection_chosen_baseline",
        False,
        "single_selection_frozen_baseline",
    )

    confidence_candidates = [
        policy
        for policy in baseline_selection
        if policy.family == "common_path_selective" and ":h=12:" in policy.name
    ]
    confidence, confidence_feasible = select_feasible(
        confidence_candidates, np.arange(165)
    )
    selected["CONFIDENCE-THRESHOLD"] = (
        confidence,
        baseline_calibration_by_name[confidence.name],
        baseline_test_by_name[confidence.name],
        "selection_chosen_baseline",
        False,
        "single_selection_frozen_baseline",
    )

    native_defer_candidates = [
        policy for policy in baseline_selection if policy.family == "la_native_defer"
    ]
    native_defer, native_defer_feasible = select_feasible(
        native_defer_candidates, np.arange(165)
    )
    selected["LA-CDM-NATIVE-DEFER"] = (
        native_defer,
        baseline_calibration_by_name[native_defer.name],
        baseline_test_by_name[native_defer.name],
        "selection_chosen_baseline",
        False,
        "single_selection_frozen_baseline",
    )

    empirical_candidates = [
        policy
        for policy in baseline_selection
        if policy.family in {"la_native_defer", "common_path_selective", "forced_horizon"}
    ]
    empirical, empirical_feasible_flag = select_feasible(
        empirical_candidates, np.arange(165)
    )
    selected["EMPIRICAL-ERM"] = (
        empirical,
        baseline_calibration_by_name[empirical.name],
        baseline_test_by_name[empirical.name],
        "selection_chosen_baseline",
        False,
        "single_selection_frozen_baseline",
    )

    summary_rows = []
    per_label_rows = []
    for display, (
        selection_policy,
        calibration_policy,
        test_policy,
        family,
        cros_certified,
        procedure,
    ) in selected.items():
        selection_stats = extended_stats(
            selection_policy, selection["labels"], label_names
        )
        calibration_stats = extended_stats(
            calibration_policy, calibration["labels"], label_names
        )
        test_stats = extended_stats(test_policy, test["labels"], label_names)
        joint_pvalue, risk_pvalue, coverage_pvalue = candidate_pvalue(
            calibration_policy, np.arange(len(calibration["labels"]))
        )
        certified = bool(cros_certified or joint_pvalue <= DELTA)
        summary_rows.append(
            {
                "method": display,
                "family": family,
                "policy": test_policy.name,
                "calibration_certified_exploratory": certified,
                "calibration_procedure": procedure,
                "calibration_joint_pvalue": joint_pvalue,
                "calibration_risk_pvalue": risk_pvalue,
                "calibration_coverage_pvalue": coverage_pvalue,
                **{f"selection_{key}": value for key, value in selection_stats.items()},
                **{f"calibration_{key}": value for key, value in calibration_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
            }
        )
        for label_index, label in enumerate(label_names):
            mask = test["labels"] == label_index
            autonomous = test_policy.autonomous[mask]
            errors = test_policy.errors[mask]
            per_label_rows.append(
                {
                    "method": display,
                    "label": label,
                    "n": int(mask.sum()),
                    "autonomous_n": int(autonomous.sum()),
                    "coverage": float(autonomous.mean()),
                    "selective_risk": float(errors[autonomous].mean())
                    if autonomous.any()
                    else np.nan,
                    "wrong_autonomous_n": int((autonomous & errors).sum()),
                    "wrong_autonomous_mass": float((autonomous & errors).mean()),
                    "mean_total_cost": float(test_policy.total_costs[mask].mean()),
                }
            )

    summary = pd.DataFrame(summary_rows)
    per_label = pd.DataFrame(per_label_rows)
    summary.to_csv(output_root / "summary.csv", index=False)
    per_label.to_csv(output_root / "per_label.csv", index=False)

    rng = np.random.default_rng(SEED + 200)
    draws = rng.integers(0, len(test["labels"]), size=(args.bootstrap_replicates, len(test["labels"])))
    bootstrapped = {
        display: bootstrap_metrics(policy_tuple[2], test["labels"], label_names, draws)
        for display, policy_tuple in selected.items()
    }
    reference = bootstrapped["CROS-MIX"]
    bootstrap_rows = []
    for comparator, values in bootstrapped.items():
        if comparator == "CROS-MIX":
            continue
        for metric in reference:
            difference = reference[metric] - values[metric]
            mean, lower, upper = interval(difference)
            bootstrap_rows.append(
                {
                    "reference": "CROS-MIX",
                    "comparator": comparator,
                    "metric": metric,
                    "difference_reference_minus_comparator": mean,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "ci_excludes_zero": bool(lower > 0 or upper < 0),
                    "replicates": args.bootstrap_replicates,
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output_root / "paired_bootstrap.csv", index=False)

    stability_rows = []
    test_indices = np.arange(len(test["labels"]))
    for replicate in range(args.randomization_replicates):
        policy = apply_mix(cros_test, support, weights, SEED + 10_000 + replicate)
        stats = extended_stats(policy, test["labels"], label_names)
        stability_rows.append({"replicate": replicate, **stats})
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output_root / "mix_randomization_stability.csv", index=False)
    expected_mix = expected_mixture_stats(cros_test, support, weights)
    key_paired_comparisons = {
        comparator: bootstrap[bootstrap["comparator"] == comparator][
            [
                "metric",
                "difference_reference_minus_comparator",
                "ci95_lower",
                "ci95_upper",
                "ci_excludes_zero",
            ]
        ].to_dict(orient="records")
        for comparator in ("LA-CDM-NATIVE", "LA-CDM-NATIVE-DEFER", "EMPIRICAL-ERM")
    }
    mix_labels = per_label[per_label["method"] == "CROS-MIX"]
    subgroup_violations = mix_labels[
        (mix_labels["coverage"] < GAMMA)
        | (mix_labels["selective_risk"] > ALPHA)
    ][["label", "n", "coverage", "selective_risk"]].to_dict(orient="records")

    metadata = {
        "status": "CROS_VS_LA_CDM_FROZEN_EXPLORATORY_COMPLETE",
        "confirmatory": False,
        "reason_not_confirmatory": "the 367-episode test labels were previously opened",
        "same_backbone": True,
        "same_action_path": True,
        "same_cost_mapping": True,
        "calibration_n": 367,
        "test_n": 367,
        "targets": {"alpha": ALPHA, "gamma": GAMMA},
        "calibration_delta": DELTA,
        "bootstrap_replicates": args.bootstrap_replicates,
        "randomization_replicates": args.randomization_replicates,
        "time_sensitive_proxy_labels": sorted(TIME_SENSITIVE_PROXY_LABELS),
        "time_sensitive_proxy_warning": (
            "wrong autonomous diagnosis among these labels is an analytic proxy, "
            "not clinician-adjudicated major-disease omission"
        ),
        "baseline_selection": {
            "best_fixed_stage": best_fixed.name,
            "confidence_threshold": confidence.name,
            "confidence_selection_empirically_feasible": confidence_feasible,
            "la_native_defer": native_defer.name,
            "la_native_defer_selection_empirically_feasible": native_defer_feasible,
            "empirical_erm": empirical.name,
            "empirical_erm_selection_feasible": empirical_feasible_flag,
        },
        "cros_mix_expected_test_metrics": expected_mix,
        "cros_mix_randomization_constraint_pass_fraction": float(
            stability["constraints_pass_descriptive"].mean()
        ),
        "key_paired_bootstrap_comparisons": key_paired_comparisons,
        "cros_mix_diagnosis_subgroup_constraint_violations": subgroup_violations,
        "subgroup_interpretation": (
            "global certification does not imply simultaneous diagnosis-subgroup guarantees"
        ),
        "frozen_manifest_sha256": sha256(freeze_path),
        "inputs_sha256": {
            "cros_results": sha256(result_path),
            "benchmark_manifest": sha256(Path(args.benchmark_manifest)),
        },
        "output_artifacts_sha256": {
            "summary": sha256(output_root / "summary.csv"),
            "paired_bootstrap": sha256(output_root / "paired_bootstrap.csv"),
            "per_label": sha256(output_root / "per_label.csv"),
            "mix_randomization_stability": sha256(
                output_root / "mix_randomization_stability.csv"
            ),
            "evaluator": sha256(Path(__file__)),
        },
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    display_columns = [
        "method",
        "calibration_certified_exploratory",
        "calibration_joint_pvalue",
        "test_selective_risk",
        "test_autonomous_coverage",
        "test_mean_total_cost",
        "test_mean_tests",
        "test_time_sensitive_proxy_miss_rate",
    ]
    report = [
        "# Frozen CROS v2 vs LA-CDM same-backbone comparison",
        "",
        "Exploratory only: the test labels were previously opened. All methods share the trained adapter, diagnostic probabilities, proposed action path, missing-result semantics and relative resource costs.",
        "",
        markdown_table(summary[display_columns]),
        "",
        "## Randomized-mixture stability",
        "",
        f"- Replicates: {args.randomization_replicates}",
        f"- Constraint-pass fraction: {metadata['cros_mix_randomization_constraint_pass_fraction']:.4f}",
        f"- Analytic expected metrics: `{json.dumps(expected_mix, sort_keys=True)}`",
        "",
        "## Diagnosis-subgroup warning",
        "",
        f"Global constraints do not hold in every diagnosis subgroup: `{json.dumps(subgroup_violations, sort_keys=True)}`",
        "",
        "The time-sensitive miss metric is a label-based analytic proxy and has not been clinician adjudicated.",
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for path in output_root.iterdir():
        os.chmod(path, 0o600)
    print(summary[display_columns].to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
