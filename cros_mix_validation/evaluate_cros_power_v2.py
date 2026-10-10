#!/usr/bin/env python3
"""Power-aware CROS repair for the trained LA-CDM common backbone.

The adapter-training split trains an auxiliary state-wise error ranker.  The
165 model-training-independent development-validation episodes freeze the
candidate thresholds and fixed testing order.  Previously opened locked
calibration/test splits are then used only for an explicitly exploratory
calibration/transfer audit.  Alpha and minimum coverage remain fixed at
0.25 and 0.70.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import binom
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from evaluate_la_cdm_common_backbone_dev import load_shards
from run_ed_cds_selective_sequential_pilot import (
    ALPHA,
    DELTA,
    GAMMA,
    DEFER_COST,
    SelectivePolicy,
    candidate_pvalue,
    selective_stats,
)


SEED = 20260902
TARGET_COVERAGES = (0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.93, 0.95)
ORDER_CAP = 12
# Design margins used only to fit the randomized policy on the selection split.
# Certification still uses the unchanged ALPHA/GAMMA/DELTA hypotheses.
MIX_TARGET_RISK = ALPHA - 0.05
MIX_TARGET_COVERAGE = GAMMA + 0.10


@dataclass(frozen=True)
class CandidateSpec:
    horizon: int
    threshold: float

    @property
    def name(self) -> str:
        return f"risk_rank:h={self.horizon}:tau={self.threshold:.10f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_tensor(data: dict[str, np.ndarray]) -> np.ndarray:
    probabilities = np.clip(data["probabilities"].astype(float), 1e-8, 1.0)
    n, stages, classes = probabilities.shape
    ordered = np.sort(probabilities, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    entropy = -np.sum(probabilities * np.log(probabilities), axis=2) / np.log(classes)
    stage = np.broadcast_to(
        np.arange(stages, dtype=float)[None, :] / max(1, stages - 1), (n, stages)
    )
    missing = data["cumulative_missing"].astype(float) / np.maximum(
        1.0, np.arange(stages, dtype=float)[None, :]
    )
    costs = data["cumulative_costs"].astype(float) / 51.9
    latencies = data["cumulative_latencies"].astype(float) / 1890.0
    stop_scores = data["stop_scores"].astype(float)
    return np.concatenate(
        (
            probabilities,
            probabilities.max(axis=2, keepdims=True),
            margin[:, :, None],
            entropy[:, :, None],
            stage[:, :, None],
            missing[:, :, None],
            costs[:, :, None],
            latencies[:, :, None],
            stop_scores[:, :, None],
        ),
        axis=2,
    )


def error_matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    predictions = data["probabilities"].argmax(axis=2)
    return predictions != data["labels"][:, None]


def new_ranker(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=seed,
    )


def fit_ranker(
    train: dict[str, np.ndarray], seed: int
) -> tuple[HistGradientBoostingClassifier, np.ndarray, dict[str, float]]:
    features = feature_tensor(train)
    targets = error_matrix(train)
    n, stages, width = features.shape
    oof = np.zeros((n, stages), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (fit_indices, holdout_indices) in enumerate(
        splitter.split(np.arange(n), train["labels"])
    ):
        model = new_ranker(seed + fold)
        model.fit(
            features[fit_indices].reshape(-1, width),
            targets[fit_indices].reshape(-1).astype(int),
        )
        oof[holdout_indices] = model.predict_proba(
            features[holdout_indices].reshape(-1, width)
        )[:, 1].reshape(len(holdout_indices), stages)
    final_model = new_ranker(seed + 100)
    final_model.fit(features.reshape(-1, width), targets.reshape(-1).astype(int))
    flat_target = targets.reshape(-1).astype(int)
    flat_oof = oof.reshape(-1)
    diagnostics = {
        "episode_n": n,
        "state_n": int(n * stages),
        "state_error_rate": float(flat_target.mean()),
        "oof_auc": float(roc_auc_score(flat_target, flat_oof)),
        "oof_brier": float(brier_score_loss(flat_target, flat_oof)),
    }
    return final_model, oof, diagnostics


def predict_risk(model: HistGradientBoostingClassifier, data: dict[str, np.ndarray]) -> np.ndarray:
    features = feature_tensor(data)
    n, stages, width = features.shape
    return model.predict_proba(features.reshape(-1, width))[:, 1].reshape(n, stages)


def freeze_specs(selection_scores: np.ndarray) -> list[CandidateSpec]:
    _, stages = selection_scores.shape
    specs: dict[tuple[int, float], CandidateSpec] = {}
    for horizon in range(stages):
        minimum = selection_scores[:, : horizon + 1].min(axis=1)
        for target in TARGET_COVERAGES:
            threshold = float(np.quantile(minimum, target, method="higher"))
            key = (horizon, round(threshold, 10))
            specs[key] = CandidateSpec(horizon=horizon, threshold=threshold)
    return list(specs.values())


def make_policy(
    data: dict[str, np.ndarray], scores: np.ndarray, spec: CandidateSpec
) -> SelectivePolicy:
    n = len(data["labels"])
    chosen = np.full(n, spec.horizon, dtype=int)
    autonomous = np.zeros(n, dtype=bool)
    for stage in range(spec.horizon + 1):
        stop = (~autonomous) & (scores[:, stage] <= spec.threshold)
        chosen[stop] = stage
        autonomous[stop] = True
    episode = np.arange(n)
    predictions = data["probabilities"][episode, chosen].argmax(axis=1)
    test_costs = data["cumulative_costs"][episode, chosen].astype(float)
    return SelectivePolicy(
        name=spec.name,
        family="learned_error_ranker",
        errors=predictions != data["labels"],
        autonomous=autonomous,
        test_costs=test_costs,
        total_costs=test_costs + (~autonomous).astype(float) * DEFER_COST,
        latencies=data["cumulative_latencies"][episode, chosen].astype(float),
        tests=chosen.astype(float),
    )


def make_policies(
    data: dict[str, np.ndarray], scores: np.ndarray, specs: list[CandidateSpec]
) -> list[SelectivePolicy]:
    return [make_policy(data, scores, spec) for spec in specs]


def projected_pvalue(policy: SelectivePolicy, selection: np.ndarray, n: int) -> float:
    stats = selective_stats(policy, selection)
    selected = int(round(stats["autonomous_coverage"] * n))
    errors = int(round(stats["selective_risk"] * selected))
    risk_p = float(binom.cdf(errors, selected, ALPHA)) if selected else 1.0
    coverage_p = float(binom.sf(selected - 1, n, GAMMA))
    return max(risk_p, coverage_p)


def fixed_order(
    policies: list[SelectivePolicy], selection: np.ndarray, calibration_n: int
) -> list[int]:
    return sorted(
        range(len(policies)),
        key=lambda index: (
            projected_pvalue(policies[index], selection, calibration_n),
            policies[index].total_costs[selection].mean(),
            policies[index].name,
        ),
    )[:ORDER_CAP]


def cheapest(
    policies: list[SelectivePolicy], indices: list[int], selection: np.ndarray
) -> int | None:
    if not indices:
        return None
    return min(
        indices,
        key=lambda index: (
            policies[index].total_costs[selection].mean(),
            -policies[index].autonomous[selection].mean(),
            policies[index].name,
        ),
    )


def fit_mix(
    policies: list[SelectivePolicy], selection: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    coverage = np.asarray([p.autonomous[selection].mean() for p in policies])
    error_mass = np.asarray(
        [np.mean(p.errors[selection] & p.autonomous[selection]) for p in policies]
    )
    cost = np.asarray([p.total_costs[selection].mean() for p in policies])
    solution = linprog(
        c=cost,
        A_ub=np.vstack(
            (error_mass - MIX_TARGET_RISK * coverage, -coverage)
        ),
        b_ub=np.asarray((0.0, -MIX_TARGET_COVERAGE)),
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
    return support, weights[support]


def apply_mix(
    policies: list[SelectivePolicy], support: np.ndarray, weights: np.ndarray, seed: int
) -> SelectivePolicy:
    rng = np.random.default_rng(seed)
    assignment_position = np.searchsorted(
        np.cumsum(weights), rng.random(len(policies[0].errors)), side="right"
    )
    assignment_position = np.minimum(assignment_position, len(support) - 1)
    assignment = support[assignment_position]
    episode = np.arange(len(assignment))

    def realized(attribute: str) -> np.ndarray:
        matrix = np.stack([getattr(p, attribute) for p in policies])
        return matrix[assignment, episode]

    return SelectivePolicy(
        name="risk_rank_mix:" + ";".join(
            f"{policies[index].name}@{weight:.8f}"
            for index, weight in zip(support, weights)
        ),
        family="learned_error_ranker_mix",
        errors=realized("errors").astype(bool),
        autonomous=realized("autonomous").astype(bool),
        test_costs=realized("test_costs").astype(float),
        total_costs=realized("total_costs").astype(float),
        latencies=realized("latencies").astype(float),
        tests=realized("tests").astype(float),
        support_size=len(support),
    )


def concatenate_policies(
    first: list[SelectivePolicy], second: list[SelectivePolicy]
) -> list[SelectivePolicy]:
    if [p.name for p in first] != [p.name for p in second]:
        raise AssertionError("policy order mismatch")
    output = []
    for left, right in zip(first, second):
        output.append(
            SelectivePolicy(
                name=left.name,
                family=left.family,
                errors=np.concatenate((left.errors, right.errors)),
                autonomous=np.concatenate((left.autonomous, right.autonomous)),
                test_costs=np.concatenate((left.test_costs, right.test_costs)),
                total_costs=np.concatenate((left.total_costs, right.total_costs)),
                latencies=np.concatenate((left.latencies, right.latencies)),
                tests=np.concatenate((left.tests, right.tests)),
            )
        )
    return output


def validate_and_repair_shard_metadata(root: Path) -> list[dict[str, object]]:
    """Correct only legacy provenance fields; never alter numerical archives."""
    corrections: list[dict[str, object]] = []
    split_roots = {
        "development": root / "train" / "shards",
        "locked_calibration": root / "locked_calibration" / "shards",
        "locked_test": root / "locked_test" / "shards",
    }
    for split, shard_root in split_roots.items():
        paths = sorted(shard_root.glob("shard_*.json"))
        if len(paths) != 11:
            raise AssertionError(f"expected 11 {split} metadata shards, got {len(paths)}")
        expected_status = f"LA_CDM_COMMON_BACKBONE_{split.upper()}_SHARD_COMPLETE"
        expected_locked = split in {"locked_calibration", "locked_test"}
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("split") != split:
                raise AssertionError(f"split mismatch in {path}")
            old_status = payload.get("status")
            old_locked = payload.get("locked_calibration_or_test_loaded")
            if old_status == expected_status and old_locked is expected_locked:
                if "metadata_correction" in payload:
                    corrections.append(
                        {
                            "path": str(path),
                            "previously_corrected": True,
                            **payload["metadata_correction"],
                        }
                    )
                continue
            payload["status"] = expected_status
            payload["locked_calibration_or_test_loaded"] = expected_locked
            payload["metadata_correction"] = {
                "reason": "legacy generator hard-coded development provenance fields",
                "numerical_npz_modified": False,
            }
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.chmod(path, 0o600)
            corrections.append(
                {
                    "path": str(path),
                    "old_status": old_status,
                    "old_locked_flag": old_locked,
                    "new_status": expected_status,
                    "new_locked_flag": expected_locked,
                }
            )
    return corrections


def result_row(
    scenario: str,
    method: str,
    selected: int | None,
    selection_policies: list[SelectivePolicy],
    calibration_policies: list[SelectivePolicy],
    evaluation_policies: list[SelectivePolicy] | None,
    calibration_indices: np.ndarray,
    evaluation_indices: np.ndarray | None,
) -> dict[str, object]:
    if selected is None:
        return {"scenario": scenario, "method": method, "certified": False}
    calibration_policy = calibration_policies[selected]
    joint_p, risk_p, coverage_p = candidate_pvalue(
        calibration_policy, calibration_indices
    )
    row: dict[str, object] = {
        "scenario": scenario,
        "method": method,
        "certified": True,
        "selected_policy": calibration_policy.name,
        "joint_pvalue": joint_p,
        "risk_pvalue": risk_p,
        "coverage_pvalue": coverage_p,
        **{f"cal_{k}": v for k, v in selective_stats(calibration_policy, calibration_indices).items()},
        "selection_mean_total_cost": float(
            selection_policies[selected].total_costs.mean()
        ),
        "cal_n": int(len(calibration_indices)),
        "cal_autonomous_n": int(calibration_policy.autonomous[calibration_indices].sum()),
        "cal_autonomous_error_n": int(
            (
                calibration_policy.errors[calibration_indices]
                & calibration_policy.autonomous[calibration_indices]
            ).sum()
        ),
    }
    if evaluation_policies is not None and evaluation_indices is not None:
        row.update(
            {
                f"eval_{k}": v
                for k, v in selective_stats(
                    evaluation_policies[selected], evaluation_indices
                ).items()
            }
        )
        row["eval_n"] = int(len(evaluation_indices))
        row["eval_autonomous_n"] = int(
            evaluation_policies[selected].autonomous[evaluation_indices].sum()
        )
        row["eval_autonomous_error_n"] = int(
            (
                evaluation_policies[selected].errors[evaluation_indices]
                & evaluation_policies[selected].autonomous[evaluation_indices]
            ).sum()
        )
        row["eval_joint_transfer"] = bool(
            row["eval_selective_risk"] <= ALPHA
            and row["eval_autonomous_coverage"] >= GAMMA
        )
    return row


def certify_scenario(
    scenario: str,
    selection_policies: list[SelectivePolicy],
    calibration_policies: list[SelectivePolicy],
    evaluation_policies: list[SelectivePolicy] | None,
    mix_selection: SelectivePolicy | None,
    mix_calibration: SelectivePolicy | None,
    mix_evaluation: SelectivePolicy | None,
) -> tuple[list[dict[str, object]], list[int]]:
    selection = np.arange(len(selection_policies[0].errors))
    calibration = np.arange(len(calibration_policies[0].errors))
    evaluation = (
        np.arange(len(evaluation_policies[0].errors))
        if evaluation_policies is not None
        else None
    )
    order = fixed_order(selection_policies, selection, len(calibration))
    pvalues = np.asarray(
        [candidate_pvalue(calibration_policies[index], calibration)[0] for index in order]
    )
    rows: list[dict[str, object]] = []

    single = order[0] if pvalues[0] <= DELTA else None
    rows.append(
        result_row(
            scenario, "cros_risk_single", single, selection_policies,
            calibration_policies, evaluation_policies, calibration, evaluation,
        )
    )

    ltt_rejected: list[int] = []
    for index, pvalue in zip(order, pvalues):
        if pvalue <= DELTA:
            ltt_rejected.append(index)
        else:
            break
    rows.append(
        result_row(
            scenario, "cros_risk_ltt", cheapest(selection_policies, ltt_rejected, selection),
            selection_policies, calibration_policies, evaluation_policies,
            calibration, evaluation,
        )
    )

    sorted_positions = np.argsort(pvalues)
    holm: list[int] = []
    for rank, position in enumerate(sorted_positions):
        if pvalues[position] <= DELTA / (len(order) - rank):
            holm.append(order[int(position)])
        else:
            break
    rows.append(
        result_row(
            scenario, "cros_risk_holm", cheapest(selection_policies, holm, selection),
            selection_policies, calibration_policies, evaluation_policies,
            calibration, evaluation,
        )
    )

    bonferroni = [
        index for index, pvalue in zip(order, pvalues) if pvalue <= DELTA / len(order)
    ]
    rows.append(
        result_row(
            scenario, "cros_risk_bonferroni",
            cheapest(selection_policies, bonferroni, selection),
            selection_policies, calibration_policies, evaluation_policies,
            calibration, evaluation,
        )
    )

    if mix_calibration is None or mix_selection is None:
        rows.append({"scenario": scenario, "method": "cros_risk_mix_single", "certified": False})
    else:
        mix_pvalue, mix_risk_pvalue, mix_coverage_pvalue = candidate_pvalue(
            mix_calibration, calibration
        )
        if mix_pvalue > DELTA:
            rows.append(
                {
                    "scenario": scenario,
                    "method": "cros_risk_mix_single",
                    "certified": False,
                    "joint_pvalue": mix_pvalue,
                    "risk_pvalue": mix_risk_pvalue,
                    "coverage_pvalue": mix_coverage_pvalue,
                }
            )
        else:
            row = {
                "scenario": scenario,
                "method": "cros_risk_mix_single",
                "certified": True,
                "selected_policy": mix_calibration.name,
                "joint_pvalue": mix_pvalue,
                "risk_pvalue": mix_risk_pvalue,
                "coverage_pvalue": mix_coverage_pvalue,
                "selection_mean_total_cost": float(mix_selection.total_costs.mean()),
                "cal_n": int(len(calibration)),
                "cal_autonomous_n": int(mix_calibration.autonomous[calibration].sum()),
                "cal_autonomous_error_n": int(
                    (
                        mix_calibration.errors[calibration]
                        & mix_calibration.autonomous[calibration]
                    ).sum()
                ),
                **{
                    f"cal_{k}": v
                    for k, v in selective_stats(mix_calibration, calibration).items()
                },
            }
            if mix_evaluation is not None and evaluation is not None:
                row.update(
                    {
                        f"eval_{k}": v
                        for k, v in selective_stats(mix_evaluation, evaluation).items()
                    }
                )
                row["eval_n"] = int(len(evaluation))
                row["eval_autonomous_n"] = int(
                    mix_evaluation.autonomous[evaluation].sum()
                )
                row["eval_autonomous_error_n"] = int(
                    (
                        mix_evaluation.errors[evaluation]
                        & mix_evaluation.autonomous[evaluation]
                    ).sum()
                )
                row["eval_joint_transfer"] = bool(
                    row["eval_selective_risk"] <= ALPHA
                    and row["eval_autonomous_coverage"] >= GAMMA
                )
            rows.append(row)
    return rows, order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="cros_mix_validation/results/la_cdm_cros_power_v2",
    )
    parser.add_argument(
        "--validation-root",
        default=(
            "cros_mix_validation/results/"
            "la_cdm_trained_common_backbone_validation_v1/shards"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    provenance_corrections = validate_and_repair_shard_metadata(root)
    train = load_shards(root / "train" / "shards", 935)
    selection_data = load_shards(Path(args.validation_root), 165)
    locked_cal = load_shards(root / "locked_calibration" / "shards", 367)
    locked_test = load_shards(root / "locked_test" / "shards", 367)

    ranker, train_oof_scores, ranker_diagnostics = fit_ranker(train, SEED)
    selection_scores = predict_risk(ranker, selection_data)
    locked_cal_scores = predict_risk(ranker, locked_cal)
    locked_test_scores = predict_risk(ranker, locked_test)
    specs = freeze_specs(selection_scores)
    ranker_path = root / "cros_power_v2_ranker.joblib"
    joblib.dump(ranker, ranker_path, compress=3)
    spec_path = root / "cros_power_v2_candidate_specs.json"
    spec_path.write_text(
        json.dumps([asdict(spec) for spec in specs], indent=2) + "\n",
        encoding="utf-8",
    )
    train_policies = make_policies(train, train_oof_scores, specs)
    selection_policies = make_policies(selection_data, selection_scores, specs)
    locked_cal_policies = make_policies(locked_cal, locked_cal_scores, specs)
    locked_test_policies = make_policies(locked_test, locked_test_scores, specs)
    pooled_policies = concatenate_policies(locked_cal_policies, locked_test_policies)

    mix_fit = fit_mix(selection_policies, np.arange(165))
    if mix_fit is None:
        mix_selection = mix_cal = mix_test = mix_pooled = None
        mix_support: list[dict[str, object]] = []
    else:
        support, weights = mix_fit
        mix_selection = apply_mix(selection_policies, support, weights, SEED + 1)
        mix_cal = apply_mix(locked_cal_policies, support, weights, SEED + 2)
        mix_test = apply_mix(locked_test_policies, support, weights, SEED + 3)
        mix_pooled = apply_mix(pooled_policies, support, weights, SEED + 4)
        mix_support = [
            {"policy": selection_policies[index].name, "weight": float(weight)}
            for index, weight in zip(support, weights)
        ]

    locked_rows, locked_order = certify_scenario(
        "locked_calibration_367_to_locked_test_367_exploratory",
        selection_policies,
        locked_cal_policies,
        locked_test_policies,
        mix_selection,
        mix_cal,
        mix_test,
    )
    pooled_rows, pooled_order = certify_scenario(
        "pooled_previously_opened_calibration_734_power_audit",
        selection_policies,
        pooled_policies,
        None,
        mix_selection,
        mix_pooled,
        None,
    )
    rows = pd.DataFrame(locked_rows + pooled_rows)
    result_path = root / "cros_power_v2_results.csv"
    rows.to_csv(result_path, index=False)

    selection = np.arange(165)
    order_payload = {
        "locked367": [
            {
                "rank": rank + 1,
                "policy": selection_policies[index].name,
                "projected_joint_pvalue": projected_pvalue(
                    selection_policies[index], selection, 367
                ),
                **selective_stats(selection_policies[index], selection),
            }
            for rank, index in enumerate(locked_order)
        ],
        "pooled734": [
            {
                "rank": rank + 1,
                "policy": selection_policies[index].name,
                "projected_joint_pvalue": projected_pvalue(
                    selection_policies[index], selection, 734
                ),
                **selective_stats(selection_policies[index], selection),
            }
            for rank, index in enumerate(pooled_order)
        ],
    }
    order_path = root / "cros_power_v2_order.json"
    order_path.write_text(
        json.dumps(order_payload, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "status": "CROS_POWER_V2_EXPLORATORY_COMPLETE",
        "confirmatory": False,
        "reason_not_confirmatory": (
            "locked calibration/test labels were opened in prior development; "
            "pooled audit has no untouched evaluation cohort"
        ),
        "targets_unchanged": {"alpha": ALPHA, "gamma": GAMMA, "delta": DELTA},
        "risk_ranker": ranker_diagnostics,
        "candidate_n": len(specs),
        "fixed_order_cap": ORDER_CAP,
        "selection_n": 165,
        "locked_calibration_n": 367,
        "locked_test_n": 367,
        "pooled_power_calibration_n": 734,
        "mix_targets": {
            "selection_risk": MIX_TARGET_RISK,
            "selection_coverage": MIX_TARGET_COVERAGE,
        },
        "mix_design_margins": {
            "risk_below_alpha": ALPHA - MIX_TARGET_RISK,
            "coverage_above_gamma": MIX_TARGET_COVERAGE - GAMMA,
        },
        "randomization_seed": SEED,
        "mix_support": mix_support,
        "locked_calibration_or_test_loaded": True,
        "shard_metadata_provenance_corrections": provenance_corrections,
        "frozen_artifacts_sha256": {
            "ranker": sha256(ranker_path),
            "candidate_specs": sha256(spec_path),
            "fixed_order": sha256(order_path),
            "results": sha256(result_path),
            "evaluator": sha256(Path(__file__)),
        },
    }
    metadata_path = root / "cros_power_v2_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    freeze_manifest = {
        "status": "CROS_POWER_V2_FROZEN_FOR_FUTURE_UNTOUCHED_COHORT",
        "confirmatory_result_in_this_run": False,
        "targets": {"alpha": ALPHA, "gamma": GAMMA, "delta": DELTA},
        "risk_ranker_training_n": 935,
        "policy_selection_n": 165,
        "candidate_n": len(specs),
        "fixed_order_cap": ORDER_CAP,
        "randomization_seed": SEED,
        "mix_support": mix_support,
        "artifacts_sha256": {
            **metadata["frozen_artifacts_sha256"],
            "metadata": sha256(metadata_path),
        },
        "input_shards_sha256": {
            "ranker_training": {
                path.name: sha256(path)
                for path in sorted((root / "train" / "shards").glob("shard_*.npz"))
            },
            "policy_selection": {
                path.name: sha256(path)
                for path in sorted(Path(args.validation_root).glob("shard_*.npz"))
            },
        },
    }
    (root / "cros_power_v2_freeze_manifest.json").write_text(
        json.dumps(freeze_manifest, indent=2) + "\n", encoding="utf-8"
    )
    for path in root.glob("cros_power_v2_*"):
        os.chmod(path, 0o600)
    print(rows.to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
