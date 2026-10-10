#!/usr/bin/env python3
"""Freeze the primary ED/Ext-CDS selective-mixture protocol before calibration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from run_ed_cds_selective_sequential_pilot import (
    ALPHA,
    DEFER_COST,
    DELTA,
    FROZEN_TARGET_COVERAGE,
    FROZEN_TARGET_RISK,
    GAMMA,
    make_policies,
)


ROOT = Path("cros_mix_validation/results/ed_cds_sequential_v1")
BENCHMARK = Path("cros_mix_validation/data/derived/ed_cds_action_v1/manifest.json")
OUTPUT = Path("cros_mix_validation/results/ed_cds_locked_protocol_v1/protocol.json")
RANDOMIZATION_SEED = 20261417


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite the frozen protocol")
    state_manifest_path = ROOT / "state_manifest.json"
    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    probabilities = {}
    labels = None
    for state in state_manifest["states"]:
        state_id = int(state["state_id"])
        path = ROOT / "states" / f"state_{state_id:03d}.npz"
        data = np.load(path)
        probabilities[state_id] = data["probabilities"].astype(float)
        if labels is None:
            labels = data["labels"].astype(int)
    assert labels is not None
    policies = make_policies(labels, probabilities, state_manifest, benchmark["actions"])
    coverage = np.asarray([policy.autonomous.mean() for policy in policies])
    error_mass = np.asarray(
        [np.mean(policy.errors & policy.autonomous) for policy in policies]
    )
    cost = np.asarray([policy.total_costs.mean() for policy in policies])
    solution = linprog(
        c=cost,
        A_ub=np.vstack(
            (
                error_mass - FROZEN_TARGET_RISK * coverage,
                -coverage,
            )
        ),
        b_ub=np.asarray((0.0, -FROZEN_TARGET_COVERAGE)),
        A_eq=np.ones((1, len(policies))),
        b_eq=np.ones(1),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError("development mixture LP is infeasible")
    weights = np.where(solution.x < 1e-9, 0.0, solution.x)
    weights /= weights.sum()
    support = np.flatnonzero(weights > 0)
    selection_risk = float(
        np.dot(weights, error_mass) / np.dot(weights, coverage)
    )
    selection_coverage = float(np.dot(weights, coverage))
    payload = {
        "status": "ED_CDS_SELECTIVE_PROTOCOL_FROZEN_BEFORE_CALIBRATION",
        "version": "ed_cds_locked_protocol_v1",
        "primary_method": "cros_mix_frozen",
        "primary_endpoint": {
            "selective_diagnostic_risk_max": ALPHA,
            "autonomous_coverage_min": GAMMA,
            "familywise_delta": DELTA,
            "joint_test": "intersection-union max of exact binomial risk and coverage p-values",
        },
        "selection_mixture_target": {
            "selective_risk": FROZEN_TARGET_RISK,
            "autonomous_coverage": FROZEN_TARGET_COVERAGE,
        },
        "defer_semantics": {
            "terminal_decision": "defer_to_clinician",
            "relative_burden": DEFER_COST,
            "counted_as_correct": False,
            "reported_as": "unresolved/deferred",
        },
        "diagnostic_model": {
            "type": "state-specific word+character TF-IDF one-vs-rest logistic regression",
            "training": "all 1100 development episodes",
            "calibration_use": "prediction only; no fitting",
            "test_use": "forbidden unless the primary calibration certificate passes",
        },
        "randomization": {
            "seed": RANDOMIZATION_SEED,
            "assignment": "SHA256(seed:anonymous_row_index) mapped through frozen cumulative weights",
        },
        "candidate_policy_n": len(policies),
        "mixture_support": [
            {"policy": policies[index].name, "weight": float(weights[index])}
            for index in support
        ],
        "development_oof_selection_metrics": {
            "selective_risk": selection_risk,
            "autonomous_coverage": selection_coverage,
            "mean_total_cost": float(np.dot(weights, cost)),
        },
        "locked_calibration_opened": False,
        "locked_test_opened": False,
        "sha256": {
            "benchmark_manifest": sha256(BENCHMARK),
            "state_manifest": sha256(state_manifest_path),
            "policy_code": sha256(
                Path("cros_mix_validation/run_ed_cds_selective_sequential_pilot.py")
            ),
            "locked_state_scorer": sha256(
                Path("cros_mix_validation/score_ed_cds_locked_state.py")
            ),
            "calibration_evaluator": sha256(
                Path("cros_mix_validation/evaluate_ed_cds_locked_calibration.py")
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False, mode=0o700)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(OUTPUT, 0o400)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
