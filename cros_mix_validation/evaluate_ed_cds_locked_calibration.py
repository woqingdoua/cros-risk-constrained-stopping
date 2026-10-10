#!/usr/bin/env python3
"""Evaluate the single frozen primary mixture on locked calibration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from run_ed_cds_selective_sequential_pilot import (
    ALPHA,
    DELTA,
    GAMMA,
    candidate_pvalue,
    deterministic_bounds,
    make_policies,
)


PROTOCOL_PATH = Path(
    "cros_mix_validation/results/ed_cds_locked_protocol_v1/protocol.json"
)
STATE_MANIFEST_PATH = Path(
    "cros_mix_validation/results/ed_cds_sequential_v1/state_manifest.json"
)
BENCHMARK_PATH = Path(
    "cros_mix_validation/data/derived/ed_cds_action_v1/manifest.json"
)
OUTPUT_PATH = Path(
    "cros_mix_validation/results/ed_cds_locked_calibration_v1/primary_certificate.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_from_hash(seed: int, row_index: int) -> float:
    digest = hashlib.sha256(f"{seed}:{row_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError("refusing to overwrite the locked calibration certificate")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_hashes = protocol["sha256"]
    observed_hashes = {
        "benchmark_manifest": sha256(BENCHMARK_PATH),
        "state_manifest": sha256(STATE_MANIFEST_PATH),
        "policy_code": sha256(
            Path("cros_mix_validation/run_ed_cds_selective_sequential_pilot.py")
        ),
        "locked_state_scorer": sha256(
            Path("cros_mix_validation/score_ed_cds_locked_state.py")
        ),
        "calibration_evaluator": sha256(Path(__file__)),
    }
    if observed_hashes != expected_hashes:
        raise AssertionError("frozen protocol code/data hash mismatch")
    if protocol["locked_calibration_opened"] or protocol["locked_test_opened"]:
        raise AssertionError("protocol was not frozen in the expected pre-calibration state")

    state_manifest = json.loads(STATE_MANIFEST_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    root = Path("cros_mix_validation/results/ed_cds_locked_calibration_v1/states")
    probabilities = {}
    labels = None
    row_indices = None
    for state in state_manifest["states"]:
        state_id = int(state["state_id"])
        data = np.load(root / f"state_{state_id:03d}.npz")
        probabilities[state_id] = data["probabilities"].astype(float)
        if labels is None:
            labels = data["labels"].astype(int)
            row_indices = data["row_indices"].astype(int)
        else:
            if not np.array_equal(labels, data["labels"]):
                raise AssertionError("label order mismatch across states")
            if not np.array_equal(row_indices, data["row_indices"]):
                raise AssertionError("row order mismatch across states")
    assert labels is not None and row_indices is not None
    policies = make_policies(labels, probabilities, state_manifest, benchmark["actions"])
    by_name = {policy.name: policy for policy in policies}
    support_spec = protocol["mixture_support"]
    support = [by_name[item["policy"]] for item in support_spec]
    weights = np.asarray([float(item["weight"]) for item in support_spec])
    weights /= weights.sum()
    cumulative = np.cumsum(weights)
    uniforms = np.asarray(
        [
            uniform_from_hash(protocol["randomization"]["seed"], int(row_index))
            for row_index in row_indices
        ]
    )
    assignment = np.minimum(np.searchsorted(cumulative, uniforms, side="right"), len(support) - 1)
    episode = np.arange(len(labels))

    def realized(attribute: str) -> np.ndarray:
        matrix = np.stack([getattr(policy, attribute) for policy in support], axis=0)
        return matrix[assignment, episode]

    autonomous = realized("autonomous").astype(bool)
    errors = realized("errors").astype(bool)
    selected_n = int(autonomous.sum())
    error_n = int(np.sum(errors & autonomous))
    coverage = selected_n / len(labels)
    selective_risk = error_n / selected_n if selected_n else 1.0
    # Exact intersection-union test: both component nulls must be rejected.
    class Frozen:
        pass

    frozen = Frozen()
    frozen.autonomous = autonomous
    frozen.errors = errors
    joint_p, risk_p, coverage_p = candidate_pvalue(frozen, np.arange(len(labels)))
    selective_upper, coverage_lower = deterministic_bounds(
        frozen, np.arange(len(labels)), DELTA / 2
    )
    certified = bool(joint_p <= DELTA)
    payload = {
        "status": (
            "ED_CDS_PRIMARY_CALIBRATION_CERTIFIED"
            if certified
            else "ED_CDS_PRIMARY_CALIBRATION_NOT_CERTIFIED"
        ),
        "certified": certified,
        "test_opening_authorized_by_protocol": certified,
        "method": protocol["primary_method"],
        "calibration_n": len(labels),
        "autonomous_n": selected_n,
        "error_n_among_autonomous": error_n,
        "selective_risk": selective_risk,
        "autonomous_coverage": coverage,
        "defer_rate": 1.0 - coverage,
        "joint_pvalue": joint_p,
        "risk_pvalue": risk_p,
        "coverage_pvalue": coverage_p,
        "simultaneous_one_sided_bounds_delta_0.05": {
            "selective_risk_upper": selective_upper,
            "autonomous_coverage_lower": coverage_lower,
        },
        "targets": {
            "selective_risk_max": ALPHA,
            "autonomous_coverage_min": GAMMA,
            "delta": DELTA,
        },
        "mean_test_cost": float(realized("test_costs").mean()),
        "mean_total_cost": float(realized("total_costs").mean()),
        "mean_latency": float(realized("latencies").mean()),
        "mean_tests": float(realized("tests").mean()),
        "mixture_support_size": len(support),
        "locked_test_loaded": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(OUTPUT_PATH, 0o400)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
