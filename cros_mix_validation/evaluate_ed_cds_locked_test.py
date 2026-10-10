#!/usr/bin/env python3
"""One-time locked-test evaluation of the calibration-certified primary mixture."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from run_ed_cds_selective_sequential_pilot import ALPHA, GAMMA, make_policies


PROTOCOL_PATH = Path(
    "cros_mix_validation/results/ed_cds_locked_protocol_v1/protocol.json"
)
CERTIFICATE_PATH = Path(
    "cros_mix_validation/results/ed_cds_locked_calibration_v1/primary_certificate.json"
)
STATE_MANIFEST_PATH = Path(
    "cros_mix_validation/results/ed_cds_sequential_v1/state_manifest.json"
)
BENCHMARK_PATH = Path(
    "cros_mix_validation/data/derived/ed_cds_action_v1/manifest.json"
)
OUTPUT_ROOT = Path("cros_mix_validation/results/ed_cds_locked_test_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uniform_from_hash(seed: int, row_index: int) -> float:
    digest = hashlib.sha256(f"{seed}:{row_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


def exact_interval(events: int, n: int, confidence: float = 0.95) -> list[float]:
    if n <= 0:
        return [0.0, 1.0]
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if events == 0 else float(beta.ppf(tail, events, n - events + 1))
    upper = 1.0 if events == n else float(beta.ppf(1.0 - tail, events + 1, n - events))
    return [lower, upper]


def main() -> None:
    output_path = OUTPUT_ROOT / "primary_test_result.json"
    if output_path.exists():
        raise FileExistsError("refusing to overwrite the one-time locked test result")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    certificate = json.loads(CERTIFICATE_PATH.read_text(encoding="utf-8"))
    if not certificate.get("certified") or not certificate.get(
        "test_opening_authorized_by_protocol"
    ):
        raise PermissionError("primary calibration did not authorize test opening")
    if certificate["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise AssertionError("certificate/protocol hash mismatch")
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
        "calibration_evaluator": sha256(
            Path("cros_mix_validation/evaluate_ed_cds_locked_calibration.py")
        ),
    }
    if observed_hashes != expected_hashes:
        raise AssertionError("frozen protocol code/data hash mismatch")

    state_manifest = json.loads(STATE_MANIFEST_PATH.read_text(encoding="utf-8"))
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    probabilities = {}
    labels = None
    row_indices = None
    state_root = OUTPUT_ROOT / "states"
    for state in state_manifest["states"]:
        state_id = int(state["state_id"])
        data = np.load(state_root / f"state_{state_id:03d}.npz")
        probabilities[state_id] = data["probabilities"].astype(float)
        if labels is None:
            labels = data["labels"].astype(int)
            row_indices = data["row_indices"].astype(int)
        else:
            if not np.array_equal(labels, data["labels"]):
                raise AssertionError("label order mismatch")
            if not np.array_equal(row_indices, data["row_indices"]):
                raise AssertionError("row order mismatch")
    assert labels is not None and row_indices is not None
    policies = make_policies(labels, probabilities, state_manifest, benchmark["actions"])
    by_name = {policy.name: policy for policy in policies}
    support_spec = protocol["mixture_support"]
    support = [by_name[item["policy"]] for item in support_spec]
    weights = np.asarray([float(item["weight"]) for item in support_spec])
    weights /= weights.sum()
    uniforms = np.asarray(
        [
            uniform_from_hash(protocol["randomization"]["seed"], int(row_index))
            for row_index in row_indices
        ]
    )
    assignment = np.minimum(
        np.searchsorted(np.cumsum(weights), uniforms, side="right"), len(support) - 1
    )
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
    result = {
        "status": "ED_CDS_PRIMARY_LOCKED_TEST_COMPLETE",
        "method": protocol["primary_method"],
        "test_n": len(labels),
        "autonomous_n": selected_n,
        "deferred_n": len(labels) - selected_n,
        "error_n_among_autonomous": error_n,
        "selective_risk": selective_risk,
        "selective_risk_exact95": exact_interval(error_n, selected_n),
        "autonomous_coverage": coverage,
        "autonomous_coverage_exact95": exact_interval(selected_n, len(labels)),
        "defer_rate": 1.0 - coverage,
        "joint_target_transfers": bool(selective_risk <= ALPHA and coverage >= GAMMA),
        "targets": {
            "selective_risk_max": ALPHA,
            "autonomous_coverage_min": GAMMA,
        },
        "mean_test_cost": float(realized("test_costs").mean()),
        "mean_total_cost": float(realized("total_costs").mean()),
        "mean_latency": float(realized("latencies").mean()),
        "mean_tests": float(realized("tests").mean()),
        "any_protocol_or_threshold_refit_after_calibration": False,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "certificate_sha256": sha256(CERTIFICATE_PATH),
        "test_evaluator_sha256": sha256(Path(__file__)),
    }

    class_rows = []
    class_names = list(benchmark["labels"])
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_index
        class_autonomous = autonomous & mask
        class_n = int(mask.sum())
        class_selected = int(class_autonomous.sum())
        class_errors = int(np.sum(errors & class_autonomous))
        class_rows.append(
            {
                "diagnosis": class_name,
                "n": class_n,
                "autonomous_n": class_selected,
                "deferred_n": class_n - class_selected,
                "autonomous_coverage": class_selected / class_n,
                "error_n_among_autonomous": class_errors,
                "selective_error": (
                    class_errors / class_selected if class_selected else np.nan
                ),
            }
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    pd.DataFrame(class_rows).to_csv(OUTPUT_ROOT / "class_metrics.csv", index=False)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.chmod(output_path, 0o400)
    os.chmod(OUTPUT_ROOT / "class_metrics.csv", 0o400)
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(class_rows).to_string(index=False))


if __name__ == "__main__":
    main()
