#!/usr/bin/env python3
"""Independent count-level verification for the exploratory CROS v2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


ALPHA = 0.25
GAMMA = 0.70
DELTA = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_indices(root: Path, expected_n: int) -> np.ndarray:
    paths = sorted(root.glob("shard_*.npz"))
    if len(paths) != 11:
        raise AssertionError(f"expected 11 shards in {root}, got {len(paths)}")
    values = np.concatenate([np.load(path)["row_indices"] for path in paths])
    if len(values) != expected_n or len(np.unique(values)) != expected_n:
        raise AssertionError(f"invalid row indices in {root}")
    return values


def close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-14))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="cros_mix_validation/results/la_cdm_cros_power_v2"
    )
    parser.add_argument(
        "--selection-root",
        default=(
            "cros_mix_validation/results/"
            "la_cdm_trained_common_backbone_validation_v1/shards"
        ),
    )
    parser.add_argument(
        "--evaluator-path",
        default="cros_mix_validation/evaluate_cros_power_v2.py",
    )
    args = parser.parse_args()
    root = Path(args.root)
    result_path = root / "cros_power_v2_results.csv"
    metadata_path = root / "cros_power_v2_metadata.json"
    freeze_path = root / "cros_power_v2_freeze_manifest.json"

    split_rows = {
        "train": row_indices(root / "train" / "shards", 935),
        "selection": row_indices(Path(args.selection_root), 165),
        "calibration": row_indices(root / "locked_calibration" / "shards", 367),
        "test": row_indices(root / "locked_test" / "shards", 367),
    }
    overlap: dict[str, int] = {}
    names = list(split_rows)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            key = f"{left}__{right}"
            overlap[key] = int(
                len(set(split_rows[left].tolist()) & set(split_rows[right].tolist()))
            )
    if any(overlap.values()):
        raise AssertionError(f"episode overlap detected: {overlap}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["confirmatory"] is not False:
        raise AssertionError("previously opened data must not be called confirmatory")
    if metadata["targets_unchanged"] != {
        "alpha": ALPHA,
        "gamma": GAMMA,
        "delta": DELTA,
    }:
        raise AssertionError("certification targets changed")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "CROS_POWER_V2_FROZEN_FOR_FUTURE_UNTOUCHED_COHORT":
        raise AssertionError("unexpected freeze status")
    artifact_paths = {
        "ranker": root / "cros_power_v2_ranker.joblib",
        "candidate_specs": root / "cros_power_v2_candidate_specs.json",
        "fixed_order": root / "cros_power_v2_order.json",
        "results": result_path,
        "evaluator": Path(args.evaluator_path),
        "metadata": metadata_path,
    }
    verified_hashes: dict[str, str] = {}
    for name, path in artifact_paths.items():
        actual = sha256(path)
        expected = freeze["artifacts_sha256"][name]
        if actual != expected:
            raise AssertionError(f"frozen artifact hash mismatch: {name}")
        verified_hashes[name] = actual
    input_roots = {
        "ranker_training": root / "train" / "shards",
        "policy_selection": Path(args.selection_root),
    }
    for group, expected_paths in freeze["input_shards_sha256"].items():
        for filename, expected in expected_paths.items():
            actual = sha256(input_roots[group] / filename)
            if actual != expected:
                raise AssertionError(f"frozen input hash mismatch: {group}/{filename}")

    results = pd.read_csv(result_path)
    locked = results[
        results["scenario"]
        == "locked_calibration_367_to_locked_test_367_exploratory"
    ]
    if len(locked) != 5 or not locked["certified"].all():
        raise AssertionError("all five CROS variants were expected to certify")

    checks: list[dict[str, object]] = []
    for row in locked.itertuples(index=False):
        n = int(row.cal_n)
        autonomous_n = int(row.cal_autonomous_n)
        error_n = int(row.cal_autonomous_error_n)
        risk = error_n / autonomous_n
        coverage = autonomous_n / n
        risk_pvalue = float(binom.cdf(error_n, autonomous_n, ALPHA))
        coverage_pvalue = float(binom.sf(autonomous_n - 1, n, GAMMA))
        joint_pvalue = max(risk_pvalue, coverage_pvalue)
        if not all(
            (
                close(risk, float(row.cal_selective_risk)),
                close(coverage, float(row.cal_autonomous_coverage)),
                close(risk_pvalue, float(row.risk_pvalue)),
                close(coverage_pvalue, float(row.coverage_pvalue)),
                close(joint_pvalue, float(row.joint_pvalue)),
            )
        ):
            raise AssertionError(f"stored statistic mismatch for {row.method}")
        if joint_pvalue > DELTA or not bool(row.eval_joint_transfer):
            raise AssertionError(f"failed certificate or transfer for {row.method}")
        if row.method == "cros_risk_bonferroni" and joint_pvalue > DELTA / 12:
            raise AssertionError("Bonferroni certificate exceeds delta/12")
        eval_n = int(row.eval_n)
        eval_autonomous_n = int(row.eval_autonomous_n)
        eval_error_n = int(row.eval_autonomous_error_n)
        if not close(eval_error_n / eval_autonomous_n, float(row.eval_selective_risk)):
            raise AssertionError(f"test risk count mismatch for {row.method}")
        if not close(eval_autonomous_n / eval_n, float(row.eval_autonomous_coverage)):
            raise AssertionError(f"test coverage count mismatch for {row.method}")
        checks.append(
            {
                "method": row.method,
                "calibration_n": n,
                "calibration_autonomous_n": autonomous_n,
                "calibration_autonomous_error_n": error_n,
                "recomputed_joint_pvalue": joint_pvalue,
                "test_n": eval_n,
                "test_autonomous_n": eval_autonomous_n,
                "test_autonomous_error_n": eval_error_n,
                "test_transfer": True,
            }
        )

    report = {
        "status": "CROS_POWER_V2_COUNT_VERIFICATION_PASSED",
        "confirmatory": False,
        "split_row_intersections": overlap,
        "targets": {"alpha": ALPHA, "gamma": GAMMA, "delta": DELTA},
        "frozen_artifacts_verified": verified_hashes,
        "freeze_manifest_sha256": sha256(freeze_path),
        "checks": checks,
        "input_sha256": {
            "results": sha256(result_path),
            "metadata": sha256(metadata_path),
        },
    }
    output_path = root / "cros_power_v2_verification.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_path.chmod(0o600)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
