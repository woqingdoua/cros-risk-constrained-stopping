#!/usr/bin/env python3
"""Independent table/count verification for the same-backbone comparison."""

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
CROS_METHODS = {"CROS-SINGLE", "CROS-LTT", "CROS-BONFERRONI", "CROS-MIX"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-14))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="cros_mix_validation/results/cros_vs_la_cdm_frozen_v1"
    )
    parser.add_argument(
        "--evaluator-path",
        default="cros_mix_validation/evaluate_cros_vs_la_cdm_frozen.py",
    )
    args = parser.parse_args()
    root = Path(args.root)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["confirmatory"] is not False:
        raise AssertionError("opened test data must remain exploratory")

    paths = {
        "summary": root / "summary.csv",
        "paired_bootstrap": root / "paired_bootstrap.csv",
        "per_label": root / "per_label.csv",
        "mix_randomization_stability": root / "mix_randomization_stability.csv",
        "evaluator": Path(args.evaluator_path),
    }
    verified_hashes = {}
    for name, path in paths.items():
        actual = sha256(path)
        if actual != metadata["output_artifacts_sha256"][name]:
            raise AssertionError(f"artifact hash mismatch: {name}")
        verified_hashes[name] = actual

    summary = pd.read_csv(paths["summary"])
    if len(summary) != 11 or set(summary["method"]).intersection(CROS_METHODS) != CROS_METHODS:
        raise AssertionError("unexpected method table")
    count_checks = []
    for row in summary.itertuples(index=False):
        n = int(row.calibration_n)
        autonomous_n = int(row.calibration_autonomous_n)
        error_n = int(row.calibration_autonomous_error_n)
        risk = error_n / autonomous_n
        coverage = autonomous_n / n
        risk_pvalue = float(binom.cdf(error_n, autonomous_n, ALPHA))
        coverage_pvalue = float(binom.sf(autonomous_n - 1, n, GAMMA))
        joint = max(risk_pvalue, coverage_pvalue)
        if not all(
            (
                close(risk, row.calibration_selective_risk),
                close(coverage, row.calibration_autonomous_coverage),
                close(risk_pvalue, row.calibration_risk_pvalue),
                close(coverage_pvalue, row.calibration_coverage_pvalue),
                close(joint, row.calibration_joint_pvalue),
            )
        ):
            raise AssertionError(f"calibration count mismatch: {row.method}")
        expected_certified = row.method in CROS_METHODS
        if bool(row.calibration_certified_exploratory) != expected_certified:
            raise AssertionError(f"unexpected certificate status: {row.method}")
        if expected_certified and joint > DELTA:
            raise AssertionError(f"CROS raw p-value exceeds delta: {row.method}")
        if not expected_certified and joint <= DELTA:
            raise AssertionError(f"baseline unexpectedly certifies: {row.method}")
        count_checks.append(
            {
                "method": row.method,
                "n": n,
                "autonomous_n": autonomous_n,
                "autonomous_error_n": error_n,
                "joint_pvalue": joint,
                "certified": expected_certified,
            }
        )

    per_label = pd.read_csv(paths["per_label"])
    label_n_checks = per_label.groupby("method", observed=True)["n"].sum()
    if not (label_n_checks == 367).all():
        raise AssertionError("per-label sample counts do not sum to 367")

    bootstrap = pd.read_csv(paths["paired_bootstrap"])
    required = bootstrap[
        (bootstrap["comparator"] == "LA-CDM-NATIVE")
        & bootstrap["metric"].isin(
            ["selective_risk", "error_mass", "total_cost", "time_sensitive_proxy_miss_rate"]
        )
    ]
    if len(required) != 4 or not required["ci_excludes_zero"].all():
        raise AssertionError("CROS-MIX/LA-CDM paired effects are not all resolved")
    if not (required["ci95_upper"] < 0).all():
        raise AssertionError("CROS-MIX expected lower risk/cost/miss than LA-CDM")

    stability = pd.read_csv(paths["mix_randomization_stability"])
    if len(stability) != 1000:
        raise AssertionError("unexpected randomization replicate count")
    pass_fraction = float(stability["constraints_pass_descriptive"].mean())
    if not close(pass_fraction, metadata["cros_mix_randomization_constraint_pass_fraction"]):
        raise AssertionError("randomization stability mismatch")

    report = {
        "status": "CROS_VS_LA_CDM_FROZEN_VERIFICATION_PASSED",
        "confirmatory": False,
        "verified_artifacts_sha256": verified_hashes,
        "calibration_count_checks": count_checks,
        "per_label_total_n": {key: int(value) for key, value in label_n_checks.items()},
        "cros_mix_randomization_constraint_pass_fraction": pass_fraction,
        "cros_mix_vs_la_cdm_paired_ci_checks_passed": True,
        "diagnosis_subgroup_violation_n": len(
            metadata["cros_mix_diagnosis_subgroup_constraint_violations"]
        ),
    }
    output = root / "verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.chmod(0o600)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
