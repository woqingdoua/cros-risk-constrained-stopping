#!/usr/bin/env python3
"""Extract restricted same-patient inputs from frozen CROS caches in the controlled environment.

No model loading, fitting, selection, retuning, or external data transfer.
Patient-level files are written only inside the server run directory.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXPECTED_EVALUATOR_SHA = "e1acfc26796b857e7c3ce57f600b76c59c4c201cc74afa10df7b4eaafd92eb05"
EXPECTED_LOADER_SHA = "c0d014f8d98469730e9698c26e5c6fc8ff911f43dd878360af24d2ae9eb57361"
EXPECTED_ANALYZER_SHA = "2b799edd9b48b6f6ffbc1e5b6406ed4951d809170b2cdb786cf3cd076e10d5fc"
EXPECTED_OUTCOMES_SHA = "a14939975b05941e6f48ba3f6e001ccff11051226acb59f48580cdddb14064a6"
EXPECTED_STATES_SHA = "bb835ec19374d1d02db664d05392e9851020c6fffce5c9cfa538a472eb082ef5"
DET_NAME = "risk_rank:h=3:tau=0.3286273681"
MIX_SEED = 20260905
FIELDS = ["patient_key", "policy", "autonomous", "error_stage0", "error_stop", "error_full"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def import_verified_functions(path, names, namespace):
    # Execute only these reviewed pure cached-array functions, never module imports/main.
    text = path.read_text()
    nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    for name in names:
        require(name in nodes, "Required reviewed source function missing")
        source = ast.get_source_segment(text, nodes[name])
        exec(compile("from __future__ import annotations\n" + source, str(path), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    root, run = args.root.resolve(), args.run_dir.resolve()
    require(run.is_dir(), "Upload the reviewed analysis code to a new run directory first")
    private = run / "private"
    require(not private.exists() and not (run / "aggregate").exists(), "Refusing to overwrite prior analysis")
    bundle = root / "results/genai4health_additions_20260828T150000Z/restricted_bundle"
    v2 = root / "results/la_cdm_cros_power_v2"
    audit = root / "results/genai4health_additions_20260828T150000Z/aggregate_audit"
    evaluator = root / "evaluate_cros_power_v2.py"
    loader = root / "evaluate_la_cdm_common_backbone_dev.py"
    analyzer = run / "analyze_same_accepted_cached.py"
    outcomes_path, states_path = bundle / "policy_outcomes.csv", bundle / "ranker_states.csv"
    for path, expected in [(evaluator, EXPECTED_EVALUATOR_SHA), (loader, EXPECTED_LOADER_SHA),
                           (analyzer, EXPECTED_ANALYZER_SHA), (outcomes_path, EXPECTED_OUTCOMES_SHA),
                           (states_path, EXPECTED_STATES_SHA)]:
        require(path.is_file() and sha(path) == expected, "Reviewed source/cache hash mismatch; fail closed")
    freeze_path = v2 / "cros_power_v2_freeze_manifest.json"
    specs_path = v2 / "cros_power_v2_candidate_specs.json"
    order_path = v2 / "cros_power_v2_order.json"
    freeze = json.loads(freeze_path.read_text())
    require(freeze["randomization_seed"] == 20260902, "Frozen base RNG seed mismatch")
    for key, path in [("evaluator", evaluator), ("candidate_specs", specs_path), ("fixed_order", order_path)]:
        require(sha(path) == freeze["artifacts_sha256"][key], "Frozen artifact hash mismatch")
    prov_path = audit / "provenance_audit.json"
    historical = json.loads(prov_path.read_text())
    historical_shards = historical["current_shard_sha256"]["evaluation"]
    shard_paths = sorted((v2 / "locked_test/shards").glob("shard_*.npz"))
    require(len(shard_paths) == 11 and {p.name for p in shard_paths} == set(historical_shards),
            "Evaluation shard set differs from preserved bundle")
    for path in shard_paths:
        require(sha(path) == historical_shards[path.name], "Evaluation shard hash mismatch")
    split_path = audit / "split_audit.json"
    split = json.loads(split_path.read_text())
    require(split["one_episode_per_patient"] and split["builder_deduplicates_subject_before_split"]
            and split["all_intersections_zero"], "Preserved patient-separation audit failed")
    require(sha(root / "build_ed_cds_action_benchmark.py") == split["builder_sha256"] and
            sha(root / "data/derived/ed_cds_action_v1/manifest.json") == split["benchmark_manifest_sha256"],
            "Patient-separation source evidence changed")

    ns = {"np": np, "Path": Path, "SelectivePolicy": SimpleNamespace, "DEFER_COST": 10.0}
    import_verified_functions(loader, ["load_shards"], ns)
    import_verified_functions(evaluator, ["make_policy", "apply_mix"], ns)
    data = ns["load_shards"](v2 / "locked_test/shards", 367)
    require(len(set(map(int, data["row_indices"]))) == 367, "Duplicate source episode rows")
    require(data["probabilities"].shape == (367, 13, 9), "Canonical cached probability shape changed")
    keys = [f"evaluation:runlocal:{i:04d}" for i in range(367)]
    key_index = {k: i for i, k in enumerate(keys)}
    state_rows = read_csv(states_path)
    require(len(state_rows) == 367 * 13 and all(r["split"] == "evaluation" for r in state_rows),
            "Unexpected ranker-state split/count")
    seen, scores = set(), np.empty((367, 13), dtype=float)
    errors = data["probabilities"].argmax(axis=2) != data["labels"][:, None]
    for row in state_rows:
        require(row["patient_key"] in key_index, "Noncanonical run-local patient key")
        i, stage = key_index[row["patient_key"]], int(row["stage"])
        require(0 <= stage <= 12 and (i, stage) not in seen, "Duplicate/invalid patient stage")
        seen.add((i, stage))
        require(int(row["error"]) == int(errors[i, stage]), "Per-patient cached state error mismatch")
        probability = np.asarray([float(row[f"p{j}"]) for j in range(9)])
        require(np.array_equal(probability, data["probabilities"][i, stage].astype(float)),
                "CSV-to-shard per-patient/stage probability mapping mismatch")
        scores[i, stage] = float(row["cros_full_score"])
    require(len(seen) == 4771, "Missing canonical patient-stage state")
    require((int(errors[:, 0].sum()), int(errors[:, 12].sum())) == (104, 126),
            "Whole-cohort initial/full errors differ from prespecified guards")

    specs = []
    for item in json.loads(specs_path.read_text()):
        specs.append(SimpleNamespace(horizon=item["horizon"], threshold=item["threshold"],
                                     name=f"risk_rank:h={item['horizon']}:tau={item['threshold']:.10f}"))
    names = [s.name for s in specs]
    require(len(specs) == 130 and len(set(names)) == 130, "Frozen candidate list changed")
    # Reproduce the original full candidate-list call; no new candidate selection occurs.
    policies = [ns["make_policy"](data, scores, spec) for spec in specs]
    support_names = [item["policy"] for item in freeze["mix_support"]]
    weights = np.asarray([item["weight"] for item in freeze["mix_support"]], dtype=float)
    require(len(support_names) == 3 and (weights >= 0).all() and abs(weights.sum() - 1) < 1e-10,
            "Frozen mixture weights invalid")
    support = np.asarray([names.index(name) for name in support_names], dtype=int)
    selection_path = audit / "policy_selection.json"
    selection = json.loads(selection_path.read_text())
    require(selection["CROS-FULL"]["selected_policy"] == DET_NAME, "Frozen deterministic selection changed")
    det = policies[names.index(DET_NAME)]
    all_outcomes = [r for r in read_csv(outcomes_path) if r["split"] == "evaluation"]
    for alias, policy in [("CROS-FULL", det)] + [(name, policies[names.index(name)]) for name in support_names]:
        rows = [r for r in all_outcomes if r["policy"] == alias]
        require(len(rows) == 367 and {r["patient_key"] for r in rows} == set(keys),
                "Primitive controller patient mapping incomplete")
        for row in rows:
            i = key_index[row["patient_key"]]
            require(row["autonomous"] in ["0", "1"] and row["error"] in ["0", "1"],
                    "Nonbinary cached terminal outcome")
            require(int(row["autonomous"]) == int(policy.autonomous[i]) and
                    int(row["error"]) == int(policy.errors[i]) and
                    float(row["tests"]) == float(policy.tests[i]),
                    "Primitive per-patient acceptance/stage/terminal-error mismatch")
            require(abs(float(row["total_cost"]) - float(policy.total_costs[i])) < 3e-6,
                    "Primitive per-patient cost mismatch")

    require(np.random.default_rng(MIX_SEED).bit_generator.__class__.__name__ == "PCG64",
            "Unexpected original default_rng implementation")
    rng = np.random.default_rng(MIX_SEED)
    positions = np.searchsorted(np.cumsum(weights), rng.random(367), side="right")
    positions = np.minimum(positions, len(support) - 1)
    assignment = support[positions]
    mix = ns["apply_mix"](policies, support, weights, MIX_SEED)
    # Verify every patient and every copied terminal field against the original evaluator.
    for attribute in ["errors", "autonomous", "test_costs", "total_costs", "latencies", "tests"]:
        expected = np.stack([getattr(p, attribute) for p in policies])[assignment, np.arange(367)]
        require(np.array_equal(getattr(mix, attribute), expected), "Original mixture per-patient realization mismatch")
    for policy, expected in [(det, (297, 52)), (mix, (288, 47))]:
        stop = policy.tests.astype(int)
        require(np.array_equal(policy.tests, stop.astype(float)), "Noninteger realized stopping stage")
        require(np.array_equal(policy.errors, errors[np.arange(367), stop]),
                "Realized per-patient terminal error differs from mapped stopping state")
        require((int(policy.autonomous.sum()), int((policy.errors & policy.autonomous).sum())) == expected,
                "Prespecified accepted/error guard mismatch; fail closed")

    # No patient-level values enter the provenance sidecar or stdout.
    row_order_hash = hashlib.sha256(json.dumps({"global_indices": data["global_indices"].tolist(),
                                              "row_indices": data["row_indices"].tolist()},
                                             sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assignment_hash = hashlib.sha256(assignment.astype("<i8").tobytes()).hexdigest()
    source_paths = {"policy_outcomes.csv": outcomes_path, "ranker_states.csv": states_path,
                    "evaluator.py": evaluator, "loader.py": loader, "freeze_manifest.json": freeze_path,
                    "candidate_specs.json": specs_path, "fixed_order.json": order_path,
                    "provenance_audit.json": prov_path, "split_audit.json": split_path,
                    "policy_selection.json": selection_path, "extractor.py": Path(__file__),
                    "analysis.py": analyzer}
    source_paths.update({"locked_test/" + p.name: p for p in shard_paths})
    command = f"python3 {Path(__file__).resolve()} --root {root} --run-dir {run}"
    provenance = {
        "source_sha256": {name: sha(path) for name, path in source_paths.items()},
        "extraction_command": command,
        "validation": {"all_4771_patient_stage_probabilities_and_errors_match_shards": True,
                       "all_primitive_patient_acceptance_stages_errors_and_costs_match": True,
                       "all_original_mix_terminal_fields_match_per_patient": True,
                       "all_terminal_errors_match_their_mapped_stopping_states": True,
                       "whole_cohort_stage0_full_errors": [104, 126],
                       "one_episode_per_patient_preserved_audit_verified": True},
        "policy_provenance": {
            "CROS-Det": {"legacy_alias": "CROS-FULL", "frozen_policy": DET_NAME,
                         "horizon": specs[names.index(DET_NAME)].horizon,
                         "full_precision_threshold": specs[names.index(DET_NAME)].threshold,
                         "accepted_n": 297, "stopping_errors": 52},
            "CROS-Mix-realized": {
                "original_assignment_verified": True, "assignment_sha256": assignment_hash,
                "assignment_source": {"method": "exact reconstruction; no historical vector retained",
                                      "original_evaluator_sha256": sha(evaluator), "loader_sha256": sha(loader),
                                      "rng": "numpy.random.default_rng / PCG64", "numpy_version": np.__version__,
                                      "historical_numpy_version_recorded": False,
                                      "seed": MIX_SEED, "support": support_names,
                                      "support_candidate_indices": support.tolist(), "weights": weights.tolist(),
                                      "row_order": "sorted shards; concatenate; argsort(global_indices)",
                                      "row_order_sha256": row_order_hash,
                                      "assignment_encoding": "little-endian int64 frozen candidate indices in global-index order",
                                      "per_patient_reference_function_check": "all six original apply_mix fields equal direct assigned primitive fields"},
                "accepted_n": 288, "stopping_errors": 47}},
        "interpretation": "Retrospective exploratory comparison within each fixed accepted cohort; no retuning or new safety certificate.",
        "full_workup_semantics": "stage 12 on the frozen cached common path, including recorded missing-result behavior"}
    private.mkdir(mode=0o700)
    with (private / "same_accepted_input.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for label, policy in [("CROS-Det", det), ("CROS-Mix-realized", mix)]:
            for i, key in enumerate(keys):
                writer.writerow({"patient_key": key, "policy": label, "autonomous": int(policy.autonomous[i]),
                                 "error_stage0": int(errors[i, 0]), "error_stop": int(policy.errors[i]),
                                 "error_full": int(errors[i, 12])})
    np.save(private / "original_realized_candidate_assignment.npy", assignment.astype("<i8"))
    np.savez_compressed(private / "stopping_stages.npz", deterministic=det.tests.astype(int),
                        mixture=mix.tests.astype(int), global_indices=data["global_indices"],
                        row_indices=data["row_indices"])
    (private / "extraction_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": "VERIFIED_CACHED_EXTRACTION_COMPLETE", "evaluation_n": 367,
                      "det_accepted_errors": [297, 52], "mix_accepted_errors": [288, 47],
                      "assignment_sha256": assignment_hash, "row_order_sha256": row_order_hash}))


if __name__ == "__main__":
    main()
