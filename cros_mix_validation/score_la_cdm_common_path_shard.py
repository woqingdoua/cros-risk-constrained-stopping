#!/usr/bin/env python3
"""Score a forced-continuation LA-CDM path for common-backbone controls.

At each state, the same zero-shot hypothesis agent scores diagnoses and the
same decision agent scores STOP plus every remaining test.  The native LA-CDM
stop time is recorded, while an alternate-controller path is extended with the
highest-scoring non-STOP action.  This permits stopping controllers to be
compared without changing the diagnostic or action-proposal backbone.

An optional PEFT adapter can be loaded over the base model.  This is a
prompt-transfer common-backbone experiment: every controller sees exactly the
same adapted letter-scoring interface, but it is not claimed to reproduce the
official free-form LA-CDM rollout prompt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from score_la_cdm_zs_port_shard import (
    LETTERS,
    decision_prompt,
    hypothesis_prompt,
    load_episodes,
    option_permutation,
    score_prompts,
    sha256,
    token_forms,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tests", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--expected-split-label", default="development")
    parser.add_argument(
        "--development-path",
        default=(
            "cros_mix_validation/data/derived/ed_cds_action_v1/"
            "development_episodes.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="cros_mix_validation/data/derived/ed_cds_action_v1/manifest.json",
    )
    parser.add_argument(
        "--output-root",
        default="cros_mix_validation/results/la_cdm_common_backbone_dev_v1/shards",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    manifest_path = Path(args.benchmark_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = list(manifest["labels"])
    action_specs = dict(manifest["actions"])
    all_actions = list(action_specs)
    if len(classes) > len(LETTERS) or len(all_actions) + 1 > len(LETTERS):
        raise AssertionError("option alphabet is too small")

    development_path = Path(args.development_path)
    all_episodes = load_episodes(development_path, args.expected_split_label)
    global_indices = np.array_split(np.arange(len(all_episodes)), args.num_shards)[
        args.shard_index
    ]
    episodes = [all_episodes[int(index)] for index in global_indices]
    n = len(episodes)
    class_to_index = {name: index for index, name in enumerate(classes)}
    labels = np.asarray(
        [class_to_index[str(episode["label"])] for episode in episodes], dtype=np.int8
    )
    anonymous_rows = np.asarray(
        [int(episode["row_index"]) for episode in episodes], dtype=np.int16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    option_ids, option_forms = token_forms(tokenizer, len(LETTERS))
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            local_files_only=True,
        )
    model = model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    stages = args.max_tests + 1
    acquired: list[list[str]] = [[] for _ in range(n)]
    probabilities = np.zeros((n, stages, len(classes)), dtype=np.float32)
    confidences = np.zeros((n, stages), dtype=np.float32)
    cumulative_costs = np.zeros((n, stages), dtype=np.float32)
    cumulative_latencies = np.zeros((n, stages), dtype=np.float32)
    cumulative_missing = np.zeros((n, stages), dtype=np.int8)
    stop_scores = np.zeros((n, stages), dtype=np.float32)
    actions = np.full((n, args.max_tests), -1, dtype=np.int8)
    native_stop_stage = np.full(n, args.max_tests, dtype=np.int8)
    native_has_stopped = np.zeros(n, dtype=bool)

    all_indices = np.arange(n)
    for step in range(stages):
        h_orders = [
            option_permutation(
                args.seed,
                int(anonymous_rows[index]),
                step,
                "common_hypothesis",
                len(classes),
            )
            for index in all_indices
        ]
        h_prompts = [
            hypothesis_prompt(tokenizer, episodes[index], acquired[index], classes, order)
            for index, order in zip(all_indices, h_orders)
        ]
        h_letter_prob = score_prompts(
            model,
            tokenizer,
            h_prompts,
            option_ids[: len(classes)],
            args.batch_size,
            device,
        )
        hypotheses = np.zeros(n, dtype=np.int8)
        for index, order in enumerate(h_orders):
            class_prob = np.zeros(len(classes), dtype=np.float32)
            class_prob[order] = h_letter_prob[index]
            probabilities[index, step] = class_prob
            hypotheses[index] = int(np.argmax(class_prob))
            confidences[index, step] = float(class_prob[hypotheses[index]])
        if step == args.max_tests:
            native_stop_stage[~native_has_stopped] = step
            native_has_stopped[:] = True
            break

        d_prompts: list[str] = []
        d_orders: list[np.ndarray] = []
        d_candidates: list[list[str]] = []
        for index in all_indices:
            remaining = [action for action in all_actions if action not in acquired[index]]
            candidates = remaining + ["diagnosis"]
            order = option_permutation(
                args.seed,
                int(anonymous_rows[index]),
                step,
                "common_decision",
                len(candidates),
            )
            d_candidates.append(candidates)
            d_orders.append(order)
            d_prompts.append(
                decision_prompt(
                    tokenizer,
                    episodes[index],
                    acquired[index],
                    classes[int(hypotheses[index])],
                    float(confidences[index, step]),
                    remaining,
                    action_specs,
                    order,
                    args.max_tests,
                )
            )
        option_n = len(d_candidates[0])
        if any(len(candidates) != option_n for candidates in d_candidates):
            raise AssertionError("forced paths should have equal action counts per stage")
        d_letter_prob = score_prompts(
            model,
            tokenizer,
            d_prompts,
            option_ids[:option_n],
            args.batch_size,
            device,
        )
        for index, (candidates, order) in enumerate(zip(d_candidates, d_orders)):
            candidate_prob = np.zeros(len(candidates), dtype=np.float32)
            candidate_prob[order] = d_letter_prob[index]
            diagnosis_index = len(candidates) - 1
            stop_scores[index, step] = candidate_prob[diagnosis_index]
            native_stop_now = int(np.argmax(candidate_prob)) == diagnosis_index
            if native_stop_now and not native_has_stopped[index]:
                native_stop_stage[index] = step
                native_has_stopped[index] = True
            # Forced continuation uses the decision agent's highest-scoring test.
            test_index = int(np.argmax(candidate_prob[:-1]))
            action = candidates[test_index]
            action_id = all_actions.index(action)
            actions[index, step] = action_id
            acquired[index].append(action)
            cumulative_costs[index, step + 1] = (
                cumulative_costs[index, step]
                + float(action_specs[action]["resource_cost"])
            )
            cumulative_latencies[index, step + 1] = (
                cumulative_latencies[index, step]
                + float(action_specs[action]["latency_minutes"])
            )
            payload = str(episodes[index]["action_payloads"].get(action, "")).strip()
            cumulative_missing[index, step + 1] = (
                cumulative_missing[index, step] + int(not bool(payload))
            )
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "step": step,
                    "native_stopped_cumulative": int(native_has_stopped.sum()),
                    "n": n,
                }
            ),
            flush=True,
        )

    if not native_has_stopped.all():
        raise AssertionError("native stop stage missing")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    npz_path = output_root / f"shard_{args.shard_index:02d}.npz"
    json_path = output_root / f"shard_{args.shard_index:02d}.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite common-backbone shard")
    np.savez_compressed(
        npz_path,
        global_indices=global_indices.astype(np.int16),
        row_indices=anonymous_rows,
        labels=labels,
        probabilities=probabilities,
        confidences=confidences,
        cumulative_costs=cumulative_costs,
        cumulative_latencies=cumulative_latencies,
        cumulative_missing=cumulative_missing,
        stop_scores=stop_scores,
        actions=actions,
        native_stop_stage=native_stop_stage,
    )
    split_slug = args.expected_split_label.strip().lower()
    locked_split_loaded = split_slug in {"locked_calibration", "locked_test"}
    metadata = {
        "status": (
            "LA_CDM_COMMON_BACKBONE_"
            + split_slug.upper()
            + "_SHARD_COMPLETE"
        ),
        "split": args.expected_split_label,
        "locked_calibration_or_test_loaded": locked_split_loaded,
        "method": (
            "LA-CDM trained-adapter prompt-transfer forced-continuation common path"
            if args.adapter_path
            else "LA-CDM-ZS-port forced-continuation common path"
        ),
        "model": "Qwen2.5-7B-Instruct",
        "model_config_sha256": sha256(Path(args.model_path) / "config.json"),
        "adapter_path": args.adapter_path,
        "adapter_config_sha256": (
            sha256(Path(args.adapter_path) / "adapter_config.json")
            if args.adapter_path
            else None
        ),
        "adapter_model_sha256": (
            sha256(Path(args.adapter_path) / "adapter_model.safetensors")
            if args.adapter_path
            else None
        ),
        "development_sha256": sha256(development_path),
        "benchmark_manifest_sha256": sha256(manifest_path),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "shard_n": n,
        "max_tests": args.max_tests,
        "seed": args.seed,
        "option_token_forms": option_forms,
        "contains_clinical_text": False,
        "counterfactual_claim": False,
        "official_free_form_prompt_equivalence_claim": False,
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.chmod(npz_path, 0o600)
    os.chmod(json_path, 0o600)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
