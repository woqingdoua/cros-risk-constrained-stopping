#!/usr/bin/env python3
"""Zero-shot, structured LA-CDM port for the ED/Ext-CDS action benchmark.

This is deliberately named LA-CDM-ZS-port: it preserves the official split
between a hypothesis agent and a decision agent, but does not claim to be the
official GRPO/SFT-trained checkpoint.  Only anonymous predictions and action
indices are persisted; clinical text remains in the restricted episode files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


NO_RECORDED_RESULT = "[NO_RECORDED_RESULT]"
LETTERS = tuple("ABCDEFGHIJKLM")


def load_episodes(path: Path, expected_split: str) -> list[dict[str, object]]:
    episodes: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            if episode.get("split") != expected_split:
                raise AssertionError("split label mismatch")
            episodes.append(episode)
    return episodes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, (limit - 15) // 2)
    return text[:half] + "\n[TRUNCATED]\n" + text[-half:]


def option_permutation(seed: int, row_index: int, step: int, namespace: str, n: int) -> np.ndarray:
    material = f"{seed}:{row_index}:{step}:{namespace}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(derived).permutation(n)


def evidence_document(episode: dict[str, object], acquired: list[str]) -> str:
    chunks = [
        "Initial emergency-department presentation:",
        truncate_middle(str(episode["initial_text"]), 3400),
    ]
    payloads = episode["action_payloads"]
    for number, action in enumerate(acquired, start=1):
        payload = str(payloads.get(action, "")).strip()
        observation = payload if payload else NO_RECORDED_RESULT
        chunks.extend(
            (
                f"Test {number} ({action.replace('_', ' ')}):",
                truncate_middle(observation, 900),
            )
        )
    document = "\n".join(chunks)
    return truncate_middle(document, 10500)


def chat_prompt(tokenizer, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def hypothesis_prompt(
    tokenizer,
    episode: dict[str, object],
    acquired: list[str],
    classes: list[str],
    order: np.ndarray,
) -> str:
    options = "\n".join(
        f"{LETTERS[position]}. {classes[int(class_index)].replace('_', ' ')}"
        for position, class_index in enumerate(order)
    )
    content = (
        "You are the hypothesis agent in a sequential diagnostic system. "
        "Use only the evidence displayed below. A missing recorded result is "
        "unknown and must never be interpreted as normal. Choose the single "
        "most likely primary diagnosis.\n\n"
        + evidence_document(episode, acquired)
        + "\n\nCandidate diagnoses:\n"
        + options
        + "\n\nReturn exactly one option letter and no other text."
    )
    return chat_prompt(tokenizer, content)


def decision_prompt(
    tokenizer,
    episode: dict[str, object],
    acquired: list[str],
    hypothesis: str,
    confidence: float,
    remaining: list[str],
    action_specs: dict[str, dict[str, float]],
    order: np.ndarray,
    max_tests: int,
) -> str:
    candidates = remaining + ["diagnosis"]
    rendered: list[str] = []
    for position, candidate_index in enumerate(order):
        candidate = candidates[int(candidate_index)]
        if candidate == "diagnosis":
            description = f"FINAL DIAGNOSIS: {hypothesis.replace('_', ' ')}"
        else:
            spec = action_specs[candidate]
            description = (
                f"ORDER {candidate.replace('_', ' ')} "
                f"(resource cost {float(spec['resource_cost']):g}; "
                f"nominal wait {float(spec['latency_minutes']):g} minutes)"
            )
        rendered.append(f"{LETTERS[position]}. {description}")
    content = (
        "You are the decision agent in a hypothesis-driven sequential diagnostic "
        "system. At each step choose exactly one unrequested test, or stop and "
        "issue the current hypothesis as the final diagnosis. Balance diagnostic "
        "value against resource cost and delay. Never repeat a test. A missing "
        "recorded result is unknown, not normal. You may order at most "
        f"{max_tests} tests.\n\n"
        + evidence_document(episode, acquired)
        + f"\n\nCurrent hypothesis: {hypothesis.replace('_', ' ')}\n"
        + f"Normalized hypothesis confidence: {confidence:.4f}\n"
        + f"Tests already ordered: {len(acquired)} of {max_tests}\n\n"
        + "Available actions:\n"
        + "\n".join(rendered)
        + "\n\nReturn exactly one option letter and no other text."
    )
    return chat_prompt(tokenizer, content)


def token_forms(tokenizer, n: int) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    forms: list[str] = []
    for letter in LETTERS[:n]:
        candidates = (letter, " " + letter)
        encodings = [tokenizer.encode(value, add_special_tokens=False) for value in candidates]
        valid = [(value, encoded[0]) for value, encoded in zip(candidates, encodings) if len(encoded) == 1]
        if not valid:
            raise AssertionError(f"no single-token form for {letter}")
        form, token_id = valid[0]
        forms.append(form)
        ids.append(int(token_id))
    return ids, forms


@torch.inference_mode()
def score_prompts(
    model,
    tokenizer,
    prompts: list[str],
    option_token_ids: list[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        logits = output.logits[:, -1, option_token_ids].float()
        rows.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", choices=("locked_calibration", "locked_test"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tests", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--data-root",
        default="cros_mix_validation/data/derived/ed_cds_action_v1",
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="cros_mix_validation/data/derived/ed_cds_action_v1/manifest.json",
    )
    parser.add_argument(
        "--output-root",
        default="cros_mix_validation/results/la_cdm_zs_port_qwen7_v1",
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

    data_path = Path(args.data_root) / f"{args.split}_episodes.jsonl.gz"
    episodes_all = load_episodes(data_path, args.split)
    shard_global = np.array_split(np.arange(len(episodes_all)), args.num_shards)[args.shard_index]
    episodes = [episodes_all[int(index)] for index in shard_global]
    n = len(episodes)
    class_to_index = {name: index for index, name in enumerate(classes)}
    labels = np.asarray([class_to_index[str(episode["label"])] for episode in episodes], dtype=np.int8)
    anonymous_rows = np.asarray([int(episode["row_index"]) for episode in episodes], dtype=np.int16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
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
    ).to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    acquired: list[list[str]] = [[] for _ in range(n)]
    active = np.ones(n, dtype=bool)
    predictions = np.full(n, -1, dtype=np.int8)
    final_confidence = np.zeros(n, dtype=np.float32)
    final_probabilities = np.zeros((n, len(classes)), dtype=np.float32)
    costs = np.zeros(n, dtype=np.float32)
    latencies = np.zeros(n, dtype=np.float32)
    missing_results = np.zeros(n, dtype=np.int8)
    forced_at_max = np.zeros(n, dtype=bool)
    action_matrix = np.full((n, args.max_tests), -1, dtype=np.int8)

    for step in range(args.max_tests + 1):
        active_indices = np.flatnonzero(active)
        if len(active_indices) == 0:
            break
        h_orders = [
            option_permutation(args.seed, int(anonymous_rows[i]), step, "hypothesis", len(classes))
            for i in active_indices
        ]
        h_prompts = [
            hypothesis_prompt(tokenizer, episodes[int(i)], acquired[int(i)], classes, order)
            for i, order in zip(active_indices, h_orders)
        ]
        h_letter_prob = score_prompts(
            model,
            tokenizer,
            h_prompts,
            option_ids[: len(classes)],
            args.batch_size,
            device,
        )
        current_predictions: dict[int, int] = {}
        current_confidences: dict[int, float] = {}
        for local, (i, order) in enumerate(zip(active_indices, h_orders)):
            class_prob = np.zeros(len(classes), dtype=np.float32)
            class_prob[order] = h_letter_prob[local]
            prediction = int(np.argmax(class_prob))
            current_predictions[int(i)] = prediction
            current_confidences[int(i)] = float(class_prob[prediction])
            final_probabilities[int(i)] = class_prob
            final_confidence[int(i)] = float(class_prob[prediction])

        if step == args.max_tests:
            for i in active_indices:
                predictions[int(i)] = current_predictions[int(i)]
                forced_at_max[int(i)] = True
                active[int(i)] = False
            break

        d_prompts: list[str] = []
        d_orders: list[np.ndarray] = []
        d_candidates: list[list[str]] = []
        for i in active_indices:
            idx = int(i)
            remaining = [action for action in all_actions if action not in acquired[idx]]
            candidates = remaining + ["diagnosis"]
            order = option_permutation(
                args.seed, int(anonymous_rows[idx]), step, "decision", len(candidates)
            )
            d_candidates.append(candidates)
            d_orders.append(order)
            d_prompts.append(
                decision_prompt(
                    tokenizer,
                    episodes[idx],
                    acquired[idx],
                    classes[current_predictions[idx]],
                    current_confidences[idx],
                    remaining,
                    action_specs,
                    order,
                    args.max_tests,
                )
            )
        max_options = max(len(candidates) for candidates in d_candidates)
        d_letter_prob = score_prompts(
            model,
            tokenizer,
            d_prompts,
            option_ids[:max_options],
            args.batch_size,
            device,
        )
        for local, i in enumerate(active_indices):
            idx = int(i)
            candidates = d_candidates[local]
            order = d_orders[local]
            selected_position = int(np.argmax(d_letter_prob[local, : len(candidates)]))
            choice = candidates[int(order[selected_position])]
            if choice == "diagnosis":
                predictions[idx] = current_predictions[idx]
                active[idx] = False
                continue
            action_index = all_actions.index(choice)
            action_matrix[idx, step] = action_index
            acquired[idx].append(choice)
            costs[idx] += float(action_specs[choice]["resource_cost"])
            latencies[idx] += float(action_specs[choice]["latency_minutes"])
            payload = str(episodes[idx]["action_payloads"].get(choice, "")).strip()
            if not payload:
                missing_results[idx] += 1
        print(
            json.dumps(
                {
                    "split": args.split,
                    "shard": args.shard_index,
                    "step": step,
                    "active_after_step": int(active.sum()),
                    "diagnosed": int((predictions >= 0).sum()),
                }
            ),
            flush=True,
        )

    if np.any(predictions < 0):
        raise AssertionError("some episodes have no final prediction")
    output_dir = Path(args.output_root) / args.split / "shards"
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    npz_path = output_dir / f"shard_{args.shard_index:02d}.npz"
    json_path = output_dir / f"shard_{args.shard_index:02d}.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite an LA-CDM port shard")
    np.savez_compressed(
        npz_path,
        global_indices=shard_global.astype(np.int16),
        row_indices=anonymous_rows,
        labels=labels,
        predictions=predictions,
        final_confidence=final_confidence,
        final_probabilities=final_probabilities,
        costs=costs,
        latencies=latencies,
        tests=np.asarray([len(value) for value in acquired], dtype=np.int8),
        missing_results=missing_results,
        forced_at_max=forced_at_max,
        actions=action_matrix,
    )
    metadata = {
        "status": "LA_CDM_ZS_PORT_SHARD_COMPLETE",
        "method": "la_cdm_zs_port",
        "fidelity": (
            "official two-agent hypothesis/decision loop and 12-test action semantics; "
            "zero-shot structured option scoring, no official SFT/GRPO adapter"
        ),
        "split": args.split,
        "split_n": len(episodes_all),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "shard_n": n,
        "model": "Qwen2.5-7B-Instruct",
        "model_config_sha256": sha256(Path(args.model_path) / "config.json"),
        "data_sha256": sha256(data_path),
        "benchmark_manifest_sha256": sha256(manifest_path),
        "max_tests": args.max_tests,
        "seed": args.seed,
        "option_token_forms": option_forms,
        "contains_clinical_text": False,
        "locked_split_used": True,
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.chmod(npz_path, 0o600)
    os.chmod(json_path, 0o600)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
