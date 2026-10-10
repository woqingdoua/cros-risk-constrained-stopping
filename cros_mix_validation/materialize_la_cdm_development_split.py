#!/usr/bin/env python3
"""Create restricted train/validation files from ED/Ext-CDS development only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=(
            "cros_mix_validation/data/derived/ed_cds_action_v1/"
            "development_episodes.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="cros_mix_validation/data/derived/la_cdm_train_v1",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    source = Path(args.source)
    episodes = []
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            if episode.get("split") != "development":
                raise AssertionError("source contains a non-development episode")
            episodes.append(episode)
    labels = np.asarray([str(episode["label"]) for episode in episodes])
    rng = np.random.default_rng(args.seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        validation_n = max(1, int(round(len(indices) * args.validation_fraction)))
        validation_indices.extend(indices[:validation_n].tolist())
        train_indices.extend(indices[validation_n:].tolist())
    train_indices.sort()
    validation_indices.sort()
    if set(train_indices) & set(validation_indices):
        raise AssertionError("train/validation overlap")
    if sorted(train_indices + validation_indices) != list(range(len(episodes))):
        raise AssertionError("train/validation do not partition development")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    paths = {
        "train": output_root / "train_episodes.jsonl.gz",
        "validation": output_root / "validation_episodes.jsonl.gz",
    }
    for split, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
    ):
        with gzip.open(paths[split], "wt", encoding="utf-8") as handle:
            for index in indices:
                record = dict(episodes[index])
                record["la_cdm_split"] = split
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        os.chmod(paths[split], 0o600)
    manifest = {
        "status": "LA_CDM_DEVELOPMENT_TRAIN_VALIDATION_SPLIT_FROZEN",
        "version": "la_cdm_train_v1",
        "source_split": "development only",
        "locked_calibration_or_test_loaded": False,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "counts": {
            "source": len(episodes),
            "train": len(train_indices),
            "validation": len(validation_indices),
        },
        "label_counts": {
            split: dict(
                sorted(
                    Counter(str(episodes[index]["label"]) for index in indices).items()
                )
            )
            for split, indices in (
                ("train", train_indices),
                ("validation", validation_indices),
            )
        },
        "sha256": {
            "source": sha256(source),
            "train": sha256(paths["train"]),
            "validation": sha256(paths["validation"]),
        },
        "privacy": "clinical text remains mode-0600 on the server",
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
