#!/usr/bin/env python3
"""Score one frozen ED/Ext-CDS action state with development-only OOF models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

from run_ed_cds_action_pilot import document, load_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument(
        "--state-manifest",
        default="cros_mix_validation/results/ed_cds_sequential_v1/state_manifest.json",
    )
    parser.add_argument(
        "--development-path",
        default=(
            "cros_mix_validation/data/derived/ed_cds_action_v1/"
            "development_episodes.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="cros_mix_validation/results/ed_cds_sequential_v1/states",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--word-features", type=int, default=25000)
    parser.add_argument("--char-features", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    state_manifest = json.loads(Path(args.state_manifest).read_text(encoding="utf-8"))
    states = state_manifest["states"]
    if not 0 <= args.state_index < len(states):
        raise ValueError("invalid state index")
    state = states[args.state_index]
    acquired = tuple(state["acquired"])
    episodes = load_episodes(Path(args.development_path))
    labels_text = np.asarray([str(episode["label"]) for episode in episodes])
    classes = sorted(set(labels_text))
    class_to_index = {label: index for index, label in enumerate(classes)}
    labels = np.asarray([class_to_index[label] for label in labels_text], dtype=int)
    probabilities = np.zeros((len(labels), len(classes)), dtype=np.float32)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    summary_path = output_root / f"state_{args.state_index:03d}.json"
    prediction_path = output_root / f"state_{args.state_index:03d}.npz"
    if summary_path.exists() or prediction_path.exists():
        raise FileExistsError("refusing to overwrite a state result")

    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    for fold, (train_indices, evaluation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        train_documents = [document(episodes[int(i)], acquired) for i in train_indices]
        evaluation_documents = [
            document(episodes[int(i)], acquired) for i in evaluation_indices
        ]
        word = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.997,
            max_features=args.word_features,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            min_df=3,
            max_features=args.char_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        train_matrix = sp.hstack(
            (word.fit_transform(train_documents), char.fit_transform(train_documents)),
            format="csr",
        )
        evaluation_matrix = sp.hstack(
            (word.transform(evaluation_documents), char.transform(evaluation_documents)),
            format="csr",
        )
        binary = LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=300,
            random_state=args.seed + args.state_index * 10 + fold,
            solver="liblinear",
        )
        model = OneVsRestClassifier(binary, n_jobs=1)
        model.fit(train_matrix, labels[train_indices])
        probabilities[evaluation_indices] = model.predict_proba(evaluation_matrix)
        print(
            f"state={args.state_index} fold={fold} features={train_matrix.shape[1]}",
            flush=True,
        )

    predictions = probabilities.argmax(axis=1)
    error = predictions != labels
    summary = {
        "status": "ED_CDS_SEQUENTIAL_STATE_COMPLETE",
        "state_index": args.state_index,
        "acquired": list(acquired),
        "development_n": len(labels),
        "risk": float(error.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "locked_calibration_or_test_loaded": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        prediction_path,
        labels=labels.astype(np.int8),
        probabilities=probabilities.astype(np.float32),
        predictions=predictions.astype(np.int8),
    )
    os.chmod(summary_path, 0o600)
    os.chmod(prediction_path, 0o600)
    os.chmod(output_root, 0o700)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
