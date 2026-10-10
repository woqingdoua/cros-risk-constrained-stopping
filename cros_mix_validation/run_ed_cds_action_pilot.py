#!/usr/bin/env python3
"""Development-only pilot for the ED/Ext-CDS diagnostic action benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import beta, binom
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

from ed_cds_action_env import NO_RECORDED_RESULT


ALPHAS = (0.15, 0.20, 0.25, 0.30)
DELTA = 0.05
ROTATIONS = 3
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass
class Policy:
    name: str
    family: str
    errors: np.ndarray
    costs: np.ndarray
    latencies: np.ndarray
    tests: np.ndarray
    predictions: np.ndarray


def load_episodes(path: Path) -> list[dict[str, object]]:
    episodes = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            if episode.get("split") != "development":
                raise AssertionError("development pilot opened a non-development episode")
            episodes.append(episode)
    return episodes


def observation(episode: dict[str, object], action: str) -> str:
    payload = str(episode["action_payloads"].get(action, "")).strip()
    return payload if payload else f"[{action.upper()}] {NO_RECORDED_RESULT}"


def document(episode: dict[str, object], actions: list[str] | tuple[str, ...]) -> str:
    chunks = [str(episode["initial_text"])]
    chunks.extend(observation(episode, action) for action in actions)
    return "\n".join(chunks)


def augmented_training_documents(
    episodes: list[dict[str, object]],
    indices: np.ndarray,
    actions: list[str],
    seed: int,
) -> tuple[list[str], list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    documents: list[str] = []
    labels: list[str] = []
    weights: list[float] = []
    for index in indices:
        episode = episodes[int(index)]
        variants: list[tuple[str, ...]] = [(), tuple(actions)]
        variants.extend((action,) for action in actions)
        for fraction in (0.25, 0.50, 0.75):
            size = max(1, int(round(len(actions) * fraction)))
            variants.append(tuple(rng.choice(actions, size=size, replace=False)))
        unique_variants = list(dict.fromkeys(variants))
        episode_weight = 1.0 / len(unique_variants)
        for variant in unique_variants:
            documents.append(document(episode, variant))
            labels.append(str(episode["label"]))
            weights.append(episode_weight)
    return documents, labels, np.asarray(weights, dtype=float)


def predict_documents(model, vectorizer, documents: list[str]) -> np.ndarray:
    return model.predict_proba(vectorizer.transform(documents)).astype(np.float32)


def learned_action_order(
    episodes: list[dict[str, object]],
    indices: np.ndarray,
    actions: list[str],
    costs: dict[str, float],
    model,
    vectorizer,
    class_to_index: dict[str, int],
) -> tuple[list[str], dict[str, float]]:
    baseline = predict_documents(
        model, vectorizer, [document(episodes[int(i)], ()) for i in indices]
    )
    labels = np.asarray(
        [class_to_index[str(episodes[int(i)]["label"])] for i in indices]
    )
    baseline_error = (baseline.argmax(axis=1) != labels).astype(float)
    scores = {}
    for action in actions:
        after = predict_documents(
            model,
            vectorizer,
            [document(episodes[int(i)], (action,)) for i in indices],
        )
        after_error = (after.argmax(axis=1) != labels).astype(float)
        improvement = float((baseline_error - after_error).mean())
        scores[action] = improvement / max(costs[action], 1e-9)
    order = sorted(actions, key=lambda action: (-scores[action], costs[action], action))
    return order, scores


def canonical_sequences(actions: list[str], costs: dict[str, float]) -> dict[str, list[str]]:
    clinical = [
        "recheck_vitals",
        "cbc",
        "metabolic_panel",
        "hepatic_panel",
        "lipase",
        "urinalysis",
        "ct",
        "ultrasound",
        "xray",
        "ecg",
        "mri",
        "microbiology",
    ]
    imaging_early = [
        "recheck_vitals",
        "cbc",
        "ct",
        "ultrasound",
        "metabolic_panel",
        "hepatic_panel",
        "lipase",
        "urinalysis",
        "xray",
        "mri",
        "ecg",
        "microbiology",
    ]
    labs_first = [
        "recheck_vitals",
        "cbc",
        "metabolic_panel",
        "hepatic_panel",
        "lipase",
        "urinalysis",
        "microbiology",
        "ecg",
        "xray",
        "ultrasound",
        "ct",
        "mri",
    ]
    expected = set(actions)
    for sequence in (clinical, imaging_early, labs_first):
        if set(sequence) != expected:
            raise AssertionError("canonical action sequence does not match manifest")
    return {
        "clinical": clinical,
        "imaging_early": imaging_early,
        "labs_first": labs_first,
        "cost_ascending": sorted(actions, key=lambda action: (costs[action], action)),
    }


def exact_null_pvalue(errors: int, n: int, alpha: float) -> float:
    return float(binom.cdf(errors, n, alpha))


def exact_upper_bound(errors: int, n: int, delta: float) -> float:
    if errors >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


def screen_candidates(
    policies: list[Policy], selection: np.ndarray, alpha: float, cap: int = 24
) -> list[int]:
    risks = np.asarray([policy.errors[selection].mean() for policy in policies])
    costs = np.asarray([policy.costs[selection].mean() for policy in policies])
    selected = set()
    for index in range(len(policies)):
        dominated = np.any(
            (risks <= risks[index] + 1e-12)
            & (costs <= costs[index] + 1e-12)
            & ((risks < risks[index] - 1e-12) | (costs < costs[index] - 1e-12))
        )
        if not dominated:
            selected.add(index)
    for target in np.linspace(max(0.02, alpha * 0.45), alpha, 8):
        feasible = np.flatnonzero(risks <= target)
        if len(feasible):
            selected.add(int(feasible[np.argmin(costs[feasible])]))
    selected.add(int(np.argmin(risks)))
    selected.add(int(np.argmin(costs)))
    ordered = sorted(selected, key=lambda i: (costs[i], risks[i], policies[i].name))
    if len(ordered) <= cap:
        return ordered
    positions = np.unique(np.linspace(0, len(ordered) - 1, cap).round().astype(int))
    return [ordered[position] for position in positions]


def choose_cheapest(
    policies: list[Policy], indices: list[int], selection: np.ndarray
) -> int | None:
    if not indices:
        return None
    return min(
        indices,
        key=lambda index: (
            policies[index].costs[selection].mean(),
            policies[index].errors[selection].mean(),
            policies[index].name,
        ),
    )


def evaluate_deterministic(policy: Policy, evaluation: np.ndarray) -> dict[str, float]:
    return {
        "risk": float(policy.errors[evaluation].mean()),
        "cost": float(policy.costs[evaluation].mean()),
        "latency": float(policy.latencies[evaluation].mean()),
        "tests": float(policy.tests[evaluation].mean()),
        "support_size": 1,
    }


def evaluate_mixture(
    policies: list[Policy], indices: list[int], weights: np.ndarray, evaluation: np.ndarray
) -> dict[str, float]:
    return {
        "risk": float(
            sum(
                weight * policies[index].errors[evaluation].mean()
                for index, weight in zip(indices, weights)
            )
        ),
        "cost": float(
            sum(
                weight * policies[index].costs[evaluation].mean()
                for index, weight in zip(indices, weights)
            )
        ),
        "latency": float(
            sum(
                weight * policies[index].latencies[evaluation].mean()
                for index, weight in zip(indices, weights)
            )
        ),
        "tests": float(
            sum(
                weight * policies[index].tests[evaluation].mean()
                for index, weight in zip(indices, weights)
            )
        ),
        "support_size": int(np.sum(weights > 1e-8)),
    }


def method_evaluation(
    policies: list[Policy],
    selection: np.ndarray,
    calibration: np.ndarray,
    evaluation: np.ndarray,
    alpha: float,
    per_rotation_delta: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    hpi = next(index for index, policy in enumerate(policies) if policy.name == "hpi_only")
    full = next(index for index, policy in enumerate(policies) if policy.name == "full_workup")
    for method, index in (("hpi_only", hpi), ("full_workup", full)):
        rows.append({"method": method, "available": True, **evaluate_deterministic(policies[index], evaluation)})

    empirical_feasible = [
        index
        for index, policy in enumerate(policies)
        if policy.errors[selection].mean() <= alpha
    ]
    empirical = choose_cheapest(policies, empirical_feasible, selection)
    rows.append(
        {
            "method": "empirical_erm",
            "available": empirical is not None,
            **(
                evaluate_deterministic(policies[empirical], evaluation)
                if empirical is not None
                else {"risk": None, "cost": None, "latency": None, "tests": None, "support_size": 0}
            ),
        }
    )

    screened = screen_candidates(policies, selection, alpha)
    errors = np.asarray([int(policies[i].errors[calibration].sum()) for i in screened])
    n = len(calibration)
    pvalues = np.asarray([exact_null_pvalue(int(error), n, alpha) for error in errors])

    fixed_order = sorted(
        range(len(screened)),
        key=lambda j: (
            policies[screened[j]].errors[selection].mean(),
            -policies[screened[j]].costs[selection].mean(),
            policies[screened[j]].name,
        ),
    )
    rejected = []
    for position in fixed_order:
        if pvalues[position] <= per_rotation_delta:
            rejected.append(screened[position])
        else:
            break
    ltt = choose_cheapest(policies, rejected, selection)
    rows.append(
        {
            "method": "ltt_fixed_sequence",
            "available": ltt is not None,
            **(
                evaluate_deterministic(policies[ltt], evaluation)
                if ltt is not None
                else {"risk": None, "cost": None, "latency": None, "tests": None, "support_size": 0}
            ),
        }
    )

    order = np.argsort(pvalues)
    holm_positions = []
    for rank, position in enumerate(order):
        if pvalues[position] <= per_rotation_delta / (len(screened) - rank):
            holm_positions.append(int(position))
        else:
            break
    holm = choose_cheapest(
        policies, [screened[position] for position in holm_positions], selection
    )
    rows.append(
        {
            "method": "holm_stepdown",
            "available": holm is not None,
            **(
                evaluate_deterministic(policies[holm], evaluation)
                if holm is not None
                else {"risk": None, "cost": None, "latency": None, "tests": None, "support_size": 0}
            ),
        }
    )

    bonf_positions = np.flatnonzero(
        pvalues <= per_rotation_delta / len(screened)
    ).tolist()
    bonferroni = choose_cheapest(
        policies, [screened[position] for position in bonf_positions], selection
    )
    rows.append(
        {
            "method": "bonferroni",
            "available": bonferroni is not None,
            **(
                evaluate_deterministic(policies[bonferroni], evaluation)
                if bonferroni is not None
                else {"risk": None, "cost": None, "latency": None, "tests": None, "support_size": 0}
            ),
        }
    )

    simultaneous_delta = per_rotation_delta / len(screened)
    upper = np.asarray(
        [exact_upper_bound(int(error), n, simultaneous_delta) for error in errors]
    )
    certified_positions = np.flatnonzero(upper <= alpha).tolist()
    cros_det = choose_cheapest(
        policies, [screened[position] for position in certified_positions], selection
    )
    rows.append(
        {
            "method": "cros_det",
            "available": cros_det is not None,
            **(
                evaluate_deterministic(policies[cros_det], evaluation)
                if cros_det is not None
                else {"risk": None, "cost": None, "latency": None, "tests": None, "support_size": 0}
            ),
        }
    )

    selection_costs = np.asarray(
        [policies[index].costs[selection].mean() for index in screened]
    )
    solution = linprog(
        c=selection_costs,
        A_ub=upper[None, :],
        b_ub=np.asarray([alpha]),
        A_eq=np.ones((1, len(screened))),
        b_eq=np.ones(1),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if solution.success:
        weights = np.where(solution.x < 1e-9, 0.0, solution.x)
        weights /= weights.sum()
        mix_metrics = evaluate_mixture(policies, screened, weights, evaluation)
    else:
        mix_metrics = {"risk": None, "cost": None, "latency": None, "tests": None, "support_size": 0}
    rows.append(
        {"method": "cros_mix", "available": bool(solution.success), **mix_metrics}
    )
    for row in rows:
        row["screened_candidate_n"] = len(screened)
        row["transfer"] = bool(row["available"] and row["risk"] <= alpha)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development-path",
        default=(
            "cros_mix_validation/data/derived/ed_cds_action_v1/"
            "development_episodes.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--manifest-path",
        default="cros_mix_validation/data/derived/ed_cds_action_v1/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="cros_mix_validation/results/ed_cds_action_pilot_v1",
    )
    parser.add_argument("--model-folds", type=int, default=5)
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=20260910)
    parser.add_argument("--max-features", type=int, default=50000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    episodes = load_episodes(Path(args.development_path))
    manifest = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
    actions = list(manifest["actions"])
    costs = {
        action: float(manifest["actions"][action]["resource_cost"])
        for action in actions
    }
    latencies = {
        action: float(manifest["actions"][action]["latency_minutes"])
        for action in actions
    }
    labels_text = np.asarray([str(episode["label"]) for episode in episodes])
    classes = sorted(set(labels_text))
    class_to_index = {label: index for index, label in enumerate(classes)}
    labels = np.asarray([class_to_index[label] for label in labels_text], dtype=int)
    n = len(episodes)
    if n != 1100:
        raise AssertionError(f"unexpected development size {n}")

    sequence_names = ["clinical", "imaging_early", "labs_first", "cost_ascending", "learned_voi"]
    prefix_probabilities = {
        sequence: np.zeros((len(actions) + 1, n, len(classes)), dtype=np.float32)
        for sequence in sequence_names
    }
    prefix_costs = {
        sequence: np.zeros((len(actions) + 1, n), dtype=np.float32)
        for sequence in sequence_names
    }
    prefix_latencies = {
        sequence: np.zeros((len(actions) + 1, n), dtype=np.float32)
        for sequence in sequence_names
    }
    learned_orders = []
    fold_assignments = np.full(n, -1, dtype=int)
    splitter = StratifiedKFold(
        n_splits=args.model_folds, shuffle=True, random_state=args.seed_start
    )

    for fold, (train_indices, evaluation_indices) in enumerate(
        splitter.split(np.zeros(n), labels)
    ):
        fold_assignments[evaluation_indices] = fold
        train_documents, train_labels, train_weights = augmented_training_documents(
            episodes, train_indices, actions, args.seed_start + fold
        )
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.997,
            max_features=args.max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        train_matrix = vectorizer.fit_transform(train_documents)
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=2e-6,
            max_iter=2000,
            tol=1e-4,
            class_weight="balanced",
            random_state=args.seed_start + fold,
        )
        model.fit(train_matrix, train_labels, sample_weight=train_weights)
        learned_order, learned_scores = learned_action_order(
            episodes,
            train_indices,
            actions,
            costs,
            model,
            vectorizer,
            class_to_index,
        )
        learned_orders.append(
            {"fold": fold, "order": learned_order, "score_per_cost": learned_scores}
        )
        sequences = canonical_sequences(actions, costs)
        sequences["learned_voi"] = learned_order
        evaluation_episodes = [episodes[int(index)] for index in evaluation_indices]
        for sequence_name, sequence in sequences.items():
            acquired: list[str] = []
            cumulative_cost = 0.0
            cumulative_latency = 0.0
            for horizon in range(len(actions) + 1):
                documents = [document(episode, acquired) for episode in evaluation_episodes]
                probability = predict_documents(model, vectorizer, documents)
                class_positions = np.asarray(
                    [class_to_index[str(label)] for label in model.classes_], dtype=int
                )
                prefix_probabilities[sequence_name][horizon, evaluation_indices[:, None], class_positions] = probability
                prefix_costs[sequence_name][horizon, evaluation_indices] = cumulative_cost
                prefix_latencies[sequence_name][horizon, evaluation_indices] = cumulative_latency
                if horizon < len(actions):
                    action = sequence[horizon]
                    acquired.append(action)
                    cumulative_cost += costs[action]
                    cumulative_latency += latencies[action]
        print(
            f"model_fold={fold} train={len(train_indices)} evaluation={len(evaluation_indices)} "
            f"features={train_matrix.shape[1]} learned_order={','.join(learned_order)}",
            flush=True,
        )

    if np.any(fold_assignments < 0):
        raise AssertionError("OOF model coverage is incomplete")

    policies: list[Policy] = []
    for sequence_name in sequence_names:
        probabilities = prefix_probabilities[sequence_name]
        for horizon in range(len(actions) + 1):
            predictions = probabilities[horizon].argmax(axis=1)
            name = f"fixed:{sequence_name}:k={horizon}"
            if sequence_name == "clinical" and horizon == 0:
                name = "hpi_only"
            if sequence_name == "clinical" and horizon == len(actions):
                name = "full_workup"
            # Avoid duplicate HPI/full policies from the other action orders.
            if horizon in {0, len(actions)} and sequence_name != "clinical":
                continue
            policies.append(
                Policy(
                    name=name,
                    family=f"fixed_{sequence_name}",
                    errors=(predictions != labels).astype(float),
                    costs=prefix_costs[sequence_name][horizon].astype(float),
                    latencies=prefix_latencies[sequence_name][horizon].astype(float),
                    tests=np.full(n, horizon, dtype=float),
                    predictions=predictions,
                )
            )
        for threshold in THRESHOLDS:
            chosen_horizon = np.full(n, len(actions), dtype=int)
            for episode in range(n):
                for horizon in range(len(actions)):
                    if probabilities[horizon, episode].max() >= threshold:
                        chosen_horizon[episode] = horizon
                        break
            predictions = probabilities[
                chosen_horizon, np.arange(n)
            ].argmax(axis=1)
            policies.append(
                Policy(
                    name=f"confidence:{sequence_name}:tau={threshold:g}",
                    family=f"confidence_{sequence_name}",
                    errors=(predictions != labels).astype(float),
                    costs=prefix_costs[sequence_name][chosen_horizon, np.arange(n)].astype(float),
                    latencies=prefix_latencies[sequence_name][chosen_horizon, np.arange(n)].astype(float),
                    tests=chosen_horizon.astype(float),
                    predictions=predictions,
                )
            )

    state_rows = []
    clinical = prefix_probabilities["clinical"]
    for horizon in range(len(actions) + 1):
        predictions = clinical[horizon].argmax(axis=1)
        state_rows.append(
            {
                "sequence": "clinical",
                "horizon": horizon,
                "risk": float(np.mean(predictions != labels)),
                "accuracy": float(np.mean(predictions == labels)),
                "resource_cost": float(prefix_costs["clinical"][horizon].mean()),
                "latency_minutes": float(prefix_latencies["clinical"][horizon].mean()),
            }
        )
    pd.DataFrame(state_rows).to_csv(output_dir / "clinical_sequence_states.csv", index=False)

    rotation_rows = []
    seed_rows = []
    per_rotation_delta = DELTA / ROTATIONS
    for alpha in ALPHAS:
        for split_seed in range(args.seed_start, args.seed_start + args.num_seeds):
            permutation = np.random.default_rng(split_seed).permutation(n)
            folds = np.full(n, -1, dtype=int)
            for position, index in enumerate(permutation):
                folds[index] = position % ROTATIONS
            start = len(rotation_rows)
            for rotation in range(ROTATIONS):
                selection = np.flatnonzero(folds == rotation)
                calibration = np.flatnonzero(folds == (rotation + 1) % ROTATIONS)
                evaluation = np.flatnonzero(folds == (rotation + 2) % ROTATIONS)
                results = method_evaluation(
                    policies,
                    selection,
                    calibration,
                    evaluation,
                    alpha,
                    per_rotation_delta,
                )
                for result in results:
                    rotation_rows.append(
                        {
                            "alpha": alpha,
                            "split_seed": split_seed,
                            "rotation": rotation,
                            "selection_n": len(selection),
                            "calibration_n": len(calibration),
                            "evaluation_n": len(evaluation),
                            **result,
                        }
                    )
            seed_frame = pd.DataFrame(rotation_rows[start:])
            for method, group in seed_frame.groupby("method", sort=False):
                available = group["available"].astype(bool)
                complete = bool(available.all())
                seed_rows.append(
                    {
                        "alpha": alpha,
                        "split_seed": split_seed,
                        "method": method,
                        "available_rotations": int(available.sum()),
                        "all_rotations_available": complete,
                        "aggregate_risk": float(group["risk"].mean()) if complete else None,
                        "aggregate_cost": float(group["cost"].mean()) if complete else None,
                        "aggregate_latency": float(group["latency"].mean()) if complete else None,
                        "aggregate_tests": float(group["tests"].mean()) if complete else None,
                        "aggregate_transfer": bool(group["transfer"].all()) if complete else False,
                    }
                )

    rotations = pd.DataFrame(rotation_rows)
    seeds = pd.DataFrame(seed_rows)
    summary_rows = []
    for (alpha, method), group in seeds.groupby(["alpha", "method"], sort=False):
        valid = group.dropna(subset=["aggregate_risk", "aggregate_cost"])
        rotation_group = rotations[
            (rotations["alpha"] == alpha) & (rotations["method"] == method)
        ]
        available = rotation_group["available"].astype(bool)
        summary_rows.append(
            {
                "alpha": alpha,
                "method": method,
                "available_rotations": int(available.sum()),
                "transfer_among_available": float(
                    rotation_group.loc[available, "transfer"].astype(bool).mean()
                ) if available.any() else None,
                "complete_seed_n": int(len(valid)),
                "risk_median": float(valid["aggregate_risk"].median()) if len(valid) else None,
                "cost_median": float(valid["aggregate_cost"].median()) if len(valid) else None,
                "latency_median": float(valid["aggregate_latency"].median()) if len(valid) else None,
                "tests_median": float(valid["aggregate_tests"].median()) if len(valid) else None,
            }
        )
    summary = pd.DataFrame(summary_rows)
    rotations.to_csv(output_dir / "rotation_results.csv", index=False)
    seeds.to_csv(output_dir / "seed_results.csv", index=False)
    summary.to_csv(output_dir / "method_summary.csv", index=False)
    (output_dir / "learned_action_orders.json").write_text(
        json.dumps(learned_orders, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "anonymous_oof_policy_arrays.npz",
        labels=labels.astype(np.int8),
        fold_assignments=fold_assignments.astype(np.int8),
        policy_names=np.asarray([policy.name for policy in policies]),
        errors=np.stack([policy.errors for policy in policies]).astype(np.float32),
        costs=np.stack([policy.costs for policy in policies]).astype(np.float32),
        latencies=np.stack([policy.latencies for policy in policies]).astype(np.float32),
        tests=np.stack([policy.tests for policy in policies]).astype(np.float32),
    )
    metadata = {
        "status": "ED_CDS_ACTION_DEVELOPMENT_PILOT_COMPLETE",
        "development_n": n,
        "classes": classes,
        "actions": actions,
        "candidate_policy_n": len(policies),
        "model": "OOF word TF-IDF + class-balanced SGD log-loss",
        "model_folds": args.model_folds,
        "alphas": list(ALPHAS),
        "delta": DELTA,
        "num_seeds": args.num_seeds,
        "locked_calibration_or_test_loaded": False,
        "causal_claim": False,
        "environment_scope": (
            "retrospective logged information acquisition; missing actions have no "
            "counterfactual result and return NO_RECORDED_RESULT"
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# ED/Ext-CDS realistic-action development pilot",
        "",
        f"- development patients: {n}",
        f"- diagnosis classes: {len(classes)}",
        f"- orderable actions: {len(actions)} + STOP",
        f"- candidate stopping/action policies: {len(policies)}",
        "- locked calibration/test loaded: no",
        "- causal/counterfactual simulator claim: no",
        "",
        "## Results",
        "",
        summary.to_markdown(index=False),
        "",
        "Ext-CDS test snippets are untimestamped discharge-note extracts. This pilot",
        "tests action-aware retrospective information acquisition, not the outcome of",
        "ordering an unobserved test in clinical practice.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    for path in output_dir.iterdir():
        os.chmod(path, 0o600)
    os.chmod(output_dir, 0o700)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
