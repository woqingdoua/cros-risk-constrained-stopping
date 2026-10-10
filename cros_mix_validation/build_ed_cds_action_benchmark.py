#!/usr/bin/env python3
"""Build a protected ED/Ext-CDS sequential diagnostic-action benchmark.

The benchmark is intentionally retrospective. MIMIC-IV-Ext-CDS test snippets
are extracted from discharge notes and have no event timestamps. We therefore
model information acquisition from a logged record, not a causal simulator of
counterfactual test results. Missing test families return NO_RECORDED_RESULT.

No source identifier is written to the derived benchmark. Raw clinical text is
kept only in a mode-0600 gzip JSONL file on the protected server filesystem.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


VERSION = "ed_cds_action_v1"
TARGET_PATTERNS = {
    "appendicitis": r"\bappendicitis\b",
    "bowel_obstruction": (
        r"\b(?:small bowel obstruction|partial bowel obstruction|sbo|"
        r"large bowel obstruction|closed loop obstruction)\b"
    ),
    "biliary_disease": (
        r"\b(?:cholecystitis|cholelithiasis|choledocholithiasis|"
        r"biliary colic|gallstone disease)\b"
    ),
    "pancreatitis": r"\bpancreatitis\b",
    "diverticulitis": r"\bdiverticulitis\b",
    "gastroenteritis_colitis": r"\b(?:gastroenteritis|colitis)\b",
    "urinary_infection": (
        r"\b(?:urinary tract infection|pyelonephritis|urosepsis)\b"
    ),
    "renal_colic": (
        r"\b(?:nephrolithiasis|ureteral stone|kidney stone|renal colic)\b"
    ),
    "nonspecific_abdominal_pain": (
        r"^(?:abdominal pain|generalized abdominal pain|lower abdominal pain|"
        r"upper abdominal pain)\.?$"
    ),
}

ACTION_SPECS = {
    "recheck_vitals": {"resource_cost": 0.2, "latency_minutes": 15},
    "cbc": {"resource_cost": 1.0, "latency_minutes": 45},
    "metabolic_panel": {"resource_cost": 1.2, "latency_minutes": 45},
    "hepatic_panel": {"resource_cost": 1.2, "latency_minutes": 45},
    "lipase": {"resource_cost": 1.0, "latency_minutes": 45},
    "urinalysis": {"resource_cost": 0.8, "latency_minutes": 30},
    "ecg": {"resource_cost": 1.5, "latency_minutes": 15},
    "xray": {"resource_cost": 4.0, "latency_minutes": 45},
    "ultrasound": {"resource_cost": 6.0, "latency_minutes": 90},
    "ct": {"resource_cost": 12.0, "latency_minutes": 90},
    "mri": {"resource_cost": 20.0, "latency_minutes": 240},
    "microbiology": {"resource_cost": 3.0, "latency_minutes": 720},
}

ACTION_PATTERNS = {
    "cbc": (
        r"\b(?:cbc|complete blood count|white blood cell|wbc|hemoglobin|"
        r"hematocrit|platelet)s?\b"
    ),
    "metabolic_panel": (
        r"\b(?:bmp|cmp|basic metabolic|comprehensive metabolic|sodium|"
        r"potassium|creatinine|blood urea nitrogen|bun)\b"
    ),
    "hepatic_panel": (
        r"\b(?:liver function|hepatic function|ast|alt|bilirubin|"
        r"alkaline phosphatase|transaminase)s?\b"
    ),
    "lipase": r"\b(?:lipase|amylase)\b",
    "urinalysis": (
        r"\b(?:urinalysis|urine analysis|urine dip|leukocyte esterase|"
        r"nitrite|urine microscopy)\b"
    ),
    "ecg": r"\b(?:ecg|ekg|electrocardiogram)\b",
    "xray": r"\b(?:x-ray|xray|radiograph|chest film|cxr)\b",
    "ultrasound": r"\b(?:ultrasound|sonogram|sonography|ruq us)\b",
    "ct": r"\b(?:ct scan|computed tomography|ct abdomen|ct chest|cta)\b",
    "mri": r"\b(?:mri|magnetic resonance)\b",
    "microbiology": (
        r"\b(?:blood culture|urine culture|stool culture|wound culture|"
        r"polymerase chain reaction|pcr)\b"
    ),
}

TARGET_MASK = re.compile(
    r"\b(?:appendic\w*|cholecyst\w*|cholelith\w*|choledocholith\w*|"
    r"pancreatit\w*|diverticul\w*|gastroenterit\w*|colitis|"
    r"pyelonephrit\w*|urosepsis|nephrolith\w*|ureteral stone|"
    r"kidney stone|renal colic)\b",
    re.IGNORECASE,
)
RETROSPECTIVE_LEAKAGE = re.compile(
    r"\b(?:discharge diagnos|hospital course|post-?operative|pathology|"
    r"underwent|status post|s/p|appendectomy|cholecystectomy|colectomy|"
    r"laparoscopy|laparotomy|treated with|discharged on)\b",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_diagnosis(value: object) -> str:
    try:
        parsed = ast.literal_eval(str(value))
        text = " ".join(map(str, parsed)) if isinstance(parsed, list) else str(parsed)
    except (ValueError, SyntaxError):
        text = str(value)
    return re.sub(r"\s+", " ", text.lower()).strip()


def diagnosis_group(text: str) -> str | None:
    matches = {
        label for label, pattern in TARGET_PATTERNS.items() if re.search(pattern, text)
    }
    # Gallstone pancreatitis is a pancreatitis endpoint, not an ambiguous label.
    if "pancreatitis" in matches and "biliary_disease" in matches:
        matches.remove("biliary_disease")
    if len(matches) != 1:
        return None
    return next(iter(matches))


def mask_targets(text: object) -> str:
    return TARGET_MASK.sub("[MASKED_DIAGNOSIS]", str(text) if pd.notna(text) else "")


def split_segments(text: object) -> list[str]:
    raw = str(text) if pd.notna(text) else ""
    pieces = re.split(r"(?:\r?\n)+|(?<=[.;!?])\s+", raw)
    output = []
    seen = set()
    for piece in pieces:
        clean = re.sub(r"\s+", " ", piece).strip(" -:;\t")
        if len(clean) < 8 or RETROSPECTIVE_LEAKAGE.search(clean):
            continue
        clean = mask_targets(clean)
        clean = clean[:800]
        key = clean.lower()
        if key not in seen:
            seen.add(key)
            output.append(clean)
    return output


def action_payloads(test_text: object) -> dict[str, str]:
    buckets = {action: [] for action in ACTION_PATTERNS}
    for segment in split_segments(test_text):
        for action, pattern in ACTION_PATTERNS.items():
            if re.search(pattern, segment, flags=re.IGNORECASE):
                buckets[action].append(segment)
    return {
        action: "\n".join(f"[{action.upper()}] {segment}" for segment in segments)
        for action, segments in buckets.items()
    }


def numeric(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "unknown" if pd.isna(number) else f"{float(number):g}"


def vitals_payload(row: pd.Series) -> str:
    fields = (
        ("temperature", "temperature"),
        ("heartrate", "heart_rate"),
        ("resprate", "respiratory_rate"),
        ("o2sat", "oxygen_saturation"),
        ("sbp", "systolic_bp"),
        ("dbp", "diastolic_bp"),
    )
    return "[RECHECK_VITALS] " + "; ".join(
        f"{display}={numeric(row.get(source))}" for source, display in fields
    )


def initial_text(row: pd.Series) -> str:
    vitals = "; ".join(
        [
            f"temperature={numeric(row.get('temperature_triage'))}",
            f"heart_rate={numeric(row.get('heartrate_triage'))}",
            f"respiratory_rate={numeric(row.get('resprate_triage'))}",
            f"oxygen_saturation={numeric(row.get('o2sat_triage'))}",
            f"systolic_bp={numeric(row.get('sbp_triage'))}",
            f"diastolic_bp={numeric(row.get('dbp_triage'))}",
        ]
    )
    return "\n".join(
        [
            "[HPI] " + mask_targets(row.get("HPI", "")),
            "[CHIEF_COMPLAINT] "
            + mask_targets(row.get("chiefcomplaint_triage", "")),
            "[PATIENT_INFO] " + mask_targets(row.get("patient_info", "")),
            "[TRIAGE] " + vitals,
            f"[PAIN] {numeric(row.get('pain_triage'))}",
            f"[ACUITY] {numeric(row.get('acuity_triage'))}",
        ]
    )


def first_recheck(vitals: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    values = vitals.merge(
        stays[["stay_id", "intime", "outtime"]], on="stay_id", how="inner"
    )
    values["charttime"] = pd.to_datetime(values["charttime"], errors="coerce")
    values["intime"] = pd.to_datetime(values["intime"], errors="coerce")
    values["outtime"] = pd.to_datetime(values["outtime"], errors="coerce")
    values = values[
        values["charttime"].notna()
        & values["intime"].notna()
        & (values["charttime"] > values["intime"])
        & (values["outtime"].isna() | (values["charttime"] <= values["outtime"]))
    ].sort_values(["stay_id", "charttime"])
    return values.drop_duplicates("stay_id", keep="first")


def assign_splits(labels: pd.Series, seed: int) -> np.ndarray:
    indices = np.arange(len(labels))
    development, locked = train_test_split(
        indices,
        test_size=0.40,
        random_state=seed,
        stratify=labels,
    )
    calibration, test = train_test_split(
        locked,
        test_size=0.50,
        random_state=seed + 1,
        stratify=labels.iloc[locked],
    )
    split = np.full(len(labels), "development", dtype=object)
    split[calibration] = "locked_calibration"
    split[test] = "locked_test"
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ed-root", default="physionet.org/files/mimic-iv-ed/2.2/ed"
    )
    parser.add_argument(
        "--cds-root", default="physionet.org/files/mimic-iv-ext-cds/1.0.2"
    )
    parser.add_argument(
        "--output-dir",
        default="cros_mix_validation/data/derived/ed_cds_action_v1",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--minimum-recorded-actions", type=int, default=2)
    args = parser.parse_args()

    ed_root = Path(args.ed_root)
    cds_root = Path(args.cds_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)

    clinical_zip = cds_root / "clinical_data.csv.zip"
    with ZipFile(clinical_zip) as archive:
        with archive.open("clinical_data.csv") as handle:
            clinical = pd.read_csv(handle, low_memory=False)
    demographics = pd.read_csv(cds_root / "patient_demographics.csv")
    stays = pd.read_csv(ed_root / "edstays.csv.gz")
    triage = pd.read_csv(ed_root / "triage.csv.gz")
    triage = triage.rename(
        columns={
            column: f"{column}_triage"
            for column in triage.columns
            if column not in {"subject_id", "stay_id"}
        }
    )
    vitals = pd.read_csv(ed_root / "vitalsign.csv.gz", low_memory=False)

    frame = (
        clinical.merge(demographics, on="stay_id", validate="one_to_one")
        .merge(stays, on="stay_id", validate="one_to_one")
        .merge(
            triage,
            on=["subject_id", "stay_id"],
            validate="one_to_one",
        )
    )
    frame["normalized_primary_diagnosis"] = frame["primary_diagnosis"].map(
        normalize_diagnosis
    )
    frame["label"] = frame["normalized_primary_diagnosis"].map(diagnosis_group)
    frame = frame[frame["label"].notna()].copy()
    frame["intime_parsed"] = pd.to_datetime(frame["intime"], errors="coerce")
    frame = frame.sort_values(["subject_id", "intime_parsed", "stay_id"])
    frame = frame.drop_duplicates("subject_id", keep="first").reset_index(drop=True)

    recheck = first_recheck(vitals, stays)
    recheck_by_stay = recheck.set_index("stay_id")
    episodes = []
    for _, row in frame.iterrows():
        payloads = action_payloads(row["tests"])
        stay_id = int(row["stay_id"])
        if stay_id in recheck_by_stay.index:
            payloads["recheck_vitals"] = vitals_payload(recheck_by_stay.loc[stay_id])
        else:
            payloads["recheck_vitals"] = ""
        payloads = {action: payloads.get(action, "") for action in ACTION_SPECS}
        recorded = sum(bool(text.strip()) for text in payloads.values())
        if recorded < args.minimum_recorded_actions:
            continue
        episodes.append(
            {
                "initial_text": initial_text(row),
                "action_payloads": payloads,
                "label": str(row["label"]),
                "recorded_action_count": recorded,
            }
        )

    episodes.sort(key=lambda episode: (episode["label"], episode["initial_text"]))
    labels = pd.Series([episode["label"] for episode in episodes])
    split = assign_splits(labels, args.seed)
    for row_index, (episode, split_name) in enumerate(zip(episodes, split)):
        episode["row_index"] = row_index
        episode["split"] = str(split_name)

    episode_path = output_dir / "episodes.jsonl.gz"
    with gzip.open(episode_path, "wt", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    index = pd.DataFrame(
        {
            "row_index": np.arange(len(episodes)),
            "label": labels,
            "split": split,
            "recorded_action_count": [
                episode["recorded_action_count"] for episode in episodes
            ],
        }
    )
    for action in ACTION_SPECS:
        index[f"has_{action}"] = [
            bool(episode["action_payloads"][action].strip()) for episode in episodes
        ]
    index.to_csv(output_dir / "episode_index.csv.gz", index=False, compression="gzip")

    label_counts = (
        index.groupby(["split", "label"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    label_counts.to_csv(output_dir / "label_counts.csv", index=False)
    coverage = []
    for action in ACTION_SPECS:
        for label, group in index.groupby("label", observed=True):
            coverage.append(
                {
                    "action": action,
                    "label": label,
                    "n": len(group),
                    "recorded_n": int(group[f"has_{action}"].sum()),
                    "recorded_fraction": float(group[f"has_{action}"].mean()),
                }
            )
        coverage.append(
            {
                "action": action,
                "label": "__all__",
                "n": len(index),
                "recorded_n": int(index[f"has_{action}"].sum()),
                "recorded_fraction": float(index[f"has_{action}"].mean()),
            }
        )
    pd.DataFrame(coverage).to_csv(output_dir / "action_coverage.csv", index=False)

    manifest = {
        "status": "ED_CDS_ACTION_BENCHMARK_BUILT",
        "version": VERSION,
        "seed": args.seed,
        "episode_n": len(episodes),
        "unique_patient_policy": "earliest qualifying ED stay per subject",
        "labels": sorted(labels.unique()),
        "split_counts": index["split"].value_counts().sort_index().to_dict(),
        "minimum_recorded_actions": args.minimum_recorded_actions,
        "actions": ACTION_SPECS,
        "diagnosis_source": "Ext-CDS primary_diagnosis extracted from discharge note",
        "initial_state": "masked HPI + ED chief complaint + demographics + triage",
        "test_source": "strictly filtered/masked Ext-CDS discharge-note test snippets",
        "chronology": (
            "recheck_vitals is timestamped; all Ext-CDS test-family snippets are "
            "untimestamped and define retrospective logged information acquisition"
        ),
        "missing_action_semantics": (
            "ordering an unrecorded family returns NO_RECORDED_RESULT and still incurs cost"
        ),
        "cost_semantics": (
            "precommitted relative resource units and approximate turnaround minutes; "
            "not hospital charges"
        ),
        "privacy": (
            "derived files contain no stay_id, hadm_id, or subject_id; clinical text "
            "is restricted to mode-0600 episodes.jsonl.gz"
        ),
        "source_sha256": {
            "clinical_data_zip": sha256(clinical_zip),
            "edstays": sha256(ed_root / "edstays.csv.gz"),
            "triage": sha256(ed_root / "triage.csv.gz"),
            "vitalsign": sha256(ed_root / "vitalsign.csv.gz"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for path in output_dir.iterdir():
        os.chmod(path, 0o600)
    os.chmod(output_dir, 0o700)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
