"""Restricted ED/Ext-CDS JSONL adapter for the official LA-CDM trainer."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from torch.utils.data import Dataset


class EDCDSDataset(Dataset):
    def __init__(
        self,
        data_file: str,
        test_name_to_action_key: dict[str, str],
        small_sample: bool = False,
    ) -> None:
        self.records: list[dict[str, object]] = []
        with gzip.open(Path(data_file), "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("split") != "development":
                    raise AssertionError("LA-CDM training adapter may only read development")
                self.records.append(record)
        if small_sample:
            self.records = self.records[:2]
        self.test_name_to_action_key = dict(test_name_to_action_key)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        payloads = record["action_payloads"]
        test_results = {}
        for test_name, action_key in self.test_name_to_action_key.items():
            payload = str(payloads.get(action_key, "")).strip()
            test_results[test_name] = payload + "\n" if payload else "not available.\n"
        return {
            "patient_id": f"development-row-{int(record['row_index'])}",
            "patient_history": str(record["initial_text"]),
            "condition": str(record["label"]),
            "test_results": test_results,
        }
