#!/usr/bin/env python3
"""Auditable offline information-acquisition environment for ED/Ext-CDS."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any


NO_RECORDED_RESULT = "[NO_RECORDED_RESULT]"


@dataclass
class EDCDSActionEnv:
    episode: dict[str, Any]
    action_specs: dict[str, dict[str, float]]
    acquired: list[str] = field(default_factory=list)
    resource_cost: float = 0.0
    latency_minutes: float = 0.0
    terminated: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> str:
        self.acquired.clear()
        self.resource_cost = 0.0
        self.latency_minutes = 0.0
        self.terminated = False
        self.trace.clear()
        return str(self.episode["initial_text"])

    @property
    def available_actions(self) -> tuple[str, ...]:
        if self.terminated:
            return ()
        remaining = [action for action in self.action_specs if action not in self.acquired]
        return tuple(remaining + ["stop"])

    def order(self, action: str) -> str:
        if self.terminated:
            raise RuntimeError("episode already terminated")
        if action not in self.action_specs:
            raise ValueError(f"unknown action {action}")
        if action in self.acquired:
            raise ValueError(f"action already acquired: {action}")
        payload = str(self.episode["action_payloads"].get(action, "")).strip()
        observed = payload if payload else NO_RECORDED_RESULT
        spec = self.action_specs[action]
        self.acquired.append(action)
        self.resource_cost += float(spec["resource_cost"])
        self.latency_minutes += float(spec["latency_minutes"])
        self.trace.append(
            {
                "step": len(self.trace),
                "action": action,
                "recorded_result": bool(payload),
                "resource_cost": float(spec["resource_cost"]),
                "latency_minutes": float(spec["latency_minutes"]),
                "observation_sha256": hashlib.sha256(
                    observed.encode("utf-8")
                ).hexdigest(),
            }
        )
        return observed

    def stop(self, prediction: str) -> dict[str, Any]:
        if self.terminated:
            raise RuntimeError("episode already terminated")
        self.terminated = True
        correct = str(prediction) == str(self.episode["label"])
        event = {
            "step": len(self.trace),
            "action": "stop",
            "prediction": str(prediction),
            "correct": correct,
            "cumulative_resource_cost": self.resource_cost,
            "cumulative_latency_minutes": self.latency_minutes,
        }
        self.trace.append(event)
        return event

    def document(self) -> str:
        chunks = [str(self.episode["initial_text"])]
        for action in self.acquired:
            payload = str(self.episode["action_payloads"].get(action, "")).strip()
            chunks.append(payload if payload else f"[{action.upper()}] {NO_RECORDED_RESULT}")
        return "\n".join(chunks)
