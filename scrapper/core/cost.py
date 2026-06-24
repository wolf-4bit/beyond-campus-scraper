"""Per-run LLM cost & token tracking.

A single module-level `tracker` accumulates one record per LLM call. It is
reset at the start of each pipeline run and rendered into a summary table at
the end. Thread-safe so it stays correct if LangGraph runs tasks in threads.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class CallRecord:
    model: str
    stage: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


@dataclass
class CostTracker:
    records: list[CallRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        model: str,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None:
        with self._lock:
            self.records.append(
                CallRecord(model, stage, prompt_tokens, completion_tokens, cost)
            )

    def reset(self) -> None:
        with self._lock:
            self.records.clear()

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    def as_dict(self) -> dict:
        """Structured snapshot, suitable for returning or serializing to JSON."""
        groups = self._grouped()
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_calls": len(self.records),
            "breakdown": [
                {
                    "stage": stage,
                    "model": model,
                    "calls": g["calls"],
                    "prompt_tokens": g["prompt"],
                    "completion_tokens": g["completion"],
                    "cost_usd": round(g["cost"], 6),
                }
                for (stage, model), g in groups.items()
            ],
        }

    def _grouped(self) -> "OrderedDict[tuple[str, str], dict]":
        groups: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
        for r in self.records:
            key = (r.stage, r.model)
            g = groups.setdefault(
                key, {"calls": 0, "prompt": 0, "completion": 0, "cost": 0.0}
            )
            g["calls"] += 1
            g["prompt"] += r.prompt_tokens
            g["completion"] += r.completion_tokens
            g["cost"] += r.cost
        return groups

    def summary(self) -> str:
        """Render a human-readable cost table."""
        if not self.records:
            return "=== Cost Analysis ===\nNo LLM calls recorded."

        groups = self._grouped()
        rows = [
            (
                stage,
                model,
                str(g["calls"]),
                f"{g['prompt']:,}",
                f"{g['completion']:,}",
                f"${g['cost']:.4f}",
            )
            for (stage, model), g in groups.items()
        ]
        header = ("Stage", "Model", "Calls", "Prompt", "Completion", "Cost")
        total = (
            "TOTAL",
            "",
            str(len(self.records)),
            f"{self.total_prompt_tokens:,}",
            f"{self.total_completion_tokens:,}",
            f"${self.total_cost:.4f}",
        )

        widths = [
            max(len(header[i]), len(total[i]), *(len(r[i]) for r in rows))
            for i in range(len(header))
        ]

        def fmt(cols: tuple) -> str:
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols)).rstrip()

        sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
        lines = ["=== Cost Analysis ===", fmt(header), sep]
        lines += [fmt(r) for r in rows]
        lines += [sep, fmt(total)]
        return "\n".join(lines)


# Module-level singleton shared across the run.
tracker = CostTracker()
