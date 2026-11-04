"""Per-task budget meter, shared by the orchestrator and the gateway threads."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Usage:
    wall_s: float = 0.0        # wall-clock time charged: elapsed time minus queue_s
    gpu_hours: float = 0.0
    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    queue_s: float = 0.0       # time spent waiting on infrastructure, not charged (see Budget)

    def as_dict(self) -> dict:
        return {
            "wall_s": round(self.wall_s, 3), "gpu_hours": round(self.gpu_hours, 6),
            "usd": round(self.usd, 6), "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens), "queue_s": round(self.queue_s, 3),
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Usage":
        d = d or {}
        return cls(float(d.get("wall_s", 0)), float(d.get("gpu_hours", 0)), float(d.get("usd", 0)),
                   int(d.get("input_tokens", 0)), int(d.get("output_tokens", 0)),
                   float(d.get("queue_s", 0)))

    def minus(self, other: "Usage") -> "Usage":
        return Usage(self.wall_s - other.wall_s, self.gpu_hours - other.gpu_hours,
                     self.usd - other.usd, self.input_tokens - other.input_tokens,
                     self.output_tokens - other.output_tokens, self.queue_s - other.queue_s)


@dataclass
class Budget:
    """Limits come from the resolved tier; `carried` is usage already spent (resume).

    Wall time is charged for the time the agent can work. Time spent waiting on infrastructure
    (a SLURM job pending in the queue, the model endpoint unreachable) is recorded with
    `add_wait` and left out of `wall_s`; overlapping waits count once. It is reported as
    `queue_s`. A wait still going on (a job pending in the queue) is recorded through `waiting`
    and counts up to the present, so a deadline that leaves waits out moves on while it lasts."""

    limits: dict
    carried: Usage = field(default_factory=Usage)
    usd_metering: bool = True
    _start: float = field(default_factory=time.monotonic)
    _extra: Usage = field(default_factory=Usage)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _waits: list = field(default_factory=list)
    _live: dict = field(default_factory=dict)   # waits not over yet: key -> (start, end or None)
    val_calls: int = 0

    def add_wait(self, start: float, end: float) -> None:
        """Leave the interval [start, end] (time.monotonic() values) out of the wall time."""
        if end > start:
            with self._lock:
                self._waits.append((float(start), float(end)))

    @contextmanager
    def waiting(self) -> Iterator[Callable[[float, float | None], None]]:
        """Yield a `Job.on_queue` callback that records the job's queue wait while it lasts:
        (start, None) is a wait still going on, (start, end) one that ended. A wait still open
        when the block ends is closed there."""
        key = object()

        def update(start: float, end: float | None = None) -> None:
            with self._lock:
                self._live[key] = (float(start), None if end is None else float(end))

        try:
            yield update
        finally:
            with self._lock:
                span = self._live.pop(key, None)
                if span is not None:
                    self._waits.append((span[0], time.monotonic() if span[1] is None else span[1]))

    def _waited(self, now: float) -> float:
        total, cur_start, cur_end = 0.0, None, None
        spans = [*self._waits, *((a, now if b is None else b) for a, b in self._live.values())]
        for a, b in sorted((max(a, self._start), min(b, now)) for a, b in spans):
            if b <= a:
                continue
            if cur_end is not None and a <= cur_end:
                cur_end = max(cur_end, b)
            else:
                if cur_end is not None:
                    total += cur_end - cur_start
                cur_start, cur_end = a, b
        if cur_end is not None:
            total += cur_end - cur_start
        return total

    def used(self) -> Usage:
        with self._lock:
            now = time.monotonic()
            waited = self._waited(now)
            return Usage(
                wall_s=self.carried.wall_s + (now - self._start) - waited,
                queue_s=self.carried.queue_s + waited,
                gpu_hours=self.carried.gpu_hours + self._extra.gpu_hours,
                usd=self.carried.usd + self._extra.usd,
                input_tokens=self.carried.input_tokens + self._extra.input_tokens,
                output_tokens=self.carried.output_tokens + self._extra.output_tokens,
            )

    def charge(self, gpu_hours: float = 0.0, usd: float = 0.0, input_tokens: int = 0,
               output_tokens: int = 0) -> None:
        with self._lock:
            self._extra.gpu_hours += gpu_hours
            self._extra.usd += usd
            self._extra.input_tokens += input_tokens
            self._extra.output_tokens += output_tokens

    def remaining(self) -> dict:
        used = self.used()
        tokens = used.input_tokens + used.output_tokens
        return {
            "wall_s": max(0.0, self.limits["wall_hours"] * 3600 - used.wall_s),
            "gpu_hours": max(0.0, self.limits["gpu_hours"] - used.gpu_hours),
            "usd": max(0.0, self.limits["usd"] - used.usd) if self.usd_metering else None,
            "tokens": max(0, int(self.limits["tokens"]) - tokens),
            "val_calls": max(0, int(self.limits["val_max_calls"]) - self.val_calls),
        }

    def fractions(self) -> dict:
        used = self.used()
        fr = {
            "wall": used.wall_s / max(1e-9, self.limits["wall_hours"] * 3600),
            "tokens": (used.input_tokens + used.output_tokens) / max(1, self.limits["tokens"]),
        }
        if self.limits["gpu_hours"] > 0:
            fr["gpu"] = used.gpu_hours / self.limits["gpu_hours"]
        if self.usd_metering and self.limits["usd"] > 0:
            fr["usd"] = used.usd / self.limits["usd"]
        return fr

    def fraction_used(self) -> float:
        """Share of the binding resource consumed, in [0, 1+]."""
        return max(self.fractions().values())

    def exhausted(self) -> str | None:
        """Name of the first exhausted resource, or None."""
        for name, frac in self.fractions().items():
            if frac >= 1.0:
                return f"budget_{name}"
        return None

    def gpu_allowed(self, gpu_hours: float) -> bool:
        return self.used().gpu_hours + gpu_hours <= self.limits["gpu_hours"] + 1e-9
