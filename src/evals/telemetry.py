"""Granular execution telemetry tracing latency, runtime mode, and contract verification status."""

import time
from contextlib import contextmanager
from typing import Generator, Optional
from src.schemas.models import StepTelemetry


class TelemetryTracker:
    """Thread-safe and lightweight execution telemetry tracker for EquiSight pipeline."""

    def __init__(self) -> None:
        self.steps: list[StepTelemetry] = []

    def add_step(
        self,
        step_name: str,
        agent_or_engine: str,
        mode: str,
        latency_ms: float,
        status: str,
        validation_passed: bool,
        details: str = "",
    ) -> StepTelemetry:
        """Adds a completed step telemetry record."""
        step = StepTelemetry(
            step_name=step_name,
            agent_or_engine=agent_or_engine,
            mode=mode,
            latency_ms=round(float(latency_ms), 2),
            status=status,
            validation_passed=validation_passed,
            details=details,
        )
        self.steps.append(step)
        return step

    @contextmanager
    def track_step(
        self,
        step_name: str,
        agent_or_engine: str,
        mode: str,
    ) -> Generator[dict, None, None]:
        """Context manager to measure execution wall-clock time and capture errors."""
        t0 = time.perf_counter()
        ctx = {"status": "SUCCESS", "validation_passed": True, "details": ""}
        try:
            yield ctx
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.add_step(
                step_name=step_name,
                agent_or_engine=agent_or_engine,
                mode=mode,
                latency_ms=elapsed_ms,
                status="FAILED",
                validation_passed=False,
                details=str(e),
            )
            raise
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.add_step(
                step_name=step_name,
                agent_or_engine=agent_or_engine,
                mode=mode,
                latency_ms=elapsed_ms,
                status=ctx.get("status", "SUCCESS"),
                validation_passed=ctx.get("validation_passed", True),
                details=ctx.get("details", ""),
            )

    def get_steps(self) -> list[StepTelemetry]:
        """Returns the sequential list of all recorded step telemetry objects."""
        return list(self.steps)

    def total_latency_ms(self) -> float:
        """Computes aggregate cumulative execution latency in milliseconds."""
        return round(sum(step.latency_ms for step in self.steps), 2)

    def clear(self) -> None:
        """Clears all logged telemetry records."""
        self.steps.clear()
