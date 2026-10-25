"""Telemetry tracing and 3-layer automated evaluation harness."""

from src.evals.evaluator import evaluate_research_run, MANDATORY_CFA_SECTIONS
from src.evals.telemetry import TelemetryTracker

__all__ = [
    "TelemetryTracker",
    "evaluate_research_run",
    "MANDATORY_CFA_SECTIONS",
]
