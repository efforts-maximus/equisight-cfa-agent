"""Comprehensive verification test suite for Phase 3: CrewAI Agents, Flows, and 3-Layer Eval Harness.

Tests:
1. YAML Configuration Decoupling (agents.yaml, tasks.yaml)
2. Execution Telemetry Tracking (TelemetryTracker, latencies, failure capture)
3. CrewAI Tool Grounding (RetrieveFilingEvidenceTool, RAG retrieval)
4. Deterministic Scenario Synthesizer (Ordering, probability conservation, terminal growth cap)
5. 3-Layer Evaluation Harness:
   - Baseline clean report evaluation (100% pass across all 4 tickers)
   - Layer 1 Negative Math Violations (monotonicity, DuPont exactness, CCC identity, EV bridge, g >= WACC)
   - Layer 2 Negative Completeness (missing CFA sections, missing citations, invalid schema)
   - Layer 3 Negative Numerical Hallucinations (current price, target price, WACC, upside, ROE)
6. End-to-End Stateful Flow Execution (TCS.NS, ZOMATO.NS, VARUNBEV.NS, NVDA)
7. Andrew Ng Reflection Auto-Repair Gate (corrupted draft recovery, repair attempt tracking)
8. Extraction Regex Robustness (tables, plain text, various currency symbols)
"""

import copy
import json
import re
from unittest.mock import MagicMock, patch
import pytest

from src.agents.research_flow import (
    EquiSightResearchFlow,
    EquiSightState,
    LLMExecutionError,
    execute_groq_llm_synthesis,
    format_cfa_report,
    load_agent_configs,
    load_task_configs,
    repair_cfa_report,
    select_groq_model,
    synthesize_deterministic_assumptions,
)
from src.agents.tools import RetrieveFilingEvidenceTool
from src.data.market_gateway import fetch_company_snapshot, fetch_peer_multiples
from src.engines.fsa_engine import compute_fsa_summary
from src.engines.quant_risk_engine import compute_quant_risk_summary, run_monte_carlo_dcf
from src.engines.valuation_engine import run_valuation_suite
from src.evals.evaluator import (
    MANDATORY_CFA_SECTIONS,
    evaluate_research_run,
    extract_base_wacc,
    extract_current_price,
    extract_dupont_roe,
    extract_implied_upside,
    extract_target_price,
)
from src.evals.telemetry import TelemetryTracker
from src.schemas.models import (
    CompanyFinancialSnapshot,
    DCFScenarioAssumptions,
    EvalScorecard,
    FSARatioSummary,
    QuantRiskSummary,
    SingleDCFResult,
    StepTelemetry,
    ValuationSuiteOutput,
)


# =============================================================================
# 1. YAML Configuration Decoupling Tests
# =============================================================================

class TestYAMLConfiguration:
    """Verifies that declarative YAML files conform strictly to required personas and tasks."""

    def test_agents_yaml_contains_all_specialists(self):
        configs = load_agent_configs()
        expected_agents = [
            "fsa_forensic_analyst",
            "bull_equity_researcher",
            "bear_short_seller",
            "lead_valuation_pm",
            "audit_compliance_critic",
        ]
        for agent_key in expected_agents:
            assert agent_key in configs, f"Missing agent '{agent_key}' in agents.yaml"
            cfg = configs[agent_key]
            assert "role" in cfg and len(cfg["role"].strip()) > 0
            assert "goal" in cfg and len(cfg["goal"].strip()) > 0
            assert "backstory" in cfg and len(cfg["backstory"].strip()) > 0

    def test_tasks_yaml_contains_all_institutional_tasks(self):
        configs = load_task_configs()
        expected_tasks = [
            "fsa_forensic_task",
            "bull_thesis_task",
            "bear_thesis_task",
            "synthesize_scenarios_and_report_task",
            "audit_and_repair_task",
        ]
        for task_key in expected_tasks:
            assert task_key in configs, f"Missing task '{task_key}' in tasks.yaml"
            cfg = configs[task_key]
            assert "description" in cfg and len(cfg["description"].strip()) > 0
            assert "expected_output" in cfg and len(cfg["expected_output"].strip()) > 0
            assert "agent" in cfg and len(cfg["agent"].strip()) > 0


# =============================================================================
# 2. Telemetry Tracker Tests
# =============================================================================

class TestTelemetryTracker:
    """Verifies step logging, wall-clock timing, exception handling, and serialization."""

    def test_manual_add_step(self):
        tracker = TelemetryTracker()
        step = tracker.add_step(
            step_name="test_step",
            agent_or_engine="fsa_engine",
            mode="deterministic",
            latency_ms=45.2,
            status="SUCCESS",
            validation_passed=True,
            details="Computed 5-way DuPont",
        )
        assert isinstance(step, StepTelemetry)
        assert step.step_name == "test_step"
        assert step.latency_ms == 45.2
        assert len(tracker.get_steps()) == 1
        assert tracker.total_latency_ms() == 45.2

    def test_track_step_context_manager_success(self):
        tracker = TelemetryTracker()
        with tracker.track_step("computation", "quant_engine", "deterministic") as ctx:
            ctx["details"] = "Monte carlo simulated"

        steps = tracker.get_steps()
        assert len(steps) == 1
        assert steps[0].status == "SUCCESS"
        assert steps[0].validation_passed is True
        assert steps[0].latency_ms >= 0.0
        assert steps[0].details == "Monte carlo simulated"

    def test_track_step_context_manager_catches_failure(self):
        tracker = TelemetryTracker()
        with pytest.raises(ZeroDivisionError):
            with tracker.track_step("faulty_step", "math_engine", "deterministic"):
                _ = 1 / 0

        steps = tracker.get_steps()
        assert len(steps) == 1
        assert steps[0].status == "FAILED"
        assert steps[0].validation_passed is False
        assert "division by zero" in steps[0].details

    def test_clear_telemetry(self):
        tracker = TelemetryTracker()
        tracker.add_step("s1", "a", "deterministic", 10.0, "SUCCESS", True)
        tracker.add_step("s2", "b", "deterministic", 20.0, "SUCCESS", True)
        assert tracker.total_latency_ms() == 30.0
        tracker.clear()
        assert len(tracker.get_steps()) == 0
        assert tracker.total_latency_ms() == 0.0


# =============================================================================
# 3. CrewAI Tool Grounding Tests
# =============================================================================

class TestCrewAIToolGrounding:
    """Verifies RetrieveFilingEvidenceTool wrapping TF-IDF RAG retrieval."""

    def test_tool_execution_preserves_citations(self):
        tool = RetrieveFilingEvidenceTool()
        assert tool.name == "retrieve_filing_evidence"
        result = tool.run(ticker="TCS.NS", query="revenue geographic digital transformation", top_k=2)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "[FILING-CHUNK-" in result

    def test_tool_execution_us_equity(self):
        tool = RetrieveFilingEvidenceTool()
        result = tool.run(ticker="NVDA", query="data center compute hopper blackwell", top_k=2)
        assert "[FILING-CHUNK-" in result


# =============================================================================
# 4. Deterministic Scenario Synthesizer Tests
# =============================================================================

class TestDeterministicScenarioSynthesizer:
    """Verifies bounded mathematical invariant preservation across all 4 benchmark tickers."""

    @pytest.mark.parametrize("ticker", ["TCS.NS", "ZOMATO.NS", "VARUNBEV.NS", "NVDA"])
    def test_scenario_synthesizer_invariants(self, ticker):
        snapshot = fetch_company_snapshot(ticker, use_live=False)
        fsa_summary = compute_fsa_summary(snapshot)
        quant_summary = compute_quant_risk_summary(snapshot)

        assumptions = synthesize_deterministic_assumptions(snapshot, fsa_summary, quant_summary)

        # 1. Probability weight conservation
        total_prob = assumptions.bear.probability_weight + assumptions.base.probability_weight + assumptions.bull.probability_weight
        assert abs(total_prob - 1.0) < 1e-4

        # 2. Revenue CAGR ordering: Bull >= Base >= Bear
        bull_cagr = sum(assumptions.bull.revenue_cagr_5yr) / 5.0
        base_cagr = sum(assumptions.base.revenue_cagr_5yr) / 5.0
        bear_cagr = sum(assumptions.bear.revenue_cagr_5yr) / 5.0
        assert bull_cagr >= base_cagr >= bear_cagr

        # 3. Operating margin ordering: Bull >= Base >= Bear
        assert (
            assumptions.bull.target_ebit_margin_yr5
            >= assumptions.base.target_ebit_margin_yr5
            >= assumptions.bear.target_ebit_margin_yr5
        )

        # 4. Terminal growth ordering: Bull >= Base >= Bear
        assert (
            assumptions.bull.terminal_growth_rate
            >= assumptions.base.terminal_growth_rate
            >= assumptions.bear.terminal_growth_rate
        )

        # 5. Citation grounding in qualitative points
        has_bull_cite = any("[FILING-CHUNK-" in p for p in assumptions.bull_thesis_points)
        has_bear_cite = any("[FILING-CHUNK-" in p for p in assumptions.bear_thesis_points)
        assert has_bull_cite, "Bull thesis points lack [FILING-CHUNK-X] citations"
        assert has_bear_cite, "Bear thesis points lack [FILING-CHUNK-X] citations"

        # 6. Ticker-specific debate grounding: non-IT tickers must not contain IT-specific keywords
        if ticker in ("VARUNBEV.NS", "NVDA", "ZOMATO.NS"):
            all_qualitative_text = " ".join(
                assumptions.bull_thesis_points
                + assumptions.bear_thesis_points
                + assumptions.key_catalysts
                + assumptions.key_risks
                + [assumptions.debate_synthesis_summary]
            )
            assert "BFSI" not in all_qualitative_text, f"{ticker} assumptions should not contain 'BFSI'"
            assert "subcontractor" not in all_qualitative_text, f"{ticker} assumptions should not contain 'subcontractor'"


# =============================================================================
# 5. 3-Layer Evaluation Harness: Baseline & Negative Tests
# =============================================================================

@pytest.fixture
def tcs_pipeline_artifacts():
    """Builds clean pipeline artifacts for TCS.NS to test evaluation harness layers."""
    snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
    peers = fetch_peer_multiples("TCS.NS", use_live=False)
    fsa_summary = compute_fsa_summary(snapshot)
    quant_summary = compute_quant_risk_summary(snapshot)
    assumptions = synthesize_deterministic_assumptions(snapshot, fsa_summary, quant_summary)
    val_output = run_valuation_suite(snapshot, assumptions, quant_summary, peers)
    mc_output = run_monte_carlo_dcf(snapshot, assumptions, num_simulations=2000, seed=42)
    report_md = format_cfa_report(snapshot, fsa_summary, quant_summary, assumptions, val_output, mc_output)
    return {
        "snapshot": snapshot,
        "fsa_summary": fsa_summary,
        "quant_summary": quant_summary,
        "assumptions": assumptions,
        "valuation_output": val_output,
        "monte_carlo_output": mc_output,
        "report_md": report_md,
    }


class TestEvaluationHarnessBaseline:
    """Verifies that clean deterministic pipeline runs achieve 100% scores across all 3 layers."""

    def test_clean_tcs_passes_all_layers(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=a["report_md"],
        )
        assert scorecard.passed is True
        assert scorecard.math_invariance_score == 100.0
        assert scorecard.schema_completeness_score == 100.0
        assert scorecard.numerical_faithfulness_score == 100.0
        assert scorecard.overall_eval_score == 100.0
        assert len(scorecard.flagged_discrepancies) == 0


class TestLayer1MathViolations:
    """Verifies Layer 1 detects mathematical identity violations."""

    def test_dupont_product_invariance_violation(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        # Mutate reconstructed_roe so it does not match roe
        corrupted_fsa = a["fsa_summary"].model_copy(update={"reconstructed_roe": 0.1234})
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=corrupted_fsa,
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=a["report_md"],
        )
        assert scorecard.passed is False
        assert scorecard.math_invariance_score < 100.0
        assert any("DuPont 5-Way identity violated" in err for err in scorecard.flagged_discrepancies)

    def test_ccc_working_capital_identity_violation(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        # Mutate cash_conversion_cycle to corrupt identity
        corrupted_fsa = a["fsa_summary"].model_copy(update={"cash_conversion_cycle": 999.0})
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=corrupted_fsa,
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=a["report_md"],
        )
        assert scorecard.passed is False
        assert scorecard.math_invariance_score < 100.0
        assert any("Working capital CCC identity violated" in err for err in scorecard.flagged_discrepancies)

    def test_dcf_scenario_monotonicity_violation(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        # Invert scenario pricing: Bull < Base
        corrupted_bull = a["valuation_output"].bull_dcf.model_copy(update={"intrinsic_value_per_share": 100.0})
        corrupted_val = a["valuation_output"].model_copy(update={"bull_dcf": corrupted_bull})
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=corrupted_val,
            report_markdown=a["report_md"],
        )
        assert scorecard.passed is False
        assert scorecard.math_invariance_score < 100.0
        assert any("DCF scenario monotonicity violated" in err for err in scorecard.flagged_discrepancies)

    def test_ev_to_equity_balance_sheet_bridge_violation(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        # Corrupt equity value on base scenario
        corrupted_base = a["valuation_output"].base_dcf.model_copy(update={"equity_value": 1.0})
        corrupted_val = a["valuation_output"].model_copy(update={"base_dcf": corrupted_base})
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=corrupted_val,
            report_markdown=a["report_md"],
        )
        assert scorecard.passed is False
        assert scorecard.math_invariance_score < 100.0
        assert any("EV-to-Equity balance sheet bridge violated" in err for err in scorecard.flagged_discrepancies)

    def test_terminal_growth_cap_violation(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        # Set g >= wacc directly
        corrupted_base = a["valuation_output"].base_dcf.model_copy(
            update={"terminal_growth_rate": a["valuation_output"].base_dcf.wacc + 0.01}
        )
        corrupted_val = a["valuation_output"].model_copy(update={"base_dcf": corrupted_base})
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=corrupted_val,
            report_markdown=a["report_md"],
        )
        assert scorecard.passed is False
        assert scorecard.math_invariance_score < 100.0
        assert any("Terminal growth rate cap violated" in err for err in scorecard.flagged_discrepancies)


class TestLayer2CompletenessViolations:
    """Verifies Layer 2 detects missing sections, lack of citations, and schema flaws."""

    def test_missing_mandatory_section(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        # Strip Section 3 from report
        corrupted_md = re.sub(
            r"# 3\. Forensic Earnings Quality & Solvency.*?(?=# 4\.|\Z)",
            "",
            a["report_md"],
            flags=re.DOTALL,
        )
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=corrupted_md,
        )
        assert scorecard.passed is False
        assert scorecard.schema_completeness_score < 100.0
        assert any("Missing mandatory CFA report section: '3. Forensic Earnings Quality & Solvency'" in err for err in scorecard.flagged_discrepancies)

    def test_missing_filing_citations(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        # Remove all [FILING-CHUNK-X] citation tags
        corrupted_md = re.sub(r"\[FILING-CHUNK-\d+\]", "[DISCLOSURE-REF]", a["report_md"])
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=corrupted_md,
        )
        assert scorecard.passed is False
        assert scorecard.schema_completeness_score < 100.0
        assert any("Report lacks required [FILING-CHUNK-X] citation grounding" in err for err in scorecard.flagged_discrepancies)

    def test_invalid_schema_contracts_returns_failed_scorecard(self, tcs_pipeline_artifacts):
        """Verifies that invalid or None schema contracts do not crash and cleanly return passed=False."""
        a = tcs_pipeline_artifacts
        # Pass None for snapshot and fsa_summary
        scorecard = evaluate_research_run(
            snapshot=None,  # type: ignore
            fsa_summary=None,  # type: ignore
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=a["report_md"],
        )
        assert scorecard.passed is False
        assert scorecard.overall_eval_score == 0.0
        assert any("Input data contracts invalid or missing required Pydantic models" in err for err in scorecard.flagged_discrepancies)
        assert any("CompanyFinancialSnapshot" in err for err in scorecard.flagged_discrepancies)
        assert any("FSARatioSummary" in err for err in scorecard.flagged_discrepancies)

        # Pass empty report_markdown
        scorecard_empty = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown="",
        )
        assert scorecard_empty.passed is False
        assert any("report_markdown" in err for err in scorecard_empty.flagged_discrepancies)


class TestLayer3NumericalHallucinationViolations:
    """Verifies Layer 3 catches LLM hallucinations on prices, WACC, upside %, and ROE."""

    def test_current_price_hallucination(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        corrupted_md = a["report_md"].replace(
            f"Current Price:** INR {a['snapshot'].current_price:.2f}",
            "Current Price:** INR 9,999.00",
        )
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=corrupted_md,
        )
        assert scorecard.passed is False
        assert scorecard.numerical_faithfulness_score < 100.0
        assert any("Current Price mismatch" in err for err in scorecard.flagged_discrepancies)

    def test_target_price_hallucination(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        corrupted_md = a["report_md"].replace(
            f"Target Price:** INR {a['valuation_output'].probability_weighted_intrinsic_value:.2f}",
            "Target Price:** INR 7,500.00",
        )
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=corrupted_md,
        )
        assert scorecard.passed is False
        assert scorecard.numerical_faithfulness_score < 100.0
        assert any("Target Price mismatch" in err for err in scorecard.flagged_discrepancies)

    def test_wacc_hallucination(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        corrupted_md = a["report_md"].replace(
            f"Base WACC:** {a['valuation_output'].base_wacc * 100:.2f}%",
            "Base WACC:** 5.00%",
        )
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=corrupted_md,
        )
        assert scorecard.passed is False
        assert scorecard.numerical_faithfulness_score < 100.0
        assert any("Base WACC mismatch" in err for err in scorecard.flagged_discrepancies)

    def test_implied_upside_hallucination(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        corrupted_md = a["report_md"].replace(
            f"Implied Upside:** {a['valuation_output'].implied_upside_pct * 100:.2f}%",
            "Implied Upside:** +80.00%",
        )
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=corrupted_md,
        )
        assert scorecard.passed is False
        assert scorecard.numerical_faithfulness_score < 100.0
        assert any("Implied Upside mismatch" in err for err in scorecard.flagged_discrepancies)

    def test_dupont_roe_hallucination(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts
        corrupted_md = a["report_md"].replace(
            f"5-Way DuPont ROE:** {a['fsa_summary'].roe * 100:.2f}%",
            "5-Way DuPont ROE:** 12.00%",
        )
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=corrupted_md,
        )
        assert scorecard.passed is False
        assert scorecard.numerical_faithfulness_score < 100.0
        assert any("DuPont ROE mismatch" in err for err in scorecard.flagged_discrepancies)

    def test_combined_target_price_and_wacc_negative_hallucination(self, tcs_pipeline_artifacts):
        """Verifies SPEC requirement: inject both corrupted target price and fake WACC and assert failure."""
        a = tcs_pipeline_artifacts
        corrupted_md = a["report_md"].replace(
            f"Target Price:** INR {a['valuation_output'].probability_weighted_intrinsic_value:.2f}",
            "Target Price:** INR 8,888.00",
        ).replace(
            f"Base WACC:** {a['valuation_output'].base_wacc * 100:.2f}%",
            "Base WACC:** 3.50%",
        )
        scorecard = evaluate_research_run(
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            report_markdown=corrupted_md,
        )
        assert scorecard.passed is False
        assert scorecard.numerical_faithfulness_score < 100.0
        assert any("Target Price mismatch" in err for err in scorecard.flagged_discrepancies)
        assert any("Base WACC mismatch" in err for err in scorecard.flagged_discrepancies)


# =============================================================================
# 6. End-to-End Stateful Flow Execution Tests
# =============================================================================

class TestEquiSightResearchFlow:
    """Verifies end-to-end event-driven CrewAI flow execution across India and US tickers."""

    @pytest.mark.parametrize("ticker", ["TCS.NS", "ZOMATO.NS", "VARUNBEV.NS", "NVDA"])
    def test_flow_deterministic_execution(self, ticker):
        flow = EquiSightResearchFlow()
        flow.state.ticker = ticker
        flow.state.mode = "deterministic"

        result = flow.kickoff()
        assert flow.state.status == "SUCCESS"
        assert flow.state.eval_scorecard is not None
        assert flow.state.eval_scorecard.passed is True
        assert flow.state.eval_scorecard.overall_eval_score == 100.0
        assert len(flow.state.eval_scorecard.flagged_discrepancies) == 0

        # Verify telemetry trace recorded all phases
        telemetry = flow.state.telemetry
        assert len(telemetry) >= 3
        step_names = [t.step_name for t in telemetry]
        assert "ingest_and_compute_core_engines" in step_names
        assert "conduct_debate_and_synthesize_assumptions" in step_names
        assert "generate_and_evaluate_report" in step_names

        for t in telemetry:
            assert t.status == "SUCCESS"
            assert t.validation_passed is True
            assert t.latency_ms >= 0.0

        # Verify ticker-accurate debate text in generated CFA report
        if ticker in ("VARUNBEV.NS", "NVDA", "ZOMATO.NS"):
            report = flow.state.report_markdown
            assert "BFSI" not in report, f"{ticker} report should not contain 'BFSI'"
            assert "subcontractor" not in report, f"{ticker} report should not contain 'subcontractor'"

    def test_flow_kickoff_with_inputs_dict(self):
        """Verifies that flow.kickoff(inputs={...}) properly sets ticker and mode."""
        flow = EquiSightResearchFlow()
        res = flow.kickoff(inputs={"ticker": "NVDA", "mode": "deterministic"})
        assert flow.state.ticker == "NVDA"
        assert flow.state.status == "SUCCESS"
        assert flow.state.eval_scorecard is not None
        assert flow.state.eval_scorecard.passed is True


# =============================================================================
# 7. Andrew Ng Reflection Auto-Repair Gate Tests
# =============================================================================

class TestReflectionAutoRepairGate:
    """Verifies that flagged discrepancies trigger the auto-repair loop and successfully recover 100% scores."""

    def test_reflection_repair_flow_recovers_corrupted_draft(self, tcs_pipeline_artifacts):
        a = tcs_pipeline_artifacts

        # Corrupt the draft report with hallucinated numbers and missing sections
        corrupted_draft = a["report_md"].replace(
            f"Target Price:** INR {a['valuation_output'].probability_weighted_intrinsic_value:.2f}",
            "Target Price:** INR 9,999.00",
        )
        # Drop section 6
        corrupted_draft = re.sub(
            r"# 6\. Quantitative Risk & Monte Carlo Simulation.*?\Z",
            "",
            corrupted_draft,
            flags=re.DOTALL,
        )

        flow = EquiSightResearchFlow()
        flow.state.ticker = "TCS.NS"
        flow.state.mode = "deterministic"
        # Seed state with corrupted draft
        flow.state.report_markdown = corrupted_draft

        # Run flow
        flow.kickoff()

        # The initial evaluation will fail, routing to "repair" which regenerates ground truth
        assert flow.state.status == "SUCCESS"
        assert flow.state.repair_attempts >= 1
        assert flow.state.eval_scorecard.passed is True
        assert flow.state.eval_scorecard.overall_eval_score == 100.0
        assert len(flow.state.eval_scorecard.flagged_discrepancies) == 0

        # Telemetry should capture the repair step
        step_names = [t.step_name for t in flow.state.telemetry]
        assert "repair_report" in step_names

    def test_targeted_repair_preserves_narrative(self, tcs_pipeline_artifacts):
        """Verifies that reflection auto-repair corrects figures while preserving qualitative analysis."""
        a = tcs_pipeline_artifacts
        custom_comment = "<!-- UNIQUE_ANALYST_COMMENT: Structural digital transformation moat is accelerating -->"
        draft_with_notes = custom_comment + "\n\n" + a["report_md"].replace(
            f"Target Price:** INR {a['valuation_output'].probability_weighted_intrinsic_value:.2f}",
            "Target Price:** INR 9,999.00",
        )
        repaired = repair_cfa_report(
            report_markdown=draft_with_notes,
            snapshot=a["snapshot"],
            fsa_summary=a["fsa_summary"],
            quant_summary=a["quant_summary"],
            assumptions=a["assumptions"],
            valuation_output=a["valuation_output"],
            monte_carlo_output=a["monte_carlo_output"],
        )
        assert custom_comment in repaired
        assert f"Target Price:** INR {a['valuation_output'].probability_weighted_intrinsic_value:.2f}" in repaired

    def test_flow_failed_terminal_state_when_repair_exhausted(self):
        """Verifies that if repair attempts are exhausted, routing terminates in handle_failed."""
        flow = EquiSightResearchFlow()
        flow.state.ticker = "TCS.NS"
        flow.state.mode = "deterministic"
        flow.state.repair_attempts = 2  # Already exhausted max attempts
        flow.state.report_markdown = "# Completely Unrepairable Draft Without Required Sections"

        res = flow.kickoff()
        assert flow.state.status == "FAILED"
        assert flow.state.eval_scorecard.passed is False
        assert res["status"] == "FAILED"


# =============================================================================
# 8. Extraction Regex Robustness Tests
# =============================================================================

class TestExtractionRegexes:
    """Verifies that regex pattern extractors reliably extract key financial values."""

    def test_extract_current_price_formats(self):
        sample1 = "- **Current Price:** INR 3,912.60"
        sample2 = "| Current Price | $128.50 |"
        sample3 = "The Current Trading Price of the stock is ₹ 450.25 on NSE."
        assert extract_current_price(sample1) == 3912.60
        assert extract_current_price(sample2) == 128.50
        assert extract_current_price(sample3) == 450.25

    def test_extract_target_price_formats(self):
        sample1 = "- **Target Price:** INR 4,120.50"
        sample2 = "| Probability-Weighted Intrinsic Value | $145.00 |"
        sample3 = "- **Weighted Intrinsic Value:** ₹ 520.00"
        assert extract_target_price(sample1) == 4120.50
        assert extract_target_price(sample2) == 145.00
        assert extract_target_price(sample3) == 520.00

    def test_extract_base_wacc_formats(self):
        sample1 = "- **Base WACC:** 11.85%"
        sample2 = "| Base WACC | 10.50% |"
        sample3 = "Discount Rate (WACC): 9.20%"
        assert abs(extract_base_wacc(sample1) - 0.1185) < 1e-4
        assert abs(extract_base_wacc(sample2) - 0.1050) < 1e-4
        assert abs(extract_base_wacc(sample3) - 0.0920) < 1e-4

    def test_extract_implied_upside_formats(self):
        sample1 = "- **Implied Upside:** +5.31%"
        sample2 = "- **Implied Upside / Downside:** -12.40%"
        sample3 = "| Implied Upside | +15.50% |"
        assert abs(extract_implied_upside(sample1) - 0.0531) < 1e-4
        assert abs(extract_implied_upside(sample2) - (-0.1240)) < 1e-4
        assert abs(extract_implied_upside(sample3) - 0.1550) < 1e-4

    def test_extract_dupont_roe_formats(self):
        sample1 = "- **5-Way DuPont ROE:** 48.20% (Reconstructed: 48.20%)"
        sample2 = "| DuPont ROE | 25.40% |"
        sample3 = "Return on Equity (ROE): 18.50%"
        assert abs(extract_dupont_roe(sample1) - 0.4820) < 1e-4
        assert abs(extract_dupont_roe(sample2) - 0.2540) < 1e-4
        assert abs(extract_dupont_roe(sample3) - 0.1850) < 1e-4

    def test_fractional_and_small_percentage_extraction(self):
        """Verifies that small / fractional percentages (<= 1.0%) are parsed correctly without 100x distortion."""
        upside_sample = "- **Implied Upside:** +0.50%"
        downside_sample = "- **Implied Upside:** -0.40%"
        wacc_sample = "- **Base WACC:** 0.95%"
        roe_sample = "- **5-Way DuPont ROE:** 0.80%"
        assert abs(extract_implied_upside(upside_sample) - 0.0050) < 1e-5
        assert abs(extract_implied_upside(downside_sample) - (-0.0040)) < 1e-5
        assert abs(extract_base_wacc(wacc_sample) - 0.0095) < 1e-5
        assert abs(extract_dupont_roe(roe_sample) - 0.0080) < 1e-5


# =============================================================================
# 9. Anti-Hardcoding Source-Code Audit Tests
# =============================================================================

def test_zero_hardcoded_tickers_in_engines_and_agents():
    """Permanent source-code audit asserting zero hardcoded benchmark tickers in engines and agents."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    banned_tickers = ["VARUNBEV", "ZOMATO", "NVDA", "TCS", "AAPL", "AMZN", "GOOG", "MSFT", "INFY"]

    target_dirs = [root / "src" / "engines", root / "src" / "agents"]
    violations = []

    for target_dir in target_dirs:
        for py_file in target_dir.glob("**/*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            for banned in banned_tickers:
                if banned in content:
                    violations.append(f"{py_file.name} contains banned ticker '{banned}'")

    assert not violations, f"Anti-hardcoding violation(s) detected: {violations}"


def test_dynamic_rag_synthesis_custom_ticker_fallback():
    """Verifies dynamic RAG synthesis for a custom ticker with no local filing fixture."""
    from src.data.market_gateway import fetch_company_snapshot
    from src.engines.fsa_engine import compute_fsa_summary
    from src.engines.quant_risk_engine import compute_quant_risk_summary
    from src.agents.research_flow import synthesize_deterministic_assumptions
    from src.schemas.models import CompanyFinancialSnapshot

    base_snap = fetch_company_snapshot("NVDA", use_live=False)
    snap_dict = base_snap.model_dump()
    snap_dict["ticker"] = "CUSTOM_CORP"
    snap_dict["company_name"] = "Custom Enterprise Systems Inc."
    custom_snapshot = CompanyFinancialSnapshot(**snap_dict)

    fsa = compute_fsa_summary(custom_snapshot)
    quant = compute_quant_risk_summary(custom_snapshot)
    assumptions = synthesize_deterministic_assumptions(custom_snapshot, fsa, quant)

    # Invariants must hold
    assert assumptions.bull.revenue_cagr_5yr[0] >= assumptions.base.revenue_cagr_5yr[0] >= assumptions.bear.revenue_cagr_5yr[0]
    assert assumptions.bull.target_ebit_margin_yr5 >= assumptions.base.target_ebit_margin_yr5 >= assumptions.bear.target_ebit_margin_yr5
    assert assumptions.bull.terminal_growth_rate >= assumptions.base.terminal_growth_rate >= assumptions.bear.terminal_growth_rate
    assert any("[FILING-CHUNK-" in p for p in assumptions.bull_thesis_points)
    assert any("[FILING-CHUNK-" in p for p in assumptions.bear_thesis_points)


def test_dynamic_rag_synthesis_distressed_turnaround_company():
    """Verifies dynamic synthesis for distressed company with negative historical CAGR and negative operating margins."""
    from src.data.market_gateway import fetch_company_snapshot
    from src.engines.fsa_engine import compute_fsa_summary
    from src.engines.quant_risk_engine import compute_quant_risk_summary
    from src.agents.research_flow import synthesize_deterministic_assumptions
    from src.schemas.models import CompanyFinancialSnapshot

    base_snap = fetch_company_snapshot("NVDA", use_live=False)
    snap_dict = base_snap.model_dump()
    snap_dict["ticker"] = "DISTRESSED_CO"
    snap_dict["company_name"] = "Distressed Turnaround Corp"

    # Make revenues shrink and margins negative
    for i, yr in enumerate(snap_dict["historical_years"]):
        yr["revenue"] = round(1000.0 * (0.90 ** i), 2)
        yr["cogs"] = round(yr["revenue"] * 0.70, 2)
        yr["gross_profit"] = round(yr["revenue"] * 0.30, 2)
        yr["ebit"] = round(-yr["revenue"] * 0.08, 2)
        yr["ebitda"] = round(yr["ebit"] + yr["depreciation_amortization"], 2)
        yr["ebt"] = round(yr["ebit"] - yr["interest_expense"], 2)
        yr["net_income"] = round(yr["ebt"], 2)
        yr["tax_expense"] = 0.0
        yr["operating_cash_flow"] = round(-yr["revenue"] * 0.04, 2)

    distressed_snapshot = CompanyFinancialSnapshot(**snap_dict)
    fsa = compute_fsa_summary(distressed_snapshot)
    quant = compute_quant_risk_summary(distressed_snapshot)
    assumptions = synthesize_deterministic_assumptions(distressed_snapshot, fsa, quant)

    # Invariants must hold even for distressed / negative metrics
    assert assumptions.bull.revenue_cagr_5yr[0] >= assumptions.base.revenue_cagr_5yr[0] >= assumptions.bear.revenue_cagr_5yr[0]
    assert assumptions.bull.target_ebit_margin_yr5 >= assumptions.base.target_ebit_margin_yr5 >= assumptions.bear.target_ebit_margin_yr5
    assert assumptions.bull.terminal_growth_rate >= assumptions.base.terminal_growth_rate >= assumptions.bear.terminal_growth_rate
    assert -0.30 <= assumptions.bear.target_ebit_margin_yr5 <= 0.70
    assert -0.30 <= assumptions.base.target_ebit_margin_yr5 <= 0.70
    assert -0.30 <= assumptions.bull.target_ebit_margin_yr5 <= 0.70


# =============================================================================
# 10. Groq Multi-Agent LLM Synthesis Engine & Mathematical Invariance Tests
# =============================================================================

class TestGroqMultiAgentSynthesis:
    """Verifies Groq API synthesis, credential error handling, model selection, and 100% mathematical invariance."""

    def test_llm_execution_error_missing_api_key(self, monkeypatch):
        """Verifies that calling mode='llm' without GROQ_API_KEY immediately raises LLMExecutionError."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        flow = EquiSightResearchFlow()
        flow.state.ticker = "NVDA"
        flow.state.mode = "llm"
        flow.state.use_live = False

        with pytest.raises(LLMExecutionError) as exc_info:
            flow.kickoff()

        err = exc_info.value
        assert "GROQ_API_KEY" in str(err)
        assert err.provider == "groq"

    def test_llm_execution_error_empty_or_whitespace_key(self, monkeypatch):
        """Verifies that empty string or whitespace-only GROQ_API_KEY raises LLMExecutionError."""
        for bad_key in ["", "   ", "\t\n"]:
            monkeypatch.setenv("GROQ_API_KEY", bad_key)
            flow = EquiSightResearchFlow()
            flow.state.ticker = "NVDA"
            flow.state.mode = "llm"
            flow.state.use_live = False

            with pytest.raises(LLMExecutionError) as exc_info:
                flow.kickoff()

            err = exc_info.value
            assert "GROQ_API_KEY" in str(err)
            assert err.provider == "groq"

    def test_llm_execution_error_invalid_api_key(self, monkeypatch):
        """Verifies that calling mode='llm' with invalid key raises LLMExecutionError with HTTP 401."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_invalid_test_key_nonexistent")
        flow = EquiSightResearchFlow()
        flow.state.ticker = "NVDA"
        flow.state.mode = "llm"
        flow.state.use_live = False

        with pytest.raises(LLMExecutionError) as exc_info:
            flow.kickoff()

        err = exc_info.value
        assert err.provider == "groq"
        assert err.status_code == 401 or "401" in str(err)

    def test_select_groq_model_env_override(self, monkeypatch):
        """Verifies that GROQ_MODEL environment variable takes precedence when available."""
        monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "openai/gpt-oss-120b"},
                {"id": "qwen/qwen3.8-27b"},
            ]
        }
        with patch("requests.get", return_value=mock_resp):
            selected = select_groq_model("test-key")
            assert selected == "qwen/qwen3.8-27b"

    def test_select_groq_model_priority_order(self, monkeypatch):
        """Verifies priority cascade: gpt-oss-120b -> gpt-oss-20b -> qwen3.8-27b -> llama-3.3-70b-versatile."""
        monkeypatch.delenv("GROQ_MODEL", raising=False)

        # 1. 120b available
        mock_resp1 = MagicMock(status_code=200)
        mock_resp1.json.return_value = {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "openai/gpt-oss-20b"}]}
        with patch("requests.get", return_value=mock_resp1):
            assert select_groq_model("key") == "openai/gpt-oss-120b"

        # 2. 120b unavailable, 20b available
        mock_resp2 = MagicMock(status_code=200)
        mock_resp2.json.return_value = {"data": [{"id": "openai/gpt-oss-20b"}, {"id": "qwen/qwen3.8-27b"}]}
        with patch("requests.get", return_value=mock_resp2):
            assert select_groq_model("key") == "openai/gpt-oss-20b"

        # 3. Only qwen available
        mock_resp3 = MagicMock(status_code=200)
        mock_resp3.json.return_value = {"data": [{"id": "qwen/qwen3.8-27b"}]}
        with patch("requests.get", return_value=mock_resp3):
            assert select_groq_model("key") == "qwen/qwen3.8-27b"

        # 4. Only llama available
        mock_resp4 = MagicMock(status_code=200)
        mock_resp4.json.return_value = {"data": [{"id": "llama-3.3-70b-versatile"}]}
        with patch("requests.get", return_value=mock_resp4):
            assert select_groq_model("key") == "llama-3.3-70b-versatile"

    def test_select_groq_model_rejects_audio_and_errors_when_no_chat_model(self, monkeypatch):
        """Verifies that non-chat audio/guard models are ignored and LLMExecutionError is raised if no chat model."""
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "data": [
                {"id": "whisper-large-v3"},
                {"id": "meta-llama/llama-prompt-guard-2-86m"},
            ]
        }
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(LLMExecutionError) as exc_info:
                select_groq_model("key")
            assert "No compatible chat completion models" in str(exc_info.value)
            assert exc_info.value.status_code == 200

    def test_select_groq_model_network_error(self):
        """Verifies network connection error during model query raises LLMExecutionError with status_code=None."""
        import requests
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
            with pytest.raises(LLMExecutionError) as exc_info:
                select_groq_model("key")
            assert exc_info.value.provider == "groq"
            assert exc_info.value.status_code is None

    def test_execute_groq_llm_synthesis_rate_limited(self):
        """Verifies HTTP 429 response from Groq raises LLMExecutionError with status_code=429."""
        snap = fetch_company_snapshot("NVDA", use_live=False)
        peers = fetch_peer_multiples("NVDA", use_live=False)
        fsa = compute_fsa_summary(snap)
        quant = compute_quant_risk_summary(snap)
        assumptions = synthesize_deterministic_assumptions(snap, fsa, quant)
        val = run_valuation_suite(snap, assumptions, quant, peers)

        mock_resp = MagicMock(status_code=429, text='{"error": {"message": "Rate limit exceeded"}}')
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(LLMExecutionError) as exc_info:
                execute_groq_llm_synthesis(snap, fsa, quant, val, assumptions, "test-key", "openai/gpt-oss-120b")
            assert exc_info.value.status_code == 429
            assert "429" in str(exc_info.value)

    def test_execute_groq_llm_synthesis_malformed_json_or_non_dict(self):
        """Verifies non-dict or invalid JSON response raises LLMExecutionError."""
        snap = fetch_company_snapshot("NVDA", use_live=False)
        peers = fetch_peer_multiples("NVDA", use_live=False)
        fsa = compute_fsa_summary(snap)
        quant = compute_quant_risk_summary(snap)
        assumptions = synthesize_deterministic_assumptions(snap, fsa, quant)
        val = run_valuation_suite(snap, assumptions, quant, peers)

        # 1. Invalid JSON string
        mock_resp1 = MagicMock(status_code=200)
        mock_resp1.json.return_value = {"choices": [{"message": {"content": "Not JSON at all"}}]}
        with patch("requests.post", return_value=mock_resp1):
            with pytest.raises(LLMExecutionError):
                execute_groq_llm_synthesis(snap, fsa, quant, val, assumptions, "test-key", "openai/gpt-oss-120b")

        # 2. JSON array instead of JSON object
        mock_resp2 = MagicMock(status_code=200)
        mock_resp2.json.return_value = {"choices": [{"message": {"content": '["item1", "item2"]'}}]}
        with patch("requests.post", return_value=mock_resp2):
            with pytest.raises(LLMExecutionError) as exc_info:
                execute_groq_llm_synthesis(snap, fsa, quant, val, assumptions, "test-key", "openai/gpt-oss-120b")
            assert "Expected JSON object" in str(exc_info.value)

    def test_execute_groq_llm_synthesis_citation_normalization(self):
        """Verifies that synthesis normalization automatically enforces authentic [FILING-CHUNK-X] citations."""
        snap = fetch_company_snapshot("NVDA", use_live=False)
        peers = fetch_peer_multiples("NVDA", use_live=False)
        fsa = compute_fsa_summary(snap)
        quant = compute_quant_risk_summary(snap)
        assumptions = synthesize_deterministic_assumptions(snap, fsa, quant)
        val = run_valuation_suite(snap, assumptions, quant, peers)

        mock_payload = {
            "bull_thesis_points": ["Demand is surging globally", "High gross margin resilience", "CUDA moat defense"],
            "bear_thesis_points": ["Export restrictions impact revenue", "Packaging bottlenecks persist"],
            "key_catalysts": ["Launch of next-gen architecture"],
            "key_risks": ["Foundry capacity constraints"],
            "debate_synthesis_summary": "While demand is strong, supply chain risks persist.",
        }
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(mock_payload)}}]}

        with patch("requests.post", return_value=mock_resp):
            res_assumptions, latency = execute_groq_llm_synthesis(
                snap, fsa, quant, val, assumptions, "test-key", "openai/gpt-oss-120b"
            )

        assert len(res_assumptions.bull_thesis_points) == 3
        assert len(res_assumptions.bear_thesis_points) == 3
        assert len(res_assumptions.key_catalysts) == 3
        assert len(res_assumptions.key_risks) == 3
        assert all("[FILING-CHUNK-" in p for p in res_assumptions.bull_thesis_points)
        assert all("[FILING-CHUNK-" in p for p in res_assumptions.bear_thesis_points)
        assert "[FILING-CHUNK-" in res_assumptions.debate_synthesis_summary

    def test_math_invariance_deterministic_vs_llm_mocked(self, monkeypatch):
        """Verifies 100% mathematical invariance across modes with mocked LLM output on TCS.NS."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_valid_key")

        mock_models_resp = MagicMock(status_code=200)
        mock_models_resp.json.return_value = {"data": [{"id": "openai/gpt-oss-120b"}]}

        mock_chat_payload = {
            "bull_thesis_points": [
                "Strong digital transformation deal book. [FILING-CHUNK-1]",
                "Operating margin discipline across core geographies. [FILING-CHUNK-2]",
                "Unmatched enterprise client retention. [FILING-CHUNK-4]",
            ],
            "bear_thesis_points": [
                "Discretionary tech spend headwinds in BFSI. [FILING-CHUNK-5]",
                "Working capital DSO elongation. [FILING-CHUNK-3]",
                "Subcontractor cost wage pressure. [FILING-CHUNK-2]",
            ],
            "key_catalysts": [
                "Recovery in banking vertical tech budgets. [FILING-CHUNK-1]",
                "Margin expansion from automation. [FILING-CHUNK-2]",
                "Accelerated mega-deal conversion. [FILING-CHUNK-6]",
            ],
            "key_risks": [
                "Prolonged macro slowdown in North America. [FILING-CHUNK-5]",
                "Attrition and wage inflation pressure. [FILING-CHUNK-2]",
                "Regulatory visa and cross-border restrictions. [FILING-CHUNK-5]",
            ],
            "debate_synthesis_summary": "Balanced execution between margin resilience and deal growth. [FILING-CHUNK-1] [FILING-CHUNK-5]",
        }
        mock_chat_resp = MagicMock(status_code=200)
        mock_chat_resp.json.return_value = {"choices": [{"message": {"content": json.dumps(mock_chat_payload)}}]}

        with patch("requests.get", return_value=mock_models_resp), patch("requests.post", return_value=mock_chat_resp):
            # Run deterministic
            flow_det = EquiSightResearchFlow()
            flow_det.kickoff(inputs={"ticker": "TCS.NS", "mode": "deterministic", "use_live": False})

            # Run LLM
            flow_llm = EquiSightResearchFlow()
            flow_llm.kickoff(inputs={"ticker": "TCS.NS", "mode": "llm", "use_live": False})

            assert flow_det.state.status == "SUCCESS"
            assert flow_llm.state.status == "SUCCESS"

            # Check 100% mathematical equality
            v_det = flow_det.state.valuation_output
            v_llm = flow_llm.state.valuation_output
            assert v_det.probability_weighted_intrinsic_value == v_llm.probability_weighted_intrinsic_value
            assert v_det.implied_upside_pct == v_llm.implied_upside_pct
            assert v_det.base_wacc == v_llm.base_wacc
            assert v_det.reverse_dcf_implied_revenue_cagr == v_llm.reverse_dcf_implied_revenue_cagr

            # Check scenario drivers
            for sc in ["bear", "base", "bull"]:
                d1 = getattr(flow_det.state.dcf_assumptions, sc)
                d2 = getattr(flow_llm.state.dcf_assumptions, sc)
                assert d1.revenue_cagr_5yr == d2.revenue_cagr_5yr
                assert d1.target_ebit_margin_yr5 == d2.target_ebit_margin_yr5
                assert d1.effective_tax_rate == d2.effective_tax_rate
                assert d1.terminal_growth_rate == d2.terminal_growth_rate
                assert d1.wacc_adjustment_bps == d2.wacc_adjustment_bps

            assert flow_llm.state.eval_scorecard.passed is True
            assert flow_llm.state.eval_scorecard.overall_eval_score == 100.0

    def test_math_invariance_deterministic_vs_llm(self):
        """Verifies strict 100% mathematical invariance across deterministic and live Groq llm modes."""
        import os

        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set in environment.")

        # 1. Execute deterministic pipeline
        flow_det = EquiSightResearchFlow()
        flow_det.kickoff(inputs={"ticker": "NVDA", "mode": "deterministic", "use_live": False})

        # 2. Execute live Groq LLM pipeline
        flow_llm = EquiSightResearchFlow()
        flow_llm.kickoff(inputs={"ticker": "NVDA", "mode": "llm", "use_live": False})

        assert flow_det.state.status == "SUCCESS"
        assert flow_llm.state.status == "SUCCESS"

        # 3. Verify exact mathematical equality down to the last decimal place
        det_val = flow_det.state.valuation_output
        llm_val = flow_llm.state.valuation_output

        assert det_val.probability_weighted_intrinsic_value == llm_val.probability_weighted_intrinsic_value
        assert det_val.implied_upside_pct == llm_val.implied_upside_pct
        assert det_val.base_wacc == llm_val.base_wacc
        assert det_val.reverse_dcf_implied_revenue_cagr == llm_val.reverse_dcf_implied_revenue_cagr

        det_fsa = flow_det.state.fsa_summary
        llm_fsa = flow_llm.state.fsa_summary
        assert det_fsa.reconstructed_roe == llm_fsa.reconstructed_roe
        assert det_fsa.roe == llm_fsa.roe

        # 4. Verify numerical scenario drivers equality
        for sc in ["bear", "base", "bull"]:
            d_det = getattr(flow_det.state.dcf_assumptions, sc)
            d_llm = getattr(flow_llm.state.dcf_assumptions, sc)
            assert d_det.revenue_cagr_5yr == d_llm.revenue_cagr_5yr
            assert d_det.target_ebit_margin_yr5 == d_llm.target_ebit_margin_yr5
            assert d_det.effective_tax_rate == d_llm.effective_tax_rate
            assert d_det.da_pct_revenue == d_llm.da_pct_revenue
            assert d_det.capex_pct_revenue == d_llm.capex_pct_revenue
            assert d_det.nwc_pct_revenue_change == d_llm.nwc_pct_revenue_change
            assert d_det.terminal_growth_rate == d_llm.terminal_growth_rate
            assert d_det.wacc_adjustment_bps == d_llm.wacc_adjustment_bps

        # 5. Verify LLM-specific telemetry and qualitative outputs
        assert flow_llm.state.llm_model_used.startswith("groq/")
        assert flow_llm.state.llm_latency_ms > 0
        assert len(flow_llm.state.dcf_assumptions.bull_thesis_points) == 3
        assert len(flow_llm.state.dcf_assumptions.bear_thesis_points) == 3
        assert len(flow_llm.state.dcf_assumptions.key_catalysts) == 3
        assert len(flow_llm.state.dcf_assumptions.key_risks) == 3
        assert all("[FILING-CHUNK-" in p for p in flow_llm.state.dcf_assumptions.bull_thesis_points)
        assert all("[FILING-CHUNK-" in p for p in flow_llm.state.dcf_assumptions.bear_thesis_points)
        assert "[FILING-CHUNK-" in flow_llm.state.dcf_assumptions.debate_synthesis_summary

        # 6. Verify 3-Layer Eval Scorecard passed 100/100
        assert flow_llm.state.eval_scorecard.passed is True
        assert flow_llm.state.eval_scorecard.overall_eval_score == 100.0
        assert len(flow_llm.state.eval_scorecard.flagged_discrepancies) == 0



