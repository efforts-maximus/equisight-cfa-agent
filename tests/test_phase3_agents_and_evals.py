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
import re
import pytest

from src.agents.research_flow import (
    EquiSightResearchFlow,
    EquiSightState,
    format_cfa_report,
    load_agent_configs,
    load_task_configs,
    repair_cfa_report,
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
