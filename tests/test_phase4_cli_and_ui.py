"""Phase 4 & Phase 5 Integration Test Suite: CLI and Streamlit UI Components.

Tests CLI argument parsing, multi-ticker artifact export, JSON audit bundle fidelity,
and headless Plotly chart builders in app.py with zero stubs or placeholders.
"""

import json
import os
from pathlib import Path
import pytest
import plotly.graph_objects as go

from app import (
    build_dupont_waterfall_chart,
    build_football_field_chart,
    build_monte_carlo_histogram,
    build_sensitivity_heatmap,
    execute_pipeline,
)
from main import build_parser, export_artifacts, print_executive_summary, run_cli
from src.agents.research_flow import EquiSightResearchFlow
from src.data.market_gateway import fetch_company_snapshot, fetch_peer_multiples
from src.engines.fsa_engine import compute_fsa_summary
from src.engines.quant_risk_engine import compute_quant_risk_summary, run_monte_carlo_dcf
from src.engines.valuation_engine import run_valuation_suite
from src.agents.research_flow import synthesize_deterministic_assumptions


class TestCLIArgumentParsing:
    """Tests command-line interface arguments and flag parsing."""

    def test_cli_default_arguments(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.ticker == "VARUNBEV.NS"
        assert args.mode == "deterministic"
        assert args.use_live_data is False
        assert args.output_dir == "outputs"

    def test_cli_custom_arguments(self):
        parser = build_parser()
        args = parser.parse_args([
            "--ticker", "NVDA",
            "--mode", "llm",
            "--use-live-data",
            "--output-dir", "custom_outputs",
        ])
        assert args.ticker == "NVDA"
        assert args.mode == "llm"
        assert args.use_live_data is True
        assert args.output_dir == "custom_outputs"


class TestCLIExecutionAndArtifactExport:
    """Tests running the CLI end-to-end and exporting artifacts to disk."""

    @pytest.mark.parametrize("ticker", ["VARUNBEV.NS", "ZOMATO.NS", "NVDA"])
    def test_cli_run_and_artifact_generation(self, tmp_path, ticker):
        parser = build_parser()
        out_dir = str(tmp_path / "test_out")
        args = parser.parse_args([
            "--ticker", ticker,
            "--mode", "deterministic",
            "--output-dir", out_dir,
        ])
        exit_code = run_cli(args)
        assert exit_code == 0

        clean_ticker = ticker.replace("/", "_")
        report_file = Path(out_dir) / f"{clean_ticker}_research_report.md"
        bundle_file = Path(out_dir) / f"{clean_ticker}_audit_bundle.json"

        assert report_file.exists(), f"Expected {report_file} to exist"
        assert bundle_file.exists(), f"Expected {bundle_file} to exist"

        # Verify Markdown report content
        report_text = report_file.read_text(encoding="utf-8")
        assert len(report_text) > 1000
        assert "# 1. Executive Summary & Recommendation" in report_text
        assert "# 2. Financial Statement & DuPont Analysis" in report_text
        assert "# 3. Forensic Earnings Quality & Solvency" in report_text
        assert "# 4. Dialectical Bull vs. Bear Debate" in report_text
        assert "# 5. DCF & Reverse-DCF Valuation" in report_text
        assert "# 6. Quantitative Risk & Monte Carlo Simulation" in report_text

        # Verify JSON audit bundle schema
        with open(bundle_file, "r", encoding="utf-8") as f:
            bundle_data = json.load(f)

        assert bundle_data["ticker"] == ticker
        assert bundle_data["mode"] == "deterministic"
        assert bundle_data["status"] == "SUCCESS"
        assert bundle_data["snapshot"]["ticker"] == ticker
        assert "fsa_summary" in bundle_data
        assert "quant_summary" in bundle_data
        assert "dcf_assumptions" in bundle_data
        assert "valuation_output" in bundle_data
        assert "monte_carlo_output" in bundle_data
        assert "eval_scorecard" in bundle_data
        assert bundle_data["eval_scorecard"]["passed"] is True
        assert len(bundle_data["telemetry"]) >= 3

    def test_print_executive_summary_executes_without_error(self, capsys):
        flow = EquiSightResearchFlow()
        flow.kickoff(inputs={"ticker": "TCS.NS", "mode": "deterministic"})
        print_executive_summary(flow.state)
        captured = capsys.readouterr()
        assert "EQUISIGHT CFA RESEARCH: Tata Consultancy Services Limited (TCS.NS)" in captured.out
        assert "5-WAY DUPONT ROE:" in captured.out
        assert "3-LAYER EVAL SCORECARD:" in captured.out

    def test_cli_filename_sanitization_with_special_characters(self, tmp_path):
        from types import SimpleNamespace
        dummy_state = SimpleNamespace(
            ticker="NASDAQ:NVDA/CORP",
            mode="deterministic",
            status="SUCCESS",
            repair_attempts=0,
            report_markdown="# Test Report",
            snapshot=None,
            fsa_summary=None,
            quant_summary=None,
            dcf_assumptions=None,
            valuation_output=None,
            monte_carlo_output=None,
            peer_multiples=[],
            eval_scorecard=None,
            telemetry=[],
        )
        report_file, bundle_file = export_artifacts(dummy_state, tmp_path)
        assert report_file.exists()
        assert bundle_file.exists()
        assert ":" not in report_file.name
        assert "/" not in report_file.name
        assert report_file.name == "NASDAQ_NVDA_CORP_research_report.md"

    def test_print_executive_summary_graceful_handling_of_none_state(self, capsys):
        # Must not raise AttributeError
        print_executive_summary(None)
        captured = capsys.readouterr()
        assert "[-] Insufficient analytical outputs" in captured.out


class TestHeadlessPlotlyChartBuilders:
    """Tests headless rendering of all Plotly figures in app.py."""

    @pytest.fixture
    def benchmark_state(self):
        snapshot = fetch_company_snapshot("VARUNBEV.NS", use_live=False)
        peers = fetch_peer_multiples("VARUNBEV.NS", use_live=False)
        fsa = compute_fsa_summary(snapshot)
        quant = compute_quant_risk_summary(snapshot)
        assumptions = synthesize_deterministic_assumptions(snapshot, fsa, quant)
        val = run_valuation_suite(snapshot, assumptions, quant, peers)
        mc = run_monte_carlo_dcf(snapshot, assumptions, num_simulations=1000, seed=42)
        return {
            "snapshot": snapshot,
            "fsa": fsa,
            "quant": quant,
            "assumptions": assumptions,
            "val": val,
            "mc": mc,
        }

    def test_football_field_chart_builder(self, benchmark_state):
        val = benchmark_state["val"]
        snapshot = benchmark_state["snapshot"]
        fig = build_football_field_chart(
            ranges=val.football_field_ranges,
            current_price=snapshot.current_price,
            currency=snapshot.currency,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # bar trace + midpoint scatter trace
        assert fig.layout.title.text == "Valuation Methodology Football Field"
        # Assert exact acronym casing and high-contrast font color
        assert "DCF Scenario Range (Bear-Bull)" in fig.data[0].y
        assert "WACC/g Sensitivity Range" in fig.data[0].y
        assert fig.data[0].textfont.color == "#F8FAFC"

    def test_dupont_waterfall_chart_builder(self, benchmark_state):
        fsa = benchmark_state["fsa"]
        fig = build_dupont_waterfall_chart(
            tax_burden=fsa.tax_burden,
            interest_burden=fsa.interest_burden,
            ebit_margin=fsa.ebit_margin,
            asset_turnover=fsa.asset_turnover,
            financial_leverage=fsa.financial_leverage,
            reconstructed_roe=fsa.reconstructed_roe,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 6
        assert fig.layout.title.text.startswith("2027 CFA Level 1: 5-Way DuPont")

    def test_sensitivity_heatmap_builder(self, benchmark_state):
        val = benchmark_state["val"]
        snapshot = benchmark_state["snapshot"]
        fig = build_sensitivity_heatmap(
            wacc_axes=val.wacc_sensitivity_axes,
            tgr_axes=val.tgr_sensitivity_axes,
            price_matrix=val.sensitivity_matrix_prices,
            current_price=snapshot.current_price,
            currency=snapshot.currency,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        heatmap = fig.data[0]
        assert len(heatmap.z) == 5
        assert len(heatmap.z[0]) == 5
        assert fig.layout.title.text.startswith("5x5 Intrinsic Share Price Sensitivity Matrix")

    def test_monte_carlo_histogram_builder(self, benchmark_state):
        mc = benchmark_state["mc"]
        snapshot = benchmark_state["snapshot"]
        fig = build_monte_carlo_histogram(
            simulated_values=mc.simulated_values_sample,
            p5=mc.p5,
            p50=mc.p50,
            p95=mc.p95,
            current_price=snapshot.current_price,
            currency=snapshot.currency,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.layout.title.text.startswith("Vectorized 10,000-Path Monte Carlo")


class TestWhatIfOverrides:
    """Tests real-time interactive What-If recalculations in the Streamlit terminal."""

    def test_what_if_wacc_increase_decreases_target_price(self):
        snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
        peers = fetch_peer_multiples("TCS.NS", use_live=False)
        fsa = compute_fsa_summary(snapshot)
        quant = compute_quant_risk_summary(snapshot)
        assumptions = synthesize_deterministic_assumptions(snapshot, fsa, quant)
        
        base_val = run_valuation_suite(snapshot, assumptions, quant, peers)

        # Apply +100 bps WACC adjustment
        adj_assumptions = assumptions.model_copy(deep=True)
        adj_assumptions.base.wacc_adjustment_bps += 100.0
        adj_val = run_valuation_suite(snapshot, adj_assumptions, quant, peers)

        # Higher discount rate must reduce intrinsic DCF valuation
        assert adj_val.base_dcf.intrinsic_value_per_share < base_val.base_dcf.intrinsic_value_per_share

    def test_what_if_terminal_growth_increase_increases_target_price(self):
        snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
        peers = fetch_peer_multiples("TCS.NS", use_live=False)
        fsa = compute_fsa_summary(snapshot)
        quant = compute_quant_risk_summary(snapshot)
        assumptions = synthesize_deterministic_assumptions(snapshot, fsa, quant)
        
        base_val = run_valuation_suite(snapshot, assumptions, quant, peers)

        # Apply terminal growth increase within safe bounds (g < WACC)
        adj_assumptions = assumptions.model_copy(deep=True)
        adj_assumptions.base.terminal_growth_rate = min(0.040, base_val.base_wacc - 0.010)
        adj_val = run_valuation_suite(snapshot, adj_assumptions, quant, peers)

        if adj_assumptions.base.terminal_growth_rate > assumptions.base.terminal_growth_rate:
            assert adj_val.base_dcf.intrinsic_value_per_share > base_val.base_dcf.intrinsic_value_per_share

    def test_what_if_extreme_slider_boundary_conditions(self):
        snapshot = fetch_company_snapshot("VARUNBEV.NS", use_live=False)
        peers = fetch_peer_multiples("VARUNBEV.NS", use_live=False)
        fsa = compute_fsa_summary(snapshot)
        quant = compute_quant_risk_summary(snapshot)
        assumptions = synthesize_deterministic_assumptions(snapshot, fsa, quant)
        val = run_valuation_suite(snapshot, assumptions, quant, peers)

        # Extreme -200 bps WACC adjustment and maximum terminal growth (0.050)
        adj_assumptions = assumptions.model_copy(deep=True)
        adj_assumptions.base.wacc_adjustment_bps = -200.0
        adjusted_wacc = max(0.040, val.base_wacc - 0.020)
        max_safe_tgr = max(0.010, adjusted_wacc - 0.015)
        adj_assumptions.base.terminal_growth_rate = min(0.050, max_safe_tgr)

        # Must execute cleanly without Gordon growth division-by-zero or model validator errors
        adj_val = run_valuation_suite(snapshot, adj_assumptions, quant, peers)
        assert adj_val.base_dcf.intrinsic_value_per_share > 0
        assert adj_val.base_dcf.terminal_growth_rate < adj_val.base_dcf.wacc

    def test_execute_pipeline_invalid_ticker_returns_none_gracefully(self):
        # Invalid ticker must return None and log error instead of crashing with unhandled exception
        result = execute_pipeline("NON_EXISTENT_TICKER_XYZ_123", mode="deterministic")
        assert result is None

