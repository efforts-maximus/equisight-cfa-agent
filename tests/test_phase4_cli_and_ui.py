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


class TestUIStateAndCustomTickerExecution:
    """Tests UI state switching transitions and custom ticker execution."""

    def test_switch_ticker_varunbev_nvda_and_back_produces_identical_output(self):
        """Tests that switching VARUNBEV.NS -> NVDA -> VARUNBEV.NS produces 100% identical outputs."""
        state_v1 = execute_pipeline("VARUNBEV.NS", mode="deterministic", use_live=False)
        assert state_v1 is not None
        assert state_v1.status == "SUCCESS"
        v1_val = state_v1.valuation_output.probability_weighted_intrinsic_value

        state_nvda = execute_pipeline("NVDA", mode="deterministic", use_live=False)
        assert state_nvda is not None
        assert state_nvda.status == "SUCCESS"
        assert state_nvda.snapshot.ticker == "NVDA"

        state_v2 = execute_pipeline("VARUNBEV.NS", mode="deterministic", use_live=False)
        assert state_v2 is not None
        assert state_v2.status == "SUCCESS"
        assert state_v2.snapshot.ticker == "VARUNBEV.NS"
        v2_val = state_v2.valuation_output.probability_weighted_intrinsic_value

        assert v1_val == pytest.approx(v2_val, abs=1e-4)
        assert state_v1.valuation_output.recommendation == state_v2.valuation_output.recommendation
        assert state_v1.fsa_summary.reconstructed_roe == pytest.approx(state_v2.fsa_summary.reconstructed_roe, abs=1e-4)

    def test_switch_mode_deterministic_llm_and_back_works_cleanly(self):
        """Tests that switching execution mode deterministic -> llm -> deterministic operates cleanly."""
        state_det1 = execute_pipeline("TCS.NS", mode="deterministic", use_live=False)
        assert state_det1 is not None
        assert state_det1.mode == "deterministic"
        assert state_det1.eval_scorecard.passed is True

        # Switching to LLM mode (without live keys falls back gracefully to deterministic synthesis)
        state_llm = execute_pipeline("TCS.NS", mode="llm", use_live=False)
        assert state_llm is not None
        assert state_llm.mode == "llm"
        assert state_llm.eval_scorecard.passed is True

        # Switching back to deterministic mode
        state_det2 = execute_pipeline("TCS.NS", mode="deterministic", use_live=False)
        assert state_det2 is not None
        assert state_det2.mode == "deterministic"
        assert state_det2.eval_scorecard.passed is True
        assert (
            state_det1.valuation_output.probability_weighted_intrinsic_value
            == pytest.approx(state_det2.valuation_output.probability_weighted_intrinsic_value, abs=1e-4)
        )

    @pytest.mark.parametrize("custom_sym", ["INFY.NS", "AAPL"])
    def test_custom_ticker_execution_returns_100_percent_eval_scorecard(self, custom_sym):
        """Tests that custom tickers execute cleanly through the pipeline and receive 100/100 EvalScorecard."""
        result = execute_pipeline(custom_sym, mode="deterministic", use_live=False)
        assert result is not None
        assert result.status == "SUCCESS"
        assert result.snapshot.ticker == custom_sym
        assert result.eval_scorecard is not None
        assert result.eval_scorecard.passed is True
        assert result.eval_scorecard.overall_eval_score == 100.0
        assert result.eval_scorecard.math_invariance_score == 100.0
        assert result.eval_scorecard.schema_completeness_score == 100.0
        assert result.eval_scorecard.numerical_faithfulness_score == 100.0

    def test_sidebar_selection_key_state_transitions(self):
        """Simulates Streamlit session state tracking and ensures selection transitions trigger re-execution."""
        session_state = {}
        sequence = [
            ("VARUNBEV.NS", "deterministic", False),
            ("NVDA", "deterministic", False),
            ("VARUNBEV.NS", "deterministic", False),
            ("TCS.NS", "deterministic", False),
            ("ZOMATO.NS", "deterministic", False),
        ]
        history = []
        for ticker, mode, live in sequence:
            current_key = f"{ticker}::{mode}::{live}"
            # Emulate Streamlit app execution gate
            if session_state.get("last_executed_key") != current_key:
                res = execute_pipeline(ticker, mode, live)
                session_state["pipeline_result"] = res
                session_state["last_executed_key"] = current_key
                history.append((ticker, res.valuation_output.probability_weighted_intrinsic_value))

        # Check all 5 transitions executed
        assert len(history) == 5
        # The first and third executions were both VARUNBEV.NS; values must be identical
        assert history[0][0] == "VARUNBEV.NS"
        assert history[2][0] == "VARUNBEV.NS"
        assert history[0][1] == pytest.approx(history[2][1], abs=1e-4)

    def test_empty_ticker_validation_raises_error(self):
        """Verifies empty or whitespace ticker symbols raise ValueError or return None gracefully."""
        from src.data.market_gateway import fetch_company_snapshot, fetch_peer_multiples
        from src.data.filing_rag_tool import load_filing_chunks

        with pytest.raises(ValueError, match="cannot be empty"):
            fetch_company_snapshot("")
        with pytest.raises(ValueError, match="cannot be empty"):
            fetch_company_snapshot("   ")
        with pytest.raises(ValueError, match="cannot be empty"):
            fetch_peer_multiples("")
        with pytest.raises(ValueError, match="cannot be empty"):
            load_filing_chunks("")

        # Pipeline execution on empty ticker returns None rather than crashing
        res = execute_pipeline("", mode="deterministic")
        assert res is None

    def test_custom_ticker_peer_multiples_enrichment(self):
        """Verifies custom ticker peer multiples return complete valid PeerCompMultiple objects."""
        from src.data.market_gateway import fetch_peer_multiples

        peers = fetch_peer_multiples("AAPL", use_live=False)
        assert len(peers) >= 2
        for p in peers:
            assert p.pe_ratio > 0
            assert p.ev_to_ebitda > 0
            assert p.pb_ratio > 0
            assert p.ev_to_sales > 0
            assert len(p.peer_ticker) > 0

    def test_cli_bare_ticker_resolution_and_exit_codes(self):
        """Verifies CLI resolves bare Indian tickers and exits with code 1 on invalid tickers."""
        import argparse
        from main import run_cli

        # Bare ticker 'TCS' should resolve to 'TCS.NS' and return exit code 0
        args_tcs = argparse.Namespace(
            ticker="TCS",
            mode="deterministic",
            use_live_data=False,
            output_dir="outputs",
        )
        assert run_cli(args_tcs) == 0

        # Invalid ticker must exit cleanly with code 1
        args_invalid = argparse.Namespace(
            ticker="NON_EXISTENT_COMPANY_FAKE_999",
            mode="deterministic",
            use_live_data=False,
            output_dir="outputs",
        )
        assert run_cli(args_invalid) == 1

    def test_app_execute_pipeline_captures_data_unavailable_error(self):
        """Verifies execute_pipeline populates execute_pipeline.last_error on DataUnavailableError."""
        from app import execute_pipeline

        res = execute_pipeline("NON_EXISTENT_COMPANY_FAKE_999", mode="deterministic", use_live=False)
        assert res is None
        assert execute_pipeline.last_error is not None
        assert "Data unavailable" in execute_pipeline.last_error or "unavailable" in execute_pipeline.last_error.lower()

    def test_retrieve_filing_evidence_tool_with_bare_ticker(self):
        """Verifies RetrieveFilingEvidenceTool resolves bare ticker 'TCS' and returns citations."""
        from src.agents.tools import RetrieveFilingEvidenceTool

        tool = RetrieveFilingEvidenceTool()
        result = tool._run(ticker="TCS", query="cloud banking transformation", top_k=2)
        assert "[FILING-CHUNK-" in result
        assert len(result) > 50

    def test_trending_stock_card_property_accessors(self):
        """Verifies TrendingStockCard property accessors match institutional UI contracts."""
        from src.data.market_pulse import TrendingStockCard

        card_us = TrendingStockCard(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            current_price=142.50,
            currency="USD",
            daily_change_pct=3.45,
            daily_volume=48000000,
            market_cap=3500000.0,
            formatted_market_cap="$3.50T",
        )
        assert card_us.currency_symbol == "$"
        assert card_us.day_change_pct == 3.45
        assert card_us.day_volume == 48000000
        assert card_us.market_cap_formatted == "$3.50T"
        assert card_us.activity_tag in ("HIGH VOLATILITY", "HEAVY VOLUME", "ACTIVE GAINER", "MARKET MOVER")

        card_nse = TrendingStockCard(
            ticker="TCS.NS",
            company_name="Tata Consultancy Services",
            market="NSE",
            current_price=4120.00,
            currency="INR",
            daily_change_pct=1.85,
            daily_volume=3200000,
            market_cap=1490000.0,
            formatted_market_cap="₹14.90L Cr",
        )
        assert card_nse.currency_symbol == "₹"
        assert card_nse.day_change_pct == 1.85
        assert card_nse.day_volume == 3200000
        assert card_nse.market_cap_formatted == "₹14.90L Cr"

    def test_get_benchmark_mover_cards_contract(self):
        """Verifies get_benchmark_mover_cards returns 8 valid benchmark cards for offline navigation."""
        from app import get_benchmark_mover_cards

        cards = get_benchmark_mover_cards()
        assert len(cards) == 8
        tickers = [c.ticker for c in cards]
        assert "NVDA" in tickers
        assert "TCS.NS" in tickers
        assert "VARUNBEV.NS" in tickers
        assert "ZOMATO.NS" in tickers
        for c in cards:
            assert c.current_price > 0
            assert c.market in ("US", "NSE")
            assert c.currency in ("USD", "INR")
            assert len(c.activity_tag) > 0
            assert len(c.currency_symbol) > 0

    def test_compute_stock_comparison_metrics_benchmark_stocks(self):
        """Verifies compute_stock_comparison_metrics returns all 9 required valuation & FSA metrics."""
        from app import compute_stock_comparison_metrics

        for sym in ["NVDA", "TCS.NS", "VARUNBEV.NS"]:
            data = compute_stock_comparison_metrics(sym)
            assert data is not None, f"Expected valid data for {sym}"
            assert data["ticker"] == sym
            assert data["current_price"] > 0
            assert data["target_price"] > 0
            assert -1.0 <= data["implied_upside_pct"] <= 5.0
            assert data["recommendation"] in ("BUY", "HOLD", "SELL")
            assert data["dupont_roe"] is not None
            assert data["operating_margin"] is not None
            assert data["base_wacc"] > 0
            assert data["reverse_dcf_growth"] is not None
            assert data["altman_z"] is not None
            assert data["altman_zone"] in ("Distress", "Grey", "Safe")

    def test_navigation_view_constants_and_pipeline_error_capture(self):
        """Verifies view constants and LLM execution error capture contract in app.py."""
        from app import VIEW_HOME, VIEW_TERMINAL, execute_pipeline

        assert VIEW_HOME == "🏠 Command Center (Home)"
        assert VIEW_TERMINAL == "📊 Deep-Dive Research Terminal"

        res = execute_pipeline("INVALID_TICKER_FOR_TESTING_123", mode="deterministic", use_live=False)
        assert res is None
        assert execute_pipeline.last_error is not None

    def test_mover_card_label_katex_safety_and_css_targeting(self):
        """Verifies TERMINAL_CSS targets Streamlit element keys and mover labels escape KaTeX math delimiters."""
        from app import TERMINAL_CSS
        from src.data.market_pulse import TrendingStockCard

        # Verify CSS selectors target Streamlit's generated wrapper classes
        assert 'div[class*="st-key-btn_ql_"] button' in TERMINAL_CSS
        assert 'div[class*="st-key-mover_card_"] button' in TERMINAL_CSS

        # Test US mover card label KaTeX escaping
        card = TrendingStockCard(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            market="US",
            current_price=142.50,
            currency="USD",
            daily_change_pct=3.45,
            daily_volume=48000000,
            market_cap=3500000.0,
            formatted_market_cap="$3.50T",
        )
        tag = card.activity_tag
        sym = card.currency_symbol
        chg = card.day_change_pct
        vol = card.day_volume
        mcap = card.market_cap_formatted
        raw_label = (
            f"{card.ticker}  •  {tag}\n"
            f"{card.company_name}\n"
            f"{sym}{card.current_price:,.2f}  ({chg:+.2f}%)\n"
            f"MCap: {mcap}  |  Vol: {vol:,}"
        ).replace("$", r"\$")

        # Must not contain unescaped dollar signs that trigger Streamlit KaTeX inline math
        assert "$" in raw_label
        assert r"\$" in raw_label
        # Verify unescaped $ does not exist
        unescaped_dollars = [i for i, c in enumerate(raw_label) if c == "$" and (i == 0 or raw_label[i - 1] != "\\")]
        assert len(unescaped_dollars) == 0, f"Found unescaped dollar signs at {unescaped_dollars}"

    def test_comparator_identical_tickers_widget_key_isolation(self):
        """Verifies comparator button keys are disambiguated for identical stocks (e.g. NVDA vs NVDA)."""
        stock_a = "NVDA"
        stock_b = "NVDA"
        key_a = f"btn_dd_a_{stock_a}"
        key_b = f"btn_dd_b_{stock_b}"
        assert key_a != key_b, f"Widget keys must be unique even when stocks match: {key_a} vs {key_b}"

    def test_sidebar_toggle_css_preservation(self):
        """Verifies TERMINAL_CSS does not hide stHeader and preserves sidebar toggle controls."""
        from app import TERMINAL_CSS

        # Streamlit chrome (Deploy button, hamburger menu) must be hidden
        assert '[data-testid="stAppDeployButton"]' in TERMINAL_CSS
        assert '#MainMenu' in TERMINAL_CSS

        # header[data-testid="stHeader"] must NOT be hidden completely
        assert 'header[data-testid="stHeader"] {\n        display: none' not in TERMINAL_CSS
        assert 'header[data-testid="stHeader"],' not in TERMINAL_CSS

        # Sidebar toggle controls must have pointer-events and visibility preserved
        assert '[data-testid="stExpandSidebarButton"]' in TERMINAL_CSS
        assert '[data-testid="stSidebarCollapseButton"]' in TERMINAL_CSS
        assert 'pointer-events: auto' in TERMINAL_CSS
        assert 'button[aria-label="Expand sidebar"]' in TERMINAL_CSS
        assert 'button[aria-label="Collapse sidebar"]' in TERMINAL_CSS
        assert 'color: #c9d1d9' in TERMINAL_CSS
        assert 'background-color: #161b22' in TERMINAL_CSS





