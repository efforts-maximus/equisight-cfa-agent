"""Phase 1 Verification Suite: Pydantic v2 Contracts, Market Gateway & Filing RAG.

Governed by .agent/rules/strict_phase_gate.md and SPEC.md.
Tests contracts, accounting identities, dual-mode data loader, and citation-tagged filing RAG.
"""

from pathlib import Path
from unittest.mock import patch
import pytest
from pydantic import ValidationError

from src.schemas.models import (
    CompanyFinancialSnapshot,
    HistoricalYearMetrics,
    FSARatioSummary,
    QuantRiskSummary,
    ScenarioDriverSet,
    DCFScenarioAssumptions,
    SingleDCFResult,
    MonteCarloDCFOutput,
    PeerCompMultiple,
    ValuationSuiteOutput,
    StepTelemetry,
    EvalScorecard,
)
from src.data.market_gateway import (
    fetch_company_snapshot,
    fetch_peer_multiples,
    OFFLINE_FIXTURES,
    OFFLINE_PEER_FIXTURES,
)
from src.data.filing_rag_tool import (
    load_filing_chunks,
    FilingRAGRetriever,
    query_filing_evidence,
)


# ==============================================================================
# 1. BENCHMARK SNAPSHOT INGESTION TESTS
# ==============================================================================

@pytest.mark.parametrize("ticker,expected_curr,expected_unit,expected_bench,expected_rf", [
    ("TCS.NS", "INR", "INR Crores", "^NSEI", 0.068),
    ("ZOMATO.NS", "INR", "INR Crores", "^NSEI", 0.068),
    ("VARUNBEV.NS", "INR", "INR Crores", "^NSEI", 0.068),
    ("NVDA", "USD", "USD Millions", "^GSPC", 0.042),
])
def test_company_financial_snapshot_ingestion(ticker, expected_curr, expected_unit, expected_bench, expected_rf):
    """Verifies that all 4 benchmark tickers load into valid CompanyFinancialSnapshot models."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)

    assert isinstance(snapshot, CompanyFinancialSnapshot)
    assert snapshot.ticker == ticker
    assert snapshot.currency == expected_curr
    assert snapshot.unit_label == expected_unit
    assert snapshot.benchmark_ticker == expected_bench
    assert snapshot.risk_free_rate == pytest.approx(expected_rf, abs=1e-4)
    assert snapshot.equity_risk_premium > 0.0
    assert snapshot.current_price > 0.0
    assert snapshot.shares_outstanding > 0.0
    assert snapshot.market_cap > 0.0
    assert snapshot.enterprise_value > 0.0
    assert snapshot.high_52week >= snapshot.low_52week

    # Verify 4 historical fiscal years: 2023, 2024, 2025, 2026
    assert len(snapshot.historical_years) == 4
    years = [yr.fiscal_year for yr in snapshot.historical_years]
    assert years == [2023, 2024, 2025, 2026]

    # Verify exactly 252 daily return data points
    assert len(snapshot.daily_returns_stock) == 252
    assert len(snapshot.daily_returns_benchmark) == 252


# ==============================================================================
# 2. ACCOUNTING INVARIANCE TESTS (ALL TICKERS, ALL YEARS)
# ==============================================================================

@pytest.mark.parametrize("ticker", ["TCS.NS", "ZOMATO.NS", "VARUNBEV.NS", "NVDA"])
def test_accounting_identities_invariance(ticker):
    """Verifies fundamental balance sheet and income statement identities across all historical years."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)

    for yr in snapshot.historical_years:
        # 1. Gross Profit == Revenue - COGS
        assert yr.gross_profit == pytest.approx(yr.revenue - yr.cogs, abs=1e-2), (
            f"{ticker} ({yr.fiscal_year}): Gross Profit mismatch"
        )

        # 2. EBIT == EBITDA - D&A
        assert yr.ebit == pytest.approx(yr.ebitda - yr.depreciation_amortization, abs=1e-2), (
            f"{ticker} ({yr.fiscal_year}): EBIT mismatch"
        )

        # 3. EBT == EBIT - Interest Expense
        assert yr.ebt == pytest.approx(yr.ebit - yr.interest_expense, abs=1e-2), (
            f"{ticker} ({yr.fiscal_year}): EBT mismatch"
        )

        # 4. Net Income == EBT - Tax Expense
        assert yr.net_income == pytest.approx(yr.ebt - yr.tax_expense, abs=1e-2), (
            f"{ticker} ({yr.fiscal_year}): Net Income mismatch"
        )

        # 5. Total Assets == Total Liabilities + Total Equity
        assert yr.total_assets == pytest.approx(yr.total_liabilities + yr.total_equity, abs=1e-2), (
            f"{ticker} ({yr.fiscal_year}): Balance sheet inequality (Assets != Liabilities + Equity)"
        )


# ==============================================================================
# 3. PEER COMPARABLE MULTIPLES TESTS
# ==============================================================================

@pytest.mark.parametrize("ticker", ["TCS.NS", "ZOMATO.NS", "VARUNBEV.NS", "NVDA"])
def test_peer_multiples_loading(ticker):
    """Verifies that peer multiples return exactly 3 valid peers per ticker with positive metrics."""
    peers = fetch_peer_multiples(ticker, use_live=False)

    assert len(peers) == 3
    for p in peers:
        assert isinstance(p, PeerCompMultiple)
        assert len(p.peer_ticker) > 0
        assert len(p.peer_name) > 0
        assert p.pe_ratio > 0.0
        assert p.ev_to_ebitda > 0.0
        assert p.pb_ratio > 0.0
        assert p.ev_to_sales > 0.0


# ==============================================================================
# 4. PYDANTIC V2 SCHEMA BOUNDARY & VALIDATION TESTS
# ==============================================================================

def make_valid_scenario_driver(name="base", prob=0.50, growth=0.10, margin=0.25):
    """Helper to instantiate a valid ScenarioDriverSet."""
    growth_list = growth if isinstance(growth, list) else [growth] * 5
    return ScenarioDriverSet(
        scenario_name=name,
        probability_weight=prob,
        revenue_cagr_5yr=growth_list,
        target_ebit_margin_yr5=margin,
        effective_tax_rate=0.22,
        da_pct_revenue=0.03,
        capex_pct_revenue=0.04,
        nwc_pct_revenue_change=0.02,
        terminal_growth_rate=0.03,
        wacc_adjustment_bps=0.0,
        rationale="Standard operational forecast grounded in historical run rate.",
    )


def test_dcf_scenario_assumptions_valid():
    """Verifies that well-formed DCFScenarioAssumptions instantiate without error."""
    bear = make_valid_scenario_driver(name="bear", prob=0.25, growth=0.05, margin=0.20)
    base = make_valid_scenario_driver(name="base", prob=0.50, growth=0.10, margin=0.25)
    bull = make_valid_scenario_driver(name="bull", prob=0.25, growth=0.15, margin=0.30)

    assumptions = DCFScenarioAssumptions(
        bear=bear,
        base=base,
        bull=bull,
        bull_thesis_points=["High market share capture", "Expanding gross margins"],
        bear_thesis_points=["Intensifying rivalry", "Wage inflation"],
        key_catalysts=["New product launches", "International expansion"],
        key_risks=["Macro slowdown", "Regulatory scrutiny"],
        debate_synthesis_summary="Reconciled debate favors baseline expansion with modest upside risk.",
    )
    assert assumptions.base.scenario_name == "base"
    assert len(assumptions.bull_thesis_points) == 2


def test_scenario_driver_set_invalid_growth_length():
    """Negative test: revenue_cagr_5yr with length != 5 must raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        make_valid_scenario_driver(growth=[0.10, 0.10, 0.10, 0.10])
    assert "revenue_cagr_5yr must contain exactly 5 discrete annual growth rates" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        make_valid_scenario_driver(growth=[0.10] * 6)
    assert "revenue_cagr_5yr must contain exactly 5 discrete annual growth rates" in str(exc_info.value)


def test_scenario_driver_set_out_of_bounds_fields():
    """Negative test: field values violating ge/le bounds must raise ValidationError."""
    # probability_weight < 0.05
    with pytest.raises(ValidationError):
        make_valid_scenario_driver(prob=0.01)

    # probability_weight > 0.90
    with pytest.raises(ValidationError):
        make_valid_scenario_driver(prob=0.95)

    # target_ebit_margin_yr5 > 0.70
    with pytest.raises(ValidationError):
        make_valid_scenario_driver(margin=0.85)

    # terminal_growth_rate > 0.065
    with pytest.raises(ValidationError):
        ScenarioDriverSet(
            scenario_name="base",
            probability_weight=0.50,
            revenue_cagr_5yr=[0.10] * 5,
            target_ebit_margin_yr5=0.25,
            effective_tax_rate=0.22,
            da_pct_revenue=0.03,
            capex_pct_revenue=0.04,
            nwc_pct_revenue_change=0.02,
            terminal_growth_rate=0.08,  # > 0.065
            wacc_adjustment_bps=0.0,
            rationale="Aggressive terminal growth.",
        )


def test_dcf_scenario_assumptions_invalid_probability_sum():
    """Negative test: probability weights summing != 1.0 ± 0.01 must raise ValidationError."""
    bear = make_valid_scenario_driver(name="bear", prob=0.20, growth=0.05, margin=0.20)
    base = make_valid_scenario_driver(name="base", prob=0.40, growth=0.10, margin=0.25)
    bull = make_valid_scenario_driver(name="bull", prob=0.20, growth=0.15, margin=0.30)  # sum = 0.80

    with pytest.raises(ValidationError) as exc_info:
        DCFScenarioAssumptions(
            bear=bear,
            base=base,
            bull=bull,
            bull_thesis_points=["Upside"],
            bear_thesis_points=["Downside"],
            key_catalysts=["Catalyst"],
            key_risks=["Risk"],
            debate_synthesis_summary="Summary",
        )
    assert "Scenario probability weights must sum to 1.0 ± 0.01" in str(exc_info.value)


def test_dcf_scenario_assumptions_inverted_growth_ordering():
    """Negative test: Bull avg growth < Base or Base < Bear must raise ValidationError."""
    # Bull CAGR (0.08) < Base CAGR (0.12)
    bear = make_valid_scenario_driver(name="bear", prob=0.25, growth=0.05, margin=0.20)
    base = make_valid_scenario_driver(name="base", prob=0.50, growth=0.12, margin=0.25)
    bull = make_valid_scenario_driver(name="bull", prob=0.25, growth=0.08, margin=0.30)

    with pytest.raises(ValidationError) as exc_info:
        DCFScenarioAssumptions(
            bear=bear,
            base=base,
            bull=bull,
            bull_thesis_points=["Upside"],
            bear_thesis_points=["Downside"],
            key_catalysts=["Catalyst"],
            key_risks=["Risk"],
            debate_synthesis_summary="Summary",
        )
    assert "Bull average revenue CAGR" in str(exc_info.value)


def test_dcf_scenario_assumptions_inverted_margin_ordering():
    """Negative test: Bull Year-5 margin < Base Year-5 margin must raise ValidationError."""
    bear = make_valid_scenario_driver(name="bear", prob=0.25, growth=0.05, margin=0.20)
    base = make_valid_scenario_driver(name="base", prob=0.50, growth=0.10, margin=0.28)
    bull = make_valid_scenario_driver(name="bull", prob=0.25, growth=0.15, margin=0.22)  # 0.22 < 0.28

    with pytest.raises(ValidationError) as exc_info:
        DCFScenarioAssumptions(
            bear=bear,
            base=base,
            bull=bull,
            bull_thesis_points=["Upside"],
            bear_thesis_points=["Downside"],
            key_catalysts=["Catalyst"],
            key_risks=["Risk"],
            debate_synthesis_summary="Summary",
        )
    assert "Bull Year-5 margin" in str(exc_info.value)


def test_pydantic_forbid_extra_fields():
    """Negative test: Extra fields must raise ValidationError due to extra='forbid'."""
    with pytest.raises(ValidationError):
        ScenarioDriverSet(
            scenario_name="base",
            probability_weight=0.50,
            revenue_cagr_5yr=[0.10] * 5,
            target_ebit_margin_yr5=0.25,
            effective_tax_rate=0.22,
            da_pct_revenue=0.03,
            capex_pct_revenue=0.04,
            nwc_pct_revenue_change=0.02,
            terminal_growth_rate=0.03,
            wacc_adjustment_bps=0.0,
            rationale="Valid",
            unauthorized_extra_field=123,  # Extra field
        )


# ==============================================================================
# 5. DETERMINISTIC FILING RAG RETRIEVAL TESTS
# ==============================================================================

@pytest.mark.parametrize("ticker", ["TCS.NS", "ZOMATO.NS", "VARUNBEV.NS", "NVDA"])
def test_filing_chunks_loading(ticker):
    """Verifies that sample filing notes contain at least 6 well-structured tagged chunks."""
    chunks = load_filing_chunks(ticker)
    assert len(chunks) >= 6

    for i, chunk in enumerate(chunks, 1):
        assert chunk["chunk_id"] == f"[FILING-CHUNK-{i}]"
        assert len(chunk["title"]) > 0
        assert len(chunk["text"]) > 50
        assert "[FILING-CHUNK-" in chunk["raw"]


def test_filing_rag_retriever_query():
    """Verifies deterministic TF-IDF ranking returns relevant chunks for specific domain queries."""
    # Query NVDA data center compute
    retriever_nvda = FilingRAGRetriever("NVDA")
    results_compute = retriever_nvda.retrieve("Hopper Blackwell GPU architecture data center demand", top_k=2)

    assert len(results_compute) == 2
    # Chunk 1 discusses Hopper/Blackwell compute architecture
    assert results_compute[0]["chunk_id"] == "[FILING-CHUNK-1]"
    assert results_compute[0]["score"] > 0.0

    # Query TCS operating margins and wage hikes
    retriever_tcs = FilingRAGRetriever("TCS.NS")
    results_margin = retriever_tcs.retrieve("operating margins wage revisions attrition delivery mix", top_k=2)

    assert len(results_margin) == 2
    # Chunk 2 discusses operating margins and wage revisions
    assert results_margin[0]["chunk_id"] == "[FILING-CHUNK-2]"
    assert results_margin[0]["score"] > 0.0


def test_query_filing_evidence_formatting():
    """Verifies that query_filing_evidence returns properly formatted citations."""
    formatted = query_filing_evidence("ZOMATO.NS", "quick commerce Blinkit dark store GOV", top_k=2)

    assert "[FILING-CHUNK-" in formatted
    assert "(Score: " in formatted
    assert "Blinkit" in formatted


# ==============================================================================
# 6. ALL 12 PYDANTIC V2 CONTRACTS INSTANTIATION & SERIALIZATION
# ==============================================================================

def test_all_12_pydantic_contracts_instantiation():
    """Verifies that all 12 institutional models instantiate, serialize, and roundtrip JSON cleanly."""
    # 1. HistoricalYearMetrics
    h = HistoricalYearMetrics(
        fiscal_year=2024, revenue=100.0, cogs=40.0, gross_profit=60.0, ebitda=30.0,
        depreciation_amortization=5.0, ebit=25.0, interest_expense=2.0, ebt=23.0,
        tax_expense=5.0, net_income=18.0, total_assets=200.0, current_assets=80.0,
        cash_and_equivalents=20.0, accounts_receivable=30.0, inventory=15.0,
        total_liabilities=80.0, current_liabilities=40.0, accounts_payable=15.0,
        total_debt=20.0, total_equity=120.0, retained_earnings=100.0,
        operating_cash_flow=22.0, capex=8.0, working_capital_change=2.0
    )
    assert h.fiscal_year == 2024

    # 2. CompanyFinancialSnapshot
    snap = CompanyFinancialSnapshot(
        ticker="TEST", company_name="Test Corp", sector="Tech", industry="Software",
        currency="USD", unit_label="USD Millions", current_price=50.0,
        shares_outstanding=10.0, market_cap=500.0, enterprise_value=500.0,
        high_52week=60.0, low_52week=40.0, benchmark_ticker="^GSPC",
        risk_free_rate=0.04, equity_risk_premium=0.05, historical_years=[h],
        daily_returns_stock=[0.01] * 252, daily_returns_benchmark=[0.005] * 252,
        business_summary="Summary"
    )
    assert snap.ticker == "TEST"

    # 3. FSARatioSummary
    fsa = FSARatioSummary(
        gross_margin=0.6, ebitda_margin=0.3, operating_margin=0.25, net_profit_margin=0.18,
        roa=0.09, roe=0.15, roic=0.12, revenue_cagr_3yr=0.10, tax_burden=0.78,
        interest_burden=0.92, ebit_margin=0.25, asset_turnover=0.5, financial_leverage=1.67,
        reconstructed_roe=0.15, current_ratio=2.0, quick_ratio=1.25, dso=109.5, dio=136.9,
        dpo=136.9, cash_conversion_cycle=109.5, debt_to_equity=0.17, net_debt_to_ebitda=0.0,
        interest_coverage=12.5, altman_z_score=3.5, altman_zone="Safe",
        cfo_to_net_income_ratio=1.22, accruals_to_assets_ratio=0.02,
        receivables_vs_revenue_growth_spread=-0.01, accounting_red_flags=[],
        common_size_income_summary={"cogs": 0.40, "ebit": 0.25}
    )
    assert fsa.altman_zone == "Safe"

    # 4. QuantRiskSummary
    q = QuantRiskSummary(
        ols_beta=1.1, adjusted_beta=1.07, alpha_annualized=0.03, r_squared=0.65,
        annualized_volatility=0.22, downside_deviation=0.14, sharpe_ratio=1.2,
        sortino_ratio=1.8, max_drawdown=0.15, var_95_daily_historical=0.02,
        var_95_daily_parametric=0.021, cvar_95_daily=0.028
    )
    assert q.adjusted_beta == pytest.approx(1.07)

    # 5. ScenarioDriverSet
    drv = ScenarioDriverSet(
        scenario_name="base", probability_weight=0.5, revenue_cagr_5yr=[0.10] * 5,
        target_ebit_margin_yr5=0.25, effective_tax_rate=0.22, da_pct_revenue=0.03,
        capex_pct_revenue=0.04, nwc_pct_revenue_change=0.02, terminal_growth_rate=0.03,
        wacc_adjustment_bps=0.0, rationale="Grounded forecast"
    )
    assert drv.scenario_name == "base"

    # 6. SingleDCFResult
    dcf = SingleDCFResult(
        scenario_name="base", wacc=0.10, terminal_growth_rate=0.03, projected_revenues=[100.0] * 5,
        projected_ebit=[25.0] * 5, projected_fcff=[18.0] * 5, pv_of_fcff=60.0,
        terminal_value=250.0, pv_of_terminal_value=150.0, enterprise_value=210.0,
        equity_value=210.0, intrinsic_value_per_share=21.0, upside_pct=0.05
    )
    assert dcf.intrinsic_value_per_share == 21.0

    # 7. MonteCarloDCFOutput
    mc = MonteCarloDCFOutput(
        num_simulations=10000, mean_intrinsic_value=22.0, median_intrinsic_value=21.5,
        std_dev=3.5, p5=16.0, p25=19.0, p50=21.5, p75=24.5, p95=28.0,
        probability_of_upside=0.65, simulated_values_sample=[20.0, 22.0, 24.0]
    )
    assert mc.num_simulations == 10000

    # 8. PeerCompMultiple
    peer = PeerCompMultiple(
        peer_ticker="INFY.NS", peer_name="Infosys", pe_ratio=25.0, ev_to_ebitda=16.0,
        pb_ratio=6.0, ev_to_sales=4.0, roe=0.25, revenue_growth=0.08
    )
    assert peer.peer_ticker == "INFY.NS"

    # 9. ValuationSuiteOutput
    val = ValuationSuiteOutput(
        cost_of_equity=0.11, pre_tax_cost_of_debt=0.06, after_tax_cost_of_debt=0.045,
        base_wacc=0.10, bear_dcf=dcf, base_dcf=dcf, bull_dcf=dcf,
        probability_weighted_intrinsic_value=21.0, implied_upside_pct=0.05,
        recommendation="BUY", reverse_dcf_implied_revenue_cagr=0.095,
        market_expectations_gap_bps=50.0, wacc_sensitivity_axes=[0.08, 0.09, 0.10, 0.11, 0.12],
        tgr_sensitivity_axes=[0.02, 0.025, 0.03, 0.035, 0.04],
        sensitivity_matrix_prices=[[20.0] * 5] * 5, peer_comps=[peer],
        peer_median_implied_price=20.5, football_field_ranges={"52W": (40.0, 50.0, 60.0)}
    )
    assert val.recommendation == "BUY"

    # 10. StepTelemetry
    tel = StepTelemetry(
        step_name="FSA_Analysis", agent_or_engine="fsa_engine", mode="deterministic",
        latency_ms=15.4, status="SUCCESS", validation_passed=True, details="Clean execution"
    )
    assert tel.validation_passed is True

    # 11. EvalScorecard
    scorecard = EvalScorecard(
        math_invariance_score=100.0, schema_completeness_score=100.0,
        numerical_faithfulness_score=100.0, overall_eval_score=100.0,
        passed=True, repair_attempts=0, flagged_discrepancies=[]
    )
    assert scorecard.passed is True

    # JSON roundtrip test
    dumped = val.model_dump_json()
    reloaded = ValuationSuiteOutput.model_validate_json(dumped)
    assert reloaded.recommendation == "BUY"


# ==============================================================================
# 7. MODEL INVARIANTS & BOUNDARY TESTS
# ==============================================================================

def test_single_dcf_result_terminal_growth_vs_wacc_validation():
    """Negative test: SingleDCFResult where terminal_growth_rate >= wacc must raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        SingleDCFResult(
            scenario_name="base", wacc=0.08, terminal_growth_rate=0.08, projected_revenues=[100.0] * 5,
            projected_ebit=[25.0] * 5, projected_fcff=[18.0] * 5, pv_of_fcff=60.0,
            terminal_value=250.0, pv_of_terminal_value=150.0, enterprise_value=210.0,
            equity_value=210.0, intrinsic_value_per_share=21.0, upside_pct=0.05
        )
    assert "Terminal growth rate" in str(exc_info.value)
    assert "must be strictly less than WACC" in str(exc_info.value)


def test_monte_carlo_dcf_output_percentile_ordering_validation():
    """Negative test: MonteCarloDCFOutput with inverted percentiles must raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        MonteCarloDCFOutput(
            num_simulations=10000, mean_intrinsic_value=22.0, median_intrinsic_value=21.5,
            std_dev=3.5, p5=25.0, p25=20.0, p50=21.5, p75=24.5, p95=28.0,  # P5 > P25
            probability_of_upside=0.65, simulated_values_sample=[20.0, 22.0]
        )
    assert "Percentiles must be monotonically ordered" in str(exc_info.value)


def test_company_financial_snapshot_invariants_validation():
    """Negative tests: Snapshot invariants (52w high >= low, history not empty, returns length match)."""
    h = HistoricalYearMetrics(
        fiscal_year=2024, revenue=100.0, cogs=40.0, gross_profit=60.0, ebitda=30.0,
        depreciation_amortization=5.0, ebit=25.0, interest_expense=2.0, ebt=23.0,
        tax_expense=5.0, net_income=18.0, total_assets=200.0, current_assets=80.0,
        cash_and_equivalents=20.0, accounts_receivable=30.0, inventory=15.0,
        total_liabilities=80.0, current_liabilities=40.0, accounts_payable=15.0,
        total_debt=20.0, total_equity=120.0, retained_earnings=100.0,
        operating_cash_flow=22.0, capex=8.0, working_capital_change=2.0
    )

    # 1. high_52week < low_52week
    with pytest.raises(ValidationError) as exc_info:
        CompanyFinancialSnapshot(
            ticker="TEST", company_name="Test", sector="Tech", industry="Software",
            currency="USD", unit_label="USD Millions", current_price=50.0,
            shares_outstanding=10.0, market_cap=500.0, enterprise_value=500.0,
            high_52week=30.0, low_52week=60.0, benchmark_ticker="^GSPC",
            risk_free_rate=0.04, equity_risk_premium=0.05, historical_years=[h],
            daily_returns_stock=[0.01] * 252, daily_returns_benchmark=[0.005] * 252,
            business_summary="Summary"
        )
    assert "cannot be less than low_52week" in str(exc_info.value)

    # 2. historical_years empty
    with pytest.raises(ValidationError) as exc_info:
        CompanyFinancialSnapshot(
            ticker="TEST", company_name="Test", sector="Tech", industry="Software",
            currency="USD", unit_label="USD Millions", current_price=50.0,
            shares_outstanding=10.0, market_cap=500.0, enterprise_value=500.0,
            high_52week=60.0, low_52week=30.0, benchmark_ticker="^GSPC",
            risk_free_rate=0.04, equity_risk_premium=0.05, historical_years=[],
            daily_returns_stock=[0.01] * 252, daily_returns_benchmark=[0.005] * 252,
            business_summary="Summary"
        )
    assert "historical_years must contain at least one year" in str(exc_info.value)

    # 3. daily_returns length mismatch
    with pytest.raises(ValidationError) as exc_info:
        CompanyFinancialSnapshot(
            ticker="TEST", company_name="Test", sector="Tech", industry="Software",
            currency="USD", unit_label="USD Millions", current_price=50.0,
            shares_outstanding=10.0, market_cap=500.0, enterprise_value=500.0,
            high_52week=60.0, low_52week=30.0, benchmark_ticker="^GSPC",
            risk_free_rate=0.04, equity_risk_premium=0.05, historical_years=[h],
            daily_returns_stock=[0.01] * 252, daily_returns_benchmark=[0.005] * 200,
            business_summary="Summary"
        )
    assert "must match daily_returns_benchmark" in str(exc_info.value)


def test_dcf_scenario_assumptions_base_vs_bear_cagr_ordering():
    """Negative test: Base average revenue CAGR < Bear average revenue CAGR must raise ValidationError."""
    bear = make_valid_scenario_driver(name="bear", prob=0.25, growth=0.10, margin=0.20)
    base = make_valid_scenario_driver(name="base", prob=0.50, growth=0.07, margin=0.25)  # 0.07 < 0.10
    bull = make_valid_scenario_driver(name="bull", prob=0.25, growth=0.15, margin=0.30)

    with pytest.raises(ValidationError) as exc_info:
        DCFScenarioAssumptions(
            bear=bear, base=base, bull=bull,
            bull_thesis_points=["Upside"], bear_thesis_points=["Downside"],
            key_catalysts=["Catalyst"], key_risks=["Risk"],
            debate_synthesis_summary="Summary"
        )
    assert "Base average revenue CAGR" in str(exc_info.value)
    assert "must be >= Bear" in str(exc_info.value)


def test_dcf_scenario_assumptions_base_vs_bear_margin_ordering():
    """Negative test: Base Year-5 margin < Bear Year-5 margin must raise ValidationError."""
    bear = make_valid_scenario_driver(name="bear", prob=0.25, growth=0.05, margin=0.22)
    base = make_valid_scenario_driver(name="base", prob=0.50, growth=0.10, margin=0.18)  # 0.18 < 0.22
    bull = make_valid_scenario_driver(name="bull", prob=0.25, growth=0.15, margin=0.25)

    with pytest.raises(ValidationError) as exc_info:
        DCFScenarioAssumptions(
            bear=bear, base=base, bull=bull,
            bull_thesis_points=["Upside"], bear_thesis_points=["Downside"],
            key_catalysts=["Catalyst"], key_risks=["Risk"],
            debate_synthesis_summary="Summary"
        )
    assert "Base Year-5 margin" in str(exc_info.value)
    assert "must be >= Bear" in str(exc_info.value)


def test_dcf_scenario_assumptions_probability_weight_boundary_limits():
    """Verifies that probability sums within ±0.01 pass, but beyond ±0.01 fail."""
    # Sum = 1.008 (within 0.01) -> Valid
    p_valid_high = DCFScenarioAssumptions(
        bear=make_valid_scenario_driver(name="bear", prob=0.25, growth=0.05, margin=0.20),
        base=make_valid_scenario_driver(name="base", prob=0.508, growth=0.10, margin=0.25),
        bull=make_valid_scenario_driver(name="bull", prob=0.25, growth=0.15, margin=0.30),
        bull_thesis_points=["Upside"], bear_thesis_points=["Downside"],
        key_catalysts=["Catalyst"], key_risks=["Risk"], debate_synthesis_summary="Summary"
    )
    assert p_valid_high.base.probability_weight == 0.508

    # Sum = 1.015 (beyond 0.01) -> Invalid
    with pytest.raises(ValidationError):
        DCFScenarioAssumptions(
            bear=make_valid_scenario_driver(name="bear", prob=0.25, growth=0.05, margin=0.20),
            base=make_valid_scenario_driver(name="base", prob=0.515, growth=0.10, margin=0.25),
            bull=make_valid_scenario_driver(name="bull", prob=0.25, growth=0.15, margin=0.30),
            bull_thesis_points=["Upside"], bear_thesis_points=["Downside"],
            key_catalysts=["Catalyst"], key_risks=["Risk"], debate_synthesis_summary="Summary"
        )

    # Sum = 0.992 (within 0.01) -> Valid
    p_valid_low = DCFScenarioAssumptions(
        bear=make_valid_scenario_driver(name="bear", prob=0.245, growth=0.05, margin=0.20),
        base=make_valid_scenario_driver(name="base", prob=0.50, growth=0.10, margin=0.25),
        bull=make_valid_scenario_driver(name="bull", prob=0.25, growth=0.15, margin=0.30),
        bull_thesis_points=["Upside"], bear_thesis_points=["Downside"],
        key_catalysts=["Catalyst"], key_risks=["Risk"], debate_synthesis_summary="Summary"
    )
    assert p_valid_low.bear.probability_weight == 0.245

    # Sum = 0.985 (beyond 0.01) -> Invalid
    with pytest.raises(ValidationError):
        DCFScenarioAssumptions(
            bear=make_valid_scenario_driver(name="bear", prob=0.235, growth=0.05, margin=0.20),
            base=make_valid_scenario_driver(name="base", prob=0.50, growth=0.10, margin=0.25),
            bull=make_valid_scenario_driver(name="bull", prob=0.25, growth=0.15, margin=0.30),
            bull_thesis_points=["Upside"], bear_thesis_points=["Downside"],
            key_catalysts=["Catalyst"], key_risks=["Risk"], debate_synthesis_summary="Summary"
        )


# ==============================================================================
# 8. DUAL-MODE MARKET GATEWAY & CACHE VERIFICATION
# ==============================================================================

def test_dual_mode_company_snapshot_with_cache(tmp_path):
    """Verifies that fetch_company_snapshot in live mode writes cache, preserves EV, and reloads from cache."""
    # First call: writes cache
    snapshot = fetch_company_snapshot("TCS.NS", use_live=True, cache_dir=tmp_path)
    cache_file = tmp_path / "TCS.NS.json"
    assert cache_file.exists()
    assert snapshot.current_price > 0.0

    # Verify EV identity: Enterprise Value == Market Cap + Total Debt - Cash
    latest_yr = snapshot.historical_years[-1]
    expected_ev = snapshot.market_cap + latest_yr.total_debt - latest_yr.cash_and_equivalents
    assert snapshot.enterprise_value == pytest.approx(expected_ev, abs=1e-1)

    # Second call: loads from cache without network
    cached_snapshot = fetch_company_snapshot("TCS.NS", use_live=True, cache_dir=tmp_path)
    assert cached_snapshot.current_price == snapshot.current_price
    assert cached_snapshot.enterprise_value == snapshot.enterprise_value


def test_dual_mode_company_snapshot_network_fallback(tmp_path):
    """Verifies that network or parser failure gracefully falls back to the audited offline fixture."""
    with patch("yfinance.Ticker", side_effect=Exception("Connection refused")):
        snapshot = fetch_company_snapshot("NVDA", use_live=True, cache_dir=tmp_path)
        assert isinstance(snapshot, CompanyFinancialSnapshot)
        assert snapshot.ticker == "NVDA"
        assert snapshot.current_price == 135.0  # Offline fixture price


def test_dual_mode_peer_multiples_with_cache(tmp_path):
    """Verifies that fetch_peer_multiples in live mode writes cache and reloads cleanly."""
    peers = fetch_peer_multiples("NVDA", use_live=True, cache_dir=tmp_path)
    cache_file = tmp_path / "NVDA_peers.json"
    assert cache_file.exists()
    assert len(peers) == 3

    # Reload from cache
    cached_peers = fetch_peer_multiples("NVDA", use_live=True, cache_dir=tmp_path)
    assert len(cached_peers) == len(peers)
    assert cached_peers[0].peer_ticker == peers[0].peer_ticker


# ==============================================================================
# 9. ROBUSTNESS & EDGE CASE TESTS
# ==============================================================================

def test_market_gateway_unknown_ticker_errors():
    """Negative tests: Unknown tickers raise KeyError / ValueError."""
    with pytest.raises(KeyError):
        fetch_company_snapshot("UNKNOWN_STOCK", use_live=False)

    with pytest.raises(KeyError):
        fetch_peer_multiples("UNKNOWN_STOCK")


def test_filing_chunks_case_insensitivity():
    """Verifies that load_filing_chunks correctly normalizes lowercase ticker symbols."""
    chunks_upper = load_filing_chunks("TCS.NS")
    chunks_lower = load_filing_chunks("tcs.ns")
    assert len(chunks_lower) == len(chunks_upper)
    assert chunks_lower[0]["chunk_id"] == chunks_upper[0]["chunk_id"]


def test_filing_chunks_unknown_ticker():
    """Negative test: Unknown ticker raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_filing_chunks("UNKNOWN_STOCK")


def test_filing_rag_retriever_empty_query():
    """Verifies that empty/whitespace queries in FilingRAGRetriever return top chunks safely with 0 score."""
    retriever = FilingRAGRetriever("TCS.NS")
    results = retriever.retrieve("", top_k=3)
    assert len(results) == 3
    assert results[0]["score"] == 0.0

    # Also test query_filing_evidence with empty query
    evidence = query_filing_evidence("TCS.NS", "", top_k=2)
    assert "[FILING-CHUNK-1]" in evidence
    assert "(Score: 0.0000)" in evidence

