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
    DataUnavailableError,
    format_market_cap,
    fetch_company_snapshot,
    fetch_peer_multiples,
    OFFLINE_FIXTURES,
    OFFLINE_PEER_FIXTURES,
)
from src.data.ticker_resolver import resolve_ticker_query
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
    assert snapshot.data_source == "AUDITED_OFFLINE_SNAPSHOT"

    if ticker == "TCS.NS":
        assert snapshot.market_cap > 1000000.0
        assert snapshot.shares_outstanding == 361.81
        assert snapshot.current_price == 3950.0
        assert snapshot.market_cap == pytest.approx(1429149.5, abs=1e-1)

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


def test_dual_mode_default_signatures(tmp_path):
    """Verifies that fetch_company_snapshot and fetch_peer_multiples default to use_live=True."""
    # Calling without use_live argument should default to use_live=True
    snap = fetch_company_snapshot("NVDA", cache_dir=tmp_path)
    assert snap.ticker == "NVDA"
    assert snap.data_source == "LIVE_YFINANCE"
    assert snap.current_price > 150.0  # Real live price, not hardcoded $135 fixture

    peers = fetch_peer_multiples("NVDA", cache_dir=tmp_path)
    assert len(peers) == 3


def test_cache_invalidation_stale_or_wrong_version(tmp_path):
    """Verifies that pre-v3 or expired cache files are automatically ignored and overwritten."""
    import json
    import time
    cache_file = tmp_path / "NVDA.json"

    # Write a fake v2 cache file with old price
    fake_data = {
        "ticker": "NVDA",
        "company_name": "NVIDIA",
        "sector": "Technology",
        "industry": "Semiconductors",
        "currency": "USD",
        "unit_label": "USD Millions",
        "current_price": 50.0,
        "shares_outstanding": 1000.0,
        "market_cap": 50000.0,
        "enterprise_value": 50000.0,
        "high_52week": 60.0,
        "low_52week": 40.0,
        "benchmark_ticker": "^GSPC",
        "risk_free_rate": 0.042,
        "equity_risk_premium": 0.05,
        "historical_years": [],
        "daily_returns_stock": [],
        "daily_returns_benchmark": [],
        "business_summary": "Test",
        "data_source": "LIVE_YFINANCE",
        "cache_version": "v2_legacy",
        "cached_at": time.time(),
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(fake_data, f)

    # Calling fetch should detect invalid cache_version, delete stale cache, and fetch live
    snap = fetch_company_snapshot("NVDA", use_live=True, cache_dir=tmp_path)
    assert snap.current_price > 150.0

    # Verify newly written cache has v3_live_verified
    with open(cache_file, "r", encoding="utf-8") as f:
        new_cached = json.load(f)
    assert new_cached["cache_version"] == "v3_live_verified"


def test_dual_currency_ingestion_and_fx_conversion(tmp_path):
    """Verifies that dual-currency filers (e.g. INFY.NS filing in USD, trading in INR) convert line items correctly."""
    snap = fetch_company_snapshot("INFY.NS", use_live=True, cache_dir=tmp_path)
    assert snap.currency == "INR"
    assert snap.unit_label == "INR Crores"
    assert snap.market_cap > 100000.0  # INFY market cap > 1 Lakh Cr
    assert snap.shares_outstanding > 0.0
    assert snap.market_cap == pytest.approx(snap.current_price * snap.shares_outstanding, abs=1e-1)

    # Revenue should be in hundreds of thousands of Crores (~1.7+ Lakh Cr), NOT in single digit billions of USD
    latest_yr = snap.historical_years[-1]
    assert latest_yr.revenue > 100000.0, f"Expected INR Crores revenue > 100,000 Cr, got {latest_yr.revenue}"
    assert latest_yr.capex >= 0.0  # Positive magnitude


def test_exchange_aliases_live_retrieval(tmp_path):
    """Verifies that tickers with exchange symbol renames (VARUNBEV.NS -> VBL.NS, ZOMATO.NS -> ETERNAL.NS) fetch live data."""
    snap_vbl = fetch_company_snapshot("VARUNBEV.NS", use_live=True, cache_dir=tmp_path)
    assert snap_vbl.ticker == "VARUNBEV.NS"
    assert snap_vbl.data_source == "LIVE_YFINANCE"
    assert snap_vbl.current_price > 0.0
    assert snap_vbl.market_cap == pytest.approx(snap_vbl.current_price * snap_vbl.shares_outstanding, abs=1e-1)

    snap_zomato = fetch_company_snapshot("ZOMATO.NS", use_live=True, cache_dir=tmp_path)
    assert snap_zomato.ticker == "ZOMATO.NS"
    assert snap_zomato.data_source == "LIVE_YFINANCE"
    assert snap_zomato.current_price > 0.0
    assert snap_zomato.market_cap == pytest.approx(snap_zomato.current_price * snap_zomato.shares_outstanding, abs=1e-1)

# 9. ROBUSTNESS & EDGE CASE TESTS
# ==============================================================================

def test_market_gateway_unknown_ticker_errors():
    """Negative tests: Unknown or invalid tickers raise DataUnavailableError instead of fake data."""
    with pytest.raises(DataUnavailableError):
        fetch_company_snapshot("INVALID_FAKE_TICKER_999", use_live=False)

    with pytest.raises(DataUnavailableError):
        fetch_company_snapshot("INVALID_FAKE_TICKER_999", use_live=True)

    with pytest.raises(DataUnavailableError):
        fetch_company_snapshot("UNKNOWN_STOCK", use_live=False)

    with pytest.raises(DataUnavailableError):
        fetch_company_snapshot("FOOBAR", use_live=False)

    with pytest.raises(DataUnavailableError):
        fetch_peer_multiples("INVALID_FAKE_TICKER_999")

    with pytest.raises(DataUnavailableError):
        fetch_peer_multiples("UNKNOWN_STOCK")

    with pytest.raises(DataUnavailableError):
        fetch_peer_multiples("FOOBAR")


def test_market_gateway_bare_ticker_resolution():
    """Verifies that bare Indian tickers and company names resolve cleanly in fetch functions."""
    snap = fetch_company_snapshot("TCS", use_live=False)
    assert snap.ticker == "TCS.NS"
    assert snap.market_cap > 1000000.0
    assert snap.data_source == "AUDITED_OFFLINE_SNAPSHOT"

    snap_name = fetch_company_snapshot("Tata Consultancy", use_live=False)
    assert snap_name.ticker == "TCS.NS"

    peers = fetch_peer_multiples("TCS", use_live=False)
    assert len(peers) == 3


def test_parse_yfinance_snapshot_missing_statements_error():
    """Negative tests: Missing core financial statements or rows must raise DataUnavailableError."""
    from unittest.mock import MagicMock
    from src.data.market_gateway import parse_yfinance_snapshot

    mock_ticker = MagicMock()
    mock_ticker.financials = None
    mock_ticker.balance_sheet = None
    mock_ticker.cashflow = None

    # Missing statements entirely
    with pytest.raises(DataUnavailableError) as exc_info:
        parse_yfinance_snapshot("FAKE", mock_ticker, {"currentPrice": 100.0, "sharesOutstanding": 1000.0})
    assert "Missing required financial statements" in str(exc_info.value)

    # Missing price
    with pytest.raises(DataUnavailableError) as exc_info:
        parse_yfinance_snapshot("FAKE", mock_ticker, {"currentPrice": 0.0})
    assert "No valid market price found" in str(exc_info.value)


def test_ticker_resolver():
    """Verifies that all 20 bare Indian tickers, 13 common names, and canonical symbols resolve correctly."""
    # 20 Bare Indian Tickers mapped to .NS
    bare_mappings = {
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "RELIANCE": "RELIANCE.NS",
        "VARUNBEV": "VARUNBEV.NS",
        "VBL": "VARUNBEV.NS",
        "ZOMATO": "ZOMATO.NS",
        "ETERNAL": "ZOMATO.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "SBIN": "SBIN.NS",
        "ITC": "ITC.NS",
        "WIPRO": "WIPRO.NS",
        "HCLTECH": "HCLTECH.NS",
        "LT": "LT.NS",
        "BAJFINANCE": "BAJFINANCE.NS",
        "BHARTIARTL": "BHARTIARTL.NS",
        "MARUTI": "MARUTI.NS",
        "TATAMOTORS": "TATAMOTORS.NS",
        "SUNPHARMA": "SUNPHARMA.NS",
        "TITAN": "TITAN.NS",
    }
    for bare, canonical in bare_mappings.items():
        assert resolve_ticker_query(bare) == canonical, f"Bare ticker {bare} failed to resolve to {canonical}"

    # 13 Common Company Names mapped to canonical symbols
    name_mappings = {
        "Tata Consultancy": "TCS.NS",
        "Infosys": "INFY.NS",
        "Reliance": "RELIANCE.NS",
        "Varun Beverages": "VARUNBEV.NS",
        "Zomato": "ZOMATO.NS",
        "Eternal": "ZOMATO.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Nvidia": "NVDA",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Tesla": "TSLA",
        "Amazon": "AMZN",
        "Alphabet": "GOOGL",
    }
    for name, canonical in name_mappings.items():
        assert resolve_ticker_query(name) == canonical, f"Company name '{name}' failed to resolve to {canonical}"

    # Negative test on completely bogus query
    with pytest.raises(DataUnavailableError):
        resolve_ticker_query("INVALID_FAKE_TICKER_999_NONEXISTENT_XYZ")


def test_format_market_cap():
    """Verifies market cap formatting according to institutional scale and currency."""
    assert format_market_cap(1429149.5, "INR") == "₹14.29 Lakh Cr"
    assert format_market_cap(138580.0, "INR") == "₹1.39 Lakh Cr"
    assert format_market_cap(239200.0, "INR") == "₹2.39 Lakh Cr"
    assert format_market_cap(3260250.0, "USD") == "$3.26T"
    assert format_market_cap(150000.0, "USD") == "$150.00B"
    assert format_market_cap(500.0, "USD") == "$500.00M"



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


# ==============================================================================
# 10. DAILY MARKET MOVERS & TRENDING STOCKS TESTS (market_pulse.py)
# ==============================================================================

def test_market_pulse_trending_stock_card_schema():
    """Verifies that TrendingStockCard model correctly instantiates and validates fields."""
    from src.data.market_pulse import TrendingStockCard

    card = TrendingStockCard(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        market="US",
        current_price=210.96,
        currency="USD",
        daily_change_pct=2.45,
        daily_volume=79987200,
        market_cap=5094051.0,
        formatted_market_cap="$5.09T",
    )
    assert card.ticker == "NVDA"
    assert card.market == "US"
    assert card.currency == "USD"
    assert card.current_price > 0.0
    assert card.daily_change_pct == 2.45
    assert card.formatted_market_cap == "$5.09T"


def test_market_pulse_fetch_popular_stocks_live():
    """Verifies live popular stocks fetching across US, NSE, and ALL markets."""
    from src.data.market_pulse import fetch_daily_popular_stocks, TrendingStockCard

    # Test ALL filter
    cards, is_live, msg = fetch_daily_popular_stocks(market_filter="ALL", max_items=4)
    if is_live:
        assert len(cards) <= 4
        assert len(cards) > 0
        assert "Live Yahoo Finance" in msg
        for c in cards:
            assert isinstance(c, TrendingStockCard)
            assert c.current_price > 0.0
            assert c.currency in ("USD", "INR")
            assert c.market in ("US", "NSE")
    else:
        # If network unavailable, must return 0 fabricated numbers and explicit offline message
        assert cards == []
        assert "Offline mode" in msg

    # Test US filter
    cards_us, is_live_us, _ = fetch_daily_popular_stocks(market_filter="US", max_items=2)
    if is_live_us:
        for c in cards_us:
            assert c.market == "US"
            assert c.currency == "USD"

    # Test NSE filter
    cards_nse, is_live_nse, _ = fetch_daily_popular_stocks(market_filter="NSE", max_items=2)
    if is_live_nse:
        for c in cards_nse:
            assert c.market == "NSE"
            assert c.currency == "INR"


def test_market_pulse_offline_fallback_zero_fabrication():
    """Verifies that when Yahoo Finance is unreachable, returns is_live=False, empty list, and zero fabricated numbers."""
    from unittest.mock import patch
    from src.data.market_pulse import fetch_daily_popular_stocks

    with patch("yfinance.Ticker", side_effect=RuntimeError("Connection refused by Yahoo")):
        cards, is_live, msg = fetch_daily_popular_stocks(market_filter="ALL", max_items=8)
        assert cards == []
        assert is_live is False
        assert "Offline mode: Live market data unavailable" in msg
        assert "zero fabricated numbers" in msg or "does not fabricate synthetic market movers" in msg


# ==============================================================================
# 11. UNIVERSAL MULTI-CURRENCY FX NORMALIZATION TESTS
# ==============================================================================

def test_resolve_fx_rate_subunits_live_and_fallback():
    """Verifies _resolve_fx_rate handles sub-units (GBp/ZAc), live quotes, history fallback, and static fallback."""
    from unittest.mock import MagicMock, patch
    import pandas as pd
    from src.data.market_gateway import _FX_RATE_CACHE, _FALLBACK_USD_RATES, _resolve_fx_rate

    saved_cache = dict(_FX_RATE_CACHE)
    try:
        # 1. Identity & empty cases
        _FX_RATE_CACHE.clear()
        assert _resolve_fx_rate("USD", "USD") == 1.0
        assert _resolve_fx_rate("INR", "INR") == 1.0
        assert _resolve_fx_rate("", "USD") == 1.0
        assert _resolve_fx_rate("USD", "") == 1.0
        assert _resolve_fx_rate("GBp", "GBP") == pytest.approx(0.01)
        assert _resolve_fx_rate("GBP", "GBp") == pytest.approx(100.0)
        assert _resolve_fx_rate("GBp", "GBp") == pytest.approx(1.0)
        assert _resolve_fx_rate("ZAc", "ZAR") == pytest.approx(0.01)

        # 2. Fallback table when live yfinance raises
        _FX_RATE_CACHE.clear()
        with patch("yfinance.Ticker", side_effect=RuntimeError("Offline")):
            assert _resolve_fx_rate("KRW", "USD") == pytest.approx(1.0 / 1385.0)
            assert _resolve_fx_rate("JPY", "USD") == pytest.approx(1.0 / 152.0)
            assert _resolve_fx_rate("TWD", "USD") == pytest.approx(1.0 / 32.0)
            assert _resolve_fx_rate("GBp", "USD") == pytest.approx(0.01 * _FALLBACK_USD_RATES["GBP"])
            assert _resolve_fx_rate("ZAc", "USD") == pytest.approx(0.01 * _FALLBACK_USD_RATES["ZAR"])
            assert _resolve_fx_rate("KRW", "INR") == pytest.approx((1.0 / 1385.0) / (1.0 / 86.5))
            assert _resolve_fx_rate("ZZZ", "USD") == 1.0

        # 3. Live FX via info + in-memory cache reuse
        _FX_RATE_CACHE.clear()
        fx_mock = MagicMock()
        fx_mock.info = {"symbol": "EURUSD=X", "quoteType": "CURRENCY", "regularMarketPrice": 1.11}
        with patch("yfinance.Ticker", return_value=fx_mock) as ticker_cls:
            rate1 = _resolve_fx_rate("EUR", "USD")
            rate2 = _resolve_fx_rate("EUR", "USD")
            assert rate1 == pytest.approx(1.11)
            assert rate2 == pytest.approx(1.11)
            assert ticker_cls.call_count == 1

        # 4. History 5d fallback when info has no valid price
        _FX_RATE_CACHE.clear()
        hist_mock = MagicMock()
        hist_mock.info = {"symbol": "JPYUSD=X", "quoteType": "CURRENCY", "regularMarketPrice": 0.0}
        hist_mock.history.return_value = pd.DataFrame({"Close": [0.0064, 0.0066]})
        with patch("yfinance.Ticker", return_value=hist_mock):
            assert _resolve_fx_rate("JPY", "USD") == pytest.approx(0.0066)
    finally:
        _FX_RATE_CACHE.clear()
        _FX_RATE_CACHE.update(saved_cache)


def test_parse_yfinance_snapshot_krw_full_fx_normalization():
    """Verifies mocked KRW equity (005930.KS) converts price, 52w range, market_cap, and statements cleanly to USD."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    import pandas as pd
    from src.data.market_gateway import _FX_RATE_CACHE, parse_yfinance_snapshot
    from src.engines.valuation_engine import compute_single_dcf

    saved_cache = dict(_FX_RATE_CACHE)
    try:
        _FX_RATE_CACHE.clear()

        dates = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31"])
        fin_df = pd.DataFrame(
            {
                d: [
                    300_000_000_000_000.0,  # Total Revenue (300T KRW)
                    190_000_000_000_000.0,  # Cost Of Revenue
                    40_000_000_000_000.0,   # EBIT
                    35_000_000_000_000.0,   # Reconciled Depreciation
                    800_000_000_000.0,      # Interest Expense
                    34_000_000_000_000.0,   # Net Income
                ]
                for d in dates
            },
            index=[
                "Total Revenue",
                "Cost Of Revenue",
                "EBIT",
                "Reconciled Depreciation",
                "Interest Expense",
                "Net Income",
            ],
        )
        bs_df = pd.DataFrame(
            {
                d: [
                    490_000_000_000_000.0,  # Total Assets
                    230_000_000_000_000.0,  # Current Assets
                    110_000_000_000_000.0,  # Cash And Cash Equivalents
                    40_000_000_000_000.0,   # Receivables
                    55_000_000_000_000.0,   # Inventory
                    100_000_000_000_000.0,  # Total Liabilities Net Minority Interest
                    75_000_000_000_000.0,   # Current Liabilities
                    15_000_000_000_000.0,   # Accounts Payable
                    18_000_000_000_000.0,   # Total Debt
                    350_000_000_000_000.0,  # Retained Earnings
                ]
                for d in dates
            },
            index=[
                "Total Assets",
                "Current Assets",
                "Cash And Cash Equivalents",
                "Receivables",
                "Inventory",
                "Total Liabilities Net Minority Interest",
                "Current Liabilities",
                "Accounts Payable",
                "Total Debt",
                "Retained Earnings",
            ],
        )
        cf_df = pd.DataFrame(
            {
                d: [
                    65_000_000_000_000.0,   # Operating Cash Flow
                    -45_000_000_000_000.0,  # Capital Expenditure
                    -2_000_000_000_000.0,   # Change In Working Capital
                ]
                for d in dates
            },
            index=[
                "Operating Cash Flow",
                "Capital Expenditure",
                "Change In Working Capital",
            ],
        )

        hist_dates = pd.bdate_range("2025-01-01", periods=60)
        idx_arr = np.arange(60, dtype=float)
        stock_closes = 198900.0 * (1.0 + 0.02 * np.sin(idx_arr / 4.0) + 0.001 * idx_arr)
        bench_closes = 5000.0 * (1.0 + 0.012 * np.sin(idx_arr / 4.0) + 0.0005 * idx_arr)

        stock_hist = pd.DataFrame(
            {"Close": stock_closes, "High": stock_closes * 1.02, "Low": stock_closes * 0.98},
            index=hist_dates,
        )
        bench_hist = pd.DataFrame({"Close": bench_closes}, index=hist_dates)

        samsung_mock = MagicMock()
        samsung_mock.income_stmt = fin_df
        samsung_mock.balance_sheet = bs_df
        samsung_mock.cashflow = cf_df
        samsung_mock.history.return_value = stock_hist

        bench_mock = MagicMock()
        bench_mock.history.return_value = bench_hist

        fx_rate_krw_usd = 1.0 / 1385.0
        fx_mock = MagicMock()
        fx_mock.info = {"symbol": "KRWUSD=X", "quoteType": "CURRENCY", "regularMarketPrice": fx_rate_krw_usd}

        def _ticker_router(sym: str):
            if sym == "KRWUSD=X":
                return fx_mock
            if sym == "^GSPC":
                return bench_mock
            return samsung_mock

        info = {
            "symbol": "005930.KS",
            "longName": "Samsung Electronics Co., Ltd.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "currency": "KRW",
            "financialCurrency": "KRW",
            "currentPrice": 198900.0,
            "marketCap": 1_295_222_281_601_024.0,
            "sharesOutstanding": 5_709_556_199.0,
            "fiftyTwoWeekHigh": 374500.0,
            "fiftyTwoWeekLow": 94500.0,
        }

        with patch("yfinance.Ticker", side_effect=_ticker_router):
            snap = parse_yfinance_snapshot("005930.KS", samsung_mock, info)

        assert snap.currency == "USD"
        assert snap.unit_label == "USD Millions"
        assert snap.benchmark_ticker == "^GSPC"
        assert snap.current_price == pytest.approx(round(198900.0 * fx_rate_krw_usd, 2), abs=1e-2)
        assert snap.current_price < 500.0  # Must NOT remain unconverted at 198,900
        assert snap.high_52week == pytest.approx(round(374500.0 * fx_rate_krw_usd, 2), abs=1e-2)
        assert snap.low_52week == pytest.approx(round(94500.0 * fx_rate_krw_usd, 2), abs=1e-2)

        expected_mc_musd = (1_295_222_281_601_024.0 * fx_rate_krw_usd) / 1e6
        assert snap.market_cap == pytest.approx(expected_mc_musd, rel=1e-2)
        assert snap.historical_years[-1].revenue == pytest.approx((300e12 * fx_rate_krw_usd) / 1e6, rel=1e-2)

        # Verify zero 1,000x price-to-DCF distortion
        base_driver = make_valid_scenario_driver(name="base", prob=0.50, growth=0.06, margin=0.14)
        dcf_res = compute_single_dcf(snap, base_driver, adjusted_beta=1.10)
        ratio = dcf_res.intrinsic_value_per_share / snap.current_price
        assert 0.1 < ratio < 10.0, f"Unexpected price-to-DCF ratio {ratio:.2f}"
    finally:
        _FX_RATE_CACHE.clear()
        _FX_RATE_CACHE.update(saved_cache)


def test_parse_yfinance_snapshot_gbp_subunit_and_hist_52w_conversion():
    """Verifies GBp equity converts sub-unit price and hist 52-week high/low while keeping GBP marketCap exact."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    import pandas as pd
    from src.data.market_gateway import _FX_RATE_CACHE, parse_yfinance_snapshot

    saved_cache = dict(_FX_RATE_CACHE)
    try:
        _FX_RATE_CACHE.clear()

        dates = pd.to_datetime(["2024-12-31", "2025-12-31"])
        fin_df = pd.DataFrame(
            {d: [50_000_000_000.0, 10_000_000_000.0, 12_000_000_000.0, 3_000_000_000.0, 800_000_000.0, 8_000_000_000.0] for d in dates},
            index=["Total Revenue", "Cost Of Revenue", "EBIT", "Reconciled Depreciation", "Interest Expense", "Net Income"],
        )
        bs_df = pd.DataFrame(
            {d: [100_000_000_000.0, 30_000_000_000.0, 6_000_000_000.0, 8_000_000_000.0, 5_000_000_000.0, 60_000_000_000.0, 20_000_000_000.0, 5_000_000_000.0, 25_000_000_000.0, 30_000_000_000.0] for d in dates},
            index=["Total Assets", "Current Assets", "Cash And Cash Equivalents", "Receivables", "Inventory", "Total Liabilities Net Minority Interest", "Current Liabilities", "Accounts Payable", "Total Debt", "Retained Earnings"],
        )
        cf_df = pd.DataFrame(
            {d: [11_000_000_000.0, -2_000_000_000.0, 500_000_000.0] for d in dates},
            index=["Operating Cash Flow", "Capital Expenditure", "Change In Working Capital"],
        )

        hist_dates = pd.bdate_range("2025-01-01", periods=50)
        idx_arr = np.arange(50, dtype=float)
        closes = 13946.0 + 400.0 * np.sin(idx_arr / 3.0)
        highs = np.full(50, 15000.0)
        lows = np.full(50, 10000.0)
        stock_hist = pd.DataFrame({"Close": closes, "High": highs, "Low": lows}, index=hist_dates)
        bench_hist = pd.DataFrame({"Close": 5000.0 + 50.0 * np.sin(idx_arr / 3.0)}, index=hist_dates)

        azn_mock = MagicMock()
        azn_mock.income_stmt = fin_df
        azn_mock.balance_sheet = bs_df
        azn_mock.cashflow = cf_df
        azn_mock.history.return_value = stock_hist

        bench_mock = MagicMock()
        bench_mock.history.return_value = bench_hist

        fx_mock = MagicMock()
        fx_mock.info = {"symbol": "GBPUSD=X", "quoteType": "CURRENCY", "regularMarketPrice": 1.28}

        def _router(sym: str):
            if sym == "GBPUSD=X":
                return fx_mock
            if sym == "^GSPC":
                return bench_mock
            return azn_mock

        info = {
            "symbol": "AZN.L",
            "longName": "AstraZeneca PLC",
            "currency": "GBp",
            "financialCurrency": "USD",
            "currentPrice": 13946.0,
            "marketCap": 216_290_967_552.0,
            "sharesOutstanding": 1_550_917_690.0,
            # Omit fiftyTwoWeekHigh / Low to exercise hist_stock High/Low FX scaling
        }

        with patch("yfinance.Ticker", side_effect=_router):
            snap = parse_yfinance_snapshot("AZN.L", azn_mock, info)

        assert snap.currency == "USD"
        assert snap.current_price == pytest.approx(round(13946.0 * 0.01 * 1.28, 2), abs=1e-2)
        assert snap.high_52week == pytest.approx(round(15000.0 * 0.01 * 1.28, 2), abs=1e-2)
        assert snap.low_52week == pytest.approx(round(10000.0 * 0.01 * 1.28, 2), abs=1e-2)
        assert snap.market_cap == pytest.approx((216_290_967_552.0 * 1.28) / 1e6, rel=1e-2)
        assert snap.shares_outstanding == pytest.approx(1550.92, rel=1e-2)
    finally:
        _FX_RATE_CACHE.clear()
        _FX_RATE_CACHE.update(saved_cache)


def test_extract_normalized_peer_multiples_cross_currency():
    """Verifies universal cross-currency peer multiple normalization across TWD/USD, USD/INR, JPY/USD, and USD/KRW."""
    from unittest.mock import patch
    from src.data.market_gateway import _FALLBACK_USD_RATES, _FX_RATE_CACHE, _extract_normalized_peer_multiples

    saved_cache = dict(_FX_RATE_CACHE)
    try:
        _FX_RATE_CACHE.clear()
        with patch("yfinance.Ticker", side_effect=RuntimeError("Force deterministic fallback FX")):
            # 1. TWD financialCurrency vs USD price currency (ADR scale-down distortion < 4.0)
            _, ev_eb_twd, _, ev_s_twd = _extract_normalized_peer_multiples({
                "symbol": "TSM",
                "currency": "USD",
                "financialCurrency": "TWD",
                "trailingPE": 22.0,
                "enterpriseToEbitda": 0.48,
                "priceToBook": 5.5,
                "enterpriseToRevenue": 0.35,
            })
            assert ev_eb_twd == pytest.approx(0.48 * 32.0, rel=1e-4)
            assert ev_s_twd == pytest.approx(0.35 * 32.0, rel=1e-4)

            # 2. USD financialCurrency vs INR price currency (scale-up distortion > 60.0)
            _, ev_eb_inr, _, ev_s_inr = _extract_normalized_peer_multiples({
                "symbol": "INFY.NS",
                "currency": "INR",
                "financialCurrency": "USD",
                "trailingPE": 25.0,
                "enterpriseToEbitda": 1384.0,
                "priceToBook": 7.0,
                "enterpriseToRevenue": 432.5,
            })
            assert ev_eb_inr == pytest.approx(16.0, rel=1e-4)
            assert ev_s_inr == pytest.approx(5.0, rel=1e-4)

            # 3. JPY financialCurrency vs USD price currency
            _, ev_eb_jpy, _, ev_s_jpy = _extract_normalized_peer_multiples({
                "symbol": "SONY",
                "currency": "USD",
                "financialCurrency": "JPY",
                "trailingPE": 18.0,
                "enterpriseToEbitda": 0.10,
                "priceToBook": 2.0,
                "enterpriseToRevenue": 0.02,
            })
            assert ev_eb_jpy == pytest.approx(0.10 * 152.0, rel=1e-4)
            assert ev_s_jpy == pytest.approx(0.02 * 152.0, rel=1e-4)

            # 4. USD financialCurrency vs KRW price currency
            _, ev_eb_krw, _, ev_s_krw = _extract_normalized_peer_multiples({
                "symbol": "005930.KS",
                "currency": "KRW",
                "financialCurrency": "USD",
                "trailingPE": 12.0,
                "enterpriseToEbitda": 13850.0,
                "priceToBook": 1.5,
                "enterpriseToRevenue": 2770.0,
            })
            assert ev_eb_krw == pytest.approx(10.0, rel=1e-4)
            assert ev_s_krw == pytest.approx(2.0, rel=1e-4)

            # 5. Non-distorted cross-currency (already properly normalized by Yahoo) must remain untouched
            _, ev_eb_ok, _, ev_s_ok = _extract_normalized_peer_multiples({
                "symbol": "TSM",
                "currency": "USD",
                "financialCurrency": "TWD",
                "trailingPE": 24.0,
                "enterpriseToEbitda": 18.5,
                "priceToBook": 6.0,
                "enterpriseToRevenue": 11.0,
            })
            assert ev_eb_ok == pytest.approx(18.5)
            assert ev_s_ok == pytest.approx(11.0)

            # 6. Same-currency INR fallback guard (ev_ebitda > 150.0)
            _, ev_eb_fb, _, ev_s_fb = _extract_normalized_peer_multiples({
                "symbol": "TECHM.NS",
                "currency": "INR",
                "financialCurrency": "INR",
                "trailingPE": 20.0,
                "enterpriseToEbitda": 865.0,
                "priceToBook": 4.0,
                "enterpriseToRevenue": 259.5,
            })
            assert ev_eb_fb == pytest.approx(10.0, rel=1e-4)
            assert ev_s_fb == pytest.approx(3.0, rel=1e-4)

            # 7. Sub-unit ZAc trading currency with USD financialCurrency (Yahoo computes EV in ZAR, not ZAc)
            _, ev_eb_npn, _, ev_s_npn = _extract_normalized_peer_multiples({
                "symbol": "NPN.JO",
                "currency": "ZAc",
                "financialCurrency": "USD",
                "trailingPE": 28.0,
                "enterpriseToEbitda": 981.347,
                "priceToBook": 2.4,
                "enterpriseToRevenue": 56.901,
            })
            usd_to_zar = 1.0 / _FALLBACK_USD_RATES["ZAR"]
            assert ev_eb_npn == pytest.approx(981.347 / usd_to_zar, rel=1e-4)
            assert ev_s_npn == pytest.approx(56.901 / usd_to_zar, rel=1e-4)

            # 8. GBp trading currency with GBP financialCurrency (EV already in GBP -> must stay untouched)
            _, ev_eb_gsk, _, ev_s_gsk = _extract_normalized_peer_multiples({
                "symbol": "GSK.L",
                "currency": "GBp",
                "financialCurrency": "GBP",
                "trailingPE": 14.0,
                "enterpriseToEbitda": 8.423,
                "priceToBook": 4.1,
                "enterpriseToRevenue": 2.886,
            })
            assert ev_eb_gsk == pytest.approx(8.423, rel=1e-4)
            assert ev_s_gsk == pytest.approx(2.886, rel=1e-4)

            # 9. Missing/zero EBITDA with distorted EV/Sales still normalizes EV/Sales
            _, ev_eb_zero1, _, ev_s_zero1 = _extract_normalized_peer_multiples({
                "symbol": "LOSS.NS",
                "currency": "INR",
                "financialCurrency": "USD",
                "enterpriseToEbitda": None,
                "enterpriseToRevenue": 432.5,
            })
            assert ev_eb_zero1 == 0.0
            assert ev_s_zero1 == pytest.approx(5.0, rel=1e-4)

            _, ev_eb_zero2, _, ev_s_zero2 = _extract_normalized_peer_multiples({
                "symbol": "LOSS_ADR",
                "currency": "USD",
                "financialCurrency": "TWD",
                "enterpriseToEbitda": 0.0,
                "enterpriseToRevenue": 0.10,
            })
            assert ev_eb_zero2 == 0.0
            assert ev_s_zero2 == pytest.approx(3.2, rel=1e-4)
    finally:
        _FX_RATE_CACHE.clear()
        _FX_RATE_CACHE.update(saved_cache)


def test_resolve_fx_rate_guards_against_stock_mock_leak_and_poisoned_cache():
    """Verifies _resolve_fx_rate ignores mismatched non-CURRENCY mocks, survives .info exceptions, and skips caching unknown 1.0 fallback."""
    from unittest.mock import MagicMock, PropertyMock, patch
    import numpy as np
    import pandas as pd
    from src.data.market_gateway import _FX_RATE_CACHE, _resolve_fx_rate

    saved_cache = dict(_FX_RATE_CACHE)
    try:
        # 1. Generic stock mock returning equity info & stock history must NOT leak 198,900 into FX rate
        _FX_RATE_CACHE.clear()
        stock_mock = MagicMock()
        stock_mock.info = {"symbol": "005930.KS", "currency": "KRW", "regularMarketPrice": 198900.0}
        stock_mock.history.return_value = pd.DataFrame({"Close": [198000.0, 198900.0]})
        with patch("yfinance.Ticker", return_value=stock_mock):
            rate = _resolve_fx_rate("KRW", "USD")
            assert rate == pytest.approx(1.0 / 1385.0)

        # 2. If .info raises, .history("5d") must still run and resolve live rate
        _FX_RATE_CACHE.clear()
        hist_only_mock = MagicMock()
        type(hist_only_mock).info = PropertyMock(side_effect=RuntimeError("info rate-limited"))
        hist_only_mock.history.return_value = pd.DataFrame({"Close": [0.0065, 0.0067]})
        with patch("yfinance.Ticker", return_value=hist_only_mock):
            assert _resolve_fx_rate("JPY", "USD") == pytest.approx(0.0067)

        # 3. Unknown currency pair must NOT cache emergency 1.0 default
        _FX_RATE_CACHE.clear()
        with patch("yfinance.Ticker", side_effect=RuntimeError("Offline")):
            assert _resolve_fx_rate("ZZZ", "USD") == 1.0
            assert ("ZZZ", "USD") not in _FX_RATE_CACHE

        # 4. NaN regularMarketPrice + NumPy scalar price handled cleanly
        _FX_RATE_CACHE.clear()
        np_mock = MagicMock()
        np_mock.info = {
            "symbol": "EURUSD=X",
            "quoteType": "CURRENCY",
            "regularMarketPrice": float("nan"),
            "currentPrice": np.float32(1.09),
        }
        with patch("yfinance.Ticker", return_value=np_mock):
            assert _resolve_fx_rate("EUR", "USD") == pytest.approx(1.09, rel=1e-4)
    finally:
        _FX_RATE_CACHE.clear()
        _FX_RATE_CACHE.update(saved_cache)


def test_parse_yfinance_snapshot_gbp_implied_shares_omitted_fin_curr_and_nan_price():
    """Verifies GBp defaults omitted financialCurrency to GBP, uses impliedSharesOutstanding for MC check, and rejects NaN price."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    import pandas as pd
    from src.data.market_gateway import DataUnavailableError, _FX_RATE_CACHE, parse_yfinance_snapshot

    saved_cache = dict(_FX_RATE_CACHE)
    try:
        _FX_RATE_CACHE.clear()

        dates = pd.to_datetime(["2024-12-31", "2025-12-31"])
        fin_df = pd.DataFrame(
            {d: [40_000_000_000.0, 8_000_000_000.0, 10_000_000_000.0, 2_000_000_000.0, 500_000_000.0, 7_000_000_000.0] for d in dates},
            index=["Total Revenue", "Cost Of Revenue", "EBIT", "Reconciled Depreciation", "Interest Expense", "Net Income"],
        )
        bs_df = pd.DataFrame(
            {d: [80_000_000_000.0, 24_000_000_000.0, 5_000_000_000.0, 6_000_000_000.0, 4_000_000_000.0, 45_000_000_000.0, 15_000_000_000.0, 4_000_000_000.0, 18_000_000_000.0, 25_000_000_000.0] for d in dates},
            index=["Total Assets", "Current Assets", "Cash And Cash Equivalents", "Receivables", "Inventory", "Total Liabilities Net Minority Interest", "Current Liabilities", "Accounts Payable", "Total Debt", "Retained Earnings"],
        )
        cf_df = pd.DataFrame(
            {d: [9_000_000_000.0, -1_500_000_000.0, 400_000_000.0] for d in dates},
            index=["Operating Cash Flow", "Capital Expenditure", "Change In Working Capital"],
        )

        hist_dates = pd.bdate_range("2025-01-01", periods=45)
        idx_arr = np.arange(45, dtype=float)
        closes = 13946.0 + 300.0 * np.sin(idx_arr / 3.0)
        stock_hist = pd.DataFrame({"Close": closes, "High": closes * 1.05, "Low": closes * 0.95}, index=hist_dates)
        bench_hist = pd.DataFrame({"Close": 5000.0 + 40.0 * np.sin(idx_arr / 3.0)}, index=hist_dates)

        azn_mock = MagicMock()
        azn_mock.income_stmt = fin_df
        azn_mock.balance_sheet = bs_df
        azn_mock.cashflow = cf_df
        azn_mock.history.return_value = stock_hist

        bench_mock = MagicMock()
        bench_mock.history.return_value = bench_hist

        fx_mock = MagicMock()
        fx_mock.info = {"symbol": "GBPUSD=X", "quoteType": "CURRENCY", "regularMarketPrice": 1.28}

        def _router(sym: str):
            if sym == "GBPUSD=X":
                return fx_mock
            if sym == "^GSPC":
                return bench_mock
            return azn_mock

        # NaN currentPrice must fail fast with DataUnavailableError
        with pytest.raises(DataUnavailableError, match="No valid market price"):
            parse_yfinance_snapshot("AZN.L", azn_mock, {"symbol": "AZN.L", "currentPrice": float("nan")})

        # Omit financialCurrency & sharesOutstanding (only impliedSharesOutstanding present)
        info = {
            "symbol": "AZN.L",
            "longName": "AstraZeneca PLC",
            "currency": "GBp",
            "currentPrice": 13946.0,
            "marketCap": 216_290_967_552.0,
            "impliedSharesOutstanding": 1_550_917_690.0,
            "fiftyTwoWeekHigh": 15000.0,
            "fiftyTwoWeekLow": 10000.0,
        }
        with patch("yfinance.Ticker", side_effect=_router):
            snap = parse_yfinance_snapshot("AZN.L", azn_mock, info)

        assert snap.currency == "USD"
        assert snap.market_cap == pytest.approx((216_290_967_552.0 * 1.28) / 1e6, rel=1e-2)
        assert snap.shares_outstanding == pytest.approx(1550.92, rel=1e-2)
        # Statements must scale using GBP->USD (1.28), NOT GBp->USD (0.0128)
        assert snap.historical_years[-1].revenue == pytest.approx((40_000_000_000.0 * 1.28) / 1e6, rel=1e-2)
    finally:
        _FX_RATE_CACHE.clear()
        _FX_RATE_CACHE.update(saved_cache)


def test_is_cache_valid_invalidates_unconverted_usd_high_price():
    """Verifies _is_cache_valid rejects stale unconverted USD snapshots with current_price > 25,000."""
    import time
    from src.data.market_gateway import CACHE_VERSION, _is_cache_valid

    now = time.time()
    # Flat snapshot format (actual disk cache layout)
    assert _is_cache_valid({
        "cache_version": CACHE_VERSION,
        "cached_at": now,
        "currency": "USD",
        "current_price": 198900.0,
    }) is False

    # Nested snapshot key format
    assert _is_cache_valid({
        "cache_version": CACHE_VERSION,
        "cached_at": now,
        "snapshot": {"currency": "USD", "current_price": 35000.0},
    }) is False

    # Valid USD price passes
    assert _is_cache_valid({
        "cache_version": CACHE_VERSION,
        "cached_at": now,
        "currency": "USD",
        "current_price": 143.61,
    }) is True

    # Legitimate high-priced INR stock (e.g. MRF.NS > 25,000 INR) remains valid
    assert _is_cache_valid({
        "cache_version": CACHE_VERSION,
        "cached_at": now,
        "currency": "INR",
        "current_price": 125000.0,
    }) is True



