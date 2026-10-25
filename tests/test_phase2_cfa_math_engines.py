"""Comprehensive Phase 2 test suite verifying 2027 CFA Level 1 mathematical invariance.

Tests:
1. FSA Engine:
   - 5-Way DuPont ROE Decomposition exactness (product of 5 terms == ROE within 1e-5)
   - Working Capital Cash Conversion Cycle identity (CCC == DSO + DIO - DPO within 1e-4)
   - Altman Z-Score solvency indicator and zone classification
   - Forensic red flag detection (CFO/NI < 0.80, Accruals > 10%, AR vs Rev spread > 10%)
2. Quant Risk Engine:
   - OLS benchmark linear regression and Blume Adjusted Beta identity (Beta_adj == 2/3 Beta_ols + 1/3)
   - Volatility, Downside Deviation, Sharpe & Sortino ratios, Max Drawdown
   - 95% Historical VaR, Parametric VaR, and Conditional VaR (CVaR >= VaR invariant)
   - Vectorized 10,000-path Monte Carlo DCF simulation performance (< 1.0s) and percentile ordering
3. Valuation Suite & Reverse-DCF Engine:
   - CAPM Cost of Equity and capital structure WACC
   - 2-Stage DCF scenario monotonicity (Bull > Base > Bear > 0)
   - Enterprise Value to Equity Value balance sheet bridge (EV + Cash - Debt == Equity Value within 1e-2)
   - SciPy brentq Reverse-DCF implied CAGR roundtrip price recovery within 1.0%
   - 5x5 WACC vs. Terminal Growth sensitivity matrix strict monotonicity
   - Peer comparable multiples valuation and football field ranges
"""

import time
import pytest
import numpy as np

from src.data.market_gateway import fetch_company_snapshot, fetch_peer_multiples
from src.engines.fsa_engine import compute_fsa_summary
from src.engines.quant_risk_engine import compute_quant_risk_summary, run_monte_carlo_dcf
from src.engines.valuation_engine import (
    compute_peer_median_implied_price,
    compute_sensitivity_matrix,
    compute_single_dcf,
    compute_wacc,
    run_valuation_suite,
    solve_reverse_dcf_cagr,
)
from src.schemas.models import (
    CompanyFinancialSnapshot,
    DCFScenarioAssumptions,
    HistoricalYearMetrics,
    ScenarioDriverSet,
    SingleDCFResult,
)


ALL_TICKERS = ["TCS.NS", "ZOMATO.NS", "VARUNBEV.NS", "NVDA"]


def get_default_scenario_assumptions(snapshot: CompanyFinancialSnapshot) -> DCFScenarioAssumptions:
    """Helper creating realistic, model-valid 3-scenario DCF assumptions for testing."""
    rf = snapshot.risk_free_rate
    term_g_base = min(0.035, rf * 0.5)

    bear_driver = ScenarioDriverSet(
        scenario_name="bear",
        probability_weight=0.25,
        revenue_cagr_5yr=[0.05, 0.05, 0.04, 0.04, 0.03],
        target_ebit_margin_yr5=0.20,
        effective_tax_rate=0.22,
        da_pct_revenue=0.03,
        capex_pct_revenue=0.045,
        nwc_pct_revenue_change=0.02,
        terminal_growth_rate=term_g_base - 0.010,
        wacc_adjustment_bps=100.0,
        rationale="Downside scenario with competitive margin compression.",
    )

    base_driver = ScenarioDriverSet(
        scenario_name="base",
        probability_weight=0.50,
        revenue_cagr_5yr=[0.10, 0.10, 0.09, 0.08, 0.07],
        target_ebit_margin_yr5=0.25,
        effective_tax_rate=0.22,
        da_pct_revenue=0.03,
        capex_pct_revenue=0.04,
        nwc_pct_revenue_change=0.02,
        terminal_growth_rate=term_g_base,
        wacc_adjustment_bps=0.0,
        rationale="Base case operational forecast in line with historical trends.",
    )

    bull_driver = ScenarioDriverSet(
        scenario_name="bull",
        probability_weight=0.25,
        revenue_cagr_5yr=[0.16, 0.15, 0.14, 0.12, 0.11],
        target_ebit_margin_yr5=0.30,
        effective_tax_rate=0.22,
        da_pct_revenue=0.03,
        capex_pct_revenue=0.035,
        nwc_pct_revenue_change=0.015,
        terminal_growth_rate=term_g_base + 0.005,
        wacc_adjustment_bps=-75.0,
        rationale="Upside scenario driven by market share expansion and operating leverage.",
    )

    return DCFScenarioAssumptions(
        bear=bear_driver,
        base=base_driver,
        bull=bull_driver,
        bull_thesis_points=["Structural margin tailwinds", "TAM expansion"],
        bear_thesis_points=["Competitive price erosion", "Cost inflation"],
        key_catalysts=["Product suite refreshes", "Geographic expansion"],
        key_risks=["Macro downturn", "Execution bottlenecks"],
        debate_synthesis_summary="Balanced outlook favoring moderate compound growth.",
    )


# ==============================================================================
# 1. FINANCIAL STATEMENT ANALYSIS (FSA) ENGINE TESTS
# ==============================================================================

@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_fsa_dupont_5way_exactness(ticker: str):
    """Verifies that the product of the 5 DuPont terms exactly equals ROE within 1e-5."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    fsa = compute_fsa_summary(snapshot)

    # 5-way terms:
    # Tax Burden = NI / EBT
    # Interest Burden = EBT / EBIT
    # EBIT Margin = EBIT / Revenue
    # Asset Turnover = Revenue / Avg Assets
    # Financial Leverage = Avg Assets / Avg Equity
    dupont_product = (
        fsa.tax_burden
        * fsa.interest_burden
        * fsa.ebit_margin
        * fsa.asset_turnover
        * fsa.financial_leverage
    )

    assert abs(dupont_product - fsa.roe) < 1e-5, (
        f"DuPont product ({dupont_product:.6f}) differs from ROE ({fsa.roe:.6f}) for {ticker}"
    )
    assert abs(fsa.reconstructed_roe - fsa.roe) < 1e-5, (
        f"reconstructed_roe ({fsa.reconstructed_roe:.6f}) differs from roe ({fsa.roe:.6f}) for {ticker}"
    )


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_fsa_working_capital_ccc_identity(ticker: str):
    """Verifies that the Cash Conversion Cycle equals DSO + DIO - DPO within 1e-4."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    fsa = compute_fsa_summary(snapshot)

    expected_ccc = fsa.dso + fsa.dio - fsa.dpo
    assert abs(fsa.cash_conversion_cycle - expected_ccc) < 1e-4, (
        f"CCC identity violated: {fsa.cash_conversion_cycle:.4f} != {expected_ccc:.4f} for {ticker}"
    )


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_fsa_altman_z_score_and_zone(ticker: str):
    """Verifies that Altman Z-Score is computed and categorized correctly into Safe, Grey, or Distress."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    fsa = compute_fsa_summary(snapshot)

    assert isinstance(fsa.altman_z_score, float)
    assert fsa.altman_zone in ["Safe", "Grey", "Distress"]

    if fsa.altman_z_score > 2.99:
        assert fsa.altman_zone == "Safe"
    elif fsa.altman_z_score >= 1.81:
        assert fsa.altman_zone == "Grey"
    else:
        assert fsa.altman_zone == "Distress"


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_fsa_profitability_and_growth_bounds(ticker: str):
    """Verifies that core profitability ratios and common-size statements are well-formed."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    fsa = compute_fsa_summary(snapshot)

    # Common-size revenue must be 1.0
    assert fsa.common_size_income_summary["revenue"] == 1.0
    assert "gross_profit" in fsa.common_size_income_summary
    assert "ebit" in fsa.common_size_income_summary
    assert "net_income" in fsa.common_size_income_summary

    assert -1.0 <= fsa.net_profit_margin <= 1.0
    assert fsa.current_ratio > 0.0
    assert fsa.quick_ratio >= 0.0


def test_fsa_forensic_red_flags_detection():
    """Verifies that synthetic forensic accounting anomalies trigger red flags."""
    snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
    snapshot_dict = snapshot.model_dump()

    # Artificially weaken cash conversion: CFO = 0.5 * Net Income (conversion < 0.80)
    # Artificially expand accounts receivable: +50% AR growth vs +10% revenue growth
    latest_yr = snapshot_dict["historical_years"][-1]
    prior_yr = snapshot_dict["historical_years"][-2]

    latest_yr["operating_cash_flow"] = latest_yr["net_income"] * 0.50
    prior_yr["accounts_receivable"] = 10000.0
    latest_yr["accounts_receivable"] = 16000.0  # +60% AR growth
    prior_yr["revenue"] = 200000.0
    latest_yr["revenue"] = 210000.0  # +5% Rev growth -> spread = 55%

    corrupted_snapshot = CompanyFinancialSnapshot(**snapshot_dict)
    fsa = compute_fsa_summary(corrupted_snapshot)

    assert fsa.cfo_to_net_income_ratio < 0.80
    assert fsa.receivables_vs_revenue_growth_spread > 0.10
    assert len(fsa.accounting_red_flags) >= 2
    assert any("CFO to Net Income conversion weak" in flag for flag in fsa.accounting_red_flags)
    assert any("Accounts receivable growth exceeds revenue growth" in flag for flag in fsa.accounting_red_flags)


# ==============================================================================
# 2. QUANTITATIVE RISK & MONTE CARLO ENGINE TESTS
# ==============================================================================

@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_quant_blume_adjusted_beta_identity(ticker: str):
    """Verifies that Blume Adjusted Beta strictly equals (2/3)*OLS_Beta + (1/3)*1.0 within 1e-5."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    q = compute_quant_risk_summary(snapshot)

    expected_adj_beta = (2.0 / 3.0) * q.ols_beta + (1.0 / 3.0) * 1.0
    assert abs(q.adjusted_beta - expected_adj_beta) < 1e-5, (
        f"Blume beta mismatch for {ticker}: {q.adjusted_beta} != {expected_adj_beta}"
    )


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_quant_risk_metrics_invariants(ticker: str):
    """Verifies statistical risk metrics: Volatility > 0, Downside Dev > 0, and CVaR >= VaR."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    q = compute_quant_risk_summary(snapshot)

    assert q.annualized_volatility > 0.0
    assert q.downside_deviation > 0.0
    assert 0.0 <= q.max_drawdown <= 1.0

    # Conditional VaR (Expected Shortfall) must be >= 95% Historical VaR
    assert q.cvar_95_daily >= q.var_95_daily_historical - 1e-6, (
        f"CVaR ({q.cvar_95_daily}) must be >= VaR ({q.var_95_daily_historical}) for {ticker}"
    )


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_vectorized_monte_carlo_dcf_performance_and_percentiles(ticker: str):
    """Verifies that 10,000-path Monte Carlo runs in < 1.0s and percentiles are monotonically ordered."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    assumptions = get_default_scenario_assumptions(snapshot)

    start_time = time.perf_counter()
    mc_output = run_monte_carlo_dcf(snapshot, assumptions, num_simulations=10000, seed=42)
    elapsed_time = time.perf_counter() - start_time

    # Performance mandate: < 1.0s (typically < 0.1s via 2D NumPy array broadcasting)
    assert elapsed_time < 1.0, f"Monte Carlo took {elapsed_time:.3f}s, exceeding 1.0s limit"
    assert mc_output.num_simulations == 10000

    # Percentile ordering: P5 <= P25 <= P50 <= P75 <= P95
    assert mc_output.p5 <= mc_output.p25 <= mc_output.p50 <= mc_output.p75 <= mc_output.p95
    assert 0.0 <= mc_output.probability_of_upside <= 1.0
    assert len(mc_output.simulated_values_sample) == 10000


# ==============================================================================
# 3. VALUATION SUITE & REVERSE-DCF ENGINE TESTS
# ==============================================================================

@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_valuation_wacc_calculation(ticker: str):
    """Verifies that CAPM Cost of Equity and capital structure WACC satisfy cost floors."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    q = compute_quant_risk_summary(snapshot)
    re, rd_pre, rd_post, wacc = compute_wacc(snapshot, q.adjusted_beta)

    rf = snapshot.risk_free_rate
    assert re > rf
    assert rd_pre >= rf + 0.005
    assert rd_post <= rd_pre
    assert wacc >= rf + 0.010


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_valuation_ev_to_equity_bridge_invariance(ticker: str):
    """Verifies that Enterprise Value + Cash - Debt == Equity Value holds within 1e-2."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    assumptions = get_default_scenario_assumptions(snapshot)
    q = compute_quant_risk_summary(snapshot)

    latest = snapshot.historical_years[-1]
    cash = latest.cash_and_equivalents
    debt = latest.total_debt

    for driver in [assumptions.bear, assumptions.base, assumptions.bull]:
        dcf = compute_single_dcf(snapshot, driver, q.adjusted_beta)
        reconstructed_equity = dcf.enterprise_value + cash - debt
        assert abs(dcf.equity_value - reconstructed_equity) < 1e-2, (
            f"EV-to-Equity bridge violated for {ticker} ({driver.scenario_name}): "
            f"EV({dcf.enterprise_value}) + Cash({cash}) - Debt({debt}) != Eq({dcf.equity_value})"
        )


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_valuation_dcf_scenario_monotonicity(ticker: str):
    """Verifies that Bull Intrinsic Price > Base Intrinsic Price > Bear Intrinsic Price > 0."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    assumptions = get_default_scenario_assumptions(snapshot)
    q = compute_quant_risk_summary(snapshot)

    bear_dcf = compute_single_dcf(snapshot, assumptions.bear, q.adjusted_beta)
    base_dcf = compute_single_dcf(snapshot, assumptions.base, q.adjusted_beta)
    bull_dcf = compute_single_dcf(snapshot, assumptions.bull, q.adjusted_beta)

    assert bear_dcf.intrinsic_value_per_share > 0.0
    assert base_dcf.intrinsic_value_per_share > bear_dcf.intrinsic_value_per_share, (
        f"Base price ({base_dcf.intrinsic_value_per_share}) not > Bear ({bear_dcf.intrinsic_value_per_share})"
    )
    assert bull_dcf.intrinsic_value_per_share > base_dcf.intrinsic_value_per_share, (
        f"Bull price ({bull_dcf.intrinsic_value_per_share}) not > Base ({base_dcf.intrinsic_value_per_share})"
    )


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_reverse_dcf_roundtrip_price_recovery(ticker: str):
    """Verifies that plugging Reverse-DCF implied CAGR into the DCF engine reproduces current price within 1.0%."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    assumptions = get_default_scenario_assumptions(snapshot)
    q = compute_quant_risk_summary(snapshot)

    implied_cagr = solve_reverse_dcf_cagr(snapshot, assumptions.base, q.adjusted_beta)

    # Plug implied_cagr back into single DCF
    driver_dict = assumptions.base.model_dump()
    driver_dict["revenue_cagr_5yr"] = [implied_cagr] * 5
    test_driver = ScenarioDriverSet(**driver_dict)

    reproduced_dcf = compute_single_dcf(snapshot, test_driver, q.adjusted_beta)
    reproduced_price = reproduced_dcf.intrinsic_value_per_share
    current_price = snapshot.current_price

    error_pct = abs(reproduced_price - current_price) / current_price
    assert error_pct < 0.01, (
        f"Reverse-DCF failed to recover price within 1.0% for {ticker}: "
        f"current={current_price}, reproduced={reproduced_price}, error={error_pct * 100:.3f}%"
    )


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_sensitivity_matrix_strict_monotonicity(ticker: str):
    """Verifies that the 5x5 WACC/g sensitivity matrix is strictly monotonic along both axes.
    
    - Down rows: Higher WACC leads to lower intrinsic price (P[r+1][c] <= P[r][c])
    - Across cols: Higher terminal growth g leads to higher intrinsic price (P[r][c+1] >= P[r][c])
    """
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    assumptions = get_default_scenario_assumptions(snapshot)
    q = compute_quant_risk_summary(snapshot)
    _, _, _, base_wacc = compute_wacc(snapshot, q.adjusted_beta)
    base_g = assumptions.base.terminal_growth_rate

    wacc_axes, tgr_axes, matrix = compute_sensitivity_matrix(snapshot, assumptions.base, base_wacc, base_g)

    assert len(matrix) == 5
    assert all(len(row) == 5 for row in matrix)

    # Verify column monotonicity (down rows: WACC increases -> price must decrease)
    for c in range(5):
        for r in range(4):
            assert matrix[r + 1][c] <= matrix[r][c], (
                f"WACC monotonicity violated for {ticker} at col {c}: row {r+1} ({matrix[r+1][c]}) > row {r} ({matrix[r][c]})"
            )

    # Verify row monotonicity (across cols: g increases -> price must increase)
    for r in range(5):
        for c in range(4):
            assert matrix[r][c + 1] >= matrix[r][c], (
                f"TGR monotonicity violated for {ticker} at row {r}: col {c+1} ({matrix[r][c+1]}) < col {c} ({matrix[r][c]})"
            )


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_run_valuation_suite_end_to_end(ticker: str):
    """Verifies that run_valuation_suite executes seamlessly across all 4 benchmark tickers."""
    snapshot = fetch_company_snapshot(ticker, use_live=False)
    assumptions = get_default_scenario_assumptions(snapshot)
    q = compute_quant_risk_summary(snapshot)
    peers = fetch_peer_multiples(ticker, use_live=False)

    suite = run_valuation_suite(snapshot, assumptions, q, peers)

    assert suite.recommendation in ["BUY", "HOLD", "SELL"]
    assert suite.probability_weighted_intrinsic_value > 0.0
    assert suite.peer_median_implied_price > 0.0
    assert len(suite.peer_comps) == 3

    # Check football field ranges
    assert len(suite.football_field_ranges) == 5
    assert "Monte Carlo (P25-P75)" in suite.football_field_ranges
    assert "Peer Comps Implied Range" in suite.football_field_ranges
    for name, (low, mid, high) in suite.football_field_ranges.items():
        assert low <= mid <= high, (
            f"Football field ordering violated for {ticker} ({name}): low={low}, mid={mid}, high={high}"
        )


# ==============================================================================
# 4. EDGE CASES, ERROR PATHS & BOUNDARY VALUE TESTS
# ==============================================================================

def test_fsa_engine_empty_historical_years_raises():
    """Negative test: compute_fsa_summary must raise ValueError if historical_years is empty."""
    snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
    snapshot_empty = CompanyFinancialSnapshot.model_construct(
        **{**snapshot.model_dump(), "historical_years": []}
    )
    with pytest.raises(ValueError) as exc:
        compute_fsa_summary(snapshot_empty)
    assert "historical_years is empty" in str(exc.value)


def test_quant_engine_insufficient_returns_raises():
    """Negative test: compute_quant_risk_summary must raise ValueError if returns series < 2."""
    snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
    snapshot_dict = snapshot.model_dump()
    snapshot_dict["daily_returns_stock"] = [0.01]
    snapshot_dict["daily_returns_benchmark"] = [0.01]
    short_snapshot = CompanyFinancialSnapshot(**snapshot_dict)

    with pytest.raises(ValueError) as exc:
        compute_quant_risk_summary(short_snapshot)
    assert "Insufficient returns data" in str(exc.value)


def test_quant_engine_max_drawdown_inception_decline():
    """Regression test: max_drawdown must measure peak-to-trough decline against 1.0 wealth inception base."""
    snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
    snapshot_dict = snapshot.model_dump()
    # Sequence where day 1 drops 10%, then makes minor positive gains that never exceed 1.0
    snapshot_dict["daily_returns_stock"] = [-0.10, 0.02, 0.02, 0.01]
    snapshot_dict["daily_returns_benchmark"] = [0.01, 0.01, 0.01, 0.01]
    snap = CompanyFinancialSnapshot(**snapshot_dict)

    q = compute_quant_risk_summary(snap)
    assert abs(q.max_drawdown - 0.10) < 1e-4, (
        f"max_drawdown failed to capture 10% inception drawdown: got {q.max_drawdown}"
    )


def test_valuation_engine_zero_debt_company():
    """Edge case: company with zero debt must compute valid WACC and DCF without division by zero."""
    snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
    snapshot_dict = snapshot.model_dump()
    for yr in snapshot_dict["historical_years"]:
        yr["total_debt"] = 0.0
        yr["interest_expense"] = 0.0
    zero_debt_snapshot = CompanyFinancialSnapshot(**snapshot_dict)

    q = compute_quant_risk_summary(zero_debt_snapshot)
    re, rd_pre, rd_post, wacc = compute_wacc(zero_debt_snapshot, q.adjusted_beta)

    assert re > zero_debt_snapshot.risk_free_rate
    assert wacc >= zero_debt_snapshot.risk_free_rate + 0.010

    assumptions = get_default_scenario_assumptions(zero_debt_snapshot)
    dcf = compute_single_dcf(zero_debt_snapshot, assumptions.base, q.adjusted_beta)
    assert dcf.intrinsic_value_per_share > 0.0
    # EV + Cash == Equity Value
    latest = zero_debt_snapshot.historical_years[-1]
    assert abs(dcf.equity_value - (dcf.enterprise_value + latest.cash_and_equivalents)) < 1e-2


def test_reverse_dcf_extreme_undervalued_and_overvalued_bounds():
    """Edge case: Reverse-DCF solver handles deep mispricing cleanly."""
    snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
    assumptions = get_default_scenario_assumptions(snapshot)
    q = compute_quant_risk_summary(snapshot)

    # 1. Artificially low price (deep value)
    snap_cheap_dict = snapshot.model_dump()
    snap_cheap_dict["current_price"] = 500.0  # ~85% below base price
    snap_cheap = CompanyFinancialSnapshot(**snap_cheap_dict)
    cagr_cheap = solve_reverse_dcf_cagr(snap_cheap, assumptions.base, q.adjusted_beta)
    assert cagr_cheap < -0.10  # expects significant contraction

    # 2. Artificially high price (hyper-growth expected)
    snap_exp_dict = snapshot.model_dump()
    snap_exp_dict["current_price"] = 15000.0  # 4x current price
    snap_exp = CompanyFinancialSnapshot(**snap_exp_dict)
    cagr_exp = solve_reverse_dcf_cagr(snap_exp, assumptions.base, q.adjusted_beta)
    assert cagr_exp > 0.20  # expects aggressive revenue growth


def test_reverse_dcf_negative_margins_solver_convergence():
    """Regression test: Reverse-DCF must solve the exact implied CAGR on negative margin companies."""
    snapshot = fetch_company_snapshot("ZOMATO.NS", use_live=False)
    assumptions = get_default_scenario_assumptions(snapshot)
    # Set negative terminal margin
    driver = assumptions.base.model_copy(update={"target_ebit_margin_yr5": -0.05})
    # Target price reachable via revenue contraction
    target_snapshot = snapshot.model_copy(update={"current_price": 16.0})

    q = compute_quant_risk_summary(target_snapshot)
    solved_cagr = solve_reverse_dcf_cagr(target_snapshot, driver, q.adjusted_beta)

    # Verify that plugging solved CAGR back into single DCF reproduces 16.0 within 1%
    test_driver = driver.model_copy(update={"revenue_cagr_5yr": [solved_cagr] * 5})
    res = compute_single_dcf(target_snapshot, test_driver, q.adjusted_beta)
    error_pct = abs(res.intrinsic_value_per_share - 16.0) / 16.0
    assert error_pct < 0.01, (
        f"Failed to recover price on negative margin company: expected 16.0, got {res.intrinsic_value_per_share:.2f}"
    )


def test_fsa_dupont_exactness_zero_ebit():
    """Edge case: DuPont identity reconstructed_roe == roe holds even if EBIT is zero."""
    snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
    snap_dict = snapshot.model_dump()
    snap_dict["historical_years"][-1]["ebit"] = 0.0
    snap_dict["historical_years"][-1]["net_income"] = 500.0
    zero_ebit_snap = CompanyFinancialSnapshot(**snap_dict)

    fsa = compute_fsa_summary(zero_ebit_snap)
    assert abs(fsa.reconstructed_roe - fsa.roe) < 1e-5, (
        f"DuPont identity failed with zero EBIT: recon={fsa.reconstructed_roe}, roe={fsa.roe}"
    )


def test_fsa_revenue_cagr_negative_revenue_safety():
    """Edge case: Non-positive revenue in history does not produce complex numbers or crash."""
    snapshot = fetch_company_snapshot("TCS.NS", use_live=False)
    snap_dict = snapshot.model_dump()
    snap_dict["historical_years"][-1]["revenue"] = -10.0
    neg_rev_snap = CompanyFinancialSnapshot.model_construct(
        **{**snap_dict, "historical_years": [HistoricalYearMetrics.model_construct(**yr) for yr in snap_dict["historical_years"]]}
    )
    fsa = compute_fsa_summary(neg_rev_snap)
    assert isinstance(fsa.revenue_cagr_3yr, float)
    assert fsa.revenue_cagr_3yr == 0.0

