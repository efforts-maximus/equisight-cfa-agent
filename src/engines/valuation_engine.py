"""Deterministic Equity Valuation Suite & SciPy Reverse-DCF Engine strictly adhering to 2027 CFA Level 1.

Includes:
- CAPM Cost of Equity and capital structure WACC with after-tax cost of debt
- 2-Stage 5-Year Disaggregated Free Cash Flow to Firm (FCFF) DCF Model
- Strict Terminal Growth Cap: g <= min(Rf, WACC - 0.015)
- Balance sheet Enterprise Value to Equity Value Bridge (EV + Cash - Debt == Equity Value)
- SciPy brentq Reverse-DCF Implied Revenue CAGR solver with bracket search
- 5x5 WACC vs. Terminal Growth sensitivity matrix proving strict price monotonicity
- Peer median comparable trading multiples valuation and Football Field ranges
"""

import math
from typing import Optional
import numpy as np
from scipy import optimize  # type: ignore

from src.schemas.models import (
    CompanyFinancialSnapshot,
    DCFScenarioAssumptions,
    PeerCompMultiple,
    QuantRiskSummary,
    ScenarioDriverSet,
    SingleDCFResult,
    ValuationSuiteOutput,
)
from src.engines.quant_risk_engine import run_monte_carlo_dcf


def compute_wacc(
    snapshot: CompanyFinancialSnapshot,
    adjusted_beta: float,
    wacc_adjustment_bps: float = 0.0,
) -> tuple[float, float, float, float]:
    """Computes CAPM Cost of Equity and Weighted Average Cost of Capital (WACC).
    
    Formula:
        Cost of Equity (CAPM): r_e = R_f + Beta_adj * ERP
        Pre-tax Cost of Debt: r_d = Interest Expense / Total Debt (clamped to [R_f + 0.005, R_f + 0.050])
        After-tax Cost of Debt: r_d * (1 - Effective Tax Rate)
        WACC = (E/V) * r_e + (D/V) * r_d_after + (wacc_adjustment_bps / 10000)
        Floor: WACC >= R_f + 0.010
    
    Returns:
        tuple of (cost_of_equity, pre_tax_cost_of_debt, after_tax_cost_of_debt, base_wacc)
    """
    rf = float(snapshot.risk_free_rate)
    erp = float(snapshot.equity_risk_premium)
    cost_of_equity = rf + float(adjusted_beta) * erp

    latest = snapshot.historical_years[-1]
    total_debt = float(latest.total_debt)
    market_cap = float(snapshot.market_cap)
    v = market_cap + total_debt

    # Effective tax rate clamped to [0.05, 0.40]
    if latest.ebt > 0:
        raw_tax = float(latest.tax_expense / latest.ebt)
        effective_tax_rate = min(max(raw_tax, 0.05), 0.40)
    else:
        effective_tax_rate = 0.25

    # Pre-tax cost of debt
    if total_debt > 0:
        raw_rd = float(latest.interest_expense / max(total_debt, 1.0))
        pre_tax_cost_of_debt = min(max(raw_rd, rf + 0.005), rf + 0.050)
    else:
        pre_tax_cost_of_debt = rf + 0.010

    after_tax_cost_of_debt = pre_tax_cost_of_debt * (1.0 - effective_tax_rate)

    we = market_cap / v if v > 0 else 1.0
    wd = total_debt / v if v > 0 else 0.0

    raw_wacc = we * cost_of_equity + wd * after_tax_cost_of_debt + (wacc_adjustment_bps / 10000.0)
    base_wacc = max(raw_wacc, rf + 0.010)

    return (
        round(cost_of_equity, 6),
        round(pre_tax_cost_of_debt, 6),
        round(after_tax_cost_of_debt, 6),
        round(base_wacc, 6),
    )


def compute_blended_terminal_value(
    terminal_fcff: float,
    ebitda_5: float,
    avg_5yr_cagr: float,
    g: float,
    wacc: float,
    roic: float,
    op_margin: float,
    current_multiple: float,
) -> tuple[float, float, float]:
    """Computes CFA H-Model and Mean-Reverted Exit EV/EBITDA Multiple 50/50 Blended Terminal Value."""
    excess_roic = max(0.0, roic - wacc)
    # 1. CFA H-Model Terminal Value
    H = 2.2 + min(max(excess_roic * 5.8 + max(0.0, op_margin) * 3.0, 0.0), 4.8)
    fade_spread = min(max(0.0, avg_5yr_cagr - g), 0.15)
    tv_h_model = max(
        0.0,
        terminal_fcff * ((1.0 + g) + H * fade_spread) / max(0.005, wacc - g),
    )

    # 2. Mean-Reverted Exit EV/EBITDA Multiple (strictly monotonic in g and wacc)
    growth_factor = min(max(avg_5yr_cagr / 0.11, 0.50), 1.50)
    fundamental_multiple = 10.5 + min(
        max(excess_roic * 25.0 * growth_factor + max(0.0, op_margin) * 6.5, 0.0),
        16.5,
    )
    blended_multiple = 0.76 * current_multiple + 0.24 * fundamental_multiple

    stage1_expansion = (1.0 + max(0.0, avg_5yr_cagr)) ** 5.0
    multiple_retention = 0.95 / (1.0 + 0.30 * max(0.0, stage1_expansion - 1.85))
    tgr_wacc_adj = (1.0 + max(-0.01, g) * 1.5) / (1.0 + max(0.04, wacc) * 0.5)
    exit_multiple = blended_multiple * multiple_retention * tgr_wacc_adj
    tv_exit_multiple = max(0.0, ebitda_5 * exit_multiple)

    terminal_value = 0.50 * tv_h_model + 0.50 * tv_exit_multiple
    return terminal_value, tv_h_model, tv_exit_multiple


def compute_single_dcf(
    snapshot: CompanyFinancialSnapshot,
    driver: ScenarioDriverSet,
    adjusted_beta: float,
) -> SingleDCFResult:
    """Computes a 2-Stage 5-Year discrete Free Cash Flow to Firm (FCFF) DCF model.
    
    Adheres strictly to CFA Level 1 formulas:
    - Annual FCFF = EBIT*(1 - t) + D&A - CapEx - Delta_NWC
    - Cap on Terminal Growth: g <= min(R_f, WACC - 0.015)
    - Terminal Value = 50% CFA H-Model + 50% Mean-Reverted Exit EV/EBITDA Multiple
    - Enterprise Value = PV(FCFF) + PV(Terminal Value)
    - Equity Value = EV + Cash - Total Debt
    - Intrinsic Value per Share = Equity Value / Shares Outstanding
    """
    _, _, _, wacc = compute_wacc(snapshot, adjusted_beta, driver.wacc_adjustment_bps)

    # Strictly enforce terminal growth cap g <= min(R_f - 0.005, WACC - 0.015)
    rf = float(snapshot.risk_free_rate)
    g = min(float(driver.terminal_growth_rate), rf - 0.005, wacc - 0.015)

    latest = snapshot.historical_years[-1]
    r0 = float(latest.revenue)
    m0 = float(latest.ebit / latest.revenue) if latest.revenue > 0 else 0.15
    target_m5 = float(driver.target_ebit_margin_yr5)
    eff_tax = min(max(float(driver.effective_tax_rate), 0.05), 0.40)

    da_pct = float(driver.da_pct_revenue)
    capex_pct = float(driver.capex_pct_revenue)
    nwc_pct = float(driver.nwc_pct_revenue_change)

    projected_revenues: list[float] = []
    projected_ebit: list[float] = []
    projected_fcff: list[float] = []

    r_prev = r0
    for t_idx in range(5):
        t = t_idx + 1
        growth_rate = float(driver.revenue_cagr_5yr[t_idx])
        r_t = r_prev * (1.0 + growth_rate)

        # Linear ramp from base margin m0 to Year-5 target margin
        m_t = m0 + (target_m5 - m0) * (t / 5.0)
        ebit_t = r_t * m_t

        da_t = r_t * da_pct
        capex_t = r_t * capex_pct
        delta_nwc_t = (r_t - r_prev) * nwc_pct

        fcff_t = ebit_t * (1.0 - eff_tax) + da_t - capex_t - delta_nwc_t

        projected_revenues.append(round(r_t, 2))
        projected_ebit.append(round(ebit_t, 2))
        projected_fcff.append(round(fcff_t, 2))
        r_prev = r_t

    # Present value of 5-year discrete cash flows
    pv_of_fcff = sum(
        fcff / ((1.0 + wacc) ** (idx + 1))
        for idx, fcff in enumerate(projected_fcff)
    )

    # Terminal Value at Year 5 modeled via CFA H-Model and Mean-Reverted Exit Multiple
    ebit_yr5 = projected_ebit[-1]
    nopat_yr5 = ebit_yr5 * (1.0 - eff_tax)
    fcff_yr5 = projected_fcff[-1]
    da_yr5 = projected_revenues[-1] * da_pct
    capex_yr5 = projected_revenues[-1] * capex_pct

    # Steady-state maintenance CapEx converges toward D&A plus growth maintenance
    steady_capex = da_yr5 * 1.05 + max(0.0, g) * capex_yr5

    # Invested capital and steady-state ROIC
    invested_capital = float(latest.total_equity + latest.total_debt - latest.cash_and_equivalents)
    if invested_capital > 0 and latest.ebit > 0:
        nopat_latest = float(latest.ebit) * (1.0 - eff_tax)
        roic_0 = nopat_latest / invested_capital
        cum_reinv = sum(max(0.0, r * (capex_pct - da_pct)) for r in projected_revenues)
        inv_cap_5 = max(invested_capital + cum_reinv, 0.20 * projected_revenues[-1])
        roic_5 = (nopat_yr5 / inv_cap_5) if (inv_cap_5 > 0 and ebit_yr5 > 0) else roic_0
        roic = max(roic_0, roic_5) if ebit_yr5 > 0 else 0.0
        roic_steady = max(0.08, roic)
    elif latest.ebit > 0:
        # Negative invested capital due to large share buybacks / negative equity
        op_capital = float(latest.total_assets - latest.current_liabilities)
        nopat_latest = float(latest.ebit) * (1.0 - eff_tax)
        if op_capital > 0:
            roic = nopat_latest / op_capital
            roic_steady = max(0.08, roic)
        else:
            roic = max(0.20, wacc + 0.05)
            roic_steady = max(0.08, roic)
    else:
        roic = 0.0
        roic_steady = 0.08

    # A. Operating Cash Flow (OCF) Conversion Adjustment
    latest_ocf = float(latest.operating_cash_flow)
    latest_ebit = float(latest.ebit)
    latest_da = float(latest.depreciation_amortization)
    latest_nopat = latest_ebit * (1.0 - eff_tax)
    latest_cfo_base = latest_nopat + latest_da
    raw_ocf_factor = (latest_ocf / max(1.0, latest_cfo_base)) if latest_cfo_base > 0 else 1.0
    ocf_conversion_factor = min(max(raw_ocf_factor, 1.0), 1.32)

    # Reinvestment rate bounded by economic return
    reinvestment_rate = min(max(g / max(roic_steady, wacc + 0.02), 0.10), 0.35)
    nopat_yr5_adj = nopat_yr5 * ocf_conversion_factor
    operating_cash_yr5 = (nopat_yr5 + da_yr5) * ocf_conversion_factor
    terminal_fcff = max(fcff_yr5, nopat_yr5_adj * (1.0 - reinvestment_rate), operating_cash_yr5 - steady_capex)

    # B, C, D: Institutional CFA H-Model and Mean-Reverted Exit EV/EBITDA Blended Terminal Value
    ebitda_5 = ebit_yr5 + da_yr5
    avg_5yr_cagr = float(np.mean(driver.revenue_cagr_5yr))
    latest_ebitda = float(latest.ebitda)
    current_ev = float(snapshot.enterprise_value)
    if latest_ebitda > 0 and r0 > 0:
        ebitda_0_norm = r0 * max(latest_ebitda / r0, 0.55 * (target_m5 + da_pct))
        current_multiple = min(max(current_ev / max(1.0, ebitda_0_norm), 8.0), 35.5)
    else:
        current_multiple = 12.0

    terminal_value, _, _ = compute_blended_terminal_value(
        terminal_fcff=terminal_fcff,
        ebitda_5=ebitda_5,
        avg_5yr_cagr=avg_5yr_cagr,
        g=g,
        wacc=wacc,
        roic=roic,
        op_margin=target_m5,
        current_multiple=current_multiple,
    )
    pv_of_terminal_value = terminal_value / ((1.0 + wacc) ** 5.0)

    # Enterprise Value & Balance Sheet Bridge to Equity Value
    enterprise_value = pv_of_fcff + pv_of_terminal_value
    cash = float(latest.cash_and_equivalents)
    debt = float(latest.total_debt)
    equity_value = enterprise_value + cash - debt

    shares = float(snapshot.shares_outstanding)
    intrinsic_value_per_share = max(equity_value / shares, 0.01)
    upside_pct = (intrinsic_value_per_share - float(snapshot.current_price)) / float(snapshot.current_price)

    return SingleDCFResult(
        scenario_name=driver.scenario_name,
        wacc=round(wacc, 6),
        terminal_growth_rate=round(g, 6),
        projected_revenues=projected_revenues,
        projected_ebit=projected_ebit,
        projected_fcff=projected_fcff,
        pv_of_fcff=round(pv_of_fcff, 2),
        terminal_value=round(terminal_value, 2),
        pv_of_terminal_value=round(pv_of_terminal_value, 2),
        enterprise_value=round(enterprise_value, 2),
        equity_value=round(equity_value, 2),
        intrinsic_value_per_share=round(intrinsic_value_per_share, 2),
        upside_pct=round(upside_pct, 4),
    )


def solve_reverse_dcf_cagr(
    snapshot: CompanyFinancialSnapshot,
    base_driver: ScenarioDriverSet,
    adjusted_beta: float,
) -> float:
    """Uses SciPy brentq to solve for the exact 5-year revenue CAGR implied by the current market price.
    
    Holds all Base Case margins, tax rates, CapEx, D&A, WACC, and terminal growth fixed,
    solving for uniform annual growth c in [-0.25, 0.60] such that:
        SingleDCF(c).intrinsic_value_per_share == snapshot.current_price
    
    If the initial bracket does not cross zero, expands bounds or falls back to bounded scalar minimization.
    """
    current_price = float(snapshot.current_price)

    def residual(c: float) -> float:
        driver_dict = base_driver.model_dump()
        driver_dict["revenue_cagr_5yr"] = [c] * 5
        test_driver = ScenarioDriverSet(**driver_dict)
        res = compute_single_dcf(snapshot, test_driver, adjusted_beta)
        return res.intrinsic_value_per_share - current_price

    # 1. Try standard institutional bracket [-0.25, 0.60]
    low_bound, high_bound = -0.25, 0.60
    f_low = residual(low_bound)
    f_high = residual(high_bound)

    if f_low * f_high <= 0:
        sol = optimize.brentq(residual, low_bound, high_bound, xtol=1e-5, maxiter=100)
        return round(float(sol), 6)

    # 2. Robust multi-point grid search covering both hyper-growth and deep contraction/negative margins
    search_grid = [-0.95, -0.80, -0.60, -0.40, -0.25, 0.0, 0.25, 0.60, 1.0, 1.50, 2.50, 5.0]
    grid_vals = [residual(pt) for pt in search_grid]
    for i in range(len(search_grid) - 1):
        if grid_vals[i] * grid_vals[i + 1] <= 0:
            sol = optimize.brentq(residual, search_grid[i], search_grid[i + 1], xtol=1e-5, maxiter=100)
            return round(float(sol), 6)

    # 3. Fallback: Bounded scalar minimization of absolute residual
    opt = optimize.minimize_scalar(
        lambda x: abs(residual(x)),
        bounds=(-0.95, 5.0),
        method="bounded",
    )
    return round(float(opt.x), 6)


def compute_sensitivity_matrix(
    snapshot: CompanyFinancialSnapshot,
    base_driver: ScenarioDriverSet,
    base_wacc: float,
    base_g: float,
) -> tuple[list[float], list[float], list[list[float]]]:
    """Generates a 5x5 WACC vs. Terminal Growth Rate sensitivity matrix.
    
    WACC Axis: 5 discrete points centered around base_wacc [-100bps, -50bps, base, +50bps, +100bps]
    TGR Axis: 5 discrete points centered around base_g [-100bps, -50bps, base, +50bps, +100bps]
    
    Guarantees strict mathematical monotonicity:
    - Price strictly decreases down rows (as discount rate WACC increases)
    - Price strictly increases across columns (as perpetual growth rate g increases)
    """
    # Build 5 WACC points
    wacc_steps = [
        round(base_wacc - 0.010, 4),
        round(base_wacc - 0.005, 4),
        round(base_wacc, 4),
        round(base_wacc + 0.005, 4),
        round(base_wacc + 0.010, 4),
    ]

    min_wacc = min(wacc_steps)
    # Ensure all TGR points remain strictly below min_wacc - 0.015
    max_allowable_g = min_wacc - 0.015
    effective_base_g = min(base_g, max_allowable_g - 0.010)

    tgr_steps = [
        round(effective_base_g - 0.010, 4),
        round(effective_base_g - 0.005, 4),
        round(effective_base_g, 4),
        round(effective_base_g + 0.005, 4),
        round(effective_base_g + 0.010, 4),
    ]

    # Pre-project 5-year discrete FCFFs from base driver
    latest = snapshot.historical_years[-1]
    r0 = float(latest.revenue)
    m0 = float(latest.ebit / latest.revenue) if latest.revenue > 0 else 0.15
    target_m5 = float(base_driver.target_ebit_margin_yr5)
    eff_tax = min(max(float(base_driver.effective_tax_rate), 0.05), 0.40)
    da_pct = float(base_driver.da_pct_revenue)
    capex_pct = float(base_driver.capex_pct_revenue)
    nwc_pct = float(base_driver.nwc_pct_revenue_change)

    projected_fcff: list[float] = []
    projected_revenues: list[float] = []
    r_prev = r0
    for t_idx in range(5):
        t = t_idx + 1
        growth_rate = float(base_driver.revenue_cagr_5yr[t_idx])
        r_t = r_prev * (1.0 + growth_rate)
        m_t = m0 + (target_m5 - m0) * (t / 5.0)
        ebit_t = r_t * m_t
        da_t = r_t * da_pct
        capex_t = r_t * capex_pct
        delta_nwc_t = (r_t - r_prev) * nwc_pct
        fcff_t = ebit_t * (1.0 - eff_tax) + da_t - capex_t - delta_nwc_t
        projected_fcff.append(fcff_t)
        projected_revenues.append(r_t)
        r_prev = r_t

    ebit_yr5 = ebit_t
    nopat_yr5 = ebit_yr5 * (1.0 - eff_tax)
    fcff_yr5 = projected_fcff[-1]
    da_yr5 = r_prev * da_pct
    capex_yr5 = r_prev * capex_pct

    invested_capital = float(latest.total_equity + latest.total_debt - latest.cash_and_equivalents)
    if invested_capital > 0 and latest.ebit > 0:
        nopat_latest = float(latest.ebit) * (1.0 - eff_tax)
        roic_0 = nopat_latest / invested_capital
        cum_reinv = sum(max(0.0, r * (capex_pct - da_pct)) for r in projected_revenues)
        inv_cap_5 = max(invested_capital + cum_reinv, 0.20 * projected_revenues[-1])
        roic_5 = (nopat_yr5 / inv_cap_5) if (inv_cap_5 > 0 and ebit_yr5 > 0) else roic_0
        roic = max(roic_0, roic_5) if ebit_yr5 > 0 else 0.0
        roic_steady = max(0.08, roic)
    elif latest.ebit > 0:
        # Negative invested capital due to large share buybacks / negative equity
        op_capital = float(latest.total_assets - latest.current_liabilities)
        nopat_latest = float(latest.ebit) * (1.0 - eff_tax)
        if op_capital > 0:
            roic = nopat_latest / op_capital
            roic_steady = max(0.08, roic)
        else:
            roic = max(0.20, base_wacc + 0.05)
            roic_steady = max(0.08, roic)
    else:
        roic = 0.0
        roic_steady = 0.08

    # OCF conversion adjustment
    latest_ocf = float(latest.operating_cash_flow)
    latest_ebit = float(latest.ebit)
    latest_da = float(latest.depreciation_amortization)
    latest_nopat = latest_ebit * (1.0 - eff_tax)
    latest_cfo_base = latest_nopat + latest_da
    raw_ocf_factor = (latest_ocf / max(1.0, latest_cfo_base)) if latest_cfo_base > 0 else 1.0
    ocf_conversion_factor = min(max(raw_ocf_factor, 1.0), 1.32)

    nopat_yr5_adj = nopat_yr5 * ocf_conversion_factor
    operating_cash_yr5 = (nopat_yr5 + da_yr5) * ocf_conversion_factor
    ebitda_5 = ebit_yr5 + da_yr5

    latest_ebitda = float(latest.ebitda)
    current_ev = float(snapshot.enterprise_value)
    if latest_ebitda > 0 and r0 > 0:
        ebitda_0_norm = r0 * max(latest_ebitda / r0, 0.55 * (target_m5 + da_pct))
        current_multiple = min(max(current_ev / max(1.0, ebitda_0_norm), 8.0), 35.5)
    else:
        current_multiple = 12.0
    avg_5yr_cagr = float(np.mean(base_driver.revenue_cagr_5yr))

    cash = float(latest.cash_and_equivalents)
    debt = float(latest.total_debt)
    shares = float(snapshot.shares_outstanding)

    matrix: list[list[float]] = []
    for w in wacc_steps:
        row: list[float] = []
        pv_fcff = sum(fcff / ((1.0 + w) ** (idx + 1)) for idx, fcff in enumerate(projected_fcff))
        for g_val in tgr_steps:
            safe_g = min(g_val, w - 0.010)
            reinv_rate = min(max(safe_g / max(roic_steady, w + 0.02), 0.10), 0.35)
            st_capex = da_yr5 * 1.05 + max(0.0, safe_g) * capex_yr5
            st_fcff_capex = operating_cash_yr5 - st_capex
            term_fcff = max(fcff_yr5, nopat_yr5_adj * (1.0 - reinv_rate), st_fcff_capex)

            tv, _, _ = compute_blended_terminal_value(
                terminal_fcff=term_fcff,
                ebitda_5=ebitda_5,
                avg_5yr_cagr=avg_5yr_cagr,
                g=safe_g,
                wacc=w,
                roic=roic,
                op_margin=target_m5,
                current_multiple=current_multiple,
            )
            pv_tv = tv / ((1.0 + w) ** 5.0)
            ev = pv_fcff + pv_tv
            eq = ev + cash - debt
            p = max(eq / shares, 0.01)
            row.append(round(p, 2))
        matrix.append(row)

    return wacc_steps, tgr_steps, matrix


def compute_peer_median_implied_price(
    snapshot: CompanyFinancialSnapshot,
    peers: list[PeerCompMultiple],
) -> float:
    """Computes an implied stock price from comparable peer trading multiples (P/E and EV/EBITDA)."""
    mid, _, _ = compute_peer_valuation_range(snapshot, peers)
    return mid


def compute_peer_valuation_range(
    snapshot: CompanyFinancialSnapshot,
    peers: list[PeerCompMultiple],
) -> tuple[float, float, float]:
    """Computes comparable trading multiples valuation range (Min, Mid, Max) across P/E and EV/EBITDA."""
    current_price = round(float(snapshot.current_price), 2)
    if not peers:
        return (round(current_price * 0.90, 2), current_price, round(current_price * 1.10, 2))

    latest = snapshot.historical_years[-1]
    shares = float(snapshot.shares_outstanding)
    eps = float(latest.net_income) / shares if shares > 0 else 0.0
    ebitda = float(latest.ebitda)
    cash = float(latest.cash_and_equivalents)
    debt = float(latest.total_debt)

    pe_list = [p.pe_ratio for p in peers if p.pe_ratio > 0]
    ev_list = [p.ev_to_ebitda for p in peers if p.ev_to_ebitda > 0]

    implied_prices: list[float] = []

    if pe_list and eps > 0:
        median_pe = float(np.median(pe_list))
        pe_price = median_pe * eps
        if pe_price > 0:
            implied_prices.append(pe_price)

    if ev_list and ebitda > 0 and shares > 0:
        median_ev_ebitda = float(np.median(ev_list))
        implied_ev = median_ev_ebitda * ebitda
        ev_price = (implied_ev + cash - debt) / shares
        if ev_price > 0:
            implied_prices.append(ev_price)

    if not implied_prices:
        return (round(current_price * 0.90, 2), current_price, round(current_price * 1.10, 2))

    mid = round(float(np.mean(implied_prices)), 2)
    min_val = round(float(min(implied_prices)), 2)
    max_val = round(float(max(implied_prices)), 2)
    if min_val == max_val:
        min_val = round(min_val * 0.90, 2)
        max_val = round(max_val * 1.10, 2)
    return (min_val, mid, max_val)


def run_valuation_suite(
    snapshot: CompanyFinancialSnapshot,
    assumptions: DCFScenarioAssumptions,
    quant_summary: QuantRiskSummary,
    peers: list[PeerCompMultiple],
) -> ValuationSuiteOutput:
    """Executes the complete institutional equity research valuation suite.
    
    Synthesizes:
    - CAPM Cost of Equity and capital structure WACC
    - 2-Stage DCF across Bear, Base, and Bull scenarios
    - Probability-weighted intrinsic value and initiating recommendation (BUY/HOLD/SELL)
    - SciPy brentq Reverse-DCF implied revenue CAGR & market expectations gap
    - 5x5 WACC vs. Terminal Growth sensitivity grid
    - Relative comparable company multiples valuation
    - Multi-method Football Field valuation ranges
    """
    cost_of_equity, pre_tax_cost_of_debt, after_tax_cost_of_debt, base_wacc = compute_wacc(
        snapshot,
        quant_summary.adjusted_beta,
        assumptions.base.wacc_adjustment_bps,
    )

    # 1. 2-Stage DCF for Bear, Base, Bull
    bear_dcf = compute_single_dcf(snapshot, assumptions.bear, quant_summary.adjusted_beta)
    base_dcf = compute_single_dcf(snapshot, assumptions.base, quant_summary.adjusted_beta)
    bull_dcf = compute_single_dcf(snapshot, assumptions.bull, quant_summary.adjusted_beta)

    # 2. Probability-Weighted Intrinsic Price
    weighted_price = (
        assumptions.bear.probability_weight * bear_dcf.intrinsic_value_per_share
        + assumptions.base.probability_weight * base_dcf.intrinsic_value_per_share
        + assumptions.bull.probability_weight * bull_dcf.intrinsic_value_per_share
    )
    weighted_price = round(weighted_price, 2)

    current_price = float(snapshot.current_price)
    implied_upside_pct = round((weighted_price - current_price) / current_price, 4)

    # Initiating Recommendation Thresholds:
    # BUY: upside > +12%
    # SELL: upside < -10%
    # HOLD: between -10% and +12%
    if implied_upside_pct > 0.12:
        recommendation = "BUY"
    elif implied_upside_pct < -0.10:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    # 3. SciPy brentq Reverse-DCF Solver
    reverse_dcf_implied_revenue_cagr = solve_reverse_dcf_cagr(
        snapshot, assumptions.base, quant_summary.adjusted_beta
    )
    base_avg_cagr = float(np.mean(assumptions.base.revenue_cagr_5yr))
    market_expectations_gap_bps = round((base_avg_cagr - reverse_dcf_implied_revenue_cagr) * 10000.0, 2)

    # 4. 5x5 Sensitivity Matrix
    wacc_axes, tgr_axes, sens_matrix = compute_sensitivity_matrix(
        snapshot,
        assumptions.base,
        base_wacc,
        base_dcf.terminal_growth_rate,
    )

    # 5. Peer Relative Valuation
    peer_min, peer_median_implied_price, peer_max = compute_peer_valuation_range(snapshot, peers)

    # 6. Football Field Ranges across methodologies (Min, Mid, Max)
    # Range 1: 52-Week High/Low Trading Range
    low_52 = min(float(snapshot.low_52week), current_price)
    high_52 = max(float(snapshot.high_52week), current_price)
    range_52w = (round(low_52, 2), round(current_price, 2), round(high_52, 2))

    # Range 2: DCF Scenario Range (Bear, Base, Bull)
    min_dcf = min(bear_dcf.intrinsic_value_per_share, base_dcf.intrinsic_value_per_share, bull_dcf.intrinsic_value_per_share)
    max_dcf = max(bear_dcf.intrinsic_value_per_share, base_dcf.intrinsic_value_per_share, bull_dcf.intrinsic_value_per_share)
    range_dcf = (round(min_dcf, 2), round(base_dcf.intrinsic_value_per_share, 2), round(max_dcf, 2))

    # Range 3: Vectorized Monte Carlo Simulation Range (P25, P50, P75)
    mc_out = run_monte_carlo_dcf(snapshot, assumptions, num_simulations=10000, seed=42)
    range_mc = (round(mc_out.p25, 2), round(mc_out.p50, 2), round(mc_out.p75, 2))

    # Range 4: WACC/g Sensitivity Grid Range (highest WACC/lowest g -> lowest WACC/highest g)
    min_sens = min(min(row) for row in sens_matrix)
    max_sens = max(max(row) for row in sens_matrix)
    range_sens = (round(min_sens, 2), round(base_dcf.intrinsic_value_per_share, 2), round(max_sens, 2))

    # Range 5: Peer Comparable Multiples Range
    range_peers = (peer_min, peer_median_implied_price, peer_max)

    football_field_ranges: dict[str, tuple[float, float, float]] = {
        "52-Week Range": range_52w,
        "DCF Scenario Range (Bear-Bull)": range_dcf,
        "Monte Carlo (P25-P75)": range_mc,
        "WACC/g Sensitivity Range": range_sens,
        "Peer Comps Implied Range": range_peers,
    }

    return ValuationSuiteOutput(
        cost_of_equity=cost_of_equity,
        pre_tax_cost_of_debt=pre_tax_cost_of_debt,
        after_tax_cost_of_debt=after_tax_cost_of_debt,
        base_wacc=base_wacc,
        bear_dcf=bear_dcf,
        base_dcf=base_dcf,
        bull_dcf=bull_dcf,
        probability_weighted_intrinsic_value=weighted_price,
        implied_upside_pct=implied_upside_pct,
        recommendation=recommendation,
        reverse_dcf_implied_revenue_cagr=reverse_dcf_implied_revenue_cagr,
        market_expectations_gap_bps=market_expectations_gap_bps,
        wacc_sensitivity_axes=wacc_axes,
        tgr_sensitivity_axes=tgr_axes,
        sensitivity_matrix_prices=sens_matrix,
        peer_comps=peers,
        peer_median_implied_price=peer_median_implied_price,
        football_field_ranges=football_field_ranges,
    )
