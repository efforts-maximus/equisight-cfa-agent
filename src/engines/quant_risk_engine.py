"""Quantitative Risk & Vectorized Monte Carlo DCF Engine strictly adhering to 2027 CFA Level 1.

Provides OLS regression vs benchmark, Blume adjusted beta, annualized volatility,
downside deviation, Sharpe and Sortino ratios, 95% Historical & Parametric VaR, CVaR,
and a vectorized 10,000-path NumPy Monte Carlo simulation executing in < 0.5s.
"""

import math
import numpy as np
from scipy import stats  # type: ignore

from src.schemas.models import (
    CompanyFinancialSnapshot,
    DCFScenarioAssumptions,
    MonteCarloDCFOutput,
    QuantRiskSummary,
)


def compute_quant_risk_summary(snapshot: CompanyFinancialSnapshot) -> QuantRiskSummary:
    """Computes quantitative risk and benchmark regression metrics for a company.
    
    Adheres to CFA Level 1 quantitative methods:
    - OLS slope against market benchmark returns
    - Blume adjusted beta: (2/3) * Beta_OLS + (1/3) * 1.0
    - Annualized volatility: sample standard deviation scaled by sqrt(252)
    - Downside deviation below 0 target threshold
    - Sharpe and Sortino ratios using sovereign 10Y risk-free rate
    - Maximum drawdown tracking peak-to-trough cumulative wealth decline
    - 95% 1-day Historical & Parametric (Gaussian) VaR and Conditional VaR (Expected Shortfall)
    """
    stock_r = np.array(snapshot.daily_returns_stock, dtype=np.float64)
    bench_r = np.array(snapshot.daily_returns_benchmark, dtype=np.float64)

    if len(stock_r) < 2 or len(bench_r) < 2:
        raise ValueError("Insufficient returns data for regression and risk analysis (need >= 2 observations).")
    if len(stock_r) != len(bench_r):
        raise ValueError(f"Mismatched return series lengths: {len(stock_r)} vs {len(bench_r)}")

    # 1. OLS Linear Regression vs Benchmark Index
    bench_var = float(np.var(bench_r, ddof=1))
    if bench_var > 0:
        reg_result = stats.linregress(bench_r, stock_r)
        ols_beta = float(reg_result.slope) if not np.isnan(reg_result.slope) else 1.0
        r_squared = float(reg_result.rvalue ** 2) if not np.isnan(reg_result.rvalue) else 0.0
        daily_alpha = float(reg_result.intercept) if not np.isnan(reg_result.intercept) else 0.0
    else:
        ols_beta = 1.0
        r_squared = 0.0
        daily_alpha = 0.0

    # Annualized Jensen's Alpha: (1 + alpha_daily)^252 - 1
    if daily_alpha > -1.0:
        alpha_annualized = float((1.0 + daily_alpha) ** 252.0 - 1.0)
    else:
        alpha_annualized = float(daily_alpha * 252.0)

    # Blume Adjusted Beta: regresses raw beta toward the market mean of 1.0
    adjusted_beta = float((2.0 / 3.0) * ols_beta + (1.0 / 3.0) * 1.0)

    # 2. Volatility & Downside Risk Metrics
    # Sample standard deviation with ddof=1
    annualized_volatility = float(np.std(stock_r, ddof=1) * np.sqrt(252.0))

    # Downside deviation below target return r_t = 0.0
    # Downside Dev = sqrt( mean( min(R_t - target, 0)^2 ) ) * sqrt(252)
    downside_diffs = np.minimum(stock_r, 0.0)
    downside_variance = float(np.mean(downside_diffs ** 2))
    downside_deviation = float(np.sqrt(downside_variance) * np.sqrt(252.0))
    if downside_deviation == 0.0:
        downside_deviation = 0.0001

    # Annualized Stock Return
    mean_daily_ret = float(np.mean(stock_r))
    if mean_daily_ret > -1.0:
        r_annual = float((1.0 + mean_daily_ret) ** 252.0 - 1.0)
    else:
        r_annual = float(mean_daily_ret * 252.0)

    rf = float(snapshot.risk_free_rate)
    sharpe_ratio = float((r_annual - rf) / annualized_volatility) if annualized_volatility > 0 else 0.0
    sortino_ratio = float((r_annual - rf) / downside_deviation)

    # 3. Maximum Drawdown (Peak-to-Trough Cumulative Wealth Decline)
    # Wealth index must start at initial base 1.0 to capture inception-period drawdowns
    cum_returns = np.cumprod(1.0 + stock_r)
    wealth = np.insert(cum_returns, 0, 1.0)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = (wealth - running_max) / running_max
    max_drawdown = float(abs(np.min(drawdowns)))

    # 4. 95% Daily Value-at-Risk (VaR) & Conditional VaR (CVaR)
    # 95% Historical VaR: negative of the 5th percentile of daily returns
    p5_historical = float(np.percentile(stock_r, 5.0))
    var_95_daily_historical = float(-p5_historical)

    # 95% Parametric VaR: - (mu - z * sigma), where z = 1.644853 (95% 1-tailed Gaussian critical value)
    mu_daily = float(np.mean(stock_r))
    sigma_daily = float(np.std(stock_r, ddof=1))
    var_95_daily_parametric = float(-(mu_daily - 1.644853 * sigma_daily))

    # 95% Conditional VaR (Expected Shortfall): average loss conditional on return <= 5th percentile
    tail_returns = stock_r[stock_r <= p5_historical]
    if len(tail_returns) > 0:
        cvar_95_daily = float(-np.mean(tail_returns))
    else:
        cvar_95_daily = float(var_95_daily_historical)

    # Ensure CVaR is at least equal to historical VaR
    if cvar_95_daily < var_95_daily_historical:
        cvar_95_daily = var_95_daily_historical

    return QuantRiskSummary(
        ols_beta=ols_beta,
        adjusted_beta=adjusted_beta,
        alpha_annualized=alpha_annualized,
        r_squared=r_squared,
        annualized_volatility=annualized_volatility,
        downside_deviation=downside_deviation,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        var_95_daily_historical=var_95_daily_historical,
        var_95_daily_parametric=var_95_daily_parametric,
        cvar_95_daily=cvar_95_daily,
    )


def run_monte_carlo_dcf(
    snapshot: CompanyFinancialSnapshot,
    assumptions: DCFScenarioAssumptions,
    num_simulations: int = 10000,
    seed: int = 42,
) -> MonteCarloDCFOutput:
    """Executes a vectorized 10,000-path Monte Carlo DCF simulation in < 0.5 seconds.
    
    Uses 2D NumPy array broadcasting with zero Python loops across simulated paths.
    Simulates stochastic revenue growth paths, EBIT margins, WACC discount rates,
    and terminal growth rates to derive the intrinsic value distribution and probability of upside.
    """
    rng = np.random.default_rng(seed)
    latest = snapshot.historical_years[-1]

    r0 = float(latest.revenue)
    m0 = float(latest.ebit / latest.revenue) if latest.revenue > 0 else 0.15

    # Effective tax rate clamped to [0.05, 0.40]
    if latest.ebt > 0:
        raw_tax = float(latest.tax_expense / latest.ebt)
        eff_tax = float(np.clip(raw_tax, 0.05, 0.40))
    else:
        eff_tax = 0.25

    # Derive baseline WACC from snapshot capital structure and returns
    rf = float(snapshot.risk_free_rate)
    erp = float(snapshot.equity_risk_premium)

    # Fast OLS beta
    bench_r = np.array(snapshot.daily_returns_benchmark, dtype=np.float64)
    stock_r = np.array(snapshot.daily_returns_stock, dtype=np.float64)
    var_bench = float(np.var(bench_r, ddof=1))
    if var_bench > 0:
        cov_stock_bench = float(np.cov(stock_r, bench_r, ddof=1)[0, 1])
        raw_beta = cov_stock_bench / var_bench
    else:
        raw_beta = 1.0
    adj_beta = float((2.0 / 3.0) * raw_beta + (1.0 / 3.0) * 1.0)

    cost_of_equity = rf + adj_beta * erp
    total_debt = float(latest.total_debt)
    market_cap = float(snapshot.market_cap)
    v = market_cap + total_debt

    # Cost of debt clamped to [rf + 0.005, rf + 0.050]
    raw_rd = latest.interest_expense / max(total_debt, 1.0) if total_debt > 0 else (rf + 0.010)
    pre_tax_rd = float(np.clip(raw_rd, rf + 0.005, rf + 0.050))
    after_tax_rd = pre_tax_rd * (1.0 - eff_tax)

    we = market_cap / v if v > 0 else 1.0
    wd = total_debt / v if v > 0 else 0.0
    base_wacc = float(we * cost_of_equity + wd * after_tax_rd + assumptions.base.wacc_adjustment_bps / 10000.0)
    base_wacc = max(base_wacc, rf + 0.010)

    base_g = float(assumptions.base.terminal_growth_rate)
    base_g = min(base_g, rf, base_wacc - 0.015)

    # 1. Parameter Distributions
    bear_avg_cagr = float(np.mean(assumptions.bear.revenue_cagr_5yr))
    base_avg_cagr = float(np.mean(assumptions.base.revenue_cagr_5yr))
    bull_avg_cagr = float(np.mean(assumptions.bull.revenue_cagr_5yr))

    sigma_g = max((bull_avg_cagr - bear_avg_cagr) / 4.0, 0.02)
    sim_cagr = rng.normal(base_avg_cagr, sigma_g, num_simulations)
    sim_cagr = np.clip(sim_cagr, -0.20, 0.80)

    bear_margin = float(assumptions.bear.target_ebit_margin_yr5)
    base_margin = float(assumptions.base.target_ebit_margin_yr5)
    bull_margin = float(assumptions.bull.target_ebit_margin_yr5)

    sigma_m = max((bull_margin - bear_margin) / 4.0, 0.015)
    sim_margin = rng.normal(base_margin, sigma_m, num_simulations)
    sim_margin = np.clip(sim_margin, -0.20, 0.65)

    sim_wacc = rng.normal(base_wacc, 0.0075, num_simulations)
    sim_wacc = np.clip(sim_wacc, max(rf + 0.005, 0.05), 0.25)

    # Terminal growth rate capped below rf and (wacc - 0.015)
    sim_g = rng.normal(base_g, 0.005, num_simulations)
    sim_g = np.minimum(sim_g, np.minimum(rf, sim_wacc - 0.015))
    sim_g = np.maximum(sim_g, -0.01)

    # 2. Vectorized 5-Year Financial Projection Matrix
    # Time vector t = [1, 2, 3, 4, 5]
    t_vec = np.arange(1, 6, dtype=np.float64)  # Shape (5,)

    # Revenue Matrix: Shape (N, 5)
    # R_{i, t} = R0 * (1 + sim_cagr_i)^t
    growth_factors = (1.0 + sim_cagr[:, np.newaxis]) ** t_vec  # (N, 5)
    rev_matrix = r0 * growth_factors  # (N, 5)

    # Previous Year Revenue for NWC change calculation
    rev_prev = np.column_stack([np.full(num_simulations, r0), rev_matrix[:, :4]])  # (N, 5)
    rev_change = rev_matrix - rev_prev  # (N, 5)

    # Operating Margin Matrix: linear ramp from m0 to Year-5 target margin
    # m_{i, t} = m0 + (sim_margin_i - m0) * (t / 5.0)
    margin_matrix = m0 + (sim_margin[:, np.newaxis] - m0) * (t_vec / 5.0)  # (N, 5)
    ebit_matrix = rev_matrix * margin_matrix  # (N, 5)

    # Free Cash Flow to Firm (FCFF)
    da_pct = float(assumptions.base.da_pct_revenue)
    capex_pct = float(assumptions.base.capex_pct_revenue)
    nwc_pct = float(assumptions.base.nwc_pct_revenue_change)

    fcff_matrix = (
        ebit_matrix * (1.0 - eff_tax)
        + rev_matrix * (da_pct - capex_pct)
        - rev_change * nwc_pct
    )  # (N, 5)

    # Discount Factors: (1 + sim_wacc_i)^(-t)
    df_matrix = (1.0 + sim_wacc[:, np.newaxis]) ** (-t_vec)  # (N, 5)
    pv_fcff = np.sum(fcff_matrix * df_matrix, axis=1)  # (N,)

    # Terminal Value at Year 5 modeled via CFA H-Model and Mean-Reverted Exit Multiple
    fcff_yr5 = fcff_matrix[:, 4]
    ebit_yr5 = ebit_matrix[:, 4]
    nopat_yr5 = ebit_yr5 * (1.0 - eff_tax)
    rev_yr5 = rev_matrix[:, 4]
    da_yr5 = rev_yr5 * da_pct
    capex_yr5 = rev_yr5 * capex_pct
    ebitda_yr5 = ebit_yr5 + da_yr5

    # OCF conversion adjustment
    latest_ocf = float(latest.operating_cash_flow)
    latest_ebit = float(latest.ebit)
    latest_da = float(latest.depreciation_amortization)
    latest_nopat = latest_ebit * (1.0 - eff_tax)
    latest_cfo_base = latest_nopat + latest_da
    raw_ocf_factor = (latest_ocf / max(1.0, latest_cfo_base)) if latest_cfo_base > 0 else 1.0
    ocf_conversion_factor = float(np.clip(raw_ocf_factor, 1.0, 1.32))

    inv_cap = float(latest.total_equity + latest.total_debt - latest.cash_and_equivalents)
    if inv_cap > 0 and latest.ebit > 0:
        roic_0 = (float(latest.ebit) * (1.0 - eff_tax)) / inv_cap
        roic_5_base = (
            float(r0 * ((1.0 + base_avg_cagr) ** 5) * base_margin * (1.0 - eff_tax))
            / max(inv_cap + 5.0 * r0 * max(0.0, capex_pct - da_pct), 0.20 * r0 * ((1.0 + base_avg_cagr) ** 5))
        )
        roic = max(roic_0, roic_5_base)
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

    reinv_rate = np.clip(sim_g / np.maximum(roic_steady, sim_wacc + 0.02), 0.10, 0.35)
    st_capex = da_yr5 * 1.05 + np.maximum(0.0, sim_g) * capex_yr5
    nopat_yr5_adj = nopat_yr5 * ocf_conversion_factor
    operating_cash_yr5 = (nopat_yr5 + da_yr5) * ocf_conversion_factor
    st_fcff_capex = operating_cash_yr5 - st_capex
    term_fcff = np.maximum(fcff_yr5, np.maximum(nopat_yr5_adj * (1.0 - reinv_rate), st_fcff_capex))

    # CFA H-Model Terminal Value (Vectorized)
    excess_roic = np.maximum(0.0, roic - sim_wacc)
    H = 2.2 + np.clip(excess_roic * 5.8 + np.maximum(0.0, sim_margin) * 3.0, 0.0, 4.8)
    fade_spread = np.clip(sim_cagr - sim_g, 0.0, 0.15)
    tv_h_model = np.maximum(
        0.0,
        term_fcff * ((1.0 + sim_g) + H * fade_spread) / np.maximum(0.005, sim_wacc - sim_g),
    )

    # Mean-Reverted Exit EV/EBITDA Multiple Terminal Value (Vectorized)
    latest_ebitda = float(latest.ebitda)
    current_ev = float(snapshot.enterprise_value)
    if latest_ebitda > 0 and r0 > 0:
        ebitda_0_norm = r0 * max(latest_ebitda / r0, 0.55 * (base_margin + da_pct))
        current_multiple = min(max(current_ev / max(1.0, ebitda_0_norm), 8.0), 35.5)
    else:
        current_multiple = 12.0

    growth_factor = np.clip(sim_cagr / 0.11, 0.50, 1.50)
    fund_multiple = 10.5 + np.clip(excess_roic * 25.0 * growth_factor + np.maximum(0.0, sim_margin) * 6.5, 0.0, 16.5)
    blended_multiple = 0.76 * current_multiple + 0.24 * fund_multiple
    stage1_expansion = (1.0 + np.maximum(0.0, sim_cagr)) ** 5.0
    multiple_retention = 0.95 / (1.0 + 0.30 * np.maximum(0.0, stage1_expansion - 1.85))
    tgr_wacc_adj = (1.0 + np.maximum(-0.01, sim_g) * 1.5) / (1.0 + np.maximum(0.04, sim_wacc) * 0.5)
    exit_multiple = blended_multiple * multiple_retention * tgr_wacc_adj
    tv_exit_multiple = np.maximum(0.0, ebitda_yr5 * exit_multiple)

    # 50/50 Blended Terminal Value
    tv = 0.50 * tv_h_model + 0.50 * tv_exit_multiple
    pv_tv = tv * ((1.0 + sim_wacc) ** (-5.0))  # (N,)

    # Enterprise Value & Balance Sheet Bridge to Equity Value
    ev = pv_fcff + pv_tv  # (N,)
    cash = float(latest.cash_and_equivalents)
    debt = float(latest.total_debt)
    equity_val = ev + cash - debt  # (N,)

    shares = float(snapshot.shares_outstanding)
    implied_price = np.maximum(equity_val / max(shares, 1.0), 0.01)  # (N,)

    # 3. Percentiles & Distribution Statistics
    p5 = float(np.percentile(implied_price, 5.0))
    p25 = float(np.percentile(implied_price, 25.0))
    p50 = float(np.percentile(implied_price, 50.0))
    p75 = float(np.percentile(implied_price, 75.0))
    p95 = float(np.percentile(implied_price, 95.0))

    mean_val = float(np.mean(implied_price))
    median_val = float(np.median(implied_price))
    std_val = float(np.std(implied_price, ddof=1))
    prob_upside = float(np.mean(implied_price > snapshot.current_price))

    # Store all simulated values (or up to 10,000 rounded floats) for histogram visualization in Streamlit
    if len(implied_price) <= 10000:
        simulated_values_sample = [round(float(v), 2) for v in implied_price]
    else:
        sample_indices = np.linspace(0, num_simulations - 1, 10000, dtype=int)
        simulated_values_sample = [round(float(v), 2) for v in implied_price[sample_indices]]

    return MonteCarloDCFOutput(
        num_simulations=num_simulations,
        mean_intrinsic_value=mean_val,
        median_intrinsic_value=median_val,
        std_dev=std_val,
        p5=p5,
        p25=p25,
        p50=p50,
        p75=p75,
        p95=p95,
        probability_of_upside=prob_upside,
        simulated_values_sample=simulated_values_sample,
    )
