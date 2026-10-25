"""Pydantic v2 data models and contracts for EquiSight CFA Agent."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HistoricalYearMetrics(BaseModel):
    """Annual historical financial metrics spanning Income Statement, Balance Sheet, and Cash Flow."""
    model_config = ConfigDict(extra="forbid")

    fiscal_year: int = Field(description="Fiscal year (e.g. 2023, 2024)")
    revenue: float = Field(description="Total revenue / top-line turnover")
    cogs: float = Field(description="Cost of goods sold / cost of services")
    gross_profit: float = Field(description="Gross profit (Revenue - COGS)")
    ebitda: float = Field(description="Operating profit before D&A")
    depreciation_amortization: float = Field(description="Depreciation and Amortization expense")
    ebit: float = Field(description="Operating profit / EBIT (EBITDA - D&A)")
    interest_expense: float = Field(description="Net interest and financing costs")
    ebt: float = Field(description="Earnings before tax (EBIT - Interest Expense)")
    tax_expense: float = Field(description="Income tax expense")
    net_income: float = Field(description="Net profit / bottom-line income")

    total_assets: float = Field(description="Total balance sheet assets")
    current_assets: float = Field(description="Current assets")
    cash_and_equivalents: float = Field(description="Cash and cash equivalents")
    accounts_receivable: float = Field(description="Trade and accounts receivable")
    inventory: float = Field(description="Inventories / raw materials / finished goods")
    total_liabilities: float = Field(description="Total balance sheet liabilities")
    current_liabilities: float = Field(description="Current liabilities")
    accounts_payable: float = Field(description="Trade and accounts payable")
    total_debt: float = Field(description="Total interest-bearing debt (short-term + long-term)")
    total_equity: float = Field(description="Total shareholders' equity")
    retained_earnings: float = Field(description="Accumulated retained earnings")

    operating_cash_flow: float = Field(description="Cash flow from operating activities (CFO)")
    capex: float = Field(description="Capital expenditures (CapEx)")
    working_capital_change: float = Field(description="Change in non-cash working capital")


class CompanyFinancialSnapshot(BaseModel):
    """Institutional-grade financial snapshot of a company, supporting India and US equities."""
    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(description="Ticker symbol, e.g. TCS.NS, NVDA")
    company_name: str = Field(description="Full legal corporate name")
    sector: str = Field(description="Macro sector, e.g. Technology, Consumer Staples")
    industry: str = Field(description="Specific industry, e.g. IT Services, Beverages")
    currency: Literal["INR", "USD"] = Field(description="Financial reporting currency")
    unit_label: Literal["INR Crores", "USD Millions"] = Field(description="Unit denomination")

    current_price: float = Field(description="Current market price per share")
    shares_outstanding: float = Field(description="Total diluted shares outstanding")
    market_cap: float = Field(description="Market capitalization")
    enterprise_value: float = Field(description="Enterprise value (Market Cap + Total Debt - Cash)")
    high_52week: float = Field(description="52-week trading high")
    low_52week: float = Field(description="52-week trading low")

    benchmark_ticker: Literal["^NSEI", "^GSPC"] = Field(description="Benchmark market index")
    risk_free_rate: float = Field(description="10Y Sovereign Bond Yield (e.g. 0.068 for India, 0.042 for US)")
    equity_risk_premium: float = Field(description="Expected equity risk premium (e.g. 0.060 for India, 0.050 for US)")

    historical_years: list[HistoricalYearMetrics] = Field(description="Sequential multi-year audited financial records")
    daily_returns_stock: list[float] = Field(description="Daily return series for stock (last 252 trading days)")
    daily_returns_benchmark: list[float] = Field(description="Daily return series for benchmark index (last 252 trading days)")
    business_summary: str = Field(description="Overview of business model, core segments, and geographic footprint")

    @model_validator(mode="after")
    def validate_snapshot_invariants(self) -> "CompanyFinancialSnapshot":
        if self.high_52week < self.low_52week:
            raise ValueError(
                f"high_52week ({self.high_52week}) cannot be less than low_52week ({self.low_52week})"
            )
        if len(self.historical_years) == 0:
            raise ValueError("historical_years must contain at least one year of financial metrics")
        if len(self.daily_returns_stock) != len(self.daily_returns_benchmark):
            raise ValueError(
                f"daily_returns_stock ({len(self.daily_returns_stock)}) must match daily_returns_benchmark ({len(self.daily_returns_benchmark)}) length"
            )
        return self


class FSARatioSummary(BaseModel):
    """Comprehensive Financial Statement Analysis ratios strictly adhering to 2027 CFA Level 1."""
    model_config = ConfigDict(extra="forbid")

    # Core Profitability & Growth
    gross_margin: float = Field(description="Gross profit / Revenue")
    ebitda_margin: float = Field(description="EBITDA / Revenue")
    operating_margin: float = Field(description="EBIT / Revenue")
    net_profit_margin: float = Field(description="Net Income / Revenue")
    roa: float = Field(description="Return on Assets: Net Income / Average Total Assets")
    roe: float = Field(description="Return on Equity: Net Income / Average Total Equity")
    roic: float = Field(description="Return on Invested Capital: NOPAT / Invested Capital")
    revenue_cagr_3yr: float = Field(description="3-year compound annual revenue growth rate")

    # 5-Way DuPont Decomposition
    tax_burden: float = Field(description="Net Income / EBT")
    interest_burden: float = Field(description="EBT / EBIT")
    ebit_margin: float = Field(description="EBIT / Revenue")
    asset_turnover: float = Field(description="Revenue / Average Total Assets")
    financial_leverage: float = Field(description="Average Total Assets / Average Total Equity")
    reconstructed_roe: float = Field(description="Product of the 5 DuPont terms")

    # Working Capital & Cash Conversion Cycle
    current_ratio: float = Field(description="Current Assets / Current Liabilities")
    quick_ratio: float = Field(description="(Cash + AR) / Current Liabilities")
    dso: float = Field(description="Days Sales Outstanding: (AR / Revenue) * 365")
    dio: float = Field(description="Days Inventory Outstanding: (Inventory / COGS) * 365")
    dpo: float = Field(description="Days Payables Outstanding: (Accounts Payable / COGS) * 365")
    cash_conversion_cycle: float = Field(description="Cash Conversion Cycle: DSO + DIO - DPO")

    # Solvency & Forensic Quality
    debt_to_equity: float = Field(description="Total Debt / Total Equity")
    net_debt_to_ebitda: float = Field(description="(Total Debt - Cash) / EBITDA")
    interest_coverage: float = Field(description="EBIT / Interest Expense")
    altman_z_score: float = Field(description="Altman Z-Score bankruptcy prediction indicator")
    altman_zone: str = Field(description="Altman Z-Score classification ('Safe', 'Grey', or 'Distress')")
    cfo_to_net_income_ratio: float = Field(description="Operating Cash Flow / Net Income (conversion quality)")
    accruals_to_assets_ratio: float = Field(description="Balance Sheet Accruals / Average Total Assets")
    receivables_vs_revenue_growth_spread: float = Field(description="AR YoY Growth Rate - Revenue YoY Growth Rate")
    accounting_red_flags: list[str] = Field(description="List of detected forensic warnings and accounting anomalies")
    common_size_income_summary: dict[str, float] = Field(description="Common-size income statement normalized to Revenue")


class QuantRiskSummary(BaseModel):
    """Quantitative risk metrics derived from benchmark linear regression and return distributions."""
    model_config = ConfigDict(extra="forbid")

    ols_beta: float = Field(description="Ordinary Least Squares slope vs. benchmark return")
    adjusted_beta: float = Field(description="Blume adjusted beta: (2/3) * ols_beta + (1/3) * 1.0")
    alpha_annualized: float = Field(description="Annualized Jensen's alpha intercept in decimal")
    r_squared: float = Field(description="Coefficient of determination for benchmark regression")
    annualized_volatility: float = Field(description="Sample standard deviation scaled to 252 annual trading days")
    downside_deviation: float = Field(description="Annualized downside volatility below zero target threshold")
    sharpe_ratio: float = Field(description="Annualized Sharpe ratio: (Annual Return - Rf) / Annual Volatility")
    sortino_ratio: float = Field(description="Annualized Sortino ratio: (Annual Return - Rf) / Downside Deviation")
    max_drawdown: float = Field(description="Peak-to-trough historical maximum drawdown in decimal")
    var_95_daily_historical: float = Field(description="95% confidence 1-day Historical Value-at-Risk")
    var_95_daily_parametric: float = Field(description="95% confidence 1-day Parametric (Gaussian) Value-at-Risk")
    cvar_95_daily: float = Field(description="95% confidence 1-day Conditional VaR / Expected Shortfall")


class ScenarioDriverSet(BaseModel):
    """Financial driver inputs for a single 5-year discrete DCF forecast scenario."""
    model_config = ConfigDict(extra="forbid")

    scenario_name: str = Field(description="Descriptive scenario name ('bear', 'base', or 'bull')")
    probability_weight: float = Field(ge=0.05, le=0.90, description="Assigned scenario probability weight")
    revenue_cagr_5yr: list[float] = Field(description="Yearly projected revenue growth rates for Years 1 through 5")
    target_ebit_margin_yr5: float = Field(ge=-0.30, le=0.70, description="Terminal Year 5 operating margin")
    effective_tax_rate: float = Field(ge=0.05, le=0.40, description="Corporate effective income tax rate")
    da_pct_revenue: float = Field(ge=0.005, le=0.20, description="Depreciation & Amortization as a fraction of revenue")
    capex_pct_revenue: float = Field(ge=0.005, le=0.35, description="CapEx as a fraction of revenue")
    nwc_pct_revenue_change: float = Field(ge=-0.10, le=0.20, description="Change in non-cash working capital as fraction of revenue growth")
    terminal_growth_rate: float = Field(ge=-0.01, le=0.065, description="Perpetual terminal growth rate g")
    wacc_adjustment_bps: float = Field(ge=-250.0, le=300.0, description="Basis point discount rate adjustment for scenario risk")
    rationale: str = Field(description="Qualitative justification and filing evidence support for scenario inputs")

    @field_validator("revenue_cagr_5yr")
    @classmethod
    def validate_5yr_projection_length(cls, v: list[float]) -> list[float]:
        if len(v) != 5:
            raise ValueError(f"revenue_cagr_5yr must contain exactly 5 discrete annual growth rates, got {len(v)}")
        return v


class DCFScenarioAssumptions(BaseModel):
    """Tripartite scenario drivers (Bear, Base, Bull) produced by agentic debate and validated by rules."""
    model_config = ConfigDict(extra="forbid")

    bear: ScenarioDriverSet = Field(description="Downside / Bear case parameters")
    base: ScenarioDriverSet = Field(description="Most likely / Base case parameters")
    bull: ScenarioDriverSet = Field(description="Upside / Bull case parameters")
    bull_thesis_points: list[str] = Field(description="Core qualitative upside catalysts cited from disclosures")
    bear_thesis_points: list[str] = Field(description="Core qualitative downside vulnerabilities and risks")
    key_catalysts: list[str] = Field(description="Key near-term operational and industry catalysts")
    key_risks: list[str] = Field(description="Critical headwinds, competitive pressures, and macro risks")
    debate_synthesis_summary: str = Field(description="Executive summary reconciling Bull vs Bear dialectical debate")

    @model_validator(mode="after")
    def validate_scenario_invariants(self) -> "DCFScenarioAssumptions":
        # 1. Probability weight conservation: weights must sum to 1.0 within ±0.01 tolerance
        total_prob = self.bear.probability_weight + self.base.probability_weight + self.bull.probability_weight
        if abs(total_prob - 1.0) > 0.01:
            raise ValueError(f"Scenario probability weights must sum to 1.0 ± 0.01, got {total_prob:.4f}")

        # 2. Strict growth ordering: Bull average 5yr CAGR >= Base average 5yr CAGR >= Bear average 5yr CAGR
        bear_avg_cagr = sum(self.bear.revenue_cagr_5yr) / len(self.bear.revenue_cagr_5yr)
        base_avg_cagr = sum(self.base.revenue_cagr_5yr) / len(self.base.revenue_cagr_5yr)
        bull_avg_cagr = sum(self.bull.revenue_cagr_5yr) / len(self.bull.revenue_cagr_5yr)

        if bull_avg_cagr < base_avg_cagr - 1e-6:
            raise ValueError(
                f"Bull average revenue CAGR ({bull_avg_cagr:.4f}) must be >= Base ({base_avg_cagr:.4f})"
            )
        if base_avg_cagr < bear_avg_cagr - 1e-6:
            raise ValueError(
                f"Base average revenue CAGR ({base_avg_cagr:.4f}) must be >= Bear ({bear_avg_cagr:.4f})"
            )

        # 3. Strict Year-5 operating margin ordering: Bull >= Base >= Bear
        if self.bull.target_ebit_margin_yr5 < self.base.target_ebit_margin_yr5 - 1e-6:
            raise ValueError(
                f"Bull Year-5 margin ({self.bull.target_ebit_margin_yr5:.4f}) must be >= Base ({self.base.target_ebit_margin_yr5:.4f})"
            )
        if self.base.target_ebit_margin_yr5 < self.bear.target_ebit_margin_yr5 - 1e-6:
            raise ValueError(
                f"Base Year-5 margin ({self.base.target_ebit_margin_yr5:.4f}) must be >= Bear ({self.bear.target_ebit_margin_yr5:.4f})"
            )

        return self


class SingleDCFResult(BaseModel):
    """Deterministic 5-year Free Cash Flow to Firm (FCFF) discounted cash flow valuation result."""
    model_config = ConfigDict(extra="forbid")

    scenario_name: str = Field(description="Scenario label ('bear', 'base', or 'bull')")
    wacc: float = Field(description="Weighted Average Cost of Capital applied for discounting")
    terminal_growth_rate: float = Field(description="Perpetual growth rate applied to Year 5 FCFF")
    projected_revenues: list[float] = Field(description="5-year projected revenue path")
    projected_ebit: list[float] = Field(description="5-year projected EBIT path")
    projected_fcff: list[float] = Field(description="5-year projected FCFF cash flows")
    pv_of_fcff: float = Field(description="Cumulative present value of 5-year discrete cash flows")
    terminal_value: float = Field(description="Undiscounted terminal value at Year 5")
    pv_of_terminal_value: float = Field(description="Present value of terminal value discounted at WACC")
    enterprise_value: float = Field(description="Total Enterprise Value: PV(FCFF) + PV(Terminal Value)")
    equity_value: float = Field(description="Implied Equity Value: Enterprise Value + Cash - Total Debt")
    intrinsic_value_per_share: float = Field(description="Implied intrinsic stock price per share")
    upside_pct: float = Field(description="Percentage upside/downside vs. current market price")

    @model_validator(mode="after")
    def validate_terminal_growth_vs_wacc(self) -> "SingleDCFResult":
        if self.terminal_growth_rate >= self.wacc:
            raise ValueError(
                f"Terminal growth rate ({self.terminal_growth_rate:.4f}) must be strictly less than WACC ({self.wacc:.4f})"
            )
        return self


class MonteCarloDCFOutput(BaseModel):
    """Vectorized 10,000-path Monte Carlo simulation output capturing intrinsic value distributions."""
    model_config = ConfigDict(extra="forbid")

    num_simulations: int = Field(description="Total number of stochastic paths evaluated (e.g. 10000)")
    mean_intrinsic_value: float = Field(description="Sample mean intrinsic price per share")
    median_intrinsic_value: float = Field(description="Sample median intrinsic price per share")
    std_dev: float = Field(description="Standard deviation of simulated intrinsic prices")
    p5: float = Field(description="5th percentile value (95% Value-at-Risk lower bound)")
    p25: float = Field(description="25th percentile value (first quartile)")
    p50: float = Field(description="50th percentile value (median)")
    p75: float = Field(description="75th percentile value (third quartile)")
    p95: float = Field(description="95th percentile value (upper bound)")
    probability_of_upside: float = Field(description="Fraction of simulated paths exceeding current market price")
    simulated_values_sample: list[float] = Field(description="Subsample of simulated values for histogram visualization")

    @model_validator(mode="after")
    def validate_percentiles_order(self) -> "MonteCarloDCFOutput":
        if not (self.p5 <= self.p25 <= self.p50 <= self.p75 <= self.p95):
            raise ValueError(
                f"Percentiles must be monotonically ordered: P5({self.p5:.2f}) <= P25({self.p25:.2f}) <= "
                f"P50({self.p50:.2f}) <= P75({self.p75:.2f}) <= P95({self.p95:.2f})"
            )
        return self


class PeerCompMultiple(BaseModel):
    """Comparable company valuation multiple for relative trading benchmark analysis."""
    model_config = ConfigDict(extra="forbid")

    peer_ticker: str = Field(description="Peer company ticker symbol")
    peer_name: str = Field(description="Peer company common name")
    pe_ratio: float = Field(description="Price to Earnings (P/E) multiple")
    ev_to_ebitda: float = Field(description="Enterprise Value to EBITDA multiple")
    pb_ratio: float = Field(description="Price to Book Value (P/BV) multiple")
    ev_to_sales: float = Field(description="Enterprise Value to Sales multiple")
    roe: float = Field(description="Return on Equity")
    revenue_growth: float = Field(description="YoY revenue growth rate")


class ValuationSuiteOutput(BaseModel):
    """Consolidated institutional equity research valuation suite output across multiple methodologies."""
    model_config = ConfigDict(extra="forbid")

    cost_of_equity: float = Field(description="CAPM Cost of Equity: Rf + Beta * ERP")
    pre_tax_cost_of_debt: float = Field(description="Pre-tax effective cost of debt (Interest Expense / Debt)")
    after_tax_cost_of_debt: float = Field(description="After-tax cost of debt: Rd * (1 - Tax Rate)")
    base_wacc: float = Field(description="Base Weighted Average Cost of Capital")

    bear_dcf: SingleDCFResult = Field(description="Downside DCF result")
    base_dcf: SingleDCFResult = Field(description="Base case DCF result")
    bull_dcf: SingleDCFResult = Field(description="Upside DCF result")

    probability_weighted_intrinsic_value: float = Field(description="Sum of scenario values weighted by probability")
    implied_upside_pct: float = Field(description="Weighted intrinsic value upside vs. current price in decimal")
    recommendation: Literal["BUY", "HOLD", "SELL"] = Field(description="Initiating coverage recommendation")

    reverse_dcf_implied_revenue_cagr: float = Field(description="SciPy brentq solved 5-yr revenue CAGR matching current price")
    market_expectations_gap_bps: float = Field(description="Difference between Base CAGR and Reverse-DCF CAGR in basis points")

    wacc_sensitivity_axes: list[float] = Field(description="5 discrete WACC points for sensitivity matrix")
    tgr_sensitivity_axes: list[float] = Field(description="5 discrete Terminal Growth Rate points for sensitivity matrix")
    sensitivity_matrix_prices: list[list[float]] = Field(description="5x5 intrinsic price matrix across WACC and g")

    peer_comps: list[PeerCompMultiple] = Field(description="Set of comparable peer trading multiples")
    peer_median_implied_price: float = Field(description="Relative valuation implied price derived from peer median multiples")
    football_field_ranges: dict[str, tuple[float, float, float]] = Field(
        description="Min, Mid, Max valuation ranges across methods: 52W Range, DCF Scenarios, Comps, Reverse-DCF"
    )


class StepTelemetry(BaseModel):
    """Granular execution telemetry tracing latency, mode, and contract verification status."""
    model_config = ConfigDict(extra="forbid")

    step_name: str = Field(description="Unique pipeline phase identifier")
    agent_or_engine: str = Field(description="Executing module or agent")
    mode: str = Field(description="Execution mode ('deterministic' or 'llm')")
    latency_ms: float = Field(description="Execution runtime in milliseconds")
    status: str = Field(description="Execution status ('SUCCESS', 'FAILED', 'REPAIRED')")
    validation_passed: bool = Field(description="Whether Pydantic contract validation succeeded")
    details: str = Field(description="Diagnostic context or summary description")


class EvalScorecard(BaseModel):
    """3-Layer automated evaluation harness scorecard evaluating reliability and fidelity."""
    model_config = ConfigDict(extra="forbid")

    math_invariance_score: float = Field(description="Score (0-100) verifying pure deterministic math invariance")
    schema_completeness_score: float = Field(description="Score (0-100) verifying presence of required fields and sections")
    numerical_faithfulness_score: float = Field(description="Score (0-100) verifying zero-hallucination citation against engine outputs")
    overall_eval_score: float = Field(description="Composite evaluation score (0-100)")
    passed: bool = Field(description="True if all 3 layers pass minimum acceptable thresholds")
    repair_attempts: int = Field(description="Number of auto-repair cycles executed by the reflection gate")
    flagged_discrepancies: list[str] = Field(description="Itemized discrepancies, math violations, or hallucinations flagged")
