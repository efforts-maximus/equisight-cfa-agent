"""Deterministic Financial Statement Analysis (FSA) Engine strictly adhering to 2027 CFA Level 1.

Computes 5-Way DuPont ROE Decomposition, Working Capital Cash Conversion Cycle,
Altman Z-Score Solvency Analysis, and Forensic Earnings Quality Red Flags.
"""

from src.schemas.models import CompanyFinancialSnapshot, FSARatioSummary


def compute_fsa_summary(snapshot: CompanyFinancialSnapshot) -> FSARatioSummary:
    """Computes comprehensive FSA ratios from multi-year audited financial records.
    
    Adheres strictly to CFA Level 1 formulas with mathematical invariance:
    - 5-Way DuPont identity: Tax Burden * Interest Burden * EBIT Margin * Asset Turnover * Leverage == ROE
    - Working Capital Cash Conversion Cycle: DSO + DIO - DPO
    - Altman Z-Score: 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    - Forensic flags: CFO/NI < 0.80, Accruals/Assets > 0.10, AR growth - Rev growth > 0.10
    """
    if not snapshot.historical_years:
        raise ValueError("Cannot compute FSA summary: snapshot.historical_years is empty.")

    latest = snapshot.historical_years[-1]
    prior = snapshot.historical_years[-2] if len(snapshot.historical_years) >= 2 else latest

    # 1. Multi-Year Balance Sheet Averages (CFA Level 1 standard: average balance sheet denominator)
    avg_assets = (latest.total_assets + prior.total_assets) / 2.0
    avg_equity = (latest.total_equity + prior.total_equity) / 2.0
    avg_ar = (latest.accounts_receivable + prior.accounts_receivable) / 2.0
    avg_inventory = (latest.inventory + prior.inventory) / 2.0
    avg_ap = (latest.accounts_payable + prior.accounts_payable) / 2.0

    # Ensure no division by zero for balance sheet denominators
    safe_avg_assets = avg_assets if avg_assets != 0 else 1.0
    safe_avg_equity = avg_equity if avg_equity != 0 else 1.0
    safe_revenue = latest.revenue if latest.revenue != 0 else 1.0
    safe_cogs = latest.cogs if latest.cogs > 0 else 1.0
    safe_assets = latest.total_assets if latest.total_assets > 0 else 1.0
    safe_cur_liab = latest.current_liabilities if latest.current_liabilities > 0 else 1.0

    # 2. Core Profitability & Growth Ratios
    gross_margin = latest.gross_profit / safe_revenue
    ebitda_margin = latest.ebitda / safe_revenue
    operating_margin = latest.ebit / safe_revenue
    net_profit_margin = latest.net_income / safe_revenue
    roa = latest.net_income / safe_avg_assets
    roe = latest.net_income / safe_avg_equity

    # NOPAT & ROIC
    # Effective tax rate clamped to [0.05, 0.40]
    if latest.ebt > 0:
        raw_tax_rate = latest.tax_expense / latest.ebt
        effective_tax_rate = min(max(raw_tax_rate, 0.05), 0.40)
    else:
        effective_tax_rate = 0.25
    nopat = latest.ebit * (1.0 - effective_tax_rate)
    invested_capital = latest.total_equity + latest.total_debt - latest.cash_and_equivalents
    roic = nopat / invested_capital if invested_capital > 0 else 0.0

    # 3-Year Revenue CAGR (guard against negative revenue fractional exponentiation resulting in complex numbers)
    if len(snapshot.historical_years) >= 4:
        first_rev = snapshot.historical_years[-4].revenue
        revenue_cagr_3yr = (
            (latest.revenue / first_rev) ** (1.0 / 3.0) - 1.0
            if first_rev > 0 and latest.revenue > 0
            else 0.0
        )
    elif len(snapshot.historical_years) >= 2:
        n_periods = len(snapshot.historical_years) - 1
        first_rev = snapshot.historical_years[0].revenue
        revenue_cagr_3yr = (
            (latest.revenue / first_rev) ** (1.0 / n_periods) - 1.0
            if first_rev > 0 and latest.revenue > 0
            else 0.0
        )
    else:
        revenue_cagr_3yr = 0.0

    # 3. 5-Way DuPont ROE Decomposition
    # Terms:
    # Tax Burden = NI / EBT
    # Interest Burden = EBT / EBIT
    # EBIT Margin = EBIT / Revenue
    # Asset Turnover = Revenue / Avg Assets
    # Financial Leverage = Avg Assets / Avg Equity
    # Product: (NI/EBT) * (EBT/EBIT) * (EBIT/Rev) * (Rev/AvgAssets) * (AvgAssets/AvgEquity) == NI / AvgEquity == ROE
    # Symmetrically cancel safe denominators/numerators to maintain strict mathematical invariance
    safe_ebt = latest.ebt if latest.ebt != 0 else 1.0
    safe_ebit = latest.ebit if latest.ebit != 0 else 1.0

    tax_burden = latest.net_income / safe_ebt
    interest_burden = safe_ebt / safe_ebit
    ebit_margin = safe_ebit / safe_revenue
    asset_turnover = safe_revenue / safe_avg_assets
    financial_leverage = safe_avg_assets / safe_avg_equity

    reconstructed_roe = tax_burden * interest_burden * ebit_margin * asset_turnover * financial_leverage

    # 4. Working Capital & Cash Conversion Cycle (CCC)
    current_ratio = latest.current_assets / safe_cur_liab
    quick_ratio = (latest.cash_and_equivalents + latest.accounts_receivable) / safe_cur_liab
    dso = (avg_ar / safe_revenue) * 365.0
    dio = (avg_inventory / safe_cogs) * 365.0 if latest.cogs > 0 else 0.0
    dpo = (avg_ap / safe_cogs) * 365.0 if latest.cogs > 0 else 0.0
    cash_conversion_cycle = dso + dio - dpo

    # 5. Solvency Ratios & Altman Z-Score
    safe_equity = latest.total_equity if latest.total_equity != 0 else 1.0
    debt_to_equity = latest.total_debt / safe_equity
    net_debt_to_ebitda = (latest.total_debt - latest.cash_and_equivalents) / latest.ebitda if latest.ebitda != 0 else 0.0
    interest_coverage = latest.ebit / latest.interest_expense if latest.interest_expense > 0 else 999.0

    # Altman Z-Score for public manufacturing / corporate entities:
    # X1 = (Current Assets - Current Liabilities) / Total Assets
    # X2 = Retained Earnings / Total Assets
    # X3 = EBIT / Total Assets
    # X4 = Market Value of Equity / Total Liabilities
    # X5 = Revenue / Total Assets
    # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    safe_liabilities = latest.total_liabilities if latest.total_liabilities > 0 else 1.0
    x1 = (latest.current_assets - latest.current_liabilities) / safe_assets
    x2 = latest.retained_earnings / safe_assets
    x3 = latest.ebit / safe_assets
    x4 = snapshot.market_cap / safe_liabilities
    x5 = latest.revenue / safe_assets
    altman_z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if altman_z_score > 2.99:
        altman_zone = "Safe"
    elif altman_z_score >= 1.81:
        altman_zone = "Grey"
    else:
        altman_zone = "Distress"

    # 6. Forensic Earnings Quality & Accounting Red Flags
    # Ratio 1: Operating Cash Flow / Net Income (quality of earnings)
    # Per approved plan: if NI > 0, CFO/NI; if NI <= 0 and CFO < 0, flag cash burn
    if latest.net_income > 0:
        cfo_to_net_income_ratio = latest.operating_cash_flow / latest.net_income
    else:
        cfo_to_net_income_ratio = 0.0 if latest.operating_cash_flow < 0 else 1.0

    # Ratio 2: Balance Sheet Accruals / Avg Total Assets
    # Accruals = Net Income - Operating Cash Flow
    accruals = latest.net_income - latest.operating_cash_flow
    accruals_to_assets_ratio = accruals / safe_avg_assets

    # Ratio 3: Receivables Growth vs Revenue Growth Spread (YoY)
    if len(snapshot.historical_years) >= 2 and prior.accounts_receivable > 0 and prior.revenue > 0:
        ar_growth = (latest.accounts_receivable - prior.accounts_receivable) / prior.accounts_receivable
        rev_growth = (latest.revenue - prior.revenue) / prior.revenue
        receivables_vs_revenue_growth_spread = ar_growth - rev_growth
    else:
        receivables_vs_revenue_growth_spread = 0.0

    accounting_red_flags: list[str] = []
    if latest.net_income > 0 and cfo_to_net_income_ratio < 0.80:
        accounting_red_flags.append(
            f"RED FLAG: CFO to Net Income conversion weak ({cfo_to_net_income_ratio:.2f} < 0.80), "
            "indicating poor cash generation quality."
        )
    elif latest.net_income <= 0 and latest.operating_cash_flow < 0:
        accounting_red_flags.append(
            f"RED FLAG: Negative operating cash flow ({latest.operating_cash_flow:.2f}), "
            "company is burning cash from operations."
        )
    if accruals_to_assets_ratio > 0.10:
        accounting_red_flags.append(
            f"RED FLAG: Accruals exceed 10% of total assets ({accruals_to_assets_ratio * 100:.1f}%), "
            "signaling aggressive capitalization."
        )
    if receivables_vs_revenue_growth_spread > 0.10:
        accounting_red_flags.append(
            f"RED FLAG: Accounts receivable growth exceeds revenue growth by > 10% "
            f"({receivables_vs_revenue_growth_spread * 100:.1f}% spread), "
            "indicating possible revenue pull-forward or channel stuffing."
        )

    # Common-size income statement normalized to Revenue
    common_size_income_summary: dict[str, float] = {
        "revenue": 1.0,
        "cogs": round(latest.cogs / safe_revenue, 4),
        "gross_profit": round(latest.gross_profit / safe_revenue, 4),
        "ebitda": round(latest.ebitda / safe_revenue, 4),
        "depreciation_amortization": round(latest.depreciation_amortization / safe_revenue, 4),
        "ebit": round(latest.ebit / safe_revenue, 4),
        "interest_expense": round(latest.interest_expense / safe_revenue, 4),
        "ebt": round(latest.ebt / safe_revenue, 4),
        "tax_expense": round(latest.tax_expense / safe_revenue, 4),
        "net_income": round(latest.net_income / safe_revenue, 4),
    }

    return FSARatioSummary(
        gross_margin=gross_margin,
        ebitda_margin=ebitda_margin,
        operating_margin=operating_margin,
        net_profit_margin=net_profit_margin,
        roa=roa,
        roe=roe,
        roic=roic,
        revenue_cagr_3yr=revenue_cagr_3yr,
        tax_burden=tax_burden,
        interest_burden=interest_burden,
        ebit_margin=ebit_margin,
        asset_turnover=asset_turnover,
        financial_leverage=financial_leverage,
        reconstructed_roe=reconstructed_roe,
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        dso=dso,
        dio=dio,
        dpo=dpo,
        cash_conversion_cycle=cash_conversion_cycle,
        debt_to_equity=debt_to_equity,
        net_debt_to_ebitda=net_debt_to_ebitda,
        interest_coverage=interest_coverage,
        altman_z_score=altman_z_score,
        altman_zone=altman_zone,
        cfo_to_net_income_ratio=cfo_to_net_income_ratio,
        accruals_to_assets_ratio=accruals_to_assets_ratio,
        receivables_vs_revenue_growth_spread=receivables_vs_revenue_growth_spread,
        accounting_red_flags=accounting_red_flags,
        common_size_income_summary=common_size_income_summary,
    )
