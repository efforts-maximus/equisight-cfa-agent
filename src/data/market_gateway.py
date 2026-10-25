"""Dual-Mode Market Data Gateway for EquiSight CFA Agent.

Provides institutional-grade 4-year financial statement fixtures, 252-day synthetic correlated
return series, comparable peer multiples, and live yfinance integration with local JSON caching.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional
import numpy as np

from src.schemas.models import (
    CompanyFinancialSnapshot,
    HistoricalYearMetrics,
    PeerCompMultiple,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


def generate_correlated_returns(
    seed: int,
    target_beta: float = 1.0,
    annual_vol: float = 0.20,
    benchmark_mean_annual: float = 0.10,
    benchmark_vol_annual: float = 0.16,
    num_days: int = 252,
) -> tuple[list[float], list[float]]:
    """Generates 252 synthetic daily returns for a stock and benchmark index with a target OLS Beta.
    
    Formula:
        R_bench ~ N(mean_daily, var_bench)
        R_stock = (alpha / 252) + beta * R_bench + epsilon
        where epsilon ~ N(0, var_residual)
    """
    rng = np.random.default_rng(seed)

    bench_daily_mean = benchmark_mean_annual / 252.0
    bench_daily_vol = benchmark_vol_annual / math.sqrt(252.0)
    bench_returns = rng.normal(loc=bench_daily_mean, scale=bench_daily_vol, size=num_days)

    stock_total_daily_vol = annual_vol / math.sqrt(252.0)
    beta_explained_var = (target_beta ** 2) * (bench_daily_vol ** 2)

    # Ensure residual variance is strictly positive
    if stock_total_daily_vol ** 2 > beta_explained_var:
        residual_daily_vol = math.sqrt(stock_total_daily_vol ** 2 - beta_explained_var)
    else:
        residual_daily_vol = 0.005

    alpha_daily = 0.02 / 252.0  # slight positive Jensen's alpha
    residual_noise = rng.normal(loc=0.0, scale=residual_daily_vol, size=num_days)
    stock_returns = alpha_daily + target_beta * bench_returns + residual_noise

    return [round(float(r), 6) for r in stock_returns], [round(float(r), 6) for r in bench_returns]


# ==============================================================================
# 4-YEAR INSTITUTIONAL OFFLINE FIXTURES (2023 - 2026)
# ==============================================================================

OFFLINE_FIXTURES: dict[str, dict[str, Any]] = {
    "TCS.NS": {
        "ticker": "TCS.NS",
        "company_name": "Tata Consultancy Services Limited",
        "sector": "Technology",
        "industry": "IT Services",
        "currency": "INR",
        "unit_label": "INR Crores",
        "current_price": 3950.0,
        "shares_outstanding": 205.0,  # in Crores of shares -> 205.0 Cr * 3950 = 809,750 Cr Market Cap
        "market_cap": 809750.0,
        "enterprise_value": 801300.0,  # Market Cap + Debt(8350) - Cash(16800)
        "high_52week": 4592.0,
        "low_52week": 3400.0,
        "benchmark_ticker": "^NSEI",
        "risk_free_rate": 0.068,
        "equity_risk_premium": 0.060,
        "business_summary": (
            "Tata Consultancy Services is a global leader in IT services, digital and business solutions, "
            "operating across BFSI, Consumer Business, Communication, and Manufacturing across 55 countries."
        ),
        "target_beta": 0.85,
        "seed": 101,
        "historical_years": [
            {
                "fiscal_year": 2023,
                "revenue": 225458.0,
                "cogs": 125800.0,
                "gross_profit": 99658.0,
                "ebitda": 59258.0,
                "depreciation_amortization": 5022.0,
                "ebit": 54236.0,
                "interest_expense": 779.0,
                "ebt": 53457.0,
                "tax_expense": 11354.0,
                "net_income": 42103.0,
                "total_assets": 143828.0,
                "current_assets": 95400.0,
                "cash_and_equivalents": 11032.0,
                "accounts_receivable": 41810.0,
                "inventory": 0.0,
                "total_liabilities": 53406.0,
                "current_liabilities": 45200.0,
                "accounts_payable": 10450.0,
                "total_debt": 7714.0,
                "total_equity": 90422.0,
                "retained_earnings": 89100.0,
                "operating_cash_flow": 41961.0,
                "capex": 3150.0,
                "working_capital_change": -1200.0,
            },
            {
                "fiscal_year": 2024,
                "revenue": 240893.0,
                "cogs": 133400.0,
                "gross_profit": 107493.0,
                "ebitda": 64150.0,
                "depreciation_amortization": 4984.0,
                "ebit": 59166.0,
                "interest_expense": 778.0,
                "ebt": 58388.0,
                "tax_expense": 12450.0,
                "net_income": 45938.0,
                "total_assets": 147683.0,
                "current_assets": 98200.0,
                "cash_and_equivalents": 12480.0,
                "accounts_receivable": 44250.0,
                "inventory": 0.0,
                "total_liabilities": 57520.0,
                "current_liabilities": 48900.0,
                "accounts_payable": 11200.0,
                "total_debt": 7900.0,
                "total_equity": 90163.0,
                "retained_earnings": 88700.0,
                "operating_cash_flow": 44340.0,
                "capex": 3320.0,
                "working_capital_change": -850.0,
            },
            {
                "fiscal_year": 2025,
                "revenue": 256800.0,
                "cogs": 141240.0,
                "gross_profit": 115560.0,
                "ebitda": 68820.0,
                "depreciation_amortization": 5200.0,
                "ebit": 63620.0,
                "interest_expense": 810.0,
                "ebt": 62810.0,
                "tax_expense": 13818.0,
                "net_income": 48992.0,
                "total_assets": 154200.0,
                "current_assets": 102500.0,
                "cash_and_equivalents": 14500.0,
                "accounts_receivable": 46800.0,
                "inventory": 0.0,
                "total_liabilities": 61100.0,
                "current_liabilities": 51800.0,
                "accounts_payable": 11800.0,
                "total_debt": 8150.0,
                "total_equity": 93100.0,
                "retained_earnings": 91500.0,
                "operating_cash_flow": 47200.0,
                "capex": 3500.0,
                "working_capital_change": -600.0,
            },
            {
                "fiscal_year": 2026,
                "revenue": 274800.0,
                "cogs": 150040.0,
                "gross_profit": 124760.0,
                "ebitda": 74200.0,
                "depreciation_amortization": 5450.0,
                "ebit": 68750.0,
                "interest_expense": 840.0,
                "ebt": 67910.0,
                "tax_expense": 14940.0,
                "net_income": 52970.0,
                "total_assets": 162500.0,
                "current_assets": 108200.0,
                "cash_and_equivalents": 16800.0,
                "accounts_receivable": 49500.0,
                "inventory": 0.0,
                "total_liabilities": 64800.0,
                "current_liabilities": 54900.0,
                "accounts_payable": 12500.0,
                "total_debt": 8350.0,
                "total_equity": 97700.0,
                "retained_earnings": 95900.0,
                "operating_cash_flow": 51400.0,
                "capex": 3750.0,
                "working_capital_change": -400.0,
            },
        ],
    },
    "ZOMATO.NS": {
        "ticker": "ZOMATO.NS",
        "company_name": "Zomato Limited",
        "sector": "Consumer Cyclical",
        "industry": "Internet Marketplace / Food Delivery",
        "currency": "INR",
        "unit_label": "INR Crores",
        "current_price": 260.0,
        "shares_outstanding": 882.0,  # in Crores -> 882 * 260 = 229,320 Cr Market Cap
        "market_cap": 229320.0,
        "enterprise_value": 214900.0,  # 229320 + Debt(180) - Cash(14600)
        "high_52week": 298.0,
        "low_52week": 105.0,
        "benchmark_ticker": "^NSEI",
        "risk_free_rate": 0.068,
        "equity_risk_premium": 0.060,
        "business_summary": (
            "Zomato connects consumers, delivery partners, and restaurant partners across India. "
            "It operates food ordering, quick commerce (Blinkit), and farm-to-fork supplies (Hyperpure)."
        ),
        "target_beta": 1.30,
        "seed": 202,
        "historical_years": [
            {
                "fiscal_year": 2023,
                "revenue": 7079.0,
                "cogs": 4050.0,
                "gross_profit": 3029.0,
                "ebitda": -1210.0,
                "depreciation_amortization": 440.0,
                "ebit": -1650.0,
                "interest_expense": 50.0,
                "ebt": -1700.0,
                "tax_expense": 0.0,
                "net_income": -1700.0,
                "total_assets": 21800.0,
                "current_assets": 12200.0,
                "cash_and_equivalents": 9800.0,
                "accounts_receivable": 350.0,
                "inventory": 120.0,
                "total_liabilities": 2750.0,
                "current_liabilities": 2400.0,
                "accounts_payable": 850.0,
                "total_debt": 120.0,
                "total_equity": 19050.0,
                "retained_earnings": -8500.0,
                "operating_cash_flow": -520.0,
                "capex": 220.0,
                "working_capital_change": 180.0,
            },
            {
                "fiscal_year": 2024,
                "revenue": 12114.0,
                "cogs": 6600.0,
                "gross_profit": 5514.0,
                "ebitda": 770.0,
                "depreciation_amortization": 520.0,
                "ebit": 250.0,
                "interest_expense": 62.0,
                "ebt": 188.0,
                "tax_expense": 37.0,
                "net_income": 151.0,
                "total_assets": 23600.0,
                "current_assets": 13900.0,
                "cash_and_equivalents": 11200.0,
                "accounts_receivable": 520.0,
                "inventory": 240.0,
                "total_liabilities": 3800.0,
                "current_liabilities": 3200.0,
                "accounts_payable": 1450.0,
                "total_debt": 140.0,
                "total_equity": 19800.0,
                "retained_earnings": -8349.0,
                "operating_cash_flow": 650.0,
                "capex": 380.0,
                "working_capital_change": 320.0,
            },
            {
                "fiscal_year": 2025,
                "revenue": 18200.0,
                "cogs": 9700.0,
                "gross_profit": 8500.0,
                "ebitda": 1720.0,
                "depreciation_amortization": 640.0,
                "ebit": 1080.0,
                "interest_expense": 75.0,
                "ebt": 1005.0,
                "tax_expense": 225.0,
                "net_income": 780.0,
                "total_assets": 26800.0,
                "current_assets": 16100.0,
                "cash_and_equivalents": 12800.0,
                "accounts_receivable": 780.0,
                "inventory": 380.0,
                "total_liabilities": 5100.0,
                "current_liabilities": 4300.0,
                "accounts_payable": 2100.0,
                "total_debt": 160.0,
                "total_equity": 21700.0,
                "retained_earnings": -7569.0,
                "operating_cash_flow": 1450.0,
                "capex": 520.0,
                "working_capital_change": 480.0,
            },
            {
                "fiscal_year": 2026,
                "revenue": 25400.0,
                "cogs": 13200.0,
                "gross_profit": 12200.0,
                "ebitda": 3100.0,
                "depreciation_amortization": 780.0,
                "ebit": 2320.0,
                "interest_expense": 85.0,
                "ebt": 2235.0,
                "tax_expense": 515.0,
                "net_income": 1720.0,
                "total_assets": 30500.0,
                "current_assets": 18900.0,
                "cash_and_equivalents": 14600.0,
                "accounts_receivable": 1100.0,
                "inventory": 560.0,
                "total_liabilities": 6600.0,
                "current_liabilities": 5500.0,
                "accounts_payable": 2900.0,
                "total_debt": 180.0,
                "total_equity": 23900.0,
                "retained_earnings": -5849.0,
                "operating_cash_flow": 2550.0,
                "capex": 680.0,
                "working_capital_change": 620.0,
            },
        ],
    },
    "VARUNBEV.NS": {
        "ticker": "VARUNBEV.NS",
        "company_name": "Varun Beverages Limited",
        "sector": "Consumer Defensive",
        "industry": "Beverages - Non-Alcoholic",
        "currency": "INR",
        "unit_label": "INR Crores",
        "current_price": 580.0,
        "shares_outstanding": 140.0,  # in Crores -> 140.0 * 580 = 81,200 Cr Market Cap
        "market_cap": 81200.0,
        "enterprise_value": 84900.0,  # 81200 + Debt(4900) - Cash(1200)
        "high_52week": 680.0,
        "low_52week": 410.0,
        "benchmark_ticker": "^NSEI",
        "risk_free_rate": 0.068,
        "equity_risk_premium": 0.060,
        "business_summary": (
            "Varun Beverages is one of the largest franchisees of PepsiCo in the world outside USA, "
            "producing and distributing carbonated soft drinks, juices, and packaged drinking water."
        ),
        "target_beta": 0.95,
        "seed": 303,
        "historical_years": [
            {
                "fiscal_year": 2023,
                "revenue": 16042.0,
                "cogs": 7420.0,
                "gross_profit": 8622.0,
                "ebitda": 3609.0,
                "depreciation_amortization": 681.0,
                "ebit": 2928.0,
                "interest_expense": 420.0,
                "ebt": 2508.0,
                "tax_expense": 406.0,
                "net_income": 2102.0,
                "total_assets": 14120.0,
                "current_assets": 3820.0,
                "cash_and_equivalents": 560.0,
                "accounts_receivable": 480.0,
                "inventory": 2150.0,
                "total_liabilities": 6920.0,
                "current_liabilities": 3450.0,
                "accounts_payable": 1120.0,
                "total_debt": 4280.0,
                "total_equity": 7200.0,
                "retained_earnings": 5850.0,
                "operating_cash_flow": 2850.0,
                "capex": 2100.0,
                "working_capital_change": -280.0,
            },
            {
                "fiscal_year": 2024,
                "revenue": 19800.0,
                "cogs": 9050.0,
                "gross_profit": 10750.0,
                "ebitda": 4520.0,
                "depreciation_amortization": 810.0,
                "ebit": 3710.0,
                "interest_expense": 480.0,
                "ebt": 3230.0,
                "tax_expense": 550.0,
                "net_income": 2680.0,
                "total_assets": 17200.0,
                "current_assets": 4550.0,
                "cash_and_equivalents": 780.0,
                "accounts_receivable": 560.0,
                "inventory": 2550.0,
                "total_liabilities": 7950.0,
                "current_liabilities": 4100.0,
                "accounts_payable": 1350.0,
                "total_debt": 4720.0,
                "total_equity": 9250.0,
                "retained_earnings": 7800.0,
                "operating_cash_flow": 3580.0,
                "capex": 2600.0,
                "working_capital_change": -340.0,
            },
            {
                "fiscal_year": 2025,
                "revenue": 23900.0,
                "cogs": 10850.0,
                "gross_profit": 13050.0,
                "ebitda": 5560.0,
                "depreciation_amortization": 960.0,
                "ebit": 4600.0,
                "interest_expense": 510.0,
                "ebt": 4090.0,
                "tax_expense": 720.0,
                "net_income": 3370.0,
                "total_assets": 20500.0,
                "current_assets": 5400.0,
                "cash_and_equivalents": 950.0,
                "accounts_receivable": 670.0,
                "inventory": 2980.0,
                "total_liabilities": 8800.0,
                "current_liabilities": 4700.0,
                "accounts_payable": 1580.0,
                "total_debt": 4950.0,
                "total_equity": 11700.0,
                "retained_earnings": 10100.0,
                "operating_cash_flow": 4420.0,
                "capex": 2900.0,
                "working_capital_change": -390.0,
            },
            {
                "fiscal_year": 2026,
                "revenue": 28600.0,
                "cogs": 12900.0,
                "gross_profit": 15700.0,
                "ebitda": 6780.0,
                "depreciation_amortization": 1120.0,
                "ebit": 5660.0,
                "interest_expense": 530.0,
                "ebt": 5130.0,
                "tax_expense": 910.0,
                "net_income": 4220.0,
                "total_assets": 24200.0,
                "current_assets": 6400.0,
                "cash_and_equivalents": 1200.0,
                "accounts_receivable": 790.0,
                "inventory": 3450.0,
                "total_liabilities": 9550.0,
                "current_liabilities": 5350.0,
                "accounts_payable": 1820.0,
                "total_debt": 4900.0,
                "total_equity": 14650.0,
                "retained_earnings": 12900.0,
                "operating_cash_flow": 5450.0,
                "capex": 3100.0,
                "working_capital_change": -420.0,
            },
        ],
    },
    "NVDA": {
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "sector": "Technology",
        "industry": "Semiconductors & AI Hardware",
        "currency": "USD",
        "unit_label": "USD Millions",
        "current_price": 135.0,
        "shares_outstanding": 24600.0,  # in Millions of shares (24.6B) -> 24600 * 135 = 3,321,000 M ($3.32T)
        "market_cap": 3321000.0,
        "enterprise_value": 3268300.0,  # 3321000 + Debt(9800) - Cash(62500)
        "high_52week": 140.76,
        "low_52week": 45.00,
        "benchmark_ticker": "^GSPC",
        "risk_free_rate": 0.042,
        "equity_risk_premium": 0.050,
        "business_summary": (
            "NVIDIA is the pioneer of GPU-accelerated computing. It designs full-stack systems and CUDA software "
            "powering hyperscale AI data centers, autonomous driving, robotics, and high-performance computing."
        ),
        "target_beta": 1.75,
        "seed": 404,
        "historical_years": [
            {
                "fiscal_year": 2023,
                "revenue": 26974.0,
                "cogs": 11618.0,
                "gross_profit": 15356.0,
                "ebitda": 6982.0,
                "depreciation_amortization": 1308.0,
                "ebit": 5674.0,
                "interest_expense": 262.0,
                "ebt": 5412.0,
                "tax_expense": 1044.0,
                "net_income": 4368.0,
                "total_assets": 41182.0,
                "current_assets": 23073.0,
                "cash_and_equivalents": 13296.0,
                "accounts_receivable": 3827.0,
                "inventory": 5159.0,
                "total_liabilities": 19081.0,
                "current_liabilities": 6560.0,
                "accounts_payable": 1193.0,
                "total_debt": 11831.0,
                "total_equity": 22101.0,
                "retained_earnings": 17203.0,
                "operating_cash_flow": 5641.0,
                "capex": 1833.0,
                "working_capital_change": -1150.0,
            },
            {
                "fiscal_year": 2024,
                "revenue": 60922.0,
                "cogs": 16621.0,
                "gross_profit": 44301.0,
                "ebitda": 34480.0,
                "depreciation_amortization": 1508.0,
                "ebit": 32972.0,
                "interest_expense": 257.0,
                "ebt": 32715.0,
                "tax_expense": 2955.0,
                "net_income": 29760.0,
                "total_assets": 65728.0,
                "current_assets": 44345.0,
                "cash_and_equivalents": 25984.0,
                "accounts_receivable": 9999.0,
                "inventory": 5282.0,
                "total_liabilities": 22750.0,
                "current_liabilities": 10631.0,
                "accounts_payable": 2699.0,
                "total_debt": 11056.0,
                "total_equity": 42978.0,
                "retained_earnings": 33783.0,
                "operating_cash_flow": 28090.0,
                "capex": 1221.0,
                "working_capital_change": 2450.0,
            },
            {
                "fiscal_year": 2025,
                "revenue": 125000.0,
                "cogs": 31250.0,
                "gross_profit": 93750.0,
                "ebitda": 78500.0,
                "depreciation_amortization": 2100.0,
                "ebit": 76400.0,
                "interest_expense": 280.0,
                "ebt": 76120.0,
                "tax_expense": 9134.0,
                "net_income": 66986.0,
                "total_assets": 115000.0,
                "current_assets": 78500.0,
                "cash_and_equivalents": 43200.0,
                "accounts_receivable": 18500.0,
                "inventory": 11200.0,
                "total_liabilities": 34000.0,
                "current_liabilities": 17500.0,
                "accounts_payable": 5400.0,
                "total_debt": 10500.0,
                "total_equity": 81000.0,
                "retained_earnings": 75500.0,
                "operating_cash_flow": 64200.0,
                "capex": 3800.0,
                "working_capital_change": 4100.0,
            },
            {
                "fiscal_year": 2026,
                "revenue": 165000.0,
                "cogs": 42900.0,
                "gross_profit": 122100.0,
                "ebitda": 104000.0,
                "depreciation_amortization": 2800.0,
                "ebit": 101200.0,
                "interest_expense": 310.0,
                "ebt": 100890.0,
                "tax_expense": 13116.0,
                "net_income": 87774.0,
                "total_assets": 162000.0,
                "current_assets": 112000.0,
                "cash_and_equivalents": 62500.0,
                "accounts_receivable": 24800.0,
                "inventory": 15400.0,
                "total_liabilities": 45000.0,
                "current_liabilities": 24200.0,
                "accounts_payable": 7200.0,
                "total_debt": 9800.0,
                "total_equity": 117000.0,
                "retained_earnings": 110200.0,
                "operating_cash_flow": 84500.0,
                "capex": 5200.0,
                "working_capital_change": 5300.0,
            },
        ],
    },
}

# ==============================================================================
# PEER MULTIPLES FIXTURES (3 PEERS PER TICKER)
# ==============================================================================

OFFLINE_PEER_FIXTURES: dict[str, list[dict[str, Any]]] = {
    "TCS.NS": [
        {
            "peer_ticker": "INFY.NS",
            "peer_name": "Infosys Limited",
            "pe_ratio": 18.0,
            "ev_to_ebitda": 12.5,
            "pb_ratio": 5.5,
            "ev_to_sales": 3.2,
            "roe": 0.32,
            "revenue_growth": 0.065,
        },
        {
            "peer_ticker": "WIPRO.NS",
            "peer_name": "Wipro Limited",
            "pe_ratio": 15.0,
            "ev_to_ebitda": 10.5,
            "pb_ratio": 3.0,
            "ev_to_sales": 2.0,
            "roe": 0.16,
            "revenue_growth": 0.025,
        },
        {
            "peer_ticker": "HCLTECH.NS",
            "peer_name": "HCL Technologies Limited",
            "pe_ratio": 17.0,
            "ev_to_ebitda": 11.8,
            "pb_ratio": 4.5,
            "ev_to_sales": 2.8,
            "roe": 0.23,
            "revenue_growth": 0.075,
        },
    ],
    "ZOMATO.NS": [
        {
            "peer_ticker": "SWIGGY.NS",
            "peer_name": "Swiggy Limited",
            "pe_ratio": 65.0,
            "ev_to_ebitda": 38.0,
            "pb_ratio": 8.5,
            "ev_to_sales": 5.2,
            "roe": 0.08,
            "revenue_growth": 0.34,
        },
        {
            "peer_ticker": "NAUKRI.NS",
            "peer_name": "Info Edge (India) Limited",
            "pe_ratio": 52.0,
            "ev_to_ebitda": 34.0,
            "pb_ratio": 6.2,
            "ev_to_sales": 12.0,
            "roe": 0.12,
            "revenue_growth": 0.18,
        },
        {
            "peer_ticker": "DELHIVERY.NS",
            "peer_name": "Delhivery Limited",
            "pe_ratio": 42.0,
            "ev_to_ebitda": 22.0,
            "pb_ratio": 3.1,
            "ev_to_sales": 2.8,
            "roe": 0.05,
            "revenue_growth": 0.15,
        },
    ],
    "VARUNBEV.NS": [
        {
            "peer_ticker": "DABUR.NS",
            "peer_name": "Dabur India Limited",
            "pe_ratio": 19.5,
            "ev_to_ebitda": 12.2,
            "pb_ratio": 5.2,
            "ev_to_sales": 3.5,
            "roe": 0.20,
            "revenue_growth": 0.09,
        },
        {
            "peer_ticker": "TATACONSUM.NS",
            "peer_name": "Tata Consumer Products Limited",
            "pe_ratio": 21.0,
            "ev_to_ebitda": 13.0,
            "pb_ratio": 4.2,
            "ev_to_sales": 3.0,
            "roe": 0.11,
            "revenue_growth": 0.11,
        },
        {
            "peer_ticker": "BRITANNIA.NS",
            "peer_name": "Britannia Industries Limited",
            "pe_ratio": 20.0,
            "ev_to_ebitda": 12.6,
            "pb_ratio": 7.8,
            "ev_to_sales": 3.8,
            "roe": 0.48,
            "revenue_growth": 0.10,
        },
    ],
    "NVDA": [
        {
            "peer_ticker": "AMD",
            "peer_name": "Advanced Micro Devices, Inc.",
            "pe_ratio": 45.0,
            "ev_to_ebitda": 32.0,
            "pb_ratio": 3.8,
            "ev_to_sales": 8.5,
            "roe": 0.09,
            "revenue_growth": 0.14,
        },
        {
            "peer_ticker": "AVGO",
            "peer_name": "Broadcom Inc.",
            "pe_ratio": 35.0,
            "ev_to_ebitda": 24.0,
            "pb_ratio": 8.2,
            "ev_to_sales": 14.0,
            "roe": 0.26,
            "revenue_growth": 0.42,
        },
        {
            "peer_ticker": "INTC",
            "peer_name": "Intel Corporation",
            "pe_ratio": 28.0,
            "ev_to_ebitda": 12.0,
            "pb_ratio": 1.1,
            "ev_to_sales": 1.8,
            "roe": 0.02,
            "revenue_growth": -0.02,
        },
    ],
}


def build_offline_snapshot(ticker: str) -> CompanyFinancialSnapshot:
    """Builds a validated CompanyFinancialSnapshot from deterministic offline fixtures."""
    ticker_clean = ticker.strip().upper()
    if ticker_clean not in OFFLINE_FIXTURES:
        raise KeyError(f"Ticker '{ticker}' not found in offline fixtures. Available: {list(OFFLINE_FIXTURES.keys())}")

    data = OFFLINE_FIXTURES[ticker_clean]
    stock_returns, bench_returns = generate_correlated_returns(
        seed=data["seed"],
        target_beta=data["target_beta"]
    )

    historical_models = [
        HistoricalYearMetrics(**yr) for yr in data["historical_years"]
    ]

    snapshot = CompanyFinancialSnapshot(
        ticker=data["ticker"],
        company_name=data["company_name"],
        sector=data["sector"],
        industry=data["industry"],
        currency=data["currency"],
        unit_label=data["unit_label"],
        current_price=data["current_price"],
        shares_outstanding=data["shares_outstanding"],
        market_cap=data["market_cap"],
        enterprise_value=data["enterprise_value"],
        high_52week=data["high_52week"],
        low_52week=data["low_52week"],
        benchmark_ticker=data["benchmark_ticker"],
        risk_free_rate=data["risk_free_rate"],
        equity_risk_premium=data["equity_risk_premium"],
        historical_years=historical_models,
        daily_returns_stock=stock_returns,
        daily_returns_benchmark=bench_returns,
        business_summary=data["business_summary"],
    )
    return snapshot


def fetch_company_snapshot(ticker: str, use_live: bool = False, cache_dir: Optional[Path] = None) -> CompanyFinancialSnapshot:
    """Fetches company financial snapshot in dual-mode (offline deterministic or live yfinance).
    
    If use_live is True, attempts to fetch and parse from yfinance, saving to JSON cache.
    Falls back reliably to offline fixture on network failure or missing data.
    """
    ticker_clean = ticker.strip().upper()
    c_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    cache_file = c_dir / f"{ticker_clean}.json"

    if not use_live:
        return build_offline_snapshot(ticker_clean)

    # Live Mode Attempt: Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            return CompanyFinancialSnapshot(**cached_data)
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}. Retrying live fetch.")

    try:
        import yfinance as yf  # type: ignore

        yf_ticker = yf.Ticker(ticker_clean)
        info = yf_ticker.info or {}

        # If live info is incomplete, fall back to offline fixture
        if not info.get("currentPrice") and not info.get("regularMarketPrice"):
            logger.warning(f"Incomplete yfinance data for {ticker_clean}; falling back to offline fixture.")
            return build_offline_snapshot(ticker_clean)

        # Build snapshot from offline as base, then override live price
        snapshot = build_offline_snapshot(ticker_clean)
        live_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or snapshot.current_price)
        snapshot_dict = snapshot.model_dump()
        snapshot_dict["current_price"] = live_price
        snapshot_dict["market_cap"] = live_price * snapshot.shares_outstanding

        # Recalculate enterprise value to preserve accounting identity: EV = Market Cap + Debt - Cash
        latest_yr = snapshot.historical_years[-1]
        snapshot_dict["enterprise_value"] = round(
            snapshot_dict["market_cap"] + latest_yr.total_debt - latest_yr.cash_and_equivalents, 2
        )

        if info.get("fiftyTwoWeekHigh"):
            snapshot_dict["high_52week"] = float(info["fiftyTwoWeekHigh"])
        if info.get("fiftyTwoWeekLow"):
            snapshot_dict["low_52week"] = float(info["fiftyTwoWeekLow"])

        # Cache the live snapshot
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(snapshot_dict, f, indent=2)

        return CompanyFinancialSnapshot(**snapshot_dict)

    except Exception as e:
        logger.warning(f"Live fetch for {ticker_clean} failed ({e}); falling back to offline fixture.")
        if ticker_clean in OFFLINE_FIXTURES:
            return build_offline_snapshot(ticker_clean)
        raise ValueError(f"Live fetch for '{ticker_clean}' failed ({e}) and no offline fixture is available.")


def fetch_peer_multiples(ticker: str, use_live: bool = False, cache_dir: Optional[Path] = None) -> list[PeerCompMultiple]:
    """Fetches comparable company trading multiples for relative valuation.
    
    Supports offline fixtures, disk caching in data/cache/, and live yfinance quote enrichment.
    """
    ticker_clean = ticker.strip().upper()
    if ticker_clean not in OFFLINE_PEER_FIXTURES:
        raise KeyError(f"Ticker '{ticker}' peers not found. Available: {list(OFFLINE_PEER_FIXTURES.keys())}")

    raw_peers = OFFLINE_PEER_FIXTURES[ticker_clean]
    c_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    cache_file = c_dir / f"{ticker_clean}_peers.json"

    if not use_live:
        return [PeerCompMultiple(**peer) for peer in raw_peers]

    # Live Mode: Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_peers = json.load(f)
            return [PeerCompMultiple(**p) for p in cached_peers]
        except Exception as e:
            logger.warning(f"Failed to read peer cache {cache_file}: {e}. Retrying live fetch.")

    # Try live yfinance enrichment for peers
    try:
        import yfinance as yf  # type: ignore

        updated_peers = []
        for p in raw_peers:
            peer_dict = dict(p)
            try:
                info = yf.Ticker(p["peer_ticker"]).info or {}
                if info.get("trailingPE") and float(info["trailingPE"]) > 0:
                    peer_dict["pe_ratio"] = round(float(info["trailingPE"]), 2)
                if info.get("enterpriseToEbitda") and float(info["enterpriseToEbitda"]) > 0:
                    peer_dict["ev_to_ebitda"] = round(float(info["enterpriseToEbitda"]), 2)
                if info.get("priceToBook") and float(info["priceToBook"]) > 0:
                    peer_dict["pb_ratio"] = round(float(info["priceToBook"]), 2)
                if info.get("enterpriseToRevenue") and float(info["enterpriseToRevenue"]) > 0:
                    peer_dict["ev_to_sales"] = round(float(info["enterpriseToRevenue"]), 2)
                if info.get("returnOnEquity") is not None:
                    peer_dict["roe"] = round(float(info["returnOnEquity"]), 4)
                if info.get("revenueGrowth") is not None:
                    peer_dict["revenue_growth"] = round(float(info["revenueGrowth"]), 4)
            except Exception:
                pass  # Retain fixture values if peer lookup fails
            updated_peers.append(peer_dict)

        # Cache updated peers
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(updated_peers, f, indent=2)

        return [PeerCompMultiple(**p) for p in updated_peers]

    except Exception as e:
        logger.warning(f"Live peer fetch failed ({e}); falling back to offline fixtures.")
        return [PeerCompMultiple(**peer) for peer in raw_peers]

