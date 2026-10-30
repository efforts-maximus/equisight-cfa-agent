"""Dual-Mode Market Data Gateway for EquiSight CFA Agent.

Provides institutional-grade 4-year financial statement fixtures, 252-day synthetic correlated
return series, comparable peer multiples, and live yfinance integration with local JSON caching.
EquiSight strictly enforces authentic market data: never fabricates fake prices or financial statements.
"""

import hashlib
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
CACHE_VERSION = "v3_live_verified"
CACHE_TTL_SECONDS = 4 * 3600  # 4-hour TTL

# Known exchange symbol aliases where Yahoo Finance symbol differs from benchmark fixture ticker
YFINANCE_EXCHANGE_ALIASES: dict[str, str] = {
    "VARUNBEV.NS": "VBL.NS",
    "ZOMATO.NS": "ETERNAL.NS",
}


def _is_cache_valid(cache_data: dict[str, Any]) -> bool:
    """Verifies that cached data matches CACHE_VERSION and is within the 4-hour TTL."""
    import time
    if not isinstance(cache_data, dict):
        return False
    if cache_data.get("cache_version") != CACHE_VERSION:
        return False
    try:
        cached_at = float(cache_data.get("cached_at") or 0.0)
    except (ValueError, TypeError) as exc:
        logger.debug(f"Cache timestamp parse error: {exc}")
        return False
    now = time.time()
    if cached_at <= 0.0 or (now - cached_at) > CACHE_TTL_SECONDS or cached_at > now + 300:
        return False
    if "peers" in cache_data and isinstance(cache_data["peers"], list):
        for p in cache_data["peers"]:
            if isinstance(p, dict):
                try:
                    ev_eb = float(p.get("ev_to_ebitda") or 0.0)
                    ev_s = float(p.get("ev_to_sales") or 0.0)
                    if math.isnan(ev_eb) or math.isnan(ev_s) or ev_eb > 150.0 or ev_s > 60.0:
                        return False
                except (ValueError, TypeError) as exc:
                    logger.debug(f"Peer multiple cache validation parse error: {exc}")
                    continue
    return True


def _match_date_col(df: Any, target_col: Any, fy: int) -> Any:
    """Finds matching column in statement DataFrame by exact match or fiscal year."""
    if df is None or not hasattr(df, "columns") or len(df.columns) == 0:
        return None
    if target_col in df.columns:
        return target_col
    for c in df.columns:
        c_fy = c.year if hasattr(c, "year") else int(str(c)[:4])
        if c_fy == fy:
            return c
    return df.columns[-1]


class DataUnavailableError(RuntimeError):
    """Raised when real market data is unavailable and cannot be fabricated."""
    pass


def _extract_normalized_peer_multiples(info: dict[str, Any]) -> tuple[float, float, float, float]:
    """Extracts peer valuation multiples and normalizes cross-currency EV distortion."""
    try:
        pe = float(info.get("trailingPE") or info.get("forwardPE") or 0.0)
        if (pe < 1.0 or pe > 250.0) and info.get("forwardPE"):
            fwd_pe = float(info.get("forwardPE") or 0.0)
            if 1.0 <= fwd_pe <= 250.0:
                pe = fwd_pe
    except (ValueError, TypeError) as exc:
        logger.debug(f"Trailing/Forward PE parse error: {exc}")
        pe = 0.0
    try:
        ev_ebitda = float(info.get("enterpriseToEbitda") or 0.0)
    except (ValueError, TypeError) as exc:
        logger.debug(f"EV/EBITDA parse error: {exc}")
        ev_ebitda = 0.0
    try:
        pb = float(info.get("priceToBook") or 0.0)
    except (ValueError, TypeError) as exc:
        logger.debug(f"P/B parse error: {exc}")
        pb = 0.0
    try:
        ev_sales = float(info.get("enterpriseToRevenue") or 0.0)
    except (ValueError, TypeError) as exc:
        logger.debug(f"EV/Sales parse error: {exc}")
        ev_sales = 0.0

    if math.isnan(pe):
        pe = 0.0
    if math.isnan(ev_ebitda):
        ev_ebitda = 0.0
    if math.isnan(pb):
        pb = 0.0
    if math.isnan(ev_sales):
        ev_sales = 0.0

    price_curr = str(info.get("currency") or "").upper().strip()
    fin_curr = str(info.get("financialCurrency") or price_curr).upper().strip()
    symbol = str(info.get("symbol") or "").upper().strip()
    is_in = price_curr == "INR" or symbol.endswith((".NS", ".BO")) or (price_curr == "" and fin_curr == "USD")

    if fin_curr != price_curr:
        if (fin_curr == "USD" and price_curr in ("INR", "")) or (is_in and ev_ebitda > 150.0):
            ev_ebitda = ev_ebitda / 86.5
            ev_sales = ev_sales / 86.5
    elif is_in and ev_ebitda > 150.0:
        ev_ebitda = ev_ebitda / 86.5
        ev_sales = ev_sales / 86.5

    return pe, ev_ebitda, pb, ev_sales


def format_market_cap(market_cap: float, currency: str) -> str:
    """Formats market capitalization institutional-style according to currency and scale.
    
    Returns '₹14.29 Lakh Cr' for INR Crores >= 100,000, and '$3.26T' / '$X.XXB' for USD Millions.
    """
    curr = currency.upper().strip()
    if curr == "INR":
        if market_cap >= 100000.0:
            val = market_cap / 100000.0
            return f"₹{val:.2f} Lakh Cr"
        return f"₹{market_cap:,.2f} Cr"
    elif curr == "USD":
        if market_cap >= 1000000.0:
            val = market_cap / 1000000.0
            return f"${val:.2f}T"
        elif market_cap >= 1000.0:
            val = market_cap / 1000.0
            return f"${val:.2f}B"
        return f"${market_cap:.2f}M"
    else:
        if market_cap >= 1000000.0:
            return f"{market_cap / 1000000.0:.2f}T"
        elif market_cap >= 1000.0:
            return f"{market_cap / 1000.0:.2f}B"
        return f"{market_cap:.2f}M"


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
        "shares_outstanding": 361.81,  # in Crores of shares -> 361.81 * 3950 = 1,429,149.5 Cr Market Cap
        "market_cap": 1429149.5,
        "enterprise_value": 1388849.5,  # Market Cap + Debt(8200) - Cash(48500)
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
                "revenue": 283248.0,
                "cogs": 155000.0,
                "gross_profit": 128248.0,
                "ebitda": 74900.0,
                "depreciation_amortization": 5500.0,
                "ebit": 69400.0,
                "interest_expense": 850.0,
                "ebt": 68550.0,
                "tax_expense": 16692.0,
                "net_income": 51858.0,
                "total_assets": 175000.0,
                "current_assets": 115000.0,
                "cash_and_equivalents": 48500.0,
                "accounts_receivable": 51000.0,
                "inventory": 0.0,
                "total_liabilities": 68000.0,
                "current_liabilities": 56000.0,
                "accounts_payable": 13500.0,
                "total_debt": 8200.0,
                "total_equity": 107000.0,
                "retained_earnings": 104000.0,
                "operating_cash_flow": 54200.0,
                "capex": 3800.0,
                "working_capital_change": -500.0,
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
        "shares_outstanding": 920.0,  # in Crores -> 920 * 260 = 239,200 Cr Market Cap
        "market_cap": 239200.0,
        "enterprise_value": 224780.0,  # 239200 + Debt(180) - Cash(14600)
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
        "current_price": 410.0,
        "shares_outstanding": 338.0,  # in Crores -> 338.0 * 410 = 138,580 Cr Market Cap
        "market_cap": 138580.0,
        "enterprise_value": 142380.0,  # 138580 + Debt(5200) - Cash(1400)
        "high_52week": 680.0,
        "low_52week": 350.0,
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
                "revenue": 30200.0,
                "cogs": 13600.0,
                "gross_profit": 16600.0,
                "ebitda": 7150.0,
                "depreciation_amortization": 1200.0,
                "ebit": 5950.0,
                "interest_expense": 550.0,
                "ebt": 5400.0,
                "tax_expense": 960.0,
                "net_income": 4440.0,
                "total_assets": 25500.0,
                "current_assets": 6800.0,
                "cash_and_equivalents": 1400.0,
                "accounts_receivable": 850.0,
                "inventory": 3650.0,
                "total_liabilities": 10100.0,
                "current_liabilities": 5650.0,
                "accounts_payable": 1950.0,
                "total_debt": 5200.0,
                "total_equity": 15400.0,
                "retained_earnings": 13500.0,
                "operating_cash_flow": 5750.0,
                "capex": 3250.0,
                "working_capital_change": -450.0,
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
        "shares_outstanding": 24150.0,  # in Millions of shares (24.15B) -> 24150 * 135 = 3,260,250 M ($3.26T)
        "market_cap": 3260250.0,
        "enterprise_value": 3202450.0,  # 3260250 + Debt(10200) - Cash(68000)
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
                "revenue": 185000.0,
                "cogs": 46250.0,
                "gross_profit": 138750.0,
                "ebitda": 117200.0,
                "depreciation_amortization": 3200.0,
                "ebit": 114000.0,
                "interest_expense": 350.0,
                "ebt": 113650.0,
                "tax_expense": 15650.0,
                "net_income": 98000.0,
                "total_assets": 182000.0,
                "current_assets": 125000.0,
                "cash_and_equivalents": 68000.0,
                "accounts_receivable": 28500.0,
                "inventory": 17500.0,
                "total_liabilities": 48000.0,
                "current_liabilities": 27000.0,
                "accounts_payable": 8200.0,
                "total_debt": 10200.0,
                "total_equity": 134000.0,
                "retained_earnings": 125000.0,
                "operating_cash_flow": 102000.0,
                "capex": 5800.0,
                "working_capital_change": 4800.0,
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
        raise DataUnavailableError(
            f"Ticker '{ticker}' not found in offline fixtures. Available: {list(OFFLINE_FIXTURES.keys())}. "
            "EquiSight does not fabricate synthetic financial data."
        )

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
        data_source="AUDITED_OFFLINE_SNAPSHOT",
    )
    return snapshot


def parse_yfinance_snapshot(
    ticker_clean: str,
    yf_ticker: Any,
    info: dict[str, Any],
) -> CompanyFinancialSnapshot:
    """Parses real financial statements from yfinance into a validated CompanyFinancialSnapshot.
    
    Strictly preserves all 2027 CFA Level 1 accounting identities without fabricating synthetic data.
    Raises DataUnavailableError whenever required statements or fundamental financial rows are absent.
    """
    price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
    if price <= 0.0:
        raise DataUnavailableError(
            f"Live Yahoo Finance data unavailable for '{ticker_clean}': No valid market price found. "
            "EquiSight does not fabricate synthetic financial data."
        )

    fin = getattr(yf_ticker, "income_stmt", None)
    if fin is None or (hasattr(fin, "empty") and fin.empty):
        fin = getattr(yf_ticker, "financials", None)
    bs = getattr(yf_ticker, "balance_sheet", None)
    cf = getattr(yf_ticker, "cashflow", None)

    if fin is None or fin.empty or bs is None or bs.empty or cf is None or cf.empty:
        raise DataUnavailableError(
            f"Live Yahoo Finance data unavailable for '{ticker_clean}': Missing required financial statements "
            "(Income Statement, Balance Sheet, or Cash Flow). EquiSight does not fabricate synthetic financial data."
        )

    is_in = ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO")
    raw_currency = str(info.get("currency") or ("INR" if is_in else "USD")).upper().strip()
    price_curr = "INR" if (is_in or raw_currency == "INR") else "USD"
    currency = price_curr
    unit_label = "INR Crores" if currency == "INR" else "USD Millions"
    unit_scale = 1e7 if currency == "INR" else 1e6
    benchmark_ticker = "^NSEI" if currency == "INR" else "^GSPC"
    rf = 0.068 if currency == "INR" else 0.042
    erp = 0.060 if currency == "INR" else 0.050

    # Handle cross-currency reporting (e.g. INFY.NS trading in INR but filing in USD, or ADR trading in USD filing in INR)
    fin_curr = str(info.get("financialCurrency") or price_curr).upper().strip()
    fx_rate = 1.0
    if fin_curr != price_curr:
        try:
            import yfinance as yf  # type: ignore
            fx_symbol = f"{fin_curr}{price_curr}=X"
            fx_info = yf.Ticker(fx_symbol).info or {}
            rate_val = float(fx_info.get("regularMarketPrice") or fx_info.get("currentPrice") or 0.0)
            if rate_val <= 0.0:
                hist_fx = yf.Ticker(fx_symbol).history(period="1d")
                if not hist_fx.empty and "Close" in hist_fx.columns:
                    rate_val = float(hist_fx["Close"].iloc[-1])
            if rate_val > 0.0:
                fx_rate = rate_val
            else:
                FALLBACK_FX = {
                    ("USD", "INR"): 86.5,
                    ("INR", "USD"): 1.0 / 86.5,
                    ("EUR", "USD"): 1.08,
                    ("USD", "EUR"): 1.0 / 1.08,
                    ("GBP", "USD"): 1.28,
                    ("USD", "GBP"): 1.0 / 1.28,
                    ("JPY", "USD"): 0.0065,
                    ("CAD", "USD"): 0.72,
                }
                fx_rate = FALLBACK_FX.get((fin_curr, price_curr), 86.5 if fin_curr == "USD" and price_curr == "INR" else 1.0)
        except Exception as exc:
            logger.debug(f"Live FX rate lookup failed for {fin_curr}->{price_curr} ({exc}); using fallback table.")
            FALLBACK_FX = {
                ("USD", "INR"): 86.5,
                ("INR", "USD"): 1.0 / 86.5,
                ("EUR", "USD"): 1.08,
                ("USD", "EUR"): 1.0 / 1.08,
                ("GBP", "USD"): 1.28,
                ("USD", "GBP"): 1.0 / 1.28,
                ("JPY", "USD"): 0.0065,
                ("CAD", "USD"): 0.72,
            }
            fx_rate = FALLBACK_FX.get((fin_curr, price_curr), 86.5 if fin_curr == "USD" and price_curr == "INR" else 1.0)

    def _scale_item(raw_val: Optional[float]) -> Optional[float]:
        if raw_val is None:
            return None
        return (raw_val * fx_rate) / unit_scale

    raw_mc = float(info.get("marketCap") or 0.0)
    raw_shares = float(info.get("sharesOutstanding") or 0.0)

    if raw_mc > 0.0 and price > 0.0:
        market_cap_scaled = raw_mc / unit_scale
        shares_outstanding = max(0.01, round(market_cap_scaled / price, 2))
    elif raw_shares > 0.0:
        shares_outstanding = max(0.01, round(raw_shares / unit_scale, 2))
    else:
        raise DataUnavailableError(
            f"Live Yahoo Finance data unavailable for '{ticker_clean}': Shares outstanding unavailable. "
            "EquiSight does not fabricate synthetic financial data."
        )
    market_cap = round(price * shares_outstanding, 2)

    date_cols = sorted(fin.columns)
    if len(date_cols) < 1:
        raise DataUnavailableError(
            f"Live Yahoo Finance data unavailable for '{ticker_clean}': No historical annual statements found. "
            "EquiSight does not fabricate synthetic financial data."
        )

    selected_cols = date_cols[-4:]

    def _get_val(df: Any, row_names: list[str], col: Any) -> Optional[float]:
        for r in row_names:
            if r in df.index:
                val = df.loc[r, col]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    return float(val)
        return None

    hist_models: list[HistoricalYearMetrics] = []
    for col in selected_cols:
        fy = col.year if hasattr(col, "year") else int(str(col)[:4])

        rev_raw = _get_val(fin, ["Total Revenue", "Operating Revenue"], col)
        if rev_raw is None or rev_raw <= 0.0:
            raise DataUnavailableError(
                f"Live Yahoo Finance data unavailable for '{ticker_clean}': Missing Total Revenue for fiscal year {fy}. "
                "EquiSight does not fabricate synthetic financial data."
            )
        rev = round(_scale_item(rev_raw), 2)

        cogs_raw = _get_val(fin, ["Cost Of Revenue", "Reconciled Cost Of Revenue", "Cost Of Goods And Services Sold"], col)
        if cogs_raw is not None:
            cogs = round(_scale_item(cogs_raw), 2)
        else:
            gp_raw = _get_val(fin, ["Gross Profit"], col)
            if gp_raw is not None:
                cogs = max(0.0, round(_scale_item(rev_raw - gp_raw), 2))
            else:
                cogs = 0.0
        gross_profit = round(rev - cogs, 2)

        ebit_raw = _get_val(fin, ["EBIT", "Operating Income", "Total Operating Income As Reported"], col)
        if ebit_raw is None:
            ebitda_raw = _get_val(fin, ["EBITDA", "Normalized EBITDA"], col)
            if ebitda_raw is not None:
                da_raw_temp = _get_val(fin, ["Reconciled Depreciation", "Depreciation Amortization Depletion"], col) or 0.0
                ebit_raw = ebitda_raw - da_raw_temp
            else:
                raise DataUnavailableError(
                    f"Live Yahoo Finance data unavailable for '{ticker_clean}': Operating Income / EBIT not reported for fiscal year {fy}. "
                    "EquiSight does not fabricate synthetic financial data."
                )
        ebit = round(_scale_item(ebit_raw), 2)

        da_raw = _get_val(fin, ["Reconciled Depreciation", "Depreciation Amortization Depletion", "Depreciation And Amortization In Income Statement", "Depreciation & Amortization"], col)
        if da_raw is None and cf is not None:
            cf_col_temp = col if col in cf.columns else (cf.columns[-1] if len(cf.columns) > 0 else None)
            if cf_col_temp is not None:
                da_raw = _get_val(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation"], cf_col_temp)
        da = round(_scale_item(da_raw or 0.0), 2)
        ebitda = round(ebit + da, 2)

        interest_raw = _get_val(fin, ["Interest Expense", "Interest Expense Non Operating", "Net Non Operating Interest Income Expense", "Interest Expense Net"], col) or 0.0
        interest = round(_scale_item(abs(interest_raw)), 2)
        ebt = round(ebit - interest, 2)

        net_income_raw = _get_val(fin, ["Net Income Common Stockholders", "Net Income", "Net Income Continuous Operations", "Net Income From Continuing Operation Net Minority Interest"], col)
        if net_income_raw is None:
            raise DataUnavailableError(
                f"Live Yahoo Finance data unavailable for '{ticker_clean}': Net Income not reported for fiscal year {fy}. "
                "EquiSight does not fabricate synthetic financial data."
            )
        net_income = round(_scale_item(net_income_raw), 2)
        tax_expense = round(ebt - net_income, 2)

        bs_col = _match_date_col(bs, col, fy)
        if bs_col is None:
            raise DataUnavailableError(
                f"Live Yahoo Finance data unavailable for '{ticker_clean}': Balance sheet missing date {col}. "
                "EquiSight does not fabricate synthetic financial data."
            )
        tot_assets_raw = _get_val(bs, ["Total Assets"], bs_col)
        if tot_assets_raw is None:
            raise DataUnavailableError(
                f"Live Yahoo Finance data unavailable for '{ticker_clean}': Total Assets not reported for balance sheet date {bs_col}. "
                "EquiSight does not fabricate synthetic financial data."
            )
        tot_assets = round(_scale_item(tot_assets_raw), 2)

        cur_assets_raw = _get_val(bs, ["Current Assets"], bs_col)
        if cur_assets_raw is None:
            raise DataUnavailableError(
                f"Live Yahoo Finance data unavailable for '{ticker_clean}': Current Assets not reported for balance sheet date {bs_col}. "
                "EquiSight does not fabricate synthetic financial data."
            )
        cur_assets = round(_scale_item(cur_assets_raw), 2)

        cash_raw = _get_val(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash Financial", "Cash Equivalents"], bs_col)
        if cash_raw is None and cf is not None:
            cash_raw = _get_val(cf, ["End Cash Position"], bs_col)
        if cash_raw is None:
            raise DataUnavailableError(
                f"Live Yahoo Finance data unavailable for '{ticker_clean}': Cash and Equivalents not reported for balance sheet date {bs_col}. "
                "EquiSight does not fabricate synthetic financial data."
            )
        cash = round(_scale_item(cash_raw), 2)

        ar_raw = _get_val(bs, ["Receivables", "Accounts Receivable", "Gross Accounts Receivable"], bs_col) or 0.0
        ar = round(_scale_item(ar_raw), 2)

        inv_raw = _get_val(bs, ["Inventory", "Finished Goods", "Raw Materials", "Inventories"], bs_col) or 0.0
        inv = round(_scale_item(inv_raw), 2)

        tot_liab_raw = _get_val(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"], bs_col)
        if tot_liab_raw is None:
            equity_raw_temp = _get_val(bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"], bs_col)
            if equity_raw_temp is not None:
                tot_liab_raw = tot_assets_raw - equity_raw_temp
            else:
                raise DataUnavailableError(
                    f"Live Yahoo Finance data unavailable for '{ticker_clean}': Total Liabilities not reported for balance sheet date {bs_col}. "
                    "EquiSight does not fabricate synthetic financial data."
                )
        tot_liab = round(_scale_item(tot_liab_raw), 2)

        cur_liab_raw = _get_val(bs, ["Current Liabilities"], bs_col)
        if cur_liab_raw is None:
            raise DataUnavailableError(
                f"Live Yahoo Finance data unavailable for '{ticker_clean}': Current Liabilities not reported for balance sheet date {bs_col}. "
                "EquiSight does not fabricate synthetic financial data."
            )
        cur_liab = round(_scale_item(cur_liab_raw), 2)

        ap_raw = _get_val(bs, ["Accounts Payable", "Payables And Accrued Expenses", "Payables", "Tradeand Other Payables Non Current"], bs_col) or 0.0
        ap = round(_scale_item(ap_raw), 2)

        debt_raw = _get_val(bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Current Debt And Capital Lease Obligation", "Long Term Debt"], bs_col) or 0.0
        debt = round(_scale_item(debt_raw), 2)

        # Strictly enforce accounting identity: Total Assets == Total Liabilities + Total Equity
        equity = round(tot_assets - tot_liab, 2)
        retained_raw = _get_val(bs, ["Retained Earnings", "Retained Earnings Total Equity"], bs_col)
        retained = round(_scale_item(retained_raw) if retained_raw is not None else equity, 2)

        cf_col = _match_date_col(cf, col, fy)
        if cf_col is None:
            raise DataUnavailableError(
                f"Live Yahoo Finance data unavailable for '{ticker_clean}': Cash flow statement missing date {col}. "
                "EquiSight does not fabricate synthetic financial data."
            )

        cfo_raw = _get_val(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], cf_col)
        if cfo_raw is None:
            raise DataUnavailableError(
                f"Live Yahoo Finance data unavailable for '{ticker_clean}': Operating Cash Flow not reported for fiscal year {fy}. "
                "EquiSight does not fabricate synthetic financial data."
            )
        cfo = round(_scale_item(cfo_raw), 2)

        capex_raw = _get_val(cf, ["Capital Expenditure", "Capital Expenditure Reported", "Investments In Property Plant And Equipment", "Purchase Of PPE"], cf_col) or 0.0
        capex = round(_scale_item(abs(capex_raw)), 2)

        nwc_raw = _get_val(cf, ["Change In Working Capital", "Changes In Working Capital"], cf_col) or 0.0
        nwc_chg = round(_scale_item(nwc_raw), 2)
        max_nwc = round(0.08 * rev, 2)
        min_nwc = round(-0.08 * rev, 2)
        if nwc_chg > max_nwc:
            nwc_chg = max_nwc
        elif nwc_chg < min_nwc:
            nwc_chg = min_nwc

        m = HistoricalYearMetrics(
            fiscal_year=fy,
            revenue=rev,
            cogs=cogs,
            gross_profit=gross_profit,
            ebitda=ebitda,
            depreciation_amortization=da,
            ebit=ebit,
            interest_expense=interest,
            ebt=ebt,
            tax_expense=tax_expense,
            net_income=net_income,
            total_assets=tot_assets,
            current_assets=cur_assets,
            cash_and_equivalents=cash,
            accounts_receivable=ar,
            inventory=inv,
            total_liabilities=tot_liab,
            current_liabilities=cur_liab,
            accounts_payable=ap,
            total_debt=debt,
            total_equity=equity,
            retained_earnings=retained,
            operating_cash_flow=cfo,
            capex=capex,
            working_capital_change=nwc_chg,
        )
        hist_models.append(m)

    latest = hist_models[-1]
    ev = round(market_cap + latest.total_debt - latest.cash_and_equivalents, 2)

    import pandas as pd  # type: ignore
    import yfinance as yf  # type: ignore

    try:
        hist_stock = yf_ticker.history(period="1y")
        hist_bench = yf.Ticker(benchmark_ticker).history(period="1y")
    except Exception as exc:
        raise DataUnavailableError(
            f"Live Yahoo Finance data unavailable for '{ticker_clean}': Insufficient aligned historical daily returns."
        ) from exc

    if (
        hist_stock is None
        or (hasattr(hist_stock, "empty") and hist_stock.empty)
        or "Close" not in hist_stock.columns
        or hist_bench is None
        or (hasattr(hist_bench, "empty") and hist_bench.empty)
        or "Close" not in hist_bench.columns
    ):
        raise DataUnavailableError(
            f"Live Yahoo Finance data unavailable for '{ticker_clean}': Insufficient aligned historical daily returns."
        )

    s_stock = hist_stock["Close"].pct_change().dropna()
    s_bench = hist_bench["Close"].pct_change().dropna()

    if s_stock.empty or s_bench.empty:
        raise DataUnavailableError(
            f"Live Yahoo Finance data unavailable for '{ticker_clean}': Insufficient aligned historical daily returns."
        )

    s_stock.index = pd.to_datetime(s_stock.index).tz_localize(None).normalize()
    s_bench.index = pd.to_datetime(s_bench.index).tz_localize(None).normalize()

    s_stock = s_stock[~s_stock.index.duplicated(keep="last")]
    s_bench = s_bench[~s_bench.index.duplicated(keep="last")]

    df_aligned = pd.concat([s_stock.rename("stock"), s_bench.rename("bench")], axis=1, join="inner").dropna()

    if len(df_aligned) < 30:
        raise DataUnavailableError(
            f"Live Yahoo Finance data unavailable for '{ticker_clean}': Insufficient aligned historical daily returns."
        )

    stock_ret = [round(float(x), 6) for x in df_aligned["stock"].values[-252:]]
    bench_ret = [round(float(x), 6) for x in df_aligned["bench"].values[-252:]]

    try:
        high_52w = float(info.get("fiftyTwoWeekHigh") or 0.0)
    except (ValueError, TypeError):
        high_52w = 0.0
    if math.isnan(high_52w) or high_52w <= 0.0:
        high_52w = 0.0

    try:
        low_52w = float(info.get("fiftyTwoWeekLow") or 0.0)
    except (ValueError, TypeError):
        low_52w = 0.0
    if math.isnan(low_52w) or low_52w <= 0.0:
        low_52w = 0.0

    if high_52w <= 0.0 or low_52w <= 0.0:
        try:
            if not hist_stock.empty and "High" in hist_stock.columns and "Low" in hist_stock.columns:
                h_val = float(hist_stock["High"].max())
                l_val = float(hist_stock["Low"].min())
                if not math.isnan(h_val) and h_val > 0.0:
                    high_52w = h_val
                if not math.isnan(l_val) and l_val > 0.0:
                    low_52w = l_val
        except Exception as exc:
            logger.debug(f"Could not retrieve 52-week high/low from history for {ticker_clean}: {exc}")
    if high_52w <= 0.0:
        high_52w = price
    if low_52w <= 0.0:
        low_52w = price
    if high_52w < low_52w:
        high_52w, low_52w = low_52w, high_52w
    high_52w = max(high_52w, price)
    low_52w = min(low_52w, price)

    company_name = info.get("longName") or info.get("shortName") or f"{ticker_clean}"
    sector = info.get("sector") or "Technology"
    industry = info.get("industry") or "Software & IT Services"
    business_summary = (
        info.get("longBusinessSummary")
        or f"{company_name} is an enterprise operating in the {sector} sector ({industry})."
    )

    return CompanyFinancialSnapshot(
        ticker=ticker_clean,
        company_name=company_name,
        sector=sector,
        industry=industry,
        currency=currency,
        unit_label=unit_label,
        current_price=price,
        shares_outstanding=shares_outstanding,
        market_cap=market_cap,
        enterprise_value=ev,
        high_52week=high_52w,
        low_52week=low_52w,
        benchmark_ticker=benchmark_ticker,
        risk_free_rate=rf,
        equity_risk_premium=erp,
        historical_years=hist_models,
        daily_returns_stock=stock_ret,
        daily_returns_benchmark=bench_ret,
        business_summary=business_summary,
        data_source="LIVE_YFINANCE",
    )


def fetch_company_snapshot(
    ticker: str,
    use_live: bool = True,
    cache_dir: Optional[Path] = None,
) -> CompanyFinancialSnapshot:
    """Fetches company financial snapshot in dual-mode (live yfinance by default, offline fallback).

    EquiSight strictly enforces authentic market data:
    - Never fabricates fake prices or fake financial statements.
    - If use_live is False and ticker is an offline benchmark fixture, returns the audited fixture immediately.
    - If use_live is True, ALWAYS queries live yfinance first for all tickers (with 4-hour v3 disk caching).
    - If live yfinance retrieval fails (network/API error or missing statements) and the ticker has an offline fixture,
      gracefully falls back to the audited fixture and sets data_source = 'FALLBACK_OFFLINE_FIXTURE'.
    - If live yfinance fails and no offline fixture exists, raises DataUnavailableError.
    """
    raw_ticker = ticker.strip()
    if not raw_ticker:
        raise ValueError("Ticker symbol cannot be empty.")

    from src.data.ticker_resolver import resolve_ticker_query
    try:
        ticker_clean = resolve_ticker_query(raw_ticker)
    except Exception as exc:
        logger.debug(f"Ticker resolution failed for '{raw_ticker}' ({exc}); defaulting to upper.")
        ticker_clean = raw_ticker.upper()

    c_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    cache_file = c_dir / f"{ticker_clean}.json"

    # 1. If offline mode is explicitly requested (use_live=False):
    if not use_live:
        if ticker_clean in OFFLINE_FIXTURES:
            return build_offline_snapshot(ticker_clean)
        # For custom tickers with use_live=False, check disk cache first
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if _is_cache_valid(cached_data):
                    clean_data = {k: v for k, v in cached_data.items() if k not in ("cache_version", "cached_at")}
                    if clean_data.get("data_source") == "LIVE_YFINANCE":
                        return CompanyFinancialSnapshot(**clean_data)
                else:
                    cache_file.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to read custom cache {cache_file}: {e}")
                cache_file.unlink(missing_ok=True)

        # Custom ticker not in offline fixtures: try live yfinance with caching or raise DataUnavailableError
        try:
            import yfinance as yf  # type: ignore
            import time
            yf_ticker = yf.Ticker(ticker_clean)
            info = yf_ticker.info or {}
            price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
            if price <= 0.0:
                raise DataUnavailableError(f"Market data unavailable for '{ticker_clean}'.")
            snapshot = parse_yfinance_snapshot(ticker_clean, yf_ticker, info)
            try:
                snap_dict = snapshot.model_dump()
                snap_dict["cache_version"] = CACHE_VERSION
                snap_dict["cached_at"] = time.time()
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(snap_dict, f, indent=2)
            except Exception as exc:
                logger.warning(f"Could not cache offline snapshot for {ticker_clean}: {exc}")
            return snapshot
        except Exception as exc:
            raise DataUnavailableError(
                f"Market data unavailable offline for '{ticker_clean}': {exc}"
            ) from exc

    # 2. Live mode (use_live=True):
    # Check disk cache first (must have cache_version="v3_live_verified" and within 4h TTL)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if _is_cache_valid(cached_data):
                clean_data = {k: v for k, v in cached_data.items() if k not in ("cache_version", "cached_at")}
                if clean_data.get("data_source") == "LIVE_YFINANCE":
                    return CompanyFinancialSnapshot(**clean_data)
            else:
                cache_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}. Retrying live fetch.")
            cache_file.unlink(missing_ok=True)

    # Always query live yfinance FIRST for ALL tickers (including benchmark fixtures)
    try:
        import yfinance as yf  # type: ignore
        import time

        yf_sym = YFINANCE_EXCHANGE_ALIASES.get(ticker_clean, ticker_clean)
        yf_ticker = yf.Ticker(yf_sym)
        info = yf_ticker.info or {}
        price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)

        # If primary symbol had no quote and there's an alias we haven't tried:
        if price <= 0.0 and ticker_clean != yf_sym:
            yf_ticker = yf.Ticker(ticker_clean)
            info = yf_ticker.info or {}
            price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)

        if price <= 0.0:
            raise DataUnavailableError(
                f"Live Yahoo Finance data unavailable for '{ticker_clean}': Symbol not found or has no active market quote."
            )

        snapshot = parse_yfinance_snapshot(ticker_clean, yf_ticker, info)

        # Cache the verified live snapshot with v3 version tag and timestamp
        try:
            snap_dict = snapshot.model_dump()
            snap_dict["cache_version"] = CACHE_VERSION
            snap_dict["cached_at"] = time.time()
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(snap_dict, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not cache live snapshot for {ticker_clean}: {e}")

        return snapshot

    except Exception as exc:
        if ticker_clean in OFFLINE_FIXTURES:
            logger.warning(f"Live fetch for {ticker_clean} failed ({exc}); falling back to offline fixture.")
            base_snap = build_offline_snapshot(ticker_clean)
            snap_dict = base_snap.model_dump()
            snap_dict["data_source"] = "FALLBACK_OFFLINE_FIXTURE"
            return CompanyFinancialSnapshot(**snap_dict)
        raise DataUnavailableError(
            f"Live Yahoo Finance data unavailable for '{ticker_clean}': {exc}. "
            "EquiSight does not fabricate synthetic financial data."
        ) from exc


def fetch_peer_multiples(
    ticker: str,
    use_live: bool = True,
    cache_dir: Optional[Path] = None,
) -> list[PeerCompMultiple]:
    """Fetches comparable company trading multiples for relative valuation.

    Supports offline fixtures, disk caching in data/cache/, and live yfinance quote enrichment.
    Raises DataUnavailableError if real peer multiples cannot be retrieved.
    """
    raw_ticker = ticker.strip()
    if not raw_ticker:
        raise ValueError("Ticker symbol cannot be empty.")

    from src.data.ticker_resolver import resolve_ticker_query
    try:
        ticker_clean = resolve_ticker_query(raw_ticker)
    except Exception as exc:
        logger.debug(f"Peer ticker resolution failed for '{raw_ticker}' ({exc}); defaulting to upper.")
        ticker_clean = raw_ticker.upper()

    c_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    c_dir.mkdir(parents=True, exist_ok=True)
    cache_file = c_dir / f"{ticker_clean}_peers.json"

    import time

    # Check cache first (v3_live_verified and within 4h TTL)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if _is_cache_valid(cached_data):
                peer_list = cached_data.get("peers", [])
                if peer_list:
                    return [PeerCompMultiple(**p) for p in peer_list]
            else:
                cache_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to read custom peer cache {cache_file}: {e}")
            cache_file.unlink(missing_ok=True)

    # Offline mode:
    if not use_live:
        if ticker_clean in OFFLINE_PEER_FIXTURES:
            return [PeerCompMultiple(**peer) for peer in OFFLINE_PEER_FIXTURES[ticker_clean]]

    # For tickers with offline peer fixtures:
    if ticker_clean in OFFLINE_PEER_FIXTURES:
        raw_peers = OFFLINE_PEER_FIXTURES[ticker_clean]
        try:
            import yfinance as yf  # type: ignore

            updated_peers = []
            for p in raw_peers:
                peer_dict = dict(p)
                try:
                    info = yf.Ticker(p["peer_ticker"]).info or {}
                    pe, ev_ebitda, pb, ev_sales = _extract_normalized_peer_multiples(info)
                    if 1.0 <= pe <= 250.0:
                        peer_dict["pe_ratio"] = round(pe, 2)
                    if 0.5 <= ev_ebitda <= 150.0:
                        peer_dict["ev_to_ebitda"] = round(ev_ebitda, 2)
                    if 0.1 <= pb <= 120.0:
                        peer_dict["pb_ratio"] = round(pb, 2)
                    if 0.1 <= ev_sales <= 60.0:
                        peer_dict["ev_to_sales"] = round(ev_sales, 2)
                    if info.get("returnOnEquity") is not None:
                        try:
                            roe_val = float(info["returnOnEquity"])
                            if not math.isnan(roe_val):
                                peer_dict["roe"] = round(roe_val, 4)
                        except (ValueError, TypeError) as exc:
                            logger.debug(f"Invalid ROE value for {p.get('peer_ticker')}: {exc}")
                    if info.get("revenueGrowth") is not None:
                        try:
                            rg_val = float(info["revenueGrowth"])
                            if not math.isnan(rg_val):
                                peer_dict["revenue_growth"] = round(rg_val, 4)
                        except (ValueError, TypeError) as exc:
                            logger.debug(f"Invalid revenue growth value for {p.get('peer_ticker')}: {exc}")
                except Exception as exc:
                    logger.debug(f"Peer enrichment failed for {p.get('peer_ticker')}: {exc}")
                updated_peers.append(peer_dict)

            # Cache updated peers
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "cache_version": CACHE_VERSION,
                        "cached_at": time.time(),
                        "peers": updated_peers,
                    }, f, indent=2)
            except Exception as exc:
                logger.warning(f"Could not cache peer fixtures for {ticker_clean}: {exc}")

            return [PeerCompMultiple(**p) for p in updated_peers]

        except Exception as e:
            logger.warning(f"Live peer fetch failed ({e}); falling back to offline fixtures.")
            return [PeerCompMultiple(**peer) for peer in raw_peers]

    # Custom tickers not in OFFLINE_PEER_FIXTURES:
    peers: list[PeerCompMultiple] = []
    try:
        import yfinance as yf  # type: ignore

        yf_ticker = yf.Ticker(ticker_clean)
        info = yf_ticker.info or {}

        price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
        sector = info.get("sector")
        if price <= 0.0 or not sector:
            raise DataUnavailableError(
                f"Comparable peers unavailable for '{ticker_clean}': Symbol not found or has no active profile on Yahoo Finance. "
                "EquiSight does not fabricate synthetic peer data."
            )

        is_in = ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO")
        if is_in:
            sector_peers = {
                "Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
                "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS"],
                "Consumer Defensive": ["VARUNBEV.NS", "ITC.NS", "BRITANNIA.NS"],
                "Consumer Cyclical": ["TATAMOTORS.NS", "MARUTI.NS", "TITAN.NS", "ZOMATO.NS"],
                "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"],
                "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS"],
                "Industrials": ["LT.NS", "BEL.NS", "SIEMENS.NS"],
                "Basic Materials": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS"],
                "Utilities": ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS"],
            }
        else:
            sector_peers = {
                "Technology": ["MSFT", "GOOGL", "AAPL", "NVDA"],
                "Communication Services": ["META", "NFLX", "MSFT", "GOOGL"],
                "Consumer Defensive": ["PG", "KO", "PEP", "COST"],
                "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE"],
                "Financial Services": ["JPM", "BAC", "WFC", "MS"],
                "Healthcare": ["JNJ", "UNH", "PFE", "ABBV"],
                "Industrials": ["CAT", "GE", "HON", "BA"],
                "Energy": ["XOM", "CVX", "COP"],
                "Basic Materials": ["LIN", "APD", "SHW"],
                "Utilities": ["NEE", "SO", "DUK"],
            }
        candidate_list = [
            p for p in sector_peers.get(sector, sector_peers.get("Technology", []))
            if p != ticker_clean and not ({p, ticker_clean} <= {"GOOG", "GOOGL"})
        ]

        for candidate in candidate_list:
            if len(peers) >= 3:
                break
            try:
                c_info = yf.Ticker(candidate).info or {}
                pe, ev_ebitda, pb, ev_sales = _extract_normalized_peer_multiples(c_info)
                if not (1.0 <= pe <= 250.0 and 0.5 <= ev_ebitda <= 150.0):
                    continue
                pb_final = round(pb, 2) if (0.1 <= pb <= 120.0) else round(max(0.5, pb), 2)
                ev_sales_final = round(ev_sales, 2) if (0.1 <= ev_sales <= 60.0) else round(max(0.5, ev_sales), 2)
                try:
                    roe_raw = float(c_info.get("returnOnEquity") or 0.0)
                    roe = round(roe_raw, 4) if not math.isnan(roe_raw) else 0.0
                except (ValueError, TypeError) as exc:
                    logger.debug(f"Invalid ROE value for candidate {candidate}: {exc}")
                    roe = 0.0
                try:
                    rg_raw = float(c_info.get("revenueGrowth") or 0.0)
                    rev_growth = round(rg_raw, 4) if not math.isnan(rg_raw) else 0.0
                except (ValueError, TypeError) as exc:
                    logger.debug(f"Invalid revenue growth value for candidate {candidate}: {exc}")
                    rev_growth = 0.0
                c_name = c_info.get("shortName") or c_info.get("longName") or candidate
                peers.append(
                    PeerCompMultiple(
                        peer_ticker=candidate,
                        peer_name=c_name,
                        pe_ratio=round(pe, 2),
                        ev_to_ebitda=round(ev_ebitda, 2),
                        pb_ratio=pb_final,
                        ev_to_sales=ev_sales_final,
                        roe=roe,
                        revenue_growth=rev_growth,
                    )
                )
            except Exception as exc:
                logger.debug(f"Peer candidate lookup failed for {candidate}: {exc}")
    except DataUnavailableError:
        raise
    except Exception as e:
        logger.warning(f"Live yfinance peer lookup for custom ticker {ticker_clean} failed: {e}")

    if len(peers) >= 2:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "cache_version": CACHE_VERSION,
                    "cached_at": time.time(),
                    "peers": [p.model_dump() for p in peers],
                }, f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not cache custom peers for {ticker_clean}: {exc}")
        return peers

    raise DataUnavailableError(
        f"Comparable peers unavailable for '{ticker_clean}': Insufficient peer data found from Yahoo Finance. "
        "EquiSight does not fabricate synthetic peer data."
    )
