"""Live Daily Market Movers & Popular Stocks Engine for EquiSight CFA Agent.

Fetches real-time price change %, trading volume, and market capitalization across
NSE India and US equities from Yahoo Finance, ranked dynamically by today's trading activity.
EquiSight strictly enforces authentic market data: never fabricates fake prices or movers.
"""

import logging
import time
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from src.data.market_gateway import format_market_cap

logger = logging.getLogger(__name__)


class TrendingStockCard(BaseModel):
    """Represents a live market mover stock card."""
    model_config = ConfigDict(extra="ignore")

    ticker: str = Field(description="Display ticker symbol, e.g. NVDA or TCS.NS")
    company_name: str = Field(description="Company full or short name")
    market: str = Field(description="Market identifier: US or NSE")
    current_price: float = Field(description="Latest traded price per share")
    currency: str = Field(description="Reporting currency (USD or INR)")
    daily_change_pct: float = Field(description="Percentage price change today (+/- %)")
    daily_volume: int = Field(description="Reported daily trading volume")
    market_cap: float = Field(description="Market capitalization in native unit scale")
    formatted_market_cap: str = Field(default="", description="Institutional formatted market cap")

    @property
    def activity_tag(self) -> str:
        if abs(self.daily_change_pct) >= 3.0:
            return "HIGH VOLATILITY"
        elif self.daily_volume > 10_000_000:
            return "HEAVY VOLUME"
        elif self.daily_change_pct > 0:
            return "ACTIVE GAINER"
        else:
            return "MARKET MOVER"

    @property
    def currency_symbol(self) -> str:
        return "₹" if self.currency == "INR" else "$"

    @property
    def day_change_pct(self) -> float:
        return self.daily_change_pct

    @property
    def day_volume(self) -> int:
        return self.daily_volume

    @property
    def market_cap_formatted(self) -> str:
        return self.formatted_market_cap


# Canonical popular universe across US and NSE India
POPULAR_STOCKS_METADATA: dict[str, dict[str, str]] = {
    # US Market Benchmarks & Movers
    "NVDA": {"company_name": "NVIDIA Corporation", "market": "US", "currency": "USD", "yf_symbol": "NVDA"},
    "AAPL": {"company_name": "Apple Inc.", "market": "US", "currency": "USD", "yf_symbol": "AAPL"},
    "MSFT": {"company_name": "Microsoft Corporation", "market": "US", "currency": "USD", "yf_symbol": "MSFT"},
    "AMZN": {"company_name": "Amazon.com, Inc.", "market": "US", "currency": "USD", "yf_symbol": "AMZN"},
    "GOOGL": {"company_name": "Alphabet Inc.", "market": "US", "currency": "USD", "yf_symbol": "GOOGL"},
    "META": {"company_name": "Meta Platforms, Inc.", "market": "US", "currency": "USD", "yf_symbol": "META"},
    "TSLA": {"company_name": "Tesla, Inc.", "market": "US", "currency": "USD", "yf_symbol": "TSLA"},
    "AVGO": {"company_name": "Broadcom Inc.", "market": "US", "currency": "USD", "yf_symbol": "AVGO"},
    # NSE India Benchmarks & Movers
    "TCS.NS": {"company_name": "Tata Consultancy Services", "market": "NSE", "currency": "INR", "yf_symbol": "TCS.NS"},
    "INFY.NS": {"company_name": "Infosys Limited", "market": "NSE", "currency": "INR", "yf_symbol": "INFY.NS"},
    "RELIANCE.NS": {"company_name": "Reliance Industries Limited", "market": "NSE", "currency": "INR", "yf_symbol": "RELIANCE.NS"},
    "HDFCBANK.NS": {"company_name": "HDFC Bank Limited", "market": "NSE", "currency": "INR", "yf_symbol": "HDFCBANK.NS"},
    "ICICIBANK.NS": {"company_name": "ICICI Bank Limited", "market": "NSE", "currency": "INR", "yf_symbol": "ICICIBANK.NS"},
    "SBIN.NS": {"company_name": "State Bank of India", "market": "NSE", "currency": "INR", "yf_symbol": "SBIN.NS"},
    "BHARTIARTL.NS": {"company_name": "Bharti Airtel Limited", "market": "NSE", "currency": "INR", "yf_symbol": "BHARTIARTL.NS"},
    "ITC.NS": {"company_name": "ITC Limited", "market": "NSE", "currency": "INR", "yf_symbol": "ITC.NS"},
    "VARUNBEV.NS": {"company_name": "Varun Beverages Limited", "market": "NSE", "currency": "INR", "yf_symbol": "VBL.NS"},
    "ZOMATO.NS": {"company_name": "Zomato Limited", "market": "NSE", "currency": "INR", "yf_symbol": "ETERNAL.NS"},
}

_PULSE_CACHE: dict[str, tuple[float, list[TrendingStockCard], bool, str]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes in-memory cache


def fetch_daily_popular_stocks(
    market_filter: str = "ALL",
    max_items: int = 8,
) -> tuple[list[TrendingStockCard], bool, str]:
    """Pulls live daily % price change and trading volume from Yahoo Finance across NSE India and US markets.

    Args:
        market_filter: 'ALL', 'US', or 'NSE' (or 'INDIA').
        max_items: Maximum number of stock cards to return.

    Returns:
        tuple[list[TrendingStockCard], bool, str]:
            - cards: List of trending stock cards ranked by today's trading activity.
            - is_live: True if real live data was retrieved, False if offline/unavailable.
            - status_message: Informational message regarding live data provenance or offline status.
    """
    filter_norm = market_filter.strip().upper()
    cache_key = f"{filter_norm}::{max_items}"
    now = time.time()

    if cache_key in _PULSE_CACHE:
        cached_time, cached_cards, cached_live, cached_msg = _PULSE_CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_cards, cached_live, cached_msg

    # Filter candidate universe
    candidates: list[tuple[str, dict[str, str]]] = []
    if filter_norm == "ALL":
        us_cands = [(t, m) for t, m in POPULAR_STOCKS_METADATA.items() if m["market"] == "US"]
        nse_cands = [(t, m) for t, m in POPULAR_STOCKS_METADATA.items() if m["market"] == "NSE"]
        # Interleave US and NSE candidates
        for u, n in zip(us_cands, nse_cands):
            candidates.extend([u, n])
        if len(us_cands) > len(nse_cands):
            candidates.extend(us_cands[len(nse_cands):])
        elif len(nse_cands) > len(us_cands):
            candidates.extend(nse_cands[len(us_cands):])
    elif filter_norm in ("US", "USA"):
        candidates = [(t, m) for t, m in POPULAR_STOCKS_METADATA.items() if m["market"] == "US"]
    elif filter_norm in ("NSE", "INDIA", "IND"):
        candidates = [(t, m) for t, m in POPULAR_STOCKS_METADATA.items() if m["market"] == "NSE"]

    if not candidates:
        return [], False, f"No tickers configured for market filter '{market_filter}'."

    cards: list[tuple[float, TrendingStockCard]] = []

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import yfinance as yf  # type: ignore

        def _fetch_single_pulse(item: tuple[str, dict[str, str]]) -> Optional[tuple[float, TrendingStockCard]]:
            ticker, meta = item
            yf_sym = meta.get("yf_symbol", ticker)
            try:
                t = yf.Ticker(yf_sym)
                fi = t.fast_info
                price = float(getattr(fi, "last_price", 0.0) or 0.0)
                if price <= 0.0:
                    return None

                prev_close = float(getattr(fi, "previous_close", 0.0) or 0.0)
                volume = int(getattr(fi, "last_volume", 0) or 0)
                raw_mc = float(getattr(fi, "market_cap", 0.0) or 0.0)

                currency = str(getattr(fi, "currency", meta["currency"]) or meta["currency"]).upper().strip()
                unit_scale = 1e7 if currency == "INR" else 1e6
                scaled_mc = round(raw_mc / unit_scale, 2) if raw_mc > 0.0 else 0.0
                fmt_mc = format_market_cap(scaled_mc, currency) if scaled_mc > 0.0 else "N/A"

                if prev_close > 0.0:
                    pct_change = round(((price - prev_close) / prev_close) * 100.0, 2)
                else:
                    pct_change = 0.0

                card = TrendingStockCard(
                    ticker=ticker,
                    company_name=meta["company_name"],
                    market=meta["market"],
                    current_price=round(price, 2),
                    currency=currency,
                    daily_change_pct=pct_change,
                    daily_volume=volume,
                    market_cap=round(scaled_mc, 2),
                    formatted_market_cap=fmt_mc,
                )
                activity_score = (price * volume) * (1.0 + abs(pct_change) / 100.0)
                return activity_score, card
            except Exception as item_err:
                logger.debug(f"Could not fetch pulse for {ticker}: {item_err}")
                return None

        # Take up to max_items + 4 candidates to keep network roundtrips minimal and fast
        eval_candidates = candidates[: max_items + 4]
        with ThreadPoolExecutor(max_workers=min(6, len(eval_candidates))) as executor:
            futures = [executor.submit(_fetch_single_pulse, c) for c in eval_candidates]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    cards.append(res)
    except Exception as exc:
        logger.warning(f"Live market pulse retrieval failed: {exc}")
        msg = "Offline mode: Live market data unavailable. EquiSight does not fabricate synthetic market movers."
        return [], False, msg

    if not cards:
        msg = "Offline mode: Live market data unavailable. EquiSight does not fabricate synthetic market movers."
        return [], False, msg

    # Rank dynamically by today's trading activity (highest activity first)
    cards.sort(key=lambda x: x[0], reverse=True)
    ranked_cards = [c for _, c in cards[:max_items]]

    status_msg = f"[Live Yahoo Finance] Market pulse active ({len(ranked_cards)} movers ranked by today's activity)."
    _PULSE_CACHE[cache_key] = (now, ranked_cards, True, status_msg)
    return ranked_cards, True, status_msg
