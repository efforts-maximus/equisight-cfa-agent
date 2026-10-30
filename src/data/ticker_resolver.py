"""Smart company name and auto-.NS ticker resolution module for EquiSight CFA Agent."""

import json
import logging
import urllib.parse
import urllib.request
from typing import Optional

from src.data.market_gateway import DataUnavailableError

logger = logging.getLogger(__name__)

# Bare Indian tickers that trade on NSE and should auto-resolve to .NS
BARE_INDIAN_TICKERS: dict[str, str] = {
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "VARUNBEV": "VARUNBEV.NS",
    "VBL": "VARUNBEV.NS",  # Varun Beverages canonical ticker in EquiSight fixtures
    "ZOMATO": "ZOMATO.NS",
    "ETERNAL": "ZOMATO.NS",  # Eternal / Zomato parent corporate entity
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

# Common company names mapped directly to their canonical tickers
COMMON_COMPANY_NAMES: dict[str, str] = {
    "tata consultancy": "TCS.NS",
    "tata consultancy services": "TCS.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "infosys limited": "INFY.NS",
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "varun beverages": "VARUNBEV.NS",
    "varun beverages limited": "VARUNBEV.NS",
    "vbl": "VARUNBEV.NS",
    "zomato": "ZOMATO.NS",
    "zomato limited": "ZOMATO.NS",
    "eternal": "ZOMATO.NS",
    "eternal limited": "ZOMATO.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfc bank limited": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "state bank of india": "SBIN.NS",
    "sbi": "SBIN.NS",
    "itc": "ITC.NS",
    "itc limited": "ITC.NS",
    "wipro": "WIPRO.NS",
    "wipro limited": "WIPRO.NS",
    "hcl tech": "HCLTECH.NS",
    "hcl technologies": "HCLTECH.NS",
    "larsen & toubro": "LT.NS",
    "l&t": "LT.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "tata motors": "TATAMOTORS.NS",
    "sun pharma": "SUNPHARMA.NS",
    "sun pharmaceutical": "SUNPHARMA.NS",
    "titan": "TITAN.NS",
    "titan company": "TITAN.NS",
    "nvidia": "NVDA",
    "nvidia corporation": "NVDA",
    "apple": "AAPL",
    "apple inc": "AAPL",
    "apple inc.": "AAPL",
    "microsoft": "MSFT",
    "microsoft corporation": "MSFT",
    "tesla": "TSLA",
    "tesla inc": "TSLA",
    "tesla inc.": "TSLA",
    "amazon": "AMZN",
    "amazon.com": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "ibm": "IBM",
    "international business machines": "IBM",
    "meta": "META",
    "facebook": "META",
    "broadcom": "AVGO",
    "berkshire": "BRK-B",
    "berkshire hathaway": "BRK-B",
    "netflix": "NFLX",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "walmart": "WMT",
}


def resolve_ticker_query(query: str, timeout_seconds: float = 5.0) -> str:
    """Resolves a user-supplied ticker query or company name to a canonical ticker symbol.
    
    1. Direct match on bare Indian tickers (TCS, INFY, RELIANCE, etc.) -> maps to .NS.
    2. Common corporate name dictionary resolution (e.g. 'Tata Consultancy' -> TCS.NS, 'Apple' -> AAPL).
    3. Preserves already qualified symbols (e.g. TCS.NS, NVDA, AAPL).
    4. Queries Yahoo Finance Search API for arbitrary company names.
    5. Validates direct ticker symbol format as fallback if search API yields no results.
    6. Raises DataUnavailableError if symbol/company cannot be resolved.
    """
    if not query or not query.strip():
        raise ValueError("Ticker search query cannot be empty.")

    clean_query = query.strip()
    upper_query = clean_query.upper()
    lower_query = clean_query.lower()

    # 1. Check exact bare Indian ticker mapping
    if upper_query in BARE_INDIAN_TICKERS:
        return BARE_INDIAN_TICKERS[upper_query]

    # 2. Check exact common company name mapping
    if lower_query in COMMON_COMPANY_NAMES:
        return COMMON_COMPANY_NAMES[lower_query]

    # 3. Check if lower query contains any of the canonical multi-word company names
    for name, sym in COMMON_COMPANY_NAMES.items():
        if " " in name and name in lower_query:
            return sym

    # 4. If query is already an explicit exchange-qualified ticker (e.g. .NS, .BO)
    if upper_query.endswith(".NS") or upper_query.endswith(".BO"):
        return upper_query

    # 5. Check if query is already a known clean ticker symbol (3-5 uppercase letters like NVDA, AAPL, MSFT)
    KNOWN_US_BENCHMARKS = {"NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "META", "IBM", "AVGO", "NFLX", "JPM", "WMT"}
    if upper_query in KNOWN_US_BENCHMARKS:
        return upper_query

    # 6. Fall back to live Yahoo Finance Search API
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(clean_query)}&quotesCount=5"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            quotes = data.get("quotes", [])

            # Prefer EQUITY or ETF quotes first
            for q in quotes:
                sym = q.get("symbol")
                qtype = q.get("quoteType", "")
                if sym and qtype in ("EQUITY", "ETF"):
                    return sym.upper()

            # Otherwise pick the first quote with a symbol
            if quotes and quotes[0].get("symbol"):
                return quotes[0]["symbol"].upper()

    except Exception as exc:
        logger.warning(f"Yahoo Finance search API failed for query '{clean_query}': {exc}")

    # 7. Fallback: If query looks like a standard exchange ticker (e.g. IBM, AMD, BRK-B), check yfinance quote
    import re
    if re.match(r"^[A-Z]{1,5}(-[A-Z]+|\.[A-Z]+)?$", upper_query):
        try:
            import yfinance as yf  # type: ignore
            t = yf.Ticker(upper_query)
            price = float(getattr(t.fast_info, "last_price", 0.0) or 0.0)
            if price > 0.0:
                return upper_query
        except Exception as exc:
            logger.debug(f"Fast quote fallback failed for '{upper_query}': {exc}")

    raise DataUnavailableError(
        f"Could not resolve ticker query for '{clean_query}': Symbol/company could not be resolved via Yahoo Finance. "
        "EquiSight does not fabricate synthetic financial data."
    )
