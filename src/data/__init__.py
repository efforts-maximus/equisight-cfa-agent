"""Data gateways, ticker resolver, and filing RAG tools for EquiSight."""

from src.data.market_gateway import (
    DataUnavailableError,
    format_market_cap,
    fetch_company_snapshot,
    fetch_peer_multiples,
    build_offline_snapshot,
    OFFLINE_FIXTURES,
    OFFLINE_PEER_FIXTURES,
)
from src.data.ticker_resolver import resolve_ticker_query
from src.data.filing_rag_tool import (
    load_filing_chunks,
    FilingRAGRetriever,
    query_filing_evidence,
)

__all__ = [
    "DataUnavailableError",
    "format_market_cap",
    "fetch_company_snapshot",
    "fetch_peer_multiples",
    "build_offline_snapshot",
    "OFFLINE_FIXTURES",
    "OFFLINE_PEER_FIXTURES",
    "resolve_ticker_query",
    "load_filing_chunks",
    "FilingRAGRetriever",
    "query_filing_evidence",
]
