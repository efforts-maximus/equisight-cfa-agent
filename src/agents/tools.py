"""CrewAI tool wrappers for EquiSight CFA Agent."""

import logging
from pathlib import Path
from typing import Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from src.data.filing_rag_tool import query_filing_evidence
from src.data.ticker_resolver import resolve_ticker_query

logger = logging.getLogger(__name__)


class RetrieveFilingEvidenceInput(BaseModel):
    """Input schema for RetrieveFilingEvidenceTool."""
    ticker: str = Field(description="Equity ticker symbol")
    query: str = Field(description="Search terms or topical queries (e.g. 'cloud margin inflation risks')")
    top_k: int = Field(default=3, description="Number of top passages to retrieve")


class RetrieveFilingEvidenceTool(BaseTool):
    """CrewAI tool retrieving tagged [FILING-CHUNK-X] MD&A disclosures from annual reports."""

    name: str = "retrieve_filing_evidence"
    description: str = (
        "Retrieves authoritative [FILING-CHUNK-X] MD&A disclosure passages from company regulatory "
        "filings for a given ticker and query. Preserves citation tags for grounding."
    )
    args_schema: Type[BaseModel] = RetrieveFilingEvidenceInput

    def _run(self, ticker: str, query: str, top_k: int = 3) -> str:
        """Executes TF-IDF RAG retrieval against local regulatory filings."""
        clean_ticker = ticker.strip() if ticker else ""
        try:
            clean_ticker = resolve_ticker_query(clean_ticker)
        except Exception as exc:
            logger.debug(f"Ticker resolution fallback for '{clean_ticker}': {exc}")
        return query_filing_evidence(ticker=clean_ticker, query=query, top_k=top_k)
