"""Deterministic TF-IDF Retrieval-Augmented Generation (RAG) tool for 10-K / Annual Report MD&A filings."""

import os
import re
from pathlib import Path
from typing import Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default directory containing tagged MD&A notes
DEFAULT_FILINGS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample_filings"


def load_filing_chunks(ticker: str, filings_dir: Optional[Path] = None) -> list[dict[str, str]]:
    """Loads and parses tagged MD&A paragraphs from the local text fixture for the given ticker.
    
    Each returned item is a dictionary with:
    - 'chunk_id': e.g., '[FILING-CHUNK-1]'
    - 'title': e.g., 'Revenue and Geographic Market Dynamics'
    - 'text': the full body text of the chunk
    - 'raw': complete raw chunk string
    """
    directory = Path(filings_dir) if filings_dir else DEFAULT_FILINGS_DIR
    ticker_clean = ticker.strip().upper()
    filename = directory / f"{ticker_clean}_mda_notes.txt"

    if not filename.exists():
        filename_alt = directory / f"{ticker.strip()}_mda_notes.txt"
        if filename_alt.exists():
            filename = filename_alt
        else:
            raise FileNotFoundError(f"Filing notes fixture not found for ticker '{ticker}' at {filename}")

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # Chunks are demarcated by [FILING-CHUNK-X]
    pattern = r"(\[FILING-CHUNK-\d+\])\s*\n+(.*?)(?=(?:\[FILING-CHUNK-\d+\])|\Z)"
    matches = re.findall(pattern, content, flags=re.DOTALL)

    chunks = []
    for chunk_id, body in matches:
        cleaned_body = body.strip()
        # Extract optional section title if separated by colon on the first line
        first_line = cleaned_body.split("\n")[0]
        title = first_line.split(":")[0].strip() if ":" in first_line else chunk_id

        chunks.append({
            "chunk_id": chunk_id,
            "title": title,
            "text": cleaned_body,
            "raw": f"{chunk_id}\n{cleaned_body}"
        })

    return chunks


class FilingRAGRetriever:
    """TF-IDF vectorizer and cosine-similarity search engine over tagged filing passages."""

    def __init__(self, ticker: str, filings_dir: Optional[Path] = None):
        self.ticker = ticker
        self.filings_dir = filings_dir or DEFAULT_FILINGS_DIR
        self.chunks = load_filing_chunks(ticker, self.filings_dir)

        if not self.chunks:
            raise ValueError(f"No filing chunks could be parsed for ticker '{ticker}'")

        self.corpus = [chunk["text"] for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieves top_k relevant filing chunks for a given query, ranked by cosine similarity."""
        if not query.strip():
            return [
                {**self.chunks[i], "score": 0.0}
                for i in range(min(top_k, len(self.chunks)))
            ]

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Sort indices in descending order of similarity
        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "title": self.chunks[idx]["title"],
                "text": self.chunks[idx]["text"],
                "score": float(similarities[idx]),
                "raw": self.chunks[idx]["raw"],
            })

        return results


def query_filing_evidence(ticker: str, query: str, top_k: int = 3, filings_dir: Optional[Path] = None) -> str:
    """Convenience function querying filing evidence and returning formatted passage citations.
    
    Format:
    [FILING-CHUNK-X] (Score: 0.XXX):
    <passage body>
    """
    retriever = FilingRAGRetriever(ticker=ticker, filings_dir=filings_dir)
    ranked_chunks = retriever.retrieve(query=query, top_k=top_k)

    if not ranked_chunks:
        return f"No filing evidence found for query '{query}' on ticker {ticker}."

    formatted_passages = []
    for chunk in ranked_chunks:
        formatted = f"{chunk['chunk_id']} (Score: {chunk['score']:.4f}):\n{chunk['text']}"
        formatted_passages.append(formatted)

    return "\n\n---\n\n".join(formatted_passages)
