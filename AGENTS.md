# EquiSight CFA Agent: Developer & Agent Context (AGENTS.md)

## Architecture Overview
EquiSight is an institutional-grade, neuro-symbolic equity research agent adhering strictly to the **2027 CFA Level 1 Curriculum** and **Dual-Mode Runtime Invariance**.

### Strict Neuro-Symbolic Separation
1. **100% Deterministic Financial & Quant Engines (`src/engines/`):**
   - Zero LLM math: LLMs never compute ratios, WACC, cash flows, beta regressions, or Monte Carlo percentiles.
   - `fsa_engine.py`: 5-Way DuPont ROE, Working Capital CCC, Altman Z-Score, and forensic earnings quality flags (`CFO/NI < 0.80`, `Accruals > 10%`, AR bloat).
   - `quant_risk_engine.py`: OLS and Blume Adjusted Beta, annualized volatility, Sharpe/Sortino ratios, 95% Historical VaR/CVaR, and 10,000-path vectorized NumPy Monte Carlo simulation.
   - `valuation_engine.py`: CAPM WACC, 2-Stage 5-Year FCFF DCF (Bear/Base/Bull), SciPy `brentq` Reverse-DCF implied revenue growth solver, 5x5 sensitivity matrices, and Football Field valuation ranges.

2. **Bounded Agentic Layer (`src/agents/`):**
   - Agents (`fsa_forensic_analyst`, `bull_equity_researcher`, `bear_short_seller`, `lead_valuation_pm`, `audit_compliance_critic`) only synthesize qualitative MD&A evidence citing `[FILING-CHUNK-X]` tags.
   - Andrew Ng Evaluator-Optimizer reflection auto-repair loop audits reports against 3-layer validation (Math Identity Invariance, Schema Completeness, Numerical Faithfulness).

3. **Dual-Mode Runtime Invariance:**
   - `--mode deterministic`: 100% deterministic CFA engine execution (uses live yfinance by default `use_live=True`, or audited fixtures when `use_live=False`) (`VARUNBEV.NS`, `TCS.NS`, `ZOMATO.NS`, `NVDA`).
   - `--mode llm`: Multi-agent CrewAI debate powered by `GEMINI_API_KEY` (Google Gemini), `GROQ_API_KEY`, or `OPENAI_API_KEY`, with strict zero-silent-fallback enforcement (raises LLMExecutionError if API keys are missing or invalid).

4. **Streamlit Financial Terminal (`app.py`):**
   - FactSet/Bloomberg-style dark financial dashboard with 5 interactive tabs.
   - Supports instant switching across benchmark tickers and arbitrary custom tickers (e.g. `INFY.NS`, `AAPL`, `MSFT`, `RELIANCE.NS`).
   - Live interactive What-If WACC and Terminal Growth Rate sliders with real-time recalculation.

## Setup & Running
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run tests
pytest -v

# Run CLI pipeline in deterministic or LLM mode
python main.py --ticker TCS.NS --mode deterministic
python main.py --ticker TCS.NS --mode llm

# Run Streamlit dashboard
streamlit run app.py --server.port 8501 --server.headless true

# Run Playwright browser verification
python tests/verify_ui_browser.py
python tests/verify_custom_ticker_browser.py
```


## Non-Negotiable Zero-Trust Anti-Cheat Laws
1. **Zero Synthetic Returns or Multiples in Live Mode:** `parse_yfinance_snapshot` must compute real date-aligned daily returns from `yfinance` 1Y history vs `^GSPC` / `^NSEI`?never call `generate_correlated_returns` in live mode.
2. **Zero Silent Exception Swallowing:** Never use bare `except Exception: pass`. Always log via `logger.warning()` / `logger.debug()` or raise `DataUnavailableError` / `LLMExecutionError`.
3. **Zero Ticker Hardcoding:** Never write ticker-specific `if/elif` branches in `src/engines/` or `src/agents/`.
