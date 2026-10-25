# EQUISIGHT: AUTONOMOUS FINANCIAL STATEMENT & EQUITY RESEARCH AI AGENT — ANTIGRAVITY 2.0 SPECIFICATION (`SPEC.md`)

## 1. PROJECT ARCHITECTURE & DUAL-AUDIENCE ENGINEERING MANDATE
You are building **EquiSight**, an institutional-grade **Autonomous Financial Statement & Equity Research AI Agent** in Python, orchestrated via **Google Antigravity 2.0**.

This system must simultaneously satisfy two high-bar technical evaluations:

1. **AI Startup CTO / Principal AI Engineer Bar:**
   - **Neuro-Symbolic Architecture:** Deterministic Python/NumPy/Pandas/SciPy engines (`src/engines/`) execute 100% of financial calculations. Multi-agent LLMs (`src/agents/`) perform qualitative MD&A filing RAG, conduct an evidence-grounded **Dialectical Bull vs. Bear Debate**, and output bounded **Pydantic v2** scenario driver assumptions (`DCFScenarioAssumptions`).
   - **Event-Driven CrewAI Flow + YAML Configs:** Prompts are cleanly decoupled into `src/agents/config/agents.yaml` and `tasks.yaml` and orchestrated by a stateful **CrewAI Flow** (`@start`, `@listen`, `@router`) with an **Andrew Ng Reflection / Evaluator-Optimizer Audit Gate**.
   - **3-Layer Automated Evaluation Harness (`src/evals/evaluator.py`) & Telemetry Trace (`src/evals/telemetry.py`):** Every run produces a verifiable `EvalScorecard` (checking Math Identity Invariance, Schema/Section Completeness, and Zero-Hallucination Numerical Faithfulness) plus an execution trace logging latency and validation status per step.
   - **Resilient Dual-Mode Runtime (`--mode deterministic` & `--mode llm`):** Runs 100% deterministically out-of-the-box without API keys or internet access (using a fundamental rule-based synthesis engine and rich offline fixtures), and switches seamlessly to live Multi-Agent LLM execution when `GEMINI_API_KEY`, `GROQ_API_KEY`, or `OPENAI_API_KEY` is present in `.env`.

2. **Investment Banking (IB), Venture Capital (VC), & Equity Research Bar (2027 CFA Level 1 Curriculum):**
   - **2027 CFA FSA (LM 1–6):** Common-Size Income Statement & Balance Sheet, **5-Way DuPont ROE Decomposition**, Working Capital Cash Conversion Cycle (`DSO + DIO - DPO`), **Altman Z-Score** (bankruptcy risk), and **Forensic Earnings Quality Red Flags** (`CFO/NI` conversion, Balance Sheet Accruals Ratio, and Receivables-vs-Revenue growth spread).
   - **2027 CFA Equities (LM 3, 6, 7, 8, 11, 12):** CAPM Cost of Equity ($r_e$) using OLS & Blume-Adjusted Beta ($\beta_{\text{adj}}$), Cost of Debt ($r_d$), WACC, Disaggregated 5-Year **Free Cash Flow to the Firm (FCFF)** 2-Stage DCF across `Bear`, `Base`, and `Bull` scenarios, **SciPy Reverse-DCF Market-Implied Growth Solver**, $5 \times 5$ WACC vs. Terminal Growth Sensitivity Matrix, Relative Peer Comps (`P/E`, `EV/EBITDA`, `P/BV`, `EV/Sales`), Valuation Football Field chart, and a **2027 CFA LM 11 Sell-Side Initiating Coverage Report**.
   - **2027 CFA Quant (LM 9, 10, 11):** Benchmark OLS Linear Regression (`^NSEI` Nifty 50 for Indian `.NS`/`.BO` stocks; `^GSPC` S&P 500 for US stocks), Annualized Alpha ($\alpha$), $R^2$, Annualized Volatility, Downside Deviation, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, 95% Historical & Parametric Value-at-Risk (VaR & CVaR), and a vectorized **10,000-Path NumPy Monte Carlo DCF Simulation**.

---

## 2. TARGET REPOSITORY FILE TREE

```text
equisight-cfa-agent/
├── .agent/
│   └── rules/
│       └── strict_phase_gate.md
├── .env.example
├── requirements.txt
├── SPEC.md
├── STATE.md
├── README.md
├── main.py
├── app.py
├── data/
│   ├── cache/
│   └── sample_filings/
│       ├── TCS.NS_mda_notes.txt
│       ├── ZOMATO.NS_mda_notes.txt
│       ├── VARUNBEV.NS_mda_notes.txt
│       └── NVDA_mda_notes.txt
├── outputs/
│   └── screenshots/
├── src/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── models.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── market_gateway.py
│   │   └── filing_rag_tool.py
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── fsa_engine.py
│   │   ├── valuation_engine.py
│   │   └── quant_risk_engine.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── config/
│   │   │   ├── agents.yaml
│   │   │   └── tasks.yaml
│   │   ├── tools.py
│   │   └── research_flow.py
│   └── evals/
│       ├── __init__.py
│       ├── evaluator.py
│       └── telemetry.py
└── tests/
    ├── __init__.py
    ├── test_phase1_contracts_and_data.py
    ├── test_phase2_cfa_math_engines.py
    ├── test_phase3_agents_and_evals.py
    └── test_phase4_cli_and_ui.py
```

---

## 3. PHASE 0 & PHASE 1: BLUEPRINT SYNTHESIS, PYDANTIC V2 CONTRACTS & DATA GATEWAY


Dynamic Subagent Allocation for Phase 1
Orchestrator Task: Setup .env.example, STATE.md, requirements.txt, and install packages. Synthesize clean-room architectural patterns from TauricResearch/TradingAgents, OpenBB-finance/OpenBB, microsoft/qlib, and Fincept-Corporation/FinceptTerminal.
Subagent 1A (schemas-subagent): Use context7 MCP to implement src/schemas/models.py with all 12 Pydantic v2 data contracts and validators.
Subagent 1B (data-gateway-subagent): Implement src/data/market_gateway.py (live yfinance + local cache + 4-year realistic offline fixtures for TCS.NS, ZOMATO.NS, VARUNBEV.NS, and NVDA + peers) and src/data/filing_rag_tool.py + data/sample_filings/ with tagged [FILING-CHUNK-X] paragraphs.

Detailed Phase 1 Specifications (src/schemas/models.py — 12 Pydantic v2 Models)
HistoricalYearMetrics: fiscal_year: int, revenue: float, cogs: float, gross_profit: float, ebitda: float, depreciation_amortization: float, ebit: float, interest_expense: float, ebt: float, tax_expense: float, net_income: float, total_assets: float, current_assets: float, cash_and_equivalents: float, accounts_receivable: float, inventory: float, total_liabilities: float, current_liabilities: float, accounts_payable: float, total_debt: float, total_equity: float, retained_earnings: float, operating_cash_flow: float, capex: float, working_capital_change: float.
CompanyFinancialSnapshot: ticker: str, company_name: str, sector: str, industry: str, currency: str ("INR" or "USD"), unit_label: str ("INR Crores" or "USD Millions"), current_price: float, shares_outstanding: float, market_cap: float, enterprise_value: float, high_52week: float, low_52week: float, benchmark_ticker: str ("^NSEI" or "^GSPC"), risk_free_rate: float (0.068 India, 0.042 US), equity_risk_premium: float (0.060 India, 0.050 US), historical_years: list[HistoricalYearMetrics], daily_returns_stock: list[float], daily_returns_benchmark: list[float], business_summary: str.
FSARatioSummary:
Core Profitability & Growth: gross_margin: float, ebitda_margin: float, operating_margin: float, net_profit_margin: float, roa: float, roe: float, roic: float, revenue_cagr_3yr: float.
5-Way DuPont: tax_burden: float, interest_burden: float, ebit_margin: float, asset_turnover: float, financial_leverage: float, reconstructed_roe: float.
Working Capital: current_ratio: float, quick_ratio: float, dso: float, dio: float, dpo: float, cash_conversion_cycle: float.
Solvency & Forensic: debt_to_equity: float, net_debt_to_ebitda: float, interest_coverage: float, altman_z_score: float, altman_zone: str, cfo_to_net_income_ratio: float, accruals_to_assets_ratio: float, receivables_vs_revenue_growth_spread: float, accounting_red_flags: list[str], common_size_income_summary: dict[str, float].
QuantRiskSummary: ols_beta: float, adjusted_beta: float, alpha_annualized: float, r_squared: float, annualized_volatility: float, downside_deviation: float, sharpe_ratio: float, sortino_ratio: float, max_drawdown: float, var_95_daily_historical: float, var_95_daily_parametric: float, cvar_95_daily: float.
ScenarioDriverSet: scenario_name: str, probability_weight: float = Field(ge=0.05, le=0.90), revenue_cagr_5yr: list[float], target_ebit_margin_yr5: float = Field(ge=-0.30, le=0.70), effective_tax_rate: float = Field(ge=0.05, le=0.40), da_pct_revenue: float = Field(ge=0.005, le=0.20), capex_pct_revenue: float = Field(ge=0.005, le=0.35), nwc_pct_revenue_change: float = Field(ge=-0.10, le=0.20), terminal_growth_rate: float = Field(ge=-0.01, le=0.065), wacc_adjustment_bps: float = Field(ge=-250.0, le=300.0), rationale: str (validate len(revenue_cagr_5yr) == 5).
DCFScenarioAssumptions: bear: ScenarioDriverSet, base: ScenarioDriverSet, bull: ScenarioDriverSet, bull_thesis_points: list[str], bear_thesis_points: list[str], key_catalysts: list[str], key_risks: list[str], debate_synthesis_summary: str (validate probability sum == 1.0 ± 0.01, Bull avg growth >= Base >= Bear, and Bull Year-5 margin >= Base >= Bear).
SingleDCFResult: scenario_name: str, wacc: float, terminal_growth_rate: float, projected_revenues: list[float], projected_ebit: list[float], projected_fcff: list[float], pv_of_fcff: float, terminal_value: float, pv_of_terminal_value: float, enterprise_value: float, equity_value: float, intrinsic_value_per_share: float, upside_pct: float.
MonteCarloDCFOutput: num_simulations: int, mean_intrinsic_value: float, median_intrinsic_value: float, std_dev: float, p5: float, p25: float, p50: float, p75: float, p95: float, probability_of_upside: float, simulated_values_sample: list[float].
PeerCompMultiple: peer_ticker: str, peer_name: str, pe_ratio: float, ev_to_ebitda: float, pb_ratio: float, ev_to_sales: float, roe: float, revenue_growth: float.
ValuationSuiteOutput: cost_of_equity: float, pre_tax_cost_of_debt: float, after_tax_cost_of_debt: float, base_wacc: float, bear_dcf: SingleDCFResult, base_dcf: SingleDCFResult, bull_dcf: SingleDCFResult, probability_weighted_intrinsic_value: float, implied_upside_pct: float, recommendation: str ("BUY", "HOLD", or "SELL"), reverse_dcf_implied_revenue_cagr: float, market_expectations_gap_bps: float, wacc_sensitivity_axes: list[float], tgr_sensitivity_axes: list[float], sensitivity_matrix_prices: list[list[float]], peer_comps: list[PeerCompMultiple], peer_median_implied_price: float, football_field_ranges: dict[str, tuple[float, float, float]].
StepTelemetry: step_name: str, agent_or_engine: str, mode: str, latency_ms: float, status: str, validation_passed: bool, details: str.
EvalScorecard: math_invariance_score: float, schema_completeness_score: float, numerical_faithfulness_score: float, overall_eval_score: float, passed: bool, repair_attempts: int, flagged_discrepancies: list[str].

PHASE 1 VERIFICATION GATE (tests/test_phase1_contracts_and_data.py)
Write and run pytest tests/test_phase1_contracts_and_data.py -v:
Assert TCS.NS, ZOMATO.NS, VARUNBEV.NS, and NVDA load into valid CompanyFinancialSnapshot models with accurate currencies and benchmarks.
Assert DCFScenarioAssumptions raises ValidationError on invalid probability sums or inverted Bear/Bull ordering.
Assert filing_rag_tool.py retrieves tagged [FILING-CHUNK-X] passages.
Produce Phase 1 Verification Artifact, update STATE.md, and STOP.

---

## 4. PHASE 2: DETERMINISTIC 2027 CFA MATH, REVERSE-DCF & QUANT ENGINES (src/engines/)


Dynamic Subagent Allocation for Phase 2
Subagent 2A (fsa-engine-subagent): Implement src/engines/fsa_engine.py (5-Way DuPont ROE, Working Capital CCC, Altman Z-Score, and Forensic Earnings Quality checks).
Subagent 2B (quant-engine-subagent): Implement src/engines/quant_risk_engine.py (OLS Beta, Blume Adjusted Beta, Sharpe, Sortino, Max Drawdown, 95% VaR/CVaR, and vectorized 10,000-path NumPy Monte Carlo DCF simulation).
Subagent 2C (valuation-engine-subagent): Implement src/engines/valuation_engine.py (CAPM WACC, 2-Stage FCFF DCF for Bear/Base/Bull, SciPy brentq Reverse-DCF Implied Growth Solver, $5 \times 5$ WACC/g Sensitivity Matrix, Peer Comps, and Football Field ranges).

Detailed Phase 2 Formulas

5-Way DuPont Identity (fsa_engine.py):
$$\text{ROE} = \left(\frac{\text{NI}}{\text{EBT}}\right) \times \left(\frac{\text{EBT}}{\text{EBIT}}\right) \times \left(\frac{\text{EBIT}}{\text{Revenue}}\right) \times \left(\frac{\text{Revenue}}{\text{Avg Assets}}\right) \times \left(\frac{\text{Avg Assets}}{\text{Avg Equity}}\right)$$

2-Stage FCFF DCF & Bridge (valuation_engine.py):
$$\text{FCFF}_t = \text{EBIT}_t(1 - t) + \text{D\&A}_t - \text{CapEx}_t - \Delta\text{NWC}_t, \quad \text{TV}_5 = \frac{\text{FCFF}_5(1 + g)}{\text{WACC} - g}$$
$$\text{Implied Equity Value} = \sum_{t=1}^5 \frac{\text{FCFF}_t}{(1+\text{WACC})^t} + \frac{\text{TV}_5}{(1+\text{WACC})^5} + \text{Cash} - \text{Total Debt}$$
Strictly enforce $g \le \min(r_f, \text{WACC} - 0.015)$.

Reverse-DCF Solver (valuation_engine.py): Use scipy.optimize.brentq over uniform 5-year revenue CAGR bounds [-0.25, 0.60] to solve for the exact market-implied growth rate where Intrinsic Value per Share == Current Price.

PHASE 2 VERIFICATION GATE (tests/test_phase2_cfa_math_engines.py)
Write and run pytest tests/test_phase2_cfa_math_engines.py -v:
Assert 5-Way DuPont product equals reconstructed_roe within 1e-5 across all 4 benchmark tickers.
Assert bull_dcf.intrinsic_value_per_share > base_dcf.intrinsic_value_per_share > bear_dcf.intrinsic_value_per_share > 0.
Assert Enterprise-to-Equity Value bridge (EV + Cash - Debt == Equity Value) holds within 1e-2.
Assert Reverse-DCF implied CAGR plugged back into the DCF engine reproduces current_price within 1%.
Assert $5 \times 5$ sensitivity grid is strictly monotonic and 10,000 Monte Carlo runs finish in < 1.0s with ordered percentiles (P5 < P25 < P50 < P75 < P95).
Produce Phase 2 Verification Artifact, update STATE.md, and STOP.

---

## 5. PHASE 3: CREWAI YAML AGENTS, DIALECTICAL DEBATE FLOW & 3-LAYER EVAL HARNESS


Dynamic Subagent Allocation for Phase 3
Subagent 3A (eval-harness-subagent): Implement src/evals/telemetry.py and src/evals/evaluator.py (the 3-Layer Evaluation Harness checking Math Invariance, Schema/Section Completeness, and Zero-Hallucination Numerical Faithfulness).
Subagent 3B (crewai-flow-subagent): Use context7 MCP to implement src/agents/config/agents.yaml, src/agents/config/tasks.yaml, src/agents/tools.py, and src/agents/research_flow.py supporting both --mode deterministic and --mode llm with the @router Reflection & Auto-Repair gate.

PHASE 3 VERIFICATION GATE (tests/test_phase3_agents_and_evals.py)
Write and run pytest tests/test_phase3_agents_and_evals.py -v:
Assert agents.yaml and tasks.yaml parse cleanly with all 5 agents (fsa_forensic_analyst, bull_equity_researcher, bear_short_seller, lead_valuation_pm, audit_compliance_critic) and tasks.
Run EquiSightResearchFlow end-to-end on TCS.NS, ZOMATO.NS, and VARUNBEV.NS and assert eval_scorecard.overall_eval_score == 100.0 and passed is True.
Negative Hallucination Eval Test: Inject a corrupted target price and fake WACC into a generated Markdown report string and verify evaluator.py flags the exact hallucinated numbers and sets passed = False.
Produce Phase 3 Verification Artifact, update STATE.md, and STOP.

---

## 6. PHASE 4 & PHASE 5: CLI, STREAMLIT TERMINAL, /browser VERIFICATION & README.md


Dynamic Subagent Allocation for Phase 4 & 5
Subagent 4A (cli-and-docs-subagent): Implement main.py and generate README.md (with full Architecture Diagram, 2027 CFA Formulas, and Founder/Recruiter Deep-Dive Q&A).
Subagent 4B (streamlit-ui-subagent): Implement app.py (5-tab Streamlit Institutional Equity Research Terminal with interactive WACC/g override sliders and Plotly Football Field, Reverse-DCF callout, Sensitivity Heatmap, Monte Carlo Histogram, and 5-Way DuPont charts).
Browser Subagent (/browser / playwright): Launch streamlit run app.py --server.port 8501 --server.headless true, navigate to http://localhost:8501, click through all 5 tabs, verify zero .stException errors, and capture screenshots to outputs/screenshots/.

PHASE 4 & 5 VERIFICATION GATE (tests/test_phase4_cli_and_ui.py)
Run python main.py --ticker VARUNBEV.NS --mode deterministic, python main.py --ticker ZOMATO.NS --mode deterministic, and python main.py --ticker NVDA --mode deterministic.
Run the full test suite: pytest -v (all Phase 1, 2, 3, and 4 tests must pass 100%).
Verify outputs/screenshots/ contains the captured UI screenshots and STATE.md records 100% completion.
