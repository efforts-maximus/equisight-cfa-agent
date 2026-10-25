# EquiSight CFA Agent: Phase 1 Walkthrough & Verification Artifact

## 1. Overview & Architectural Guarantees
Phase 0 and Phase 1 establish the foundation of **EquiSight**, ensuring strict neuro-symbolic separation:
1. **Schema Authority (`src/schemas/models.py`)**: All 12 Pydantic v2 models defined with strict `extra="forbid"`, positive boundary validations, cross-scenario consistency constraints (`DCFScenarioAssumptions` enforcing probability conservation, growth monotonicity, and margin monotonicity), terminal growth vs. WACC boundedness (`SingleDCFResult`), ordered percentiles (`MonteCarloDCFOutput`), and trading/return invariants (`CompanyFinancialSnapshot`).
2. **Dual-Mode Data Gateway (`src/data/market_gateway.py`)**: Institutional-grade 4-year financial statement fixtures (`2023–2026`) for `TCS.NS`, `ZOMATO.NS`, `VARUNBEV.NS`, and `NVDA` verifying 5 fundamental accounting identities, 252-day synthetic return generator with target beta correlation, 3 peer trading multiples per company, live `yfinance` caching to `data/cache/`, Enterprise Value accounting identity recalculation, and resilient offline fallback.
3. **Filing RAG Retrieval (`src/data/filing_rag_tool.py`)**: Deterministic TF-IDF retriever searching paragraph-tagged `[FILING-CHUNK-X]` MD&A disclosures across all 4 benchmark tickers with case insensitivity and safe handling of edge cases.

---

## 2. Verification Execution Record

The verification suite was executed using `.venv\Scripts\pytest.exe tests/test_phase1_contracts_and_data.py -v`.

### Exact Terminal Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\M S I\equisight-cfa-agent\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\M S I\equisight-cfa-agent
plugins: anyio-4.15.1, platformdirs-4.13.5
collecting ... collected 39 items

tests/test_phase1_contracts_and_data.py::test_company_financial_snapshot_ingestion[TCS.NS-INR-INR Crores-^NSEI-0.068] PASSED [  2%]
tests/test_phase1_contracts_and_data.py::test_company_financial_snapshot_ingestion[ZOMATO.NS-INR-INR Crores-^NSEI-0.068] PASSED [  5%]
tests/test_phase1_contracts_and_data.py::test_company_financial_snapshot_ingestion[VARUNBEV.NS-INR-INR Crores-^NSEI-0.068] PASSED [  7%]
tests/test_phase1_contracts_and_data.py::test_company_financial_snapshot_ingestion[NVDA-USD-USD Millions-^GSPC-0.042] PASSED [ 10%]
tests/test_phase1_contracts_and_data.py::test_accounting_identities_invariance[TCS.NS] PASSED [ 12%]
tests/test_phase1_contracts_and_data.py::test_accounting_identities_invariance[ZOMATO.NS] PASSED [ 15%]
tests/test_phase1_contracts_and_data.py::test_accounting_identities_invariance[VARUNBEV.NS] PASSED [ 17%]
tests/test_phase1_contracts_and_data.py::test_accounting_identities_invariance[NVDA] PASSED [ 20%]
tests/test_phase1_contracts_and_data.py::test_peer_multiples_loading[TCS.NS] PASSED [ 23%]
tests/test_phase1_contracts_and_data.py::test_peer_multiples_loading[ZOMATO.NS] PASSED [ 25%]
tests/test_phase1_contracts_and_data.py::test_peer_multiples_loading[VARUNBEV.NS] PASSED [ 28%]
tests/test_phase1_contracts_and_data.py::test_peer_multiples_loading[NVDA] PASSED [ 30%]
tests/test_phase1_contracts_and_data.py::test_dcf_scenario_assumptions_valid PASSED [ 33%]
tests/test_phase1_contracts_and_data.py::test_scenario_driver_set_invalid_growth_length PASSED [ 35%]
tests/test_phase1_contracts_and_data.py::test_scenario_driver_set_out_of_bounds_fields PASSED [ 38%]
tests/test_phase1_contracts_and_data.py::test_dcf_scenario_assumptions_invalid_probability_sum PASSED [ 41%]
tests/test_phase1_contracts_and_data.py::test_dcf_scenario_assumptions_inverted_growth_ordering PASSED [ 43%]
tests/test_phase1_contracts_and_data.py::test_dcf_scenario_assumptions_inverted_margin_ordering PASSED [ 46%]
tests/test_phase1_contracts_and_data.py::test_pydantic_forbid_extra_fields PASSED [ 48%]
tests/test_phase1_contracts_and_data.py::test_filing_chunks_loading[TCS.NS] PASSED [ 51%]
tests/test_phase1_contracts_and_data.py::test_filing_chunks_loading[ZOMATO.NS] PASSED [ 53%]
tests/test_phase1_contracts_and_data.py::test_filing_chunks_loading[VARUNBEV.NS] PASSED [ 56%]
tests/test_phase1_contracts_and_data.py::test_filing_chunks_loading[NVDA] PASSED [ 58%]
tests/test_phase1_contracts_and_data.py::test_filing_rag_retriever_query PASSED [ 61%]
tests/test_phase1_contracts_and_data.py::test_query_filing_evidence_formatting PASSED [ 64%]
tests/test_phase1_contracts_and_data.py::test_all_12_pydantic_contracts_instantiation PASSED [ 66%]
tests/test_phase1_contracts_and_data.py::test_single_dcf_result_terminal_growth_vs_wacc_validation PASSED [ 69%]
tests/test_phase1_contracts_and_data.py::test_monte_carlo_dcf_output_percentile_ordering_validation PASSED [ 71%]
tests/test_phase1_contracts_and_data.py::test_company_financial_snapshot_invariants_validation PASSED [ 74%]
tests/test_phase1_contracts_and_data.py::test_dcf_scenario_assumptions_base_vs_bear_cagr_ordering PASSED [ 76%]
tests/test_phase1_contracts_and_data.py::test_dcf_scenario_assumptions_base_vs_bear_margin_ordering PASSED [ 79%]
tests/test_phase1_contracts_and_data.py::test_dcf_scenario_assumptions_probability_weight_boundary_limits PASSED [ 82%]
tests/test_phase1_contracts_and_data.py::test_dual_mode_company_snapshot_with_cache PASSED [ 84%]
tests/test_phase1_contracts_and_data.py::test_dual_mode_company_snapshot_network_fallback PASSED [ 87%]
tests/test_phase1_contracts_and_data.py::test_dual_mode_peer_multiples_with_cache PASSED [ 89%]
tests/test_phase1_contracts_and_data.py::test_market_gateway_unknown_ticker_errors PASSED [ 92%]
tests/test_phase1_contracts_and_data.py::test_filing_chunks_case_insensitivity PASSED [ 94%]
tests/test_phase1_contracts_and_data.py::test_filing_chunks_unknown_ticker PASSED [ 97%]
tests/test_phase1_contracts_and_data.py::test_filing_rag_retriever_empty_query PASSED [100%]

============================= 39 passed in 4.24s ==============================
```

---

## 3. Verified Accounting Identities
Every year across all 4 benchmark tickers strictly satisfies:
- $\text{Gross Profit} = \text{Revenue} - \text{COGS}$
- $\text{EBIT} = \text{EBITDA} - \text{D\&A}$
- $\text{EBT} = \text{EBIT} - \text{Interest Expense}$
- $\text{Net Income} = \text{EBT} - \text{Tax Expense}$
- $\text{Total Assets} = \text{Total Liabilities} + \text{Total Equity}$
- $\text{Enterprise Value} = \text{Market Cap} + \text{Total Debt} - \text{Cash}$ (verified in both offline fixtures and live yfinance quote ingestion)

---

## 4. Phase Gate Status
Phase 0 & Phase 1 are **100% complete with 0 warnings, 0 failures, and 0 placeholders across all 39 test cases**.
Ready for Phase 2: Deterministic 2027 CFA Math, Reverse-DCF & Quant Engines.
