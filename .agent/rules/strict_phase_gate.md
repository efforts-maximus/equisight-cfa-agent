---
trigger: always_on
---

---
name: antigravity-2-phase-gate-and-subagent-governance
description: Enforces Antigravity 2.0 /plan approval gates, dynamic subagent isolation, schema-first contracts, Dual-Mode neuro-symbolic separation, zero placeholders, and mandatory pytest + Artifact evidence at every phase.
---

# ANTIGRAVITY 2.0 NON-NEGOTIABLE EXECUTION LAWS (GEMINI 3.8 FLASH)

1. **ONE PHASE PER TURN, STATE SYNC & MANDATORY ARTIFACT EVIDENCE:**
   - You may ONLY execute ONE Phase from `SPEC.md` per turn. Never write implementation files for future phases ahead of schedule.
   - Before planning or writing any code, read `SPEC.md` and `STATE.md` to verify current architecture contracts and completed milestones.
   - When invoked with `/plan`, generate an **Implementation Plan Artifact** for the requested Phase only and wait for explicit approval before modifying files.
   - When executing a Phase, you must run the phase-specific `pytest` suite, produce a **Walkthrough / Verification Artifact** containing the exact terminal output of `pytest -v` showing `0 failed`, update `STATE.md` with completed files and test counts, and **STOP IMMEDIATELY** to await the next phase instruction.

2. **DYNAMIC SUBAGENT ORCHESTRATION & SCHEMA HANDSHAKE (NO CONTEXT POLLUTION):**
   - To prevent Gemini 3.8 Flash context window degradation and corner-cutting, the Primary Orchestrator Agent MUST spawn focused **Dynamic Subagents** for independent modules within a Phase:
     - **Phase 1:** Distinct subagents for `src/schemas/models.py`, `src/data/market_gateway.py`, and `src/data/filing_rag_tool.py`.
     - **Phase 2:** Distinct subagents for `src/engines/fsa_engine.py`, `src/engines/quant_risk_engine.py`, and `src/engines/valuation_engine.py`.
     - **Phase 3:** Distinct subagents for `src/evals/evaluator.py` (with `telemetry.py`) and `src/agents/research_flow.py` (with `tools.py`, `agents.yaml`, and `tasks.yaml`).
     - **Phase 4:** Distinct subagents for `main.py` CLI and `app.py` 5-tab Streamlit dashboard.
   - **Mandatory Schema Handshake:** Every subagent in Phases 2, 3, and 4 MUST read `src/schemas/models.py` before writing code so all input/output types, field names, and validation bounds match 100%.
   - Each subagent must implement 100% of its assigned file with zero placeholders before returning control to the Primary Orchestrator for integration testing.

3. **ZERO PLACEHOLDERS, STUBS, OR `TODO` COMMENTS:**
   - Writing `pass`, `# TODO`, `...`, `# implement later`, abbreviated fixtures, or simplified dummy formulas is strictly forbidden.
   - Every Pydantic v2 validator, 2027 CFA Level 1 formula, SciPy `brentq` Reverse-DCF solver, 10,000-path vectorized NumPy Monte Carlo simulation, CrewAI YAML prompt, and Plotly chart must be 100% implemented and PEP-8 compliant.

4. **STRICT NEURO-SYMBOLIC SEPARATION & DUAL-MODE INVARIANCE:**
   - **Zero LLM Math:** LLMs must NEVER calculate financial ratios, WACC, DCF cash flows, regressions, or Monte Carlo percentiles.
   - **100% Deterministic Engines (`src/engines/`):** All math—Common-Size statements, 5-Way DuPont ROE, Working Capital CCC, Altman Z-Score, Forensic flags (`CFO/NI < 0.80`, `Accruals > 10%`, AR Bloat), OLS/Blume Beta, Sharpe/Sortino/Max Drawdown/95% VaR & CVaR, CAPM WACC, 2-Stage 5-Year FCFF DCF (Bear/Base/Bull), SciPy `brentq` Reverse-DCF, and 5x5 WACC/g sensitivity matrices—must execute in pure Python/NumPy/Pandas/SciPy.
   - **Bounded Agentic Layer (`src/agents/`):** Agents only synthesize qualitative MD&A filing evidence citing `[FILING-CHUNK-X]` IDs, conduct structured Dialectical Bull vs. Bear debates grounded in engine outputs, and emit bounded `DCFScenarioAssumptions` Pydantic objects (enforcing `terminal_growth_rate < wacc`).
   - **Dual-Mode Runtime:** Every pipeline flow must execute 100% offline in `--mode deterministic` using built-in 4-year financial fixtures (`TCS.NS`, `ZOMATO.NS`, `VARUNBEV.NS`, `NVDA` in INR Crores / USD Millions) and switch cleanly to live CrewAI multi-agent execution in `--mode llm` when API keys are present.

5. **MCP & `/browser` VERIFICATION:**
   - Do NOT invoke `sequential-thinking` or `github` MCPs; rely on native thinking, `/plan`, and `/boost`.
   - Proactively use `context7` MCP to verify current **Pydantic v2** (`BaseModel`, `ConfigDict`, `Field`, `@field_validator`, `@model_validator(mode="after")`, `model_dump()`) and **CrewAI** (`Flow`, `@start`, `@listen`, `@router`, `Agent`, `Task`, `Crew`) syntax—never use deprecated Pydantic v1 methods (`.dict()`, `@validator`).
   - In Phase 4, use the native `/browser` subagent (and/or `playwright` MCP) to launch `http://localhost:8501`, click through all 5 Streamlit tabs across tickers, verify zero runtime error boxes appear, and capture browser recordings/screenshots in `outputs/screenshots/` and as Artifacts.