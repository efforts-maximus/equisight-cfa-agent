# EquiSight: Institutional CFA Level 1 Neuro-Symbolic Equity Research Agent

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2.10-green.svg)](https://docs.pydantic.dev/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.42-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/plotly-5.24-brightgreen.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **EquiSight** is a dual-mode, institutional-grade equity research pipeline and Bloomberg/FactSet-style terminal. It marries **100% deterministic financial mathematics** (2027 CFA Level 1 curriculum, SciPy `brentq` Reverse-DCF, and 10,000-path vectorized NumPy Monte Carlo simulations) with a **bounded CrewAI multi-agent dialectical debate flow** and an **Andrew Ng Evaluator-Optimizer auto-repair reflection gate**.

---

## 1. System Architecture & Neuro-Symbolic Boundary

Modern LLMs are prone to arithmetic hallucinations, stochastic rounding drift, and non-reproducible valuation outputs. **EquiSight enforces strict Neuro-Symbolic Separation**:
- **Deterministic Symbolic Layer (`src/engines/`)**: 100% pure Python/NumPy/SciPy. LLMs are strictly forbidden from performing financial math, calculating WACC, discounting cash flows, computing regression betas, or solving reverse-DCF root equations.
- **Bounded Agentic Layer (`src/agents/`)**: 5 specialized CrewAI agent personas conduct structured Bull vs. Bear dialectical debates, extract qualitative filing disclosures citing verifiable `[FILING-CHUNK-X]` tags, and synthesize bounded Pydantic v2 DCF driver sets.
- **Evaluation & Auto-Repair Gate (`src/evals/`)**: A 3-layer automated evaluation harness auditing Math Invariance, Schema Completeness, and Numerical Faithfulness. If an LLM hallucinates a number or violates accounting identities, an Andrew Ng reflection loop catches and repairs the report before publication.

```mermaid
flowchart TD
    subgraph Data Layer ["Data Ingestion & Grounding (Dual-Mode)"]
        RawData[Market Gateway & 4-Yr Audited Statements]
        RAG[TF-IDF Filing RAG Tool: [FILING-CHUNK-X]]
    end

    subgraph Symbolic Engines ["100% Deterministic Financial Engines (Zero LLM Math)"]
        FSA[FSA Engine: 5-Way DuPont, CCC, Altman Z, Forensics]
        Quant[Quant Engine: Blume Beta, Sharpe, Sortino, 95% VaR/CVaR]
        Valuation[Valuation Engine: 2-Stage DCF, SciPy brentq Reverse-DCF, Peer Comps]
        MC[Vectorized 10,000-Path Monte Carlo Simulation]
    end

    subgraph Agentic Synthesis ["Bounded Agentic Layer (CrewAI + YAML)"]
        Analyst[FSA Forensic Analyst]
        Bull[Bull Equity Researcher]
        Bear[Bear Short Seller]
        PM[Lead Valuation PM]
        Critic[Audit & Compliance Critic]
    end

    subgraph Reflection Gate ["Andrew Ng Evaluator-Optimizer Reflection Gate"]
        Eval[3-Layer Eval Harness: Math, Schema, Faithfulness]
        Repair{Passed 100/100?}
        SurgicalFix[Reflection Auto-Repair Engine]
    end

    subgraph Presentation ["Presentation Layer"]
        CLI[Terminal CLI: main.py]
        UI[5-Tab Bloomberg Terminal: app.py]
        Artifacts[outputs/ Research Report & Audit Bundle]
    end

    RawData --> FSA
    RawData --> Quant
    RAG --> Bull
    RAG --> Bear
    FSA --> Analyst
    Quant --> Analyst

    Analyst --> Bull & Bear
    Bull & Bear --> PM
    PM --> Valuation
    PM --> MC

    Valuation --> Eval
    FSA --> Eval
    Quant --> Eval

    Eval --> Repair
    Repair -- "Violations Flagged" --> SurgicalFix --> Eval
    Repair -- "Passed" --> CLI & UI & Artifacts
```

---

## 2. Clean-Room Open-Source Inspirations

EquiSight borrows best-in-class architectural patterns from leading quantitative and multi-agent financial systems, reimplemented under a unified Pydantic v2 schema contract:

1. **`TradingAgents` (Multi-Agent Dialectical Debate)**:
   - *Pattern Adapted*: Structured adversarial debate between Bull and Bear specialists under the supervision of a Lead Portfolio Manager.
   - *EquiSight Clean-Room Distinction*: Bounded parameter spaces (`DCFScenarioAssumptions`) preventing unconstrained narrative drift; every qualitative argument must cite a `[FILING-CHUNK-X]` disclosure anchor.
2. **`OpenBB` (Modular Command Line & Extensible Data Architecture)**:
   - *Pattern Adapted*: Dual-mode gateway architecture separating live market feeds from local fixtures and deterministic testing caches.
   - *EquiSight Clean-Room Distinction*: Strict preservation of the Enterprise Value balance sheet identity ($\text{EV} \equiv \text{Market Cap} + \text{Debt} - \text{Cash}$) across both live yfinance data and offline test fixtures.
3. **`Qlib` (Deterministic Quantitative Risk & Return Series)**:
   - *Pattern Adapted*: Strict mathematical separation of factor models, regression betas, and downside volatility metrics.
   - *EquiSight Clean-Room Distinction*: Blume regression beta adjustment combined with full analytical Sortino downside deviation and 95% Parametric/Historical VaR/CVaR.
4. **`FinceptTerminal` (Bloomberg-Grade Dense Financial UX)**:
   - *Pattern Adapted*: High-density dark financial terminal UI with monospace typography, KPI status cards, and interactive sensitivity heatmaps.
   - *EquiSight Clean-Room Distinction*: Real-time interactive What-If WACC/g sliders with instant Plotly valuation football field recalibration.

---

## 3. 2027 CFA Level 1 Mathematical Formulations

### A. 5-Way DuPont Return on Equity Decomposition
Decomposes headline ROE into five operational, financing, and tax drivers:
$$\text{ROE} = \left(\frac{\text{Net Income}}{\text{EBT}}\right) \times \left(\frac{\text{EBT}}{\text{EBIT}}\right) \times \left(\frac{\text{EBIT}}{\text{Revenue}}\right) \times \left(\frac{\text{Revenue}}{\text{Average Assets}}\right) \times \left(\frac{\text{Average Assets}}{\text{Average Equity}}\right)$$
$$\text{ROE} = \text{Tax Burden} \times \text{Interest Burden} \times \text{EBIT Margin} \times \text{Asset Turnover} \times \text{Financial Leverage}$$

### B. Working Capital Cash Conversion Cycle (CCC)
$$\text{DSO} = \frac{\text{Accounts Receivable}}{\text{Revenue}} \times 365, \quad \text{DIO} = \frac{\text{Inventory}}{\text{COGS}} \times 365, \quad \text{DPO} = \frac{\text{Accounts Payable}}{\text{COGS}} \times 365$$
$$\text{CCC} = \text{DSO} + \text{DIO} - \text{DPO}$$

### C. Altman Z-Score Solvency Classification
$$\text{Z} = 1.2 X_1 + 1.4 X_2 + 3.3 X_3 + 0.6 X_4 + 1.0 X_5$$
- $X_1 = \frac{\text{Working Capital}}{\text{Total Assets}}$
- $X_2 = \frac{\text{Retained Earnings}}{\text{Total Assets}}$
- $X_3 = \frac{\text{EBIT}}{\text{Total Assets}}$
- $X_4 = \frac{\text{Market Value of Equity}}{\text{Total Liabilities}}$
- $X_5 = \frac{\text{Revenue}}{\text{Total Assets}}$
- **Classification**: $\text{Safe Zone} > 2.99$, $\text{Grey Zone} \in [1.81, 2.99]$, $\text{Distress Zone} < 1.81$.

### D. CAPM Cost of Equity & Weighted Average Cost of Capital (WACC)
$$\beta_{\text{adj}} = \frac{2}{3}\beta_{\text{OLS}} + \frac{1}{3}(1.0)$$
$$K_e = R_f + \beta_{\text{adj}} \times \text{ERP}, \quad K_{d,\text{after-tax}} = \frac{\text{Interest Expense}}{\text{Total Debt}} \times (1 - t)$$
$$\text{WACC} = \left(\frac{E}{D + E}\right) K_e + \left(\frac{D}{D + E}\right) K_{d,\text{after-tax}}$$

### E. 2-Stage 5-Year Free Cash Flow to Firm (FCFF) DCF
$$\text{FCFF}_t = \text{EBIT}_t \times (1 - t) + \text{D\&A}_t - \text{CapEx}_t - \Delta\text{NWC}_t$$
$$\text{PV}(\text{FCFF}) = \sum_{t=1}^5 \frac{\text{FCFF}_t}{(1 + \text{WACC})^t}, \quad \text{TV}_5 = \frac{\text{FCFF}_5 \times (1 + g)}{\text{WACC} - g}$$
$$\text{Enterprise Value (EV)} = \text{PV}(\text{FCFF}) + \frac{\text{TV}_5}{(1 + \text{WACC})^5}$$
$$\text{Implied Equity Value} = \text{EV} + \text{Cash} - \text{Total Debt}$$
$$\text{Intrinsic Price Per Share} = \frac{\text{Implied Equity Value}}{\text{Shares Outstanding}}$$

### F. SciPy `brentq` Reverse-DCF Root Solver
Finds the market-implied 5-year revenue CAGR $g_{\text{rev}}^*$ that sets the intrinsic price equal to the current traded price $P_0$:
$$f(g_{\text{rev}}) = P_{\text{DCF}}(g_{\text{rev}}) - P_0 = 0$$
Using Brent's method with multi-bracket safety intervals $[-0.40, +1.50]$, guaranteeing rapid quadratic convergence without grid search approximations.

### G. Vectorized 10,000-Path Monte Carlo Simulation
Simulates 10,000 joint stochastic draws:
$$g_{\text{sim}} \sim \mathcal{N}(\mu_g, \sigma_g^2), \quad m_{\text{sim}} \sim \mathcal{N}(\mu_m, \sigma_m^2), \quad WACC_{\text{sim}} \sim \mathcal{N}(\mu_W, \sigma_W^2)$$
Computes the empirical CDF, percentile values ($P_5, P_{25}, P_{50}, P_{75}, P_{95}$), and empirical probability of upside:
$$\mathbb{P}(\text{Upside}) = \frac{1}{N}\sum_{i=1}^{10,000} \mathbf{1}_{\{P_{\text{sim}, i} > P_0\}}$$

---

## 4. Quickstart Guide

### Environment Setup
```powershell
# 1. Clone repository and navigate to root
cd equisight-cfa-agent

# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### CLI Execution
Run the production CLI runner in `--mode deterministic` (offline, instant) or `--mode llm` (live multi-agent):
```powershell
# Analyze Varun Beverages Limited (India Consumer Defensive)
python main.py --ticker VARUNBEV.NS --mode deterministic

# Analyze Zomato Limited (India Tech / Platform)
python main.py --ticker ZOMATO.NS --mode deterministic

# Analyze NVIDIA Corporation (US AI Hardware)
python main.py --ticker NVDA --mode deterministic
```
Output artifacts are saved to `outputs/`:
- `outputs/<TICKER>_research_report.md`: Complete 6-section 2027 CFA Level 1 Initiating Coverage Report.
- `outputs/<TICKER>_audit_bundle.json`: Full Pydantic v2 serialized audit bundle with telemetry.

### Streamlit Institutional Terminal
Launch the FactSet/Bloomberg-style 5-tab terminal dashboard:
```powershell
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser:
- **Tab 1: Executive Summary & Football Field**: KPI cards and 5-methodology valuation comparison.
- **Tab 2: 2027 CFA FSA & Forensic Quality**: 5-Way DuPont waterfall, common-size tables, CCC cards, and Altman Z gauge.
- **Tab 3: DCF Scenarios, Reverse-DCF & Sensitivity**: 5-year cash flow projections, SciPy market expectations gap, peer comps, and $5 \times 5$ sensitivity heatmap.
- **Tab 4: Quantitative Risk & Monte Carlo**: Return statistics, Blume beta, 95% VaR/CVaR, and 10,000-path intrinsic value distribution.
- **Tab 5: Dialectical Bull/Bear Debate & Audit Trace**: Side-by-side debate cards citing `[FILING-CHUNK-X]`, Markdown report viewer, download buttons, 3-layer eval scorecard, and telemetry trace.

---

## 5. Verification & Test Suite

EquiSight features a comprehensive test suite across all 4 project phases:
```powershell
# Run the entire test suite (157 tests across all 4 phases)
pytest -v

# Run Phase 4 CLI & UI tests specifically (16 tests)
pytest tests/test_phase4_cli_and_ui.py -v
```

---

## 6. Founder & Recruiter Technical Deep-Dive Q&A

### Q1: Why pure Python/NumPy/SciPy for financial math instead of LLM calculation?
**A:** LLMs are autoregressive token predictors, not algebraic solvers. Prompting an LLM to calculate a 5-Way DuPont decomposition or a 5-year discrete FCFF DCF produces stochastic rounding errors, inconsistent tax burdens, and arithmetic hallucinations. In institutional investment banking and quantitative asset management, calculations must be 100% deterministic, auditable, and reproducible. EquiSight confines LLMs to qualitative synthesis, dialectical debate, and filing evidence extraction, delegating all math to deterministic Python engines.

### Q2: Why SciPy `brentq` multi-bracket root solver instead of naive grid search?
**A:** Naive grid search over a range of revenue CAGRs is computationally inefficient, discretely quantized, and prone to boundary clipping. SciPy's `brentq` combines root bracketing, bisection, and inverse quadratic interpolation to find the exact market-implied revenue growth rate matching current stock prices to machine precision ($< 10^{-6}$) in sub-millisecond execution time. EquiSight wraps `brentq` with dynamic multi-bracket fallback intervals ($[-0.40, +1.50]$) to handle hyper-growth and distressed balance sheets gracefully.

### Q3: Why Pydantic v2 `extra="forbid"` and `@model_validator(mode="after")`?
**A:** Pydantic v2 provides compiled Rust-speed validation. We set `extra="forbid"` on all 12 core data contracts to guarantee that neither LLMs nor external APIs inject uninspected fields or silent schema mutations. Post-initialization `@model_validator(mode="after")` decorators enforce fundamental accounting and financial invariants—such as ensuring terminal growth rate $g < \text{WACC}$, 52-week high $\ge$ 52-week low, probability weights summing to $1.0 \pm 0.01$, and scenario CAGRs obeying strict monotonic order ($\text{Bull} \ge \text{Base} \ge \text{Bear}$).

### Q4: How does the 3-Layer Automated Evaluation Harness guarantee zero numerical hallucinations?
**A:** EquiSight evaluates every generated report against three deterministic criteria:
1. **Layer 1: Math Invariance**: Verifies that the DuPont product equals reported ROE, the Working Capital CCC equals $\text{DSO} + \text{DIO} - \text{DPO}$, and the Enterprise Value bridge strictly holds.
2. **Layer 2: Schema Completeness**: Asserts the presence of all 6 mandatory 2027 CFA Level 1 sections and authentic `[FILING-CHUNK-X]` citations.
3. **Layer 3: Numerical Faithfulness**: Extracts every reported metric (Current Price, Target Price, Base WACC, Implied Upside %, DuPont ROE) via regular expressions and compares them against the underlying engine outputs within a strict $0.50\%$ tolerance. Any divergence is flagged as a hallucination.

### Q5: How does the Andrew Ng Reflection / Evaluator-Optimizer auto-repair gate work?
**A:** Instead of failing or blindly accepting a flawed report, the CrewAI `EquiSightResearchFlow` routes drafts through an Evaluator-Optimizer reflection gate. If Layer 1, 2, or 3 flags a discrepancy (e.g., a hallucinated target price or a dropped section), the `repair_cfa_report` module performs targeted regex substitution to restore mathematical fidelity while preserving the qualitative narrative synthesized by the agents. If the report fails after 2 repair cycles, the pipeline terminates in a safe `FAILED` state rather than publishing false data.

---

## 7. License
EquiSight is licensed under the [MIT License](LICENSE).
