"""3-Layer Automated Evaluation Harness for EquiSight CFA Agent.

Verifies:
- Layer 1: Deterministic 2027 CFA Level 1 Math Identity Invariance
- Layer 2: Pydantic Schema Contracts & CFA Report Section Completeness
- Layer 3: Zero-Hallucination Numerical Faithfulness against Deterministic Engine Ground Truth
"""

import math
import re
from typing import Optional

from src.schemas.models import (
    CompanyFinancialSnapshot,
    DCFScenarioAssumptions,
    EvalScorecard,
    FSARatioSummary,
    QuantRiskSummary,
    ValuationSuiteOutput,
)

MANDATORY_CFA_SECTIONS = [
    ("1. Executive Summary & Recommendation", r"#+\s*1\.\s*Executive Summary & Recommendation"),
    ("2. Financial Statement & DuPont Analysis", r"#+\s*2\.\s*Financial Statement & DuPont Analysis"),
    ("3. Forensic Earnings Quality & Solvency", r"#+\s*3\.\s*Forensic Earnings Quality & Solvency"),
    ("4. Dialectical Bull vs. Bear Debate", r"#+\s*4\.\s*Dialectical Bull vs\.?\s*Bear Debate"),
    ("5. DCF & Reverse-DCF Valuation", r"#+\s*5\.\s*DCF & Reverse-DCF Valuation"),
    ("6. Quantitative Risk & Monte Carlo Simulation", r"#+\s*6\.\s*Quantitative Risk & Monte Carlo Simulation"),
]


def _clean_num_str(raw: str) -> float:
    """Converts a raw numeric string with commas, currency symbols, and spaces to float."""
    cleaned = re.sub(r"[^\d\.\+\-]", "", raw.strip())
    if not cleaned or cleaned in ("+", "-", ".", "+.", "-."):
        raise ValueError(f"Cannot parse numeric string: {raw!r}")
    return float(cleaned)


def extract_current_price(markdown: str) -> Optional[float]:
    """Extracts Current Price from Markdown text or table."""
    if not markdown:
        return None
    patterns = [
        r"(?:Current\s+(?:Market\s+)?Price|Current\s+Trading\s+Price)[^\n\d]*?([\$₹]|INR|USD)?\s*([0-9,]+\.?[0-9]*)",
        r"\|\s*Current\s+(?:Market\s+)?Price\s*\|\s*([\$₹]|INR|USD)?\s*([0-9,]+\.?[0-9]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, markdown, flags=re.IGNORECASE)
        if match:
            try:
                return _clean_num_str(match.group(2))
            except (ValueError, IndexError):
                continue
    return None


def extract_base_wacc(markdown: str) -> Optional[float]:
    """Extracts Base WACC discount rate from Markdown text or table."""
    if not markdown:
        return None
    patterns = [
        r"(?:Base\s+WACC|Discount\s+Rate\s*\(WACC\)|Base\s+Discount\s+Rate)[^\n\d]*?([0-9,]+\.?[0-9]*)\s*%",
        r"\|\s*Base\s+WACC\s*\|\s*([0-9,]+\.?[0-9]*)\s*%?",
        r"(?:Base\s+WACC|WACC)[^\n\d]*?([0-9,]+\.?[0-9]*)\s*%",
    ]
    for pattern in patterns:
        match = re.search(pattern, markdown, flags=re.IGNORECASE)
        if match:
            try:
                has_pct = "%" in match.group(0)
                val = _clean_num_str(match.group(1))
                if has_pct or val > 1.0:
                    val /= 100.0
                return val
            except (ValueError, IndexError):
                continue
    return None


def extract_target_price(markdown: str) -> Optional[float]:
    """Extracts Target Price / Probability-Weighted Intrinsic Value from Markdown text or table."""
    if not markdown:
        return None
    patterns = [
        r"(?:Target\s+Price|Probability-Weighted\s+Intrinsic\s+Value|Weighted\s+Intrinsic\s+Value)[^\n\d]*?([\$₹]|INR|USD)?\s*([0-9,]+\.?[0-9]*)",
        r"\|\s*(?:Target\s+Price|Probability-Weighted\s+Intrinsic\s+Value|Weighted\s+Intrinsic\s+Value)\s*\|\s*([\$₹]|INR|USD)?\s*([0-9,]+\.?[0-9]*)",
        r"(?:Intrinsic\s+Value\s*\(Base\)|Target\s+Price)[^\n\d]*?([\$₹]|INR|USD)?\s*([0-9,]+\.?[0-9]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, markdown, flags=re.IGNORECASE)
        if match:
            try:
                return _clean_num_str(match.group(2))
            except (ValueError, IndexError):
                continue
    return None


def extract_implied_upside(markdown: str) -> Optional[float]:
    """Extracts Implied Upside % from Markdown text or table."""
    if not markdown:
        return None
    patterns = [
        r"(?:Implied\s+Upside(?:\s*\/\s*Downside)?|Expected\s+Upside)[^\n\d+-]*?([+-]?[0-9,]+\.?[0-9]*)\s*%",
        r"\|\s*Implied\s+Upside(?:\s*\/\s*Downside)?\s*\|\s*([+-]?[0-9,]+\.?[0-9]*)\s*%?",
    ]
    for pattern in patterns:
        match = re.search(pattern, markdown, flags=re.IGNORECASE)
        if match:
            try:
                has_pct = "%" in match.group(0)
                val = _clean_num_str(match.group(1))
                # Normalize percentage to decimal
                if has_pct or abs(val) > 1.0:
                    val /= 100.0
                return val
            except (ValueError, IndexError):
                continue
    return None


def extract_dupont_roe(markdown: str) -> Optional[float]:
    """Extracts 5-Way DuPont ROE from Markdown text or table."""
    if not markdown:
        return None
    patterns = [
        r"(?:5-Way\s+DuPont\s+ROE|DuPont\s+ROE|Reconstructed\s+ROE)[^\n\d+-]*?([+-]?[0-9,]+\.?[0-9]*)\s*%",
        r"\|\s*(?:5-Way\s+DuPont\s+ROE|DuPont\s+ROE|ROE)\s*\|\s*([+-]?[0-9,]+\.?[0-9]*)\s*%?",
        r"(?:Return\s+on\s+Equity\s*\(ROE\))[^\n\d+-]*?([+-]?[0-9,]+\.?[0-9]*)\s*%",
    ]
    for pattern in patterns:
        match = re.search(pattern, markdown, flags=re.IGNORECASE)
        if match:
            try:
                has_pct = "%" in match.group(0)
                val = _clean_num_str(match.group(1))
                if has_pct or abs(val) > 1.0:
                    val /= 100.0
                return val
            except (ValueError, IndexError):
                continue
    return None


def evaluate_research_run(
    snapshot: CompanyFinancialSnapshot,
    fsa_summary: FSARatioSummary,
    quant_summary: QuantRiskSummary,
    assumptions: DCFScenarioAssumptions,
    valuation_output: ValuationSuiteOutput,
    report_markdown: str,
    repair_attempts: int = 0,
) -> EvalScorecard:
    """Institutional 3-layer automated evaluation harness evaluating research rigor and fidelity.
    
    Layers:
    1. Math Identity Invariance (DuPont, CCC, DCF monotonicity, EV-to-Equity balance sheet bridge, g < WACC)
    2. Schema & CFA Report Section Completeness (Pydantic objects, 6 sections, [FILING-CHUNK-X] citations)
    3. Zero-Hallucination Numerical Faithfulness (Current price, WACC, target price, upside %, DuPont ROE)
    """
    flagged_discrepancies: list[str] = []

    # -------------------------------------------------------------------------
    # Input Contract Pre-Verification (Schema Integrity)
    # -------------------------------------------------------------------------
    valid_contracts = (
        isinstance(snapshot, CompanyFinancialSnapshot)
        and isinstance(fsa_summary, FSARatioSummary)
        and isinstance(quant_summary, QuantRiskSummary)
        and isinstance(assumptions, DCFScenarioAssumptions)
        and isinstance(valuation_output, ValuationSuiteOutput)
        and isinstance(report_markdown, str)
        and bool(report_markdown.strip())
    )
    if not valid_contracts:
        missing_models = []
        if not isinstance(snapshot, CompanyFinancialSnapshot):
            missing_models.append("CompanyFinancialSnapshot")
        if not isinstance(fsa_summary, FSARatioSummary):
            missing_models.append("FSARatioSummary")
        if not isinstance(quant_summary, QuantRiskSummary):
            missing_models.append("QuantRiskSummary")
        if not isinstance(assumptions, DCFScenarioAssumptions):
            missing_models.append("DCFScenarioAssumptions")
        if not isinstance(valuation_output, ValuationSuiteOutput):
            missing_models.append("ValuationSuiteOutput")
        if not (isinstance(report_markdown, str) and report_markdown.strip()):
            missing_models.append("report_markdown (non-empty string)")
        flagged_discrepancies.append(
            f"Input data contracts invalid or missing required Pydantic models: {', '.join(missing_models)}"
        )
        return EvalScorecard(
            math_invariance_score=0.0,
            schema_completeness_score=0.0,
            numerical_faithfulness_score=0.0,
            overall_eval_score=0.0,
            passed=False,
            repair_attempts=repair_attempts,
            flagged_discrepancies=flagged_discrepancies,
        )

    # -------------------------------------------------------------------------
    # Layer 1: Math Identity Invariance (5 checks, 20 pts each)
    # -------------------------------------------------------------------------
    math_checks_passed = 0

    # Check 1: 5-Way DuPont Product Invariance
    dupont_product = (
        fsa_summary.tax_burden
        * fsa_summary.interest_burden
        * fsa_summary.ebit_margin
        * fsa_summary.asset_turnover
        * fsa_summary.financial_leverage
    )
    if (
        abs(fsa_summary.reconstructed_roe - fsa_summary.roe) < 1e-4
        and abs(dupont_product - fsa_summary.roe) < 1e-4
    ):
        math_checks_passed += 1
    else:
        flagged_discrepancies.append(
            f"DuPont 5-Way identity violated: reconstructed_roe={fsa_summary.reconstructed_roe:.5f} vs roe={fsa_summary.roe:.5f}"
        )

    # Check 2: Working Capital CCC Identity: CCC == DSO + DIO - DPO
    expected_ccc = fsa_summary.dso + fsa_summary.dio - fsa_summary.dpo
    if abs(fsa_summary.cash_conversion_cycle - expected_ccc) < 1e-4:
        math_checks_passed += 1
    else:
        flagged_discrepancies.append(
            f"Working capital CCC identity violated: CCC={fsa_summary.cash_conversion_cycle:.4f} vs DSO+DIO-DPO={expected_ccc:.4f}"
        )

    # Check 3: DCF Scenario Monotonicity: Bull > Base > Bear > 0
    bull_val = valuation_output.bull_dcf.intrinsic_value_per_share
    base_val = valuation_output.base_dcf.intrinsic_value_per_share
    bear_val = valuation_output.bear_dcf.intrinsic_value_per_share
    if bull_val > base_val > bear_val > 0:
        math_checks_passed += 1
    else:
        flagged_discrepancies.append(
            f"DCF scenario monotonicity violated: Bull ({bull_val:.2f}) > Base ({base_val:.2f}) > Bear ({bear_val:.2f}) > 0 failed"
        )

    # Check 4: Enterprise Value to Equity Value Balance Sheet Bridge
    latest_yr = snapshot.historical_years[-1]
    bridge_ok = True
    for sc in [valuation_output.bear_dcf, valuation_output.base_dcf, valuation_output.bull_dcf]:
        computed_equity = sc.enterprise_value + latest_yr.cash_and_equivalents - latest_yr.total_debt
        if abs(computed_equity - sc.equity_value) > 0.02:
            bridge_ok = False
            flagged_discrepancies.append(
                f"EV-to-Equity balance sheet bridge violated in scenario '{sc.scenario_name}': "
                f"EV ({sc.enterprise_value:.2f}) + Cash ({latest_yr.cash_and_equivalents:.2f}) - "
                f"Debt ({latest_yr.total_debt:.2f}) != Equity ({sc.equity_value:.2f})"
            )
            break
    if bridge_ok:
        math_checks_passed += 1

    # Check 5: Terminal Growth Cap (g < WACC across all 3 scenarios)
    tgr_ok = (
        valuation_output.bear_dcf.terminal_growth_rate < valuation_output.bear_dcf.wacc
        and valuation_output.base_dcf.terminal_growth_rate < valuation_output.base_dcf.wacc
        and valuation_output.bull_dcf.terminal_growth_rate < valuation_output.bull_dcf.wacc
    )
    if tgr_ok:
        math_checks_passed += 1
    else:
        flagged_discrepancies.append(
            "Terminal growth rate cap violated: terminal_growth_rate >= WACC in one or more scenarios"
        )

    math_invariance_score = round(math_checks_passed * 20.0, 2)

    # -------------------------------------------------------------------------
    # Layer 2: Schema & CFA Report Section Completeness (100 pts)
    # -------------------------------------------------------------------------
    schema_score = 25.0  # Input contracts are verified above

    # Part B: 6 Mandatory CFA Level 1 Sections (50 pts total, 8.333 pts each)
    sections_found = 0
    for sec_title, sec_regex in MANDATORY_CFA_SECTIONS:
        if re.search(sec_regex, report_markdown, flags=re.IGNORECASE):
            sections_found += 1
        else:
            flagged_discrepancies.append(f"Missing mandatory CFA report section: '{sec_title}'")
    schema_score += round((sections_found / len(MANDATORY_CFA_SECTIONS)) * 50.0, 2)

    # Part C: Grounded [FILING-CHUNK-X] Citation Verification (25 pts)
    if re.search(r"\[FILING-CHUNK-\d+\]", report_markdown):
        schema_score += 25.0
    else:
        flagged_discrepancies.append("Report lacks required [FILING-CHUNK-X] citation grounding")

    schema_completeness_score = round(schema_score, 2)

    # -------------------------------------------------------------------------
    # Layer 3: Zero-Hallucination Numerical Faithfulness (5 checks, 20 pts each)
    # -------------------------------------------------------------------------
    faith_checks_passed = 0

    # 1. Current Price Check (tolerance: 0.5% or 0.10)
    rep_curr_price = extract_current_price(report_markdown)
    curr_price_tol = max(snapshot.current_price * 0.005, 0.10)
    if rep_curr_price is not None:
        if abs(rep_curr_price - snapshot.current_price) <= curr_price_tol:
            faith_checks_passed += 1
        else:
            flagged_discrepancies.append(
                f"Current Price mismatch: report stated {rep_curr_price:.2f}, "
                f"engine computed {snapshot.current_price:.2f}"
            )
    else:
        flagged_discrepancies.append(
            f"Current Price not found in report markdown (expected {snapshot.current_price:.2f})"
        )

    # 2. Base WACC Check (tolerance: 10 bps or 0.0010)
    rep_wacc = extract_base_wacc(report_markdown)
    if rep_wacc is not None:
        if abs(rep_wacc - valuation_output.base_wacc) <= 0.0010:
            faith_checks_passed += 1
        else:
            flagged_discrepancies.append(
                f"Base WACC mismatch: report stated {rep_wacc * 100:.2f}%, "
                f"engine computed {valuation_output.base_wacc * 100:.2f}%"
            )
    else:
        flagged_discrepancies.append(
            f"Base WACC not found in report markdown (expected {valuation_output.base_wacc * 100:.2f}%)"
        )

    # 3. Target Price Check (tolerance: 0.5% or 0.10)
    rep_target_price = extract_target_price(report_markdown)
    target_price_tol = max(valuation_output.probability_weighted_intrinsic_value * 0.005, 0.10)
    if rep_target_price is not None:
        if abs(rep_target_price - valuation_output.probability_weighted_intrinsic_value) <= target_price_tol:
            faith_checks_passed += 1
        else:
            flagged_discrepancies.append(
                f"Target Price mismatch: report stated {rep_target_price:.2f}, "
                f"engine computed {valuation_output.probability_weighted_intrinsic_value:.2f}"
            )
    else:
        flagged_discrepancies.append(
            f"Target Price not found in report markdown (expected {valuation_output.probability_weighted_intrinsic_value:.2f})"
        )

    # 4. Implied Upside % Check (tolerance: 50 bps or 0.0050)
    rep_upside = extract_implied_upside(report_markdown)
    if rep_upside is not None:
        if abs(rep_upside - valuation_output.implied_upside_pct) <= 0.0050:
            faith_checks_passed += 1
        else:
            flagged_discrepancies.append(
                f"Implied Upside mismatch: report stated {rep_upside * 100:.2f}%, "
                f"engine computed {valuation_output.implied_upside_pct * 100:.2f}%"
            )
    else:
        flagged_discrepancies.append(
            f"Implied Upside not found in report markdown (expected {valuation_output.implied_upside_pct * 100:.2f}%)"
        )

    # 5. 5-Way DuPont ROE Check (tolerance: 50 bps or 0.0050)
    rep_roe = extract_dupont_roe(report_markdown)
    if rep_roe is not None:
        if abs(rep_roe - fsa_summary.roe) <= 0.0050:
            faith_checks_passed += 1
        else:
            flagged_discrepancies.append(
                f"DuPont ROE mismatch: report stated {rep_roe * 100:.2f}%, "
                f"engine computed {fsa_summary.roe * 100:.2f}%"
            )
    else:
        flagged_discrepancies.append(
            f"DuPont ROE not found in report markdown (expected {fsa_summary.roe * 100:.2f}%)"
        )

    numerical_faithfulness_score = round(faith_checks_passed * 20.0, 2)

    # -------------------------------------------------------------------------
    # Composite Score & Gate Decision
    # -------------------------------------------------------------------------
    overall_eval_score = round(
        (math_invariance_score + schema_completeness_score + numerical_faithfulness_score) / 3.0, 2
    )

    passed = bool(
        math_invariance_score >= 99.99
        and schema_completeness_score >= 99.99
        and numerical_faithfulness_score >= 99.99
        and len(flagged_discrepancies) == 0
    )

    return EvalScorecard(
        math_invariance_score=math_invariance_score,
        schema_completeness_score=schema_completeness_score,
        numerical_faithfulness_score=numerical_faithfulness_score,
        overall_eval_score=overall_eval_score,
        passed=passed,
        repair_attempts=repair_attempts,
        flagged_discrepancies=flagged_discrepancies,
    )
