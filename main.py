"""EquiSight CFA Agent: Institutional Command-Line Interface.

Executes deterministic and LLM-assisted equity research flows, prints executive summary tables
to stdout, and exports audited research reports and JSON audit bundles to the output directory.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import dotenv

# Load environment variables with override
dotenv.load_dotenv(override=True)

from src.agents.research_flow import EquiSightResearchFlow, LLMExecutionError
from src.data.market_gateway import DataUnavailableError, format_market_cap
from src.data.ticker_resolver import resolve_ticker_query

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("equisight_cli")


def print_executive_summary(flow_state: Any) -> None:
    """Prints a structured Bloomberg/FactSet-style executive summary table to terminal."""
    if not flow_state or not getattr(flow_state, "snapshot", None):
        print("[-] Insufficient analytical outputs to render executive summary.")
        return

    snapshot = flow_state.snapshot
    fsa = flow_state.fsa_summary
    quant = flow_state.quant_summary
    val = flow_state.valuation_output
    eval_card = flow_state.eval_scorecard

    if not (snapshot and fsa and quant and val):
        print("[-] Insufficient analytical outputs to render executive summary.")
        return

    cur = snapshot.currency
    curr_sym = "Rs." if cur == "INR" else "$"

    border = "=" * 78
    divider = "-" * 78

    print("\n" + border)
    print(f"  EQUISIGHT CFA RESEARCH: {snapshot.company_name} ({snapshot.ticker})")
    print(f"  Sector: {snapshot.sector} | Industry: {snapshot.industry} | Units: {snapshot.unit_label}")
    print(f"  Data Source: {snapshot.data_source} | Execution Mode: {flow_state.mode.upper()}")
    if flow_state.mode == "llm" and getattr(flow_state, "llm_model_used", None):
        print(f"  LLM Model: {flow_state.llm_model_used} | Synthesis Latency: {flow_state.llm_latency_ms:.1f}ms")
    print(border)

    # 1. Initiating Coverage & Valuation
    formatted_mc = format_market_cap(snapshot.market_cap, cur)
    formatted_ev = format_market_cap(snapshot.enterprise_value, cur)
    print(f"  RECOMMENDATION:          {val.recommendation}")
    print(f"  Current Market Price:    {curr_sym} {snapshot.current_price:,.2f}")
    print(f"  12M Target Price:        {curr_sym} {val.probability_weighted_intrinsic_value:,.2f} ({val.implied_upside_pct * 100:+.2f}%)")
    print(f"  Market Cap:              {formatted_mc} ({curr_sym} {snapshot.market_cap:,.2f})")
    print(f"  Enterprise Value:        {formatted_ev} ({curr_sym} {snapshot.enterprise_value:,.2f})")
    print(f"  52-Week Range:           {curr_sym} {snapshot.low_52week:,.2f} - {curr_sym} {snapshot.high_52week:,.2f}")
    print(divider)

    # 2. Valuation & Market Expectations Gap
    print(f"  Base Case WACC:          {val.base_wacc * 100:.2f}% (Cost of Equity: {val.cost_of_equity * 100:.2f}%)")
    print(f"  Market Implied CAGR:     {val.reverse_dcf_implied_revenue_cagr * 100:.2f}% (SciPy brentq Reverse-DCF)")
    print(f"  Expectations Gap:        {val.market_expectations_gap_bps:+.1f} bps vs. Base Forecast")
    print(divider)

    # 3. 2027 CFA Level 1 FSA & DuPont Decomposition
    print(f"  5-WAY DUPONT ROE:        {fsa.reconstructed_roe * 100:.2f}% (Reported ROE: {fsa.roe * 100:.2f}%)")
    print(f"    - Tax Burden (NI/EBT):      {fsa.tax_burden:.4f}")
    print(f"    - Interest Burden (EBT/EBIT):{fsa.interest_burden:.4f}")
    print(f"    - EBIT Margin:              {fsa.ebit_margin * 100:.2f}%")
    print(f"    - Asset Turnover:           {fsa.asset_turnover:.4f}x")
    print(f"    - Financial Leverage:       {fsa.financial_leverage:.2f}x")
    print(divider)

    # 4. Solvency, Working Capital & Forensic Checks
    print(f"  Altman Z-Score:          {fsa.altman_z_score:.2f} [{fsa.altman_zone.upper()}]")
    print(f"  Cash Conversion Cycle:   {fsa.cash_conversion_cycle:.1f} days (DSO: {fsa.dso:.1f}d, DIO: {fsa.dio:.1f}d, DPO: {fsa.dpo:.1f}d)")
    print(f"  CFO / Net Income Ratio:  {fsa.cfo_to_net_income_ratio:.2f}x")
    if fsa.accounting_red_flags:
        print(f"  Forensic Red Flags:      {len(fsa.accounting_red_flags)} anomaly flag(s) detected:")
        for flag in fsa.accounting_red_flags:
            print(f"    * {flag}")
    else:
        print("  Forensic Red Flags:      0 flags (Clean Accounting Quality)")
    print(divider)

    # 5. Quantitative Risk & Monte Carlo
    print(f"  Blume Adjusted Beta:     {quant.adjusted_beta:.2f} (OLS Beta: {quant.ols_beta:.2f}, R2: {quant.r_squared:.2f})")
    print(f"  Annualized Volatility:   {quant.annualized_volatility * 100:.2f}% | Max Drawdown: {quant.max_drawdown * 100:.2f}%")
    print(f"  Sharpe Ratio:            {quant.sharpe_ratio:.2f} | Sortino Ratio: {quant.sortino_ratio:.2f}")
    print(f"  Daily 95% Historical VaR:{quant.var_95_daily_historical * 100:.2f}% | 95% CVaR: {quant.cvar_95_daily * 100:.2f}%")
    print(divider)

    # 6. Evaluation Scorecard
    if eval_card:
        status_str = "PASSED" if eval_card.passed else "FLAGGED"
        print(f"  3-LAYER EVAL SCORECARD:  {eval_card.overall_eval_score:.1f}/100 [{status_str}] (Auto-repair cycles: {eval_card.repair_attempts})")
        print(f"    - Math Invariance:          {eval_card.math_invariance_score:.1f}/100")
        print(f"    - Schema Completeness:      {eval_card.schema_completeness_score:.1f}/100")
        print(f"    - Numerical Faithfulness:   {eval_card.numerical_faithfulness_score:.1f}/100")
        if eval_card.flagged_discrepancies:
            print("    Flagged Discrepancies:")
            for disc in eval_card.flagged_discrepancies:
                print(f"      ! {disc}")
    print(border + "\n")


def export_artifacts(flow_state: Any, output_dir: Path) -> tuple[Path, Path]:
    """Saves Markdown research report and JSON audit bundle into output directory."""
    if not flow_state or not getattr(flow_state, "ticker", None):
        raise ValueError("Cannot export artifacts: flow_state has no ticker attribute.")

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_ticker = re.sub(r'[\\/*?:"<>|]', "_", flow_state.ticker)

    report_path = output_dir / f"{clean_ticker}_research_report.md"
    bundle_path = output_dir / f"{clean_ticker}_audit_bundle.json"

    # 1. Write Markdown report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(flow_state.report_markdown)

    # 2. Assemble and write JSON audit bundle
    audit_dict: dict[str, Any] = {
        "ticker": flow_state.ticker,
        "mode": flow_state.mode,
        "status": flow_state.status,
        "repair_attempts": flow_state.repair_attempts,
        "llm_model_used": getattr(flow_state, "llm_model_used", ""),
        "llm_latency_ms": getattr(flow_state, "llm_latency_ms", 0.0),
        "snapshot": flow_state.snapshot.model_dump() if flow_state.snapshot else None,
        "fsa_summary": flow_state.fsa_summary.model_dump() if flow_state.fsa_summary else None,
        "quant_summary": flow_state.quant_summary.model_dump() if flow_state.quant_summary else None,
        "dcf_assumptions": flow_state.dcf_assumptions.model_dump() if flow_state.dcf_assumptions else None,
        "valuation_output": flow_state.valuation_output.model_dump() if flow_state.valuation_output else None,
        "monte_carlo_output": flow_state.monte_carlo_output.model_dump() if flow_state.monte_carlo_output else None,
        "peer_multiples": [p.model_dump() for p in flow_state.peer_multiples],
        "eval_scorecard": flow_state.eval_scorecard.model_dump() if flow_state.eval_scorecard else None,
        "telemetry": [t.model_dump() for t in flow_state.telemetry],
    }

    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(audit_dict, f, indent=2)

    return report_path, bundle_path


def build_parser() -> argparse.ArgumentParser:
    """Constructs command line argument parser."""
    parser = argparse.ArgumentParser(
        description="EquiSight CFA Agent: Institutional Equity Research Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="VARUNBEV.NS",
        help="Equity ticker symbol (e.g. VARUNBEV.NS, TCS.NS, ZOMATO.NS, NVDA)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["deterministic", "llm"],
        default="deterministic",
        help="Pipeline runtime execution mode",
    )
    parser.add_argument(
        "--use-live-data",
        action="store_true",
        default=False,
        help="Attempt live market data retrieval via yfinance with disk cache",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Destination directory for markdown reports and audit bundles",
    )
    return parser


def run_cli(args: argparse.Namespace) -> int:
    """Executes the research flow and handles artifact export."""
    raw_query = args.ticker.strip() if args.ticker else ""
    try:
        resolved_ticker = resolve_ticker_query(raw_query)
        if resolved_ticker != raw_query:
            print(f"[*] Resolved query '{raw_query}' -> '{resolved_ticker}'")
    except DataUnavailableError as due:
        print(f"[!] Data Unavailable Error: {due}", file=sys.stderr)
        return 1
    except Exception:
        resolved_ticker = raw_query

    use_live_data = True
    print(f"\n[*] Initializing EquiSight Research Flow for '{resolved_ticker}' in '{args.mode}' mode (Live Market Data Default)...")
    flow = EquiSightResearchFlow()

    try:
        flow.kickoff(
            inputs={
                "ticker": resolved_ticker,
                "mode": args.mode,
                "use_live": use_live_data,
            }
        )
    except DataUnavailableError as due:
        print(f"[!] Data Unavailable Error: {due}", file=sys.stderr)
        return 1
    except LLMExecutionError as lee:
        print(f"[!] LLM Execution Error: {lee}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[!] Pipeline execution failed with fatal error: {exc}", file=sys.stderr)
        return 1

    if flow.state.mode == "llm" and getattr(flow.state, "llm_model_used", None):
        print(f"[*] Groq LLM Synthesis: model={flow.state.llm_model_used}, latency={flow.state.llm_latency_ms:.1f}ms")

    # Render terminal executive summary
    print_executive_summary(flow.state)

    # Save artifacts
    out_dir = Path(args.output_dir)
    report_file, bundle_file = export_artifacts(flow.state, out_dir)
    print(f"[+] Research Report exported to: {report_file}")
    print(f"[+] Audit Bundle exported to:   {bundle_file}\n")

    if flow.state.eval_scorecard and not flow.state.eval_scorecard.passed:
        print("[!] Warning: 3-Layer Evaluation reported discrepancies that could not be auto-repaired.", file=sys.stderr)
        return 1

    return 0


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    exit_code = run_cli(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
