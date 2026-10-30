import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.market_gateway import fetch_company_snapshot, fetch_peer_multiples
from src.engines.fsa_engine import compute_fsa_summary
from src.engines.quant_risk_engine import compute_quant_risk_summary
from src.agents.research_flow import synthesize_deterministic_assumptions
from src.engines.valuation_engine import run_valuation_suite

tickers = [
    "AAPL",
    "AMZN",
    "GOOG",
    "NVDA",
    "MSFT",
    "TCS.NS",
    "INFY.NS",
    "VARUNBEV.NS",
    "ZOMATO.NS",
]

results = []
for ticker in tickers:
    try:
        snapshot = fetch_company_snapshot(ticker, use_live=True)
        peers = fetch_peer_multiples(ticker, use_live=True)
        fsa = compute_fsa_summary(snapshot)
        quant = compute_quant_risk_summary(snapshot)
        assumptions = synthesize_deterministic_assumptions(snapshot, fsa, quant)
        val = run_valuation_suite(snapshot, assumptions, quant, peers)

        curr_sym = "INR " if snapshot.currency == "INR" else "USD "
        results.append({
            "ticker": ticker,
            "live_price": f"{curr_sym}{snapshot.current_price:.2f}",
            "target_price": f"{curr_sym}{val.probability_weighted_intrinsic_value:.2f}",
            "upside_pct": f"{val.implied_upside_pct * 100:+.2f}%",
            "rec": val.recommendation,
            "implied_cagr": f"{val.reverse_dcf_implied_revenue_cagr * 100:.2f}%",
        })
    except Exception as e:
        results.append({"ticker": ticker, "error": str(e)})

header = f"| {'Ticker':<12} | {'Live Price':<14} | {'12M Target Price':<18} | {'Implied Upside %':<18} | {'Recommendation':<14} | {'Reverse-DCF Implied CAGR':<25} |"
sep = f"|{'-' * 14}|{'-' * 16}|{'-' * 20}|{'-' * 20}|{'-' * 16}|{'-' * 27}|"

print("\n" + "=" * len(header))
print(header)
print(sep)
for r in results:
    if "error" in r:
        print(f"| {r['ticker']:<12} | ERROR: {r['error']:<70} |")
    else:
        print(f"| {r['ticker']:<12} | {r['live_price']:<14} | {r['target_price']:<18} | {r['upside_pct']:<18} | {r['rec']:<14} | {r['implied_cagr']:<25} |")
print("=" * len(header) + "\n")
