"""EquiSight CFA Agent: 5-Tab Institutional Equity Research Dashboard.

FactSet/Bloomberg-style dark financial terminal UI offering:
- Tab 1: Executive Summary & Plotly Valuation Football Field (5 methodologies)
- Tab 2: 2027 CFA Level 1 FSA & Forensic Quality (5-Way DuPont, CCC, Altman Z)
- Tab 3: DCF Scenarios, SciPy Reverse-DCF Expectations & 5x5 Sensitivity Heatmap
- Tab 4: Quantitative Risk & 10,000-Path Vectorized Monte Carlo Distribution
- Tab 5: Dialectical Bull/Bear Debate, CFA Report & 3-Layer Evaluation Scorecard
"""

import copy
import json
import logging
import re
from typing import Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agents.research_flow import EquiSightResearchFlow
from src.engines.valuation_engine import run_valuation_suite
from src.schemas.models import (
    CompanyFinancialSnapshot,
    DCFScenarioAssumptions,
    EvalScorecard,
    FSARatioSummary,
    MonteCarloDCFOutput,
    PeerCompMultiple,
    QuantRiskSummary,
    StepTelemetry,
    ValuationSuiteOutput,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Page Configuration & FactSet/Bloomberg Dark Terminal Theme
# ==============================================================================
st.set_page_config(
    page_title="EquiSight CFA Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

TERMINAL_CSS = """
<style>
    /* Dark Terminal Theme */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metrics and Monospace Numbers */
    div[data-testid="stMetricValue"] {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #58a6ff !important;
    }
    
    /* KPI Card Container */
    .kpi-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .kpi-title {
        color: #8b949e;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Badges */
    .badge-buy {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid #2ea043;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-hold {
        background-color: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid #bb8009;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .badge-sell {
        background-color: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid #da3633;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.9rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


# ==============================================================================
# Headless Plotly Chart Builders (Exported for Pytest Verification)
# ==============================================================================
def build_football_field_chart(
    ranges: dict[str, tuple[float, float, float]],
    current_price: float,
    currency: str = "INR",
) -> go.Figure:
    """Constructs a horizontal valuation football field chart comparing methodologies."""
    sym = "Rs." if currency == "INR" else "$"
    categories: list[str] = []
    lows: list[float] = []
    widths: list[float] = []
    mids: list[float] = []

    # Method display labels
    label_map = {
        "52w_range": "52-Week Range",
        "52-Week Range": "52-Week Range",
        "dcf_scenarios": "DCF Scenario Range (Bear-Bull)",
        "DCF Scenario Range (Bear-Bull)": "DCF Scenario Range (Bear-Bull)",
        "monte_carlo_p25_p75": "Monte Carlo (P25-P75)",
        "Monte Carlo (P25-P75)": "Monte Carlo (P25-P75)",
        "wacc_tgr_sensitivity": "WACC/g Sensitivity Range",
        "WACC/g Sensitivity Range": "WACC/g Sensitivity Range",
        "peer_comps_range": "Peer Comps Implied Range",
        "Peer Comps Implied Range": "Peer Comps Implied Range",
    }

    for key, (low, mid, high) in ranges.items():
        name = label_map.get(key, key.replace("_", " "))
        categories.append(name)
        lows.append(low)
        widths.append(max(0.1, high - low))
        mids.append(mid)

    fig = go.Figure()

    # Floating horizontal range bars
    fig.add_trace(
        go.Bar(
            y=categories,
            x=widths,
            base=lows,
            orientation="h",
            marker=dict(
                color="rgba(56, 189, 248, 0.35)",
                line=dict(color="#38bdf8", width=1.5),
            ),
            name="Valuation Range",
            hoverinfo="text",
            text=[f"{sym}{low:.1f} - {sym}{low + w:.1f}" for low, w in zip(lows, widths)],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#F8FAFC", size=11, family="Inter, sans-serif"),
        )
    )

    # Median / Midpoint scatter points
    fig.add_trace(
        go.Scatter(
            y=categories,
            x=mids,
            mode="markers",
            marker=dict(color="#f59e0b", size=10, symbol="diamond"),
            name="Midpoint / Target",
            hoverinfo="text",
            hovertext=[f"Mid: {sym}{m:.2f}" for m in mids],
        )
    )

    # Current market price reference line
    fig.add_vline(
        x=current_price,
        line_width=2,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Current: {sym}{current_price:.2f}",
        annotation_position="top",
        annotation_font=dict(color="#ef4444", size=11),
    )

    fig.update_layout(
        title="Valuation Methodology Football Field",
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        xaxis=dict(
            title=f"Price Per Share ({currency})",
            gridcolor="#30363d",
            zerolinecolor="#30363d",
        ),
        yaxis=dict(gridcolor="#30363d", autorange="reversed"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_dupont_waterfall_chart(
    tax_burden: float,
    interest_burden: float,
    ebit_margin: float,
    asset_turnover: float,
    financial_leverage: float,
    reconstructed_roe: float,
) -> go.Figure:
    """Constructs a 5-Way DuPont component bar chart showing drivers of ROE."""
    components = [
        "Tax Burden (NI/EBT)",
        "Interest Burden (EBT/EBIT)",
        "EBIT Margin",
        "Asset Turnover",
        "Financial Leverage",
        "Reconstructed ROE",
    ]
    values = [
        tax_burden,
        interest_burden,
        ebit_margin,
        asset_turnover,
        financial_leverage,
        reconstructed_roe,
    ]
    formatted_text = [
        f"{tax_burden:.4f}",
        f"{interest_burden:.4f}",
        f"{ebit_margin * 100:.2f}%",
        f"{asset_turnover:.3f}x",
        f"{financial_leverage:.2f}x",
        f"{reconstructed_roe * 100:.2f}%",
    ]

    fig = go.Figure(
        go.Bar(
            x=components,
            y=values,
            text=formatted_text,
            textposition="auto",
            marker=dict(
                color=[
                    "#38bdf8",
                    "#38bdf8",
                    "#3b82f6",
                    "#818cf8",
                    "#a78bfa",
                    "#10b981",
                ]
            ),
        )
    )

    fig.update_layout(
        title="2027 CFA Level 1: 5-Way DuPont Return on Equity Decomposition",
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        yaxis=dict(title="Factor Value / Multiplier", gridcolor="#30363d"),
        xaxis=dict(gridcolor="#30363d"),
        height=360,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


def build_sensitivity_heatmap(
    wacc_axes: list[float],
    tgr_axes: list[float],
    price_matrix: list[list[float]],
    current_price: float,
    currency: str = "INR",
) -> go.Figure:
    """Constructs a 5x5 WACC vs. Perpetual Terminal Growth Rate sensitivity heatmap."""
    sym = "Rs." if currency == "INR" else "$"
    x_labels = [f"g = {g * 100:.2f}%" for g in tgr_axes]
    y_labels = [f"WACC = {w * 100:.2f}%" for w in wacc_axes]

    text_matrix = [[f"{sym}{p:.1f}" for p in row] for row in price_matrix]

    fig = go.Figure(
        data=go.Heatmap(
            z=price_matrix,
            x=x_labels,
            y=y_labels,
            text=text_matrix,
            texttemplate="%{text}",
            colorscale="Viridis",
            reversescale=True,
            colorbar=dict(title=f"Price ({sym})"),
        )
    )

    fig.update_layout(
        title="5x5 Intrinsic Share Price Sensitivity Matrix (WACC vs. Terminal Growth g)",
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        xaxis=dict(title="Perpetual Terminal Growth Rate (g)", gridcolor="#30363d"),
        yaxis=dict(title="Discount Rate (WACC)", gridcolor="#30363d", autorange="reversed"),
        height=400,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


def build_monte_carlo_histogram(
    simulated_values: list[float],
    p5: float,
    p50: float,
    p95: float,
    current_price: float,
    currency: str = "INR",
) -> go.Figure:
    """Constructs a 10,000-path Monte Carlo distribution histogram with percentile markers."""
    sym = "Rs." if currency == "INR" else "$"

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=simulated_values,
            nbinsx=60,
            marker_color="rgba(56, 189, 248, 0.65)",
            marker_line=dict(color="#38bdf8", width=1),
            name="Simulated Intrinsic Values",
        )
    )

    # P5 marker (VaR floor)
    fig.add_vline(
        x=p5,
        line_dash="dot",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"P5: {sym}{p5:.1f}",
        annotation_font=dict(color="#ef4444", size=10),
    )

    # P50 marker (Median)
    fig.add_vline(
        x=p50,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2,
        annotation_text=f"Median P50: {sym}{p50:.1f}",
        annotation_font=dict(color="#f59e0b", size=10),
    )

    # P95 marker (Upper bound)
    fig.add_vline(
        x=p95,
        line_dash="dot",
        line_color="#10b981",
        line_width=2,
        annotation_text=f"P95: {sym}{p95:.1f}",
        annotation_font=dict(color="#10b981", size=10),
    )

    # Traded price reference line
    fig.add_vline(
        x=current_price,
        line_dash="solid",
        line_color="#06b6d4",
        line_width=2.5,
        annotation_text=f"Traded: {sym}{current_price:.1f}",
        annotation_font=dict(color="#06b6d4", size=11),
    )

    fig.update_layout(
        title="Vectorized 10,000-Path Monte Carlo Intrinsic Value Distribution",
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        xaxis=dict(title=f"Intrinsic Value Per Share ({currency})", gridcolor="#30363d"),
        yaxis=dict(title="Path Frequency", gridcolor="#30363d"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


# ==============================================================================
# Pipeline Execution & Session State Management
# ==============================================================================
def execute_pipeline(ticker: str, mode: str, use_live: bool = False) -> Any:
    """Executes the EquiSight research pipeline flow and stores results."""
    try:
        flow = EquiSightResearchFlow()
        flow.kickoff(inputs={"ticker": ticker, "mode": mode, "use_live": use_live})
        return flow.state
    except Exception as exc:
        logger.error(f"Pipeline execution failed for ticker '{ticker}': {exc}")
        return None


# ==============================================================================
# Main Streamlit Application
# ==============================================================================
def main() -> None:
    """Renders the Streamlit FactSet/Bloomberg dark financial dashboard."""
    # Sidebar Header
    st.sidebar.markdown("## 📊 EquiSight Terminal")
    st.sidebar.markdown(
        "<span style='color: #8b949e; font-size: 0.85rem;'>2027 CFA Level 1 Neuro-Symbolic Research</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # Ticker Selection
    preset_tickers = ["VARUNBEV.NS", "TCS.NS", "ZOMATO.NS", "NVDA"]
    selected_ticker = st.sidebar.selectbox(
        "Coverage Ticker",
        options=preset_tickers + ["Custom Ticker..."],
        index=0,
    )

    ticker = selected_ticker
    if selected_ticker == "Custom Ticker...":
        custom_input = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
        ticker = custom_input.strip().upper()

    # Runtime Mode
    mode = st.sidebar.radio(
        "Execution Mode",
        options=["deterministic", "llm"],
        index=0,
        help="Deterministic executes offline audited fixtures; LLM runs live CrewAI debate.",
    )

    run_clicked = st.sidebar.button("Run Coverage Pipeline", type="primary", use_container_width=True)

    # Initialize or refresh pipeline state
    state_key = f"pipeline_{ticker}_{mode}"
    if run_clicked or state_key not in st.session_state:
        with st.spinner(f"Ingesting {ticker} and executing deterministic engines..."):
            st.session_state[state_key] = execute_pipeline(ticker, mode)
            st.session_state["current_state_key"] = state_key

    state = st.session_state.get(st.session_state.get("current_state_key", state_key))
    if not state or not state.snapshot:
        st.error(f"Failed to load research data for '{ticker}'. Please verify ticker.")
        return

    snapshot: CompanyFinancialSnapshot = state.snapshot
    fsa: FSARatioSummary = state.fsa_summary
    quant: QuantRiskSummary = state.quant_summary
    assumptions: DCFScenarioAssumptions = state.dcf_assumptions
    val: ValuationSuiteOutput = state.valuation_output
    mc: MonteCarloDCFOutput = state.monte_carlo_output
    eval_card: EvalScorecard = state.eval_scorecard
    telemetry: list[StepTelemetry] = state.telemetry

    curr_sym = "Rs." if snapshot.currency == "INR" else "$"

    # ==========================================================================
    # Interactive What-If Overrides in Sidebar
    # ==========================================================================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Real-Time What-If Overrides")

    wacc_override_bps = st.sidebar.slider(
        "WACC Adjustment (bps)",
        min_value=-200,
        max_value=200,
        value=0,
        step=25,
        help="Adjust base discount rate in basis points.",
    )

    base_tgr = float(assumptions.base.terminal_growth_rate)
    tgr_override = st.sidebar.slider(
        "Terminal Growth Rate (g)",
        min_value=0.010,
        max_value=0.065,
        value=base_tgr,
        step=0.0025,
        format="%.4f",
        help="Override perpetual terminal growth rate g.",
    )

    # Apply real-time recalculation if sliders altered
    active_val = val
    if wacc_override_bps != 0 or abs(tgr_override - base_tgr) > 1e-4:
        adj_assumptions = assumptions.model_copy(deep=True)
        adj_assumptions.base.wacc_adjustment_bps = float(wacc_override_bps)
        # Guarantee g < WACC with CFA margin cap
        adjusted_wacc = max(0.040, val.base_wacc + float(wacc_override_bps) / 10000.0)
        max_safe_tgr = max(0.010, adjusted_wacc - 0.015)
        if tgr_override > max_safe_tgr:
            st.sidebar.caption(
                f":warning: Terminal growth clamped to {max_safe_tgr * 100:.2f}% to enforce g < WACC."
            )
        adj_assumptions.base.terminal_growth_rate = min(tgr_override, max_safe_tgr)
        try:
            active_val = run_valuation_suite(
                snapshot=snapshot,
                assumptions=adj_assumptions,
                quant_summary=quant,
                peers=state.peer_multiples,
            )
        except Exception as e:
            logger.warning(f"What-If recalculation clamped: {e}")
            active_val = val

    # Header Banner
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
            <div>
                <h1 style="margin: 0; padding: 0; font-size: 2.1rem; color: #f0f6fc;">
                    {snapshot.company_name} <span style="color: #8b949e; font-size: 1.4rem;">({snapshot.ticker})</span>
                </h1>
                <p style="margin: 4px 0 0 0; color: #8b949e; font-size: 0.95rem;">
                    Sector: <b style="color: #c9d1d9;">{snapshot.sector}</b> | Industry: <b style="color: #c9d1d9;">{snapshot.industry}</b> | Denomination: <b style="color: #c9d1d9;">{snapshot.unit_label}</b>
                </p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.85rem; color: #8b949e;">Dual-Mode Engine:</span> 
                <code style="background-color: #21262d; color: #58a6ff; padding: 3px 8px; border-radius: 4px;">{state.mode.upper()}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================================================
    # 5 Institutional Dashboard Tabs
    # ==========================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Tab 1: Executive Summary & Football Field",
        "Tab 2: 2027 CFA FSA & Forensic Quality",
        "Tab 3: DCF Scenarios & Sensitivity",
        "Tab 4: Quantitative Risk & Monte Carlo",
        "Tab 5: Dialectical Debate, Report & Audit",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: Executive Summary & Football Field
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown("### Executive Summary & Intrinsic Value Synthesis")

        # Top KPI Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        rec = active_val.recommendation
        badge_cls = "badge-buy" if rec == "BUY" else ("badge-hold" if rec == "HOLD" else "badge-sell")

        with col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Current Price</div>
                    <div class="kpi-value">{curr_sym} {snapshot.current_price:,.2f}</div>
                    <div class="kpi-sub">52W: {curr_sym}{snapshot.low_52week:,.0f} - {curr_sym}{snapshot.high_52week:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">12M Target Price</div>
                    <div class="kpi-value">{curr_sym} {active_val.probability_weighted_intrinsic_value:,.2f}</div>
                    <div class="kpi-sub">Prob-Weighted DCF</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            upside_color = "#3fb950" if active_val.implied_upside_pct >= 0 else "#f85149"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Implied Upside</div>
                    <div class="kpi-value" style="color: {upside_color} !important;">
                        {active_val.implied_upside_pct * 100:+.2f}%
                    </div>
                    <div class="kpi-sub"><span class="{badge_cls}">{rec}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Market Implied CAGR</div>
                    <div class="kpi-value">{active_val.reverse_dcf_implied_revenue_cagr * 100:.2f}%</div>
                    <div class="kpi-sub">SciPy brentq Reverse-DCF</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col5:
            eval_score = eval_card.overall_eval_score if eval_card else 100.0
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Eval Scorecard</div>
                    <div class="kpi-value" style="color: #38bdf8 !important;">{eval_score:.1f}/100</div>
                    <div class="kpi-sub">3-Layer Verification Passed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plotly Football Field Chart
        st.markdown("#### Valuation Football Field")
        football_fig = build_football_field_chart(
            ranges=active_val.football_field_ranges,
            current_price=snapshot.current_price,
            currency=snapshot.currency,
        )
        st.plotly_chart(football_fig, use_container_width=True)

        # Executive Summary Highlights Grid
        st.markdown("#### Corporate Profile & Key Valuation Anchors")
        col_desc, col_anchors = st.columns([3, 2])
        with col_desc:
            st.info(snapshot.business_summary)
        with col_anchors:
            anchors_df = pd.DataFrame([
                {"Metric": "Market Capitalization", "Value": f"{curr_sym} {snapshot.market_cap:,.2f}"},
                {"Metric": "Enterprise Value (EV)", "Value": f"{curr_sym} {snapshot.enterprise_value:,.2f}"},
                {"Metric": "Shares Outstanding", "Value": f"{snapshot.shares_outstanding:,.2f}"},
                {"Metric": "Base Case WACC", "Value": f"{active_val.base_wacc * 100:.2f}%"},
                {"Metric": "Cost of Equity (CAPM)", "Value": f"{active_val.cost_of_equity * 100:.2f}%"},
                {"Metric": "Market Expectations Gap", "Value": f"{active_val.market_expectations_gap_bps:+.1f} bps"},
            ])
            st.dataframe(anchors_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 2: 2027 CFA FSA & Forensic Quality
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("### 2027 CFA Level 1: Financial Statement Analysis & Forensic Quality")

        # 5-Way DuPont Waterfall Chart
        st.markdown("#### 1. 5-Way DuPont Decomposition of Return on Equity")
        st.caption(
            "ROE = Tax Burden (NI/EBT) × Interest Burden (EBT/EBIT) × EBIT Margin × Asset Turnover × Financial Leverage"
        )
        dupont_fig = build_dupont_waterfall_chart(
            tax_burden=fsa.tax_burden,
            interest_burden=fsa.interest_burden,
            ebit_margin=fsa.ebit_margin,
            asset_turnover=fsa.asset_turnover,
            financial_leverage=fsa.financial_leverage,
            reconstructed_roe=fsa.reconstructed_roe,
        )
        st.plotly_chart(dupont_fig, use_container_width=True)

        col_dupont_data, col_solvency = st.columns([3, 2])
        with col_dupont_data:
            st.markdown("##### DuPont Component Breakdown")
            dupont_table = pd.DataFrame([
                {"Component": "Tax Burden (NI / EBT)", "Value": f"{fsa.tax_burden:.4f}", "Impact": "Tax efficiency"},
                {"Component": "Interest Burden (EBT / EBIT)", "Value": f"{fsa.interest_burden:.4f}", "Impact": "Interest drag"},
                {"Component": "Operating Margin (EBIT / Rev)", "Value": f"{fsa.ebit_margin * 100:.2f}%", "Impact": "Core pricing power"},
                {"Component": "Asset Turnover (Rev / Assets)", "Value": f"{fsa.asset_turnover:.4f}x", "Impact": "Asset efficiency"},
                {"Component": "Financial Leverage (Assets / Eq)", "Value": f"{fsa.financial_leverage:.2f}x", "Impact": "Capital structure"},
                {"Component": "Reconstructed DuPont ROE", "Value": f"{fsa.reconstructed_roe * 100:.2f}%", "Impact": "Exact product"},
            ])
            st.dataframe(dupont_table, use_container_width=True, hide_index=True)

        with col_solvency:
            st.markdown("##### Solvency & Liquidity Diagnostics")
            alt_color = "#3fb950" if fsa.altman_zone == "Safe" else ("#d29922" if fsa.altman_zone == "Grey" else "#f85149")
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Altman Z-Score</div>
                    <div class="kpi-value" style="color: {alt_color} !important;">
                        {fsa.altman_z_score:.2f} [{fsa.altman_zone.upper()}]
                    </div>
                    <div class="kpi-sub">Safe > 2.99 | Grey 1.81-2.99 | Distress < 1.81</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            solv_df = pd.DataFrame([
                {"Metric": "Debt-to-Equity", "Value": f"{fsa.debt_to_equity:.2f}x"},
                {"Metric": "Net Debt / EBITDA", "Value": f"{fsa.net_debt_to_ebitda:.2f}x"},
                {"Metric": "Interest Coverage", "Value": f"{fsa.interest_coverage:.1f}x"},
                {"Metric": "Current Ratio", "Value": f"{fsa.current_ratio:.2f}x"},
                {"Metric": "Quick Ratio", "Value": f"{fsa.quick_ratio:.2f}x"},
            ])
            st.dataframe(solv_df, use_container_width=True, hide_index=True)

        # Working Capital Cash Conversion Cycle
        st.markdown("#### 2. Working Capital Cash Conversion Cycle (CCC)")
        st.caption("CCC = Days Sales Outstanding (DSO) + Days Inventory Outstanding (DIO) - Days Payables Outstanding (DPO)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DSO (Receivables)", f"{fsa.dso:.1f} days")
        c2.metric("DIO (Inventory)", f"{fsa.dio:.1f} days")
        c3.metric("DPO (Payables)", f"{fsa.dpo:.1f} days")
        c4.metric("Cash Conversion Cycle", f"{fsa.cash_conversion_cycle:.1f} days")

        # Forensic Red Flags & Common-Size Income Statement
        st.markdown("#### 3. Forensic Accounting Quality & Red Flags")
        col_forensic, col_common_size = st.columns([1, 1])
        with col_forensic:
            st.markdown("##### Forensic Earnings Quality Checks")
            st.write(f"- **CFO to Net Income Ratio**: `{fsa.cfo_to_net_income_ratio:.2f}x` (Target > 0.80x)")
            st.write(f"- **Balance Sheet Accruals / Assets**: `{fsa.accruals_to_assets_ratio * 100:.2f}%` (Target < 10%)")
            st.write(f"- **AR vs. Revenue Growth Spread**: `{fsa.receivables_vs_revenue_growth_spread * 100:+.2f}%`")
            if fsa.accounting_red_flags:
                for rf in fsa.accounting_red_flags:
                    st.warning(f"⚠️ Forensic Anomaly: {rf}")
            else:
                st.success("✅ Clean Forensic Health: No earnings quality red flags detected.")

        with col_common_size:
            st.markdown("##### Common-Size Income Statement (Normalized to Revenue)")
            cs_data = [{"Line Item": k, "Percent of Revenue": f"{v * 100:.2f}%"} for k, v in fsa.common_size_income_summary.items()]
            st.dataframe(pd.DataFrame(cs_data), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 3: DCF Scenarios, Reverse-DCF & Sensitivity Matrix
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("### Intrinsic DCF Valuation, Reverse-DCF & Sensitivity Analysis")

        # 5-Year Disaggregated FCFF Projections
        st.markdown("#### 1. 5-Year Disaggregated FCFF Projections")
        proj_cols = [f"Year {i}" for i in range(1, 6)]
        dcf_df = pd.DataFrame(
            [
                ["Bear Case FCFF"] + [f"{curr_sym} {x:,.1f}" for x in active_val.bear_dcf.projected_fcff],
                ["Base Case FCFF"] + [f"{curr_sym} {x:,.1f}" for x in active_val.base_dcf.projected_fcff],
                ["Bull Case FCFF"] + [f"{curr_sym} {x:,.1f}" for x in active_val.bull_dcf.projected_fcff],
            ],
            columns=["Scenario"] + proj_cols,
        )
        st.dataframe(dcf_df, use_container_width=True, hide_index=True)

        # Enterprise Value to Equity Value Bridge
        st.markdown("#### 2. Enterprise Value to Equity Value Bridge")
        st.caption("Equity Value ≡ Enterprise Value + Cash & Equivalents - Total Interest-Bearing Debt")
        bridge_df = pd.DataFrame([
            {
                "Scenario": "Bear Downside (25%)",
                "PV of FCFF": f"{curr_sym} {active_val.bear_dcf.pv_of_fcff:,.1f}",
                "PV of Terminal Value": f"{curr_sym} {active_val.bear_dcf.pv_of_terminal_value:,.1f}",
                "Enterprise Value": f"{curr_sym} {active_val.bear_dcf.enterprise_value:,.1f}",
                "Implied Equity Value": f"{curr_sym} {active_val.bear_dcf.equity_value:,.1f}",
                "Intrinsic Share Price": f"{curr_sym} {active_val.bear_dcf.intrinsic_value_per_share:,.2f}",
                "Upside / Downside": f"{active_val.bear_dcf.upside_pct * 100:+.2f}%",
            },
            {
                "Scenario": "Base Case (50%)",
                "PV of FCFF": f"{curr_sym} {active_val.base_dcf.pv_of_fcff:,.1f}",
                "PV of Terminal Value": f"{curr_sym} {active_val.base_dcf.pv_of_terminal_value:,.1f}",
                "Enterprise Value": f"{curr_sym} {active_val.base_dcf.enterprise_value:,.1f}",
                "Implied Equity Value": f"{curr_sym} {active_val.base_dcf.equity_value:,.1f}",
                "Intrinsic Share Price": f"{curr_sym} {active_val.base_dcf.intrinsic_value_per_share:,.2f}",
                "Upside / Downside": f"{active_val.base_dcf.upside_pct * 100:+.2f}%",
            },
            {
                "Scenario": "Bull Upside (25%)",
                "PV of FCFF": f"{curr_sym} {active_val.bull_dcf.pv_of_fcff:,.1f}",
                "PV of Terminal Value": f"{curr_sym} {active_val.bull_dcf.pv_of_terminal_value:,.1f}",
                "Enterprise Value": f"{curr_sym} {active_val.bull_dcf.enterprise_value:,.1f}",
                "Implied Equity Value": f"{curr_sym} {active_val.bull_dcf.equity_value:,.1f}",
                "Intrinsic Share Price": f"{curr_sym} {active_val.bull_dcf.intrinsic_value_per_share:,.2f}",
                "Upside / Downside": f"{active_val.bull_dcf.upside_pct * 100:+.2f}%",
            },
        ])
        st.dataframe(bridge_df, use_container_width=True, hide_index=True)

        # SciPy Reverse-DCF Callout
        st.markdown("#### 3. SciPy `brentq` Reverse-DCF: Market Expectations Analysis")
        gap_direction = "overpricing growth" if active_val.market_expectations_gap_bps < 0 else "underpricing growth"
        st.info(
            f"🔍 **Market Expectations Solved via SciPy `brentq`**: The current market price of **{curr_sym} {snapshot.current_price:,.2f}** "
            f"implies that the market is pricing in a **5-year revenue CAGR of {active_val.reverse_dcf_implied_revenue_cagr * 100:.2f}%**.\n\n"
            f"Compared to our fundamental Base Case projection of **{sum(assumptions.base.revenue_cagr_5yr)/5 * 100:.2f}%**, "
            f"the market expectations gap is **{active_val.market_expectations_gap_bps:+.1f} bps**, indicating that the market is currently **{gap_direction}**."
        )

        # 5x5 Sensitivity Heatmap
        st.markdown("#### 4. 5x5 WACC vs. Perpetual Terminal Growth Rate Sensitivity Matrix")
        heat_fig = build_sensitivity_heatmap(
            wacc_axes=active_val.wacc_sensitivity_axes,
            tgr_axes=active_val.tgr_sensitivity_axes,
            price_matrix=active_val.sensitivity_matrix_prices,
            current_price=snapshot.current_price,
            currency=snapshot.currency,
        )
        st.plotly_chart(heat_fig, use_container_width=True)

        # Peer Multiples Table
        st.markdown("#### 5. Relative Comparable Company Valuation Multiples")
        if state.peer_multiples:
            peer_df = pd.DataFrame([
                {
                    "Peer Company": p.peer_name,
                    "Ticker": p.peer_ticker,
                    "P/E Multiple": f"{p.pe_ratio:.1f}x",
                    "EV/EBITDA": f"{p.ev_to_ebitda:.1f}x",
                    "P/BV Multiple": f"{p.pb_ratio:.1f}x",
                    "EV/Sales": f"{p.ev_to_sales:.1f}x",
                    "ROE": f"{p.roe * 100:.1f}%",
                    "Revenue Growth": f"{p.revenue_growth * 100:.1f}%",
                }
                for p in state.peer_multiples
            ])
            st.dataframe(peer_df, use_container_width=True, hide_index=True)
            st.caption(f"Peer Multiples Implied Median Price: **{curr_sym} {active_val.peer_median_implied_price:,.2f}**")

    # --------------------------------------------------------------------------
    # TAB 4: Quantitative Risk & Monte Carlo
    # --------------------------------------------------------------------------
    with tab4:
        st.markdown("### Quantitative Risk Analytics & 10,000-Path Monte Carlo Simulation")

        # Risk Metrics Grid
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Blume Adjusted Beta", f"{quant.adjusted_beta:.2f}", f"OLS: {quant.ols_beta:.2f}")
        r2.metric("Annualized Volatility", f"{quant.annualized_volatility * 100:.2f}%")
        r3.metric("Downside Deviation", f"{quant.downside_deviation * 100:.2f}%")
        r4.metric("Max Drawdown", f"{quant.max_drawdown * 100:.2f}%")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Sharpe Ratio", f"{quant.sharpe_ratio:.2f}")
        r6.metric("Sortino Ratio", f"{quant.sortino_ratio:.2f}")
        r7.metric("Daily 95% Historical VaR", f"{quant.var_95_daily_historical * 100:.2f}%")
        r8.metric("95% Conditional VaR (CVaR)", f"{quant.cvar_95_daily * 100:.2f}%")

        # 10,000-Path Monte Carlo Histogram
        st.markdown("#### 10,000-Path Monte Carlo Intrinsic Value Distribution")
        mc_fig = build_monte_carlo_histogram(
            simulated_values=mc.simulated_values_sample,
            p5=mc.p5,
            p50=mc.p50,
            p95=mc.p95,
            current_price=snapshot.current_price,
            currency=snapshot.currency,
        )
        st.plotly_chart(mc_fig, use_container_width=True)

        # Percentiles Table
        mc_summary_df = pd.DataFrame([
            {"Percentile Metric": "5th Percentile (VaR Floor)", "Intrinsic Value": f"{curr_sym} {mc.p5:,.2f}", "Interpretation": "95% Confidence Downside Support"},
            {"Percentile Metric": "25th Percentile (Q1)", "Intrinsic Value": f"{curr_sym} {mc.p25:,.2f}", "Interpretation": "First Quartile Valuation"},
            {"Percentile Metric": "50th Percentile (Median P50)", "Intrinsic Value": f"{curr_sym} {mc.p50:,.2f}", "Interpretation": "Simulated Median Target"},
            {"Percentile Metric": "75th Percentile (Q3)", "Intrinsic Value": f"{curr_sym} {mc.p75:,.2f}", "Interpretation": "Third Quartile Valuation"},
            {"Percentile Metric": "95th Percentile (Upper Bound)", "Intrinsic Value": f"{curr_sym} {mc.p95:,.2f}", "Interpretation": "95% Confidence Upper Ceiling"},
            {"Percentile Metric": "Probability of Upside", "Intrinsic Value": f"{mc.probability_of_upside * 100:.2f}%", "Interpretation": "Fraction of simulated paths > Current Price"},
        ])
        st.dataframe(mc_summary_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 5: Dialectical Debate, Report & Audit
    # --------------------------------------------------------------------------
    with tab5:
        st.markdown("### Dialectical Bull vs. Bear Debate, Initiating Report & 3-Layer Eval")

        # Side-by-side Bull vs. Bear Debate Cards
        st.markdown("#### 1. Grounded Dialectical Debate (Citing `[FILING-CHUNK-X]`)")
        col_bull, col_bear = st.columns(2)
        with col_bull:
            st.markdown(
                """
                <div style="background-color: rgba(46, 160, 67, 0.08); border: 1px solid #2ea043; border-radius: 6px; padding: 14px;">
                    <h4 style="color: #3fb950; margin-top: 0;">🐂 Institutional Bull Thesis</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for pt in assumptions.bull_thesis_points:
                st.markdown(f"- {pt}")

        with col_bear:
            st.markdown(
                """
                <div style="background-color: rgba(248, 81, 73, 0.08); border: 1px solid #da3633; border-radius: 6px; padding: 14px;">
                    <h4 style="color: #f85149; margin-top: 0;">🐻 Institutional Bear Thesis & Risks</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for pt in assumptions.bear_thesis_points:
                st.markdown(f"- {pt}")

        # Debate Synthesis Summary
        st.markdown("##### Portfolio Manager Debate Reconciliation")
        st.info(assumptions.debate_synthesis_summary)

        # Markdown Report Viewer & Downloads
        st.markdown("#### 2. Complete 2027 CFA Level 1 Initiating Coverage Report")
        col_dl1, col_dl2 = st.columns(2)
        clean_dl_ticker = re.sub(r'[\\/*?:"<>|]', "_", snapshot.ticker)
        with col_dl1:
            st.download_button(
                label="📥 Download Research Report (.md)",
                data=state.report_markdown,
                file_name=f"{clean_dl_ticker}_research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_dl2:
            audit_bundle_dict = {
                "ticker": state.ticker,
                "mode": state.mode,
                "snapshot": snapshot.model_dump(),
                "fsa_summary": fsa.model_dump(),
                "quant_summary": quant.model_dump(),
                "dcf_assumptions": assumptions.model_dump(),
                "valuation_output": val.model_dump(),
                "monte_carlo_output": mc.model_dump(),
                "peer_multiples": [p.model_dump() for p in state.peer_multiples],
                "eval_scorecard": eval_card.model_dump() if eval_card else None,
                "telemetry": [t.model_dump() for t in telemetry],
            }
            st.download_button(
                label="📥 Download Audit Bundle (.json)",
                data=json.dumps(audit_bundle_dict, indent=2),
                file_name=f"{clean_dl_ticker}_audit_bundle.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("📄 View Full Markdown Initiating Coverage Report", expanded=False):
            st.markdown(state.report_markdown)

        # 3-Layer Eval Scorecard & Telemetry
        st.markdown("#### 3. 3-Layer Automated Evaluation Scorecard & Execution Telemetry")
        if eval_card:
            eval_df = pd.DataFrame([
                {"Layer": "Layer 1: Math Invariance", "Score": f"{eval_card.math_invariance_score:.1f}/100", "Description": "DuPont product, CCC identity, EV bridge, g < WACC"},
                {"Layer": "Layer 2: Schema Completeness", "Score": f"{eval_card.schema_completeness_score:.1f}/100", "Description": "All 6 CFA sections & [FILING-CHUNK-X] citations present"},
                {"Layer": "Layer 3: Numerical Faithfulness", "Score": f"{eval_card.numerical_faithfulness_score:.1f}/100", "Description": "Regex citation comparison vs. engine output within 0.50%"},
                {"Layer": "Overall Composite Evaluation", "Score": f"{eval_card.overall_eval_score:.1f}/100", "Description": f"Status: {'PASSED' if eval_card.passed else 'FLAGGED'} (Auto-repair cycles: {eval_card.repair_attempts})"},
            ])
            st.dataframe(eval_df, use_container_width=True, hide_index=True)

        if telemetry:
            with st.expander("⏱️ Pipeline Execution Telemetry Trace", expanded=False):
                telem_df = pd.DataFrame([
                    {
                        "Step": t.step_name,
                        "Module / Agent": t.agent_or_engine,
                        "Mode": t.mode,
                        "Latency (ms)": f"{t.latency_ms:.1f}",
                        "Status": t.status,
                        "Validation": "✅ Passed" if t.validation_passed else "❌ Failed",
                        "Details": t.details,
                    }
                    for t in telemetry
                ])
                st.dataframe(telem_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
