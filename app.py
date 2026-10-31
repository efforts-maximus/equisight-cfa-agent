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
import os
import re
from typing import Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import dotenv

# Load environment variables with override
dotenv.load_dotenv(override=True)

from src.agents.research_flow import EquiSightResearchFlow, LLMExecutionError
from src.data.market_gateway import DataUnavailableError, format_market_cap
from src.data.market_pulse import TrendingStockCard, fetch_daily_popular_stocks
from src.data.ticker_resolver import resolve_ticker_query
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

VIEW_HOME = "🏠 Command Center (Home)"
VIEW_TERMINAL = "📊 Deep-Dive Research Terminal"

TERMINAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Hide Streamlit Chrome & Fix Top Clipping while preserving Sidebar Toggle */
    [data-testid="stAppDeployButton"], .stDeployButton, #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        color: #c9d1d9 !important;
        pointer-events: none !important;
        z-index: 999990 !important;
    }
    /* Ensure sidebar toggle controls are clickable, fully visible, and high-contrast in dark mode */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button,
    header[data-testid="stHeader"] button[aria-label="Expand sidebar"],
    header[data-testid="stHeader"] button[data-testid="stExpandSidebarButton"],
    button[aria-label="Collapse sidebar"],
    button[aria-label="Expand sidebar"] {
        pointer-events: auto !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #c9d1d9 !important;
    }
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button,
    button[aria-label="Collapse sidebar"],
    button[aria-label="Expand sidebar"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
    }
    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stSidebarCollapseButton"] button:hover,
    button[aria-label="Collapse sidebar"]:hover,
    button[aria-label="Expand sidebar"]:hover {
        background-color: #21262d !important;
        border-color: #58a6ff !important;
        color: #58a6ff !important;
        box-shadow: 0 0 8px rgba(88, 166, 255, 0.3) !important;
    }
    /* Force inner icon / span / svg to high contrast light gray instead of muted dark gray */
    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapseButton"] *,
    button[aria-label="Collapse sidebar"] *,
    button[aria-label="Expand sidebar"] * {
        color: #c9d1d9 !important;
        fill: #c9d1d9 !important;
    }
    [data-testid="stSidebarCollapsedControl"]:hover *,
    [data-testid="stExpandSidebarButton"]:hover *,
    [data-testid="stSidebarCollapseButton"]:hover *,
    button[aria-label="Collapse sidebar"]:hover *,
    button[aria-label="Expand sidebar"]:hover * {
        color: #58a6ff !important;
        fill: #58a6ff !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
    }

    /* Dark Terminal Theme */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }
    
    /* Metrics and Monospace Numbers */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #58a6ff !important;
        font-variant-numeric: tabular-nums !important;
    }
    
    /* KPI Card Container */
    .kpi-card {
        background: linear-gradient(145deg, #161b22 0%, #0d1117 100%) !important;
        border: 1px solid #30363d;
        border-top: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
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
        font-family: 'JetBrains Mono', "SFMono-Regular", Consolas, monospace !important;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0f6fc;
        font-variant-numeric: tabular-nums;
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

    /* Sleek Quick Pill Buttons in Sidebar */
    div[class*="st-key-btn_ql_"] button,
    div[data-testid="stSidebar"] div[data-testid="stColumn"]:has(.quick-pill-btn) button,
    div[data-testid="stSidebar"] div[class*="st-key-btn_ql_"] button,
    .quick-pill-btn button,
    div.quick-pill-btn button {
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        background-color: #1E293B !important;
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        font-family: 'JetBrains Mono', "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
        font-weight: 700 !important;
        font-size: 0.68rem !important;
        padding: 0 1px !important;
        border-radius: 4px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: none !important;
    }
    div[class*="st-key-btn_ql_"] button:hover,
    div[data-testid="stSidebar"] div[data-testid="stColumn"]:has(.quick-pill-btn) button:hover,
    div[data-testid="stSidebar"] div[class*="st-key-btn_ql_"] button:hover,
    .quick-pill-btn button:hover,
    div.quick-pill-btn button:hover {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.5) !important;
        color: #38BDF8 !important;
        background: #1E293B !important;
    }
    div[class*="st-key-btn_ql_"] button div,
    div[class*="st-key-btn_ql_"] button p,
    div[class*="st-key-btn_ql_"] button span,
    div[data-testid="stSidebar"] div[data-testid="stColumn"]:has(.quick-pill-btn) button div,
    div[data-testid="stSidebar"] div[data-testid="stColumn"]:has(.quick-pill-btn) button p,
    div[data-testid="stSidebar"] div[data-testid="stColumn"]:has(.quick-pill-btn) button span {
        font-family: 'JetBrains Mono', "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
        font-weight: 700 !important;
        font-size: 0.68rem !important;
        color: #F8FAFC !important;
        white-space: nowrap !important;
        text-overflow: clip !important;
        overflow: visible !important;
        margin: 0 !important;
        padding: 0 !important;
        letter-spacing: -0.02em !important;
    }
    div[class*="st-key-btn_ql_"] button:hover div,
    div[class*="st-key-btn_ql_"] button:hover p,
    div[class*="st-key-btn_ql_"] button:hover span {
        color: #38BDF8 !important;
    }

    /* Live Market Mover Card Button Styling */
    div[class*="st-key-mover_card_"] button,
    div[data-testid="stColumn"]:has(.mover-card-btn) button,
    div.mover-card-btn button,
    div[data-testid="stHorizontalBlock"] .mover-card-btn button,
    button[key*="mover_card_"],
    button:has(div[title*="MCap"]),
    button:has(div[title*="Vol:"]) {
        height: 135px !important;
        min-height: 135px !important;
        max-height: 135px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
        white-space: pre-line !important;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    div[class*="st-key-mover_card_"] button:hover,
    div[data-testid="stColumn"]:has(.mover-card-btn) button:hover,
    div.mover-card-btn button:hover,
    div[data-testid="stHorizontalBlock"] .mover-card-btn button:hover,
    button[key*="mover_card_"]:hover,
    button:has(div[title*="MCap"]):hover,
    button:has(div[title*="Vol:"]):hover {
        border-color: #38BDF8 !important;
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.25), 0 4px 6px -2px rgba(56, 189, 248, 0.1) !important;
        transform: translateY(-2px) !important;
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%) !important;
    }
    div[class*="st-key-mover_card_"] button div,
    div[class*="st-key-mover_card_"] button p,
    div[class*="st-key-mover_card_"] button span,
    div[class*="st-key-mover_card_"] button .stMarkdownEllipsis,
    div[data-testid="stColumn"]:has(.mover-card-btn) button div,
    div[data-testid="stColumn"]:has(.mover-card-btn) button p,
    div[data-testid="stColumn"]:has(.mover-card-btn) button span,
    div[data-testid="stColumn"]:has(.mover-card-btn) button .stMarkdownEllipsis,
    div.mover-card-btn button div,
    div.mover-card-btn button p,
    button[key*="mover_card_"] div,
    button[key*="mover_card_"] p,
    button:has(div[title*="MCap"]) div,
    button:has(div[title*="MCap"]) p,
    button:has(div[title*="Vol:"]) div,
    button:has(div[title*="Vol:"]) p {
        text-align: left !important;
        white-space: pre-line !important;
        width: 100% !important;
        font-size: 0.85rem !important;
        line-height: 1.45 !important;
        color: #F8FAFC !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    /* Architecture Highlight Cards */
    .arch-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: all 0.2s ease-in-out;
    }
    .arch-card:hover {
        border-color: #38BDF8;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15);
    }
    .arch-title {
        color: #F8FAFC;
        font-weight: 700;
        font-size: 0.86rem;
        margin-bottom: 6px;
    }
    .arch-desc {
        color: #94A3B8;
        font-size: 0.76rem;
        line-height: 1.4;
    }

    /* Provenance Bar Pills */
    .provenance-pill {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
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
            textposition="none",
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

    # Vertically shifted range labels (12px above bar center so they never touch the orange diamond)
    for cat, low, w in zip(categories, lows, widths):
        fig.add_annotation(
            x=low + w / 2.0,
            y=cat,
            text=f"{sym}{low:.1f} - {sym}{low + w:.1f}",
            showarrow=False,
            yshift=12,
            font=dict(color="#F8FAFC", size=11, family="Inter, sans-serif"),
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
        margin=dict(l=20, r=30, t=50, b=40),
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
        annotation_position="top left",
        annotation_yshift=2,
        annotation_font=dict(color="#ef4444", size=10),
    )

    # P50 marker (Median)
    fig.add_vline(
        x=p50,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2,
        annotation_text=f"Median P50: {sym}{p50:.1f}",
        annotation_position="top right",
        annotation_yshift=-18,
        annotation_font=dict(color="#f59e0b", size=10),
    )

    # P95 marker (Upper bound)
    fig.add_vline(
        x=p95,
        line_dash="dot",
        line_color="#10b981",
        line_width=2,
        annotation_text=f"P95: {sym}{p95:.1f}",
        annotation_position="top left",
        annotation_yshift=-36,
        annotation_font=dict(color="#10b981", size=10),
    )

    # Traded price reference line
    fig.add_vline(
        x=current_price,
        line_dash="solid",
        line_color="#06b6d4",
        line_width=2.5,
        annotation_text=f"Traded: {sym}{current_price:.1f}",
        annotation_position="top right",
        annotation_yshift=18,
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
def execute_pipeline(ticker: str, mode: str, use_live: bool = True) -> Any:
    """Executes the EquiSight research pipeline flow and stores results."""
    execute_pipeline.last_error = None
    try:
        flow = EquiSightResearchFlow()
        flow.kickoff(inputs={"ticker": ticker, "mode": mode, "use_live": use_live})
        return flow.state
    except LLMExecutionError as lee:
        logger.error(f"LLM execution error for ticker '{ticker}': {lee}")
        execute_pipeline.last_error = f"LLMExecutionError: {lee}"
        return None
    except DataUnavailableError as due:
        logger.error(f"Data unavailable for ticker '{ticker}': {due}")
        execute_pipeline.last_error = f"DataUnavailableError: {due}"
        return None
    except Exception as exc:
        logger.error(f"Pipeline execution failed for ticker '{ticker}': {exc}")
        execute_pipeline.last_error = str(exc)
        return None

execute_pipeline.last_error = None


def get_benchmark_mover_cards() -> list[TrendingStockCard]:
    """Generates benchmark mover cards from audited institutional fixtures when offline."""
    benchmark_specs = [
        ("NVDA", "NVIDIA Corporation", "US", 142.50, "USD", 3.45, 48200000, 3500000.0, "$3.50T"),
        ("TCS.NS", "Tata Consultancy Services", "NSE", 4120.00, "INR", 1.85, 3200000, 1490000.0, "₹14.90L Cr"),
        ("VARUNBEV.NS", "Varun Beverages Limited", "NSE", 625.40, "INR", -0.75, 4100000, 203000.0, "₹2.03L Cr"),
        ("ZOMATO.NS", "Zomato Limited", "NSE", 265.80, "INR", 4.20, 15800000, 235000.0, "₹2.35L Cr"),
        ("AAPL", "Apple Inc.", "US", 232.10, "USD", 0.65, 38500000, 3550000.0, "$3.55T"),
        ("INFY.NS", "Infosys Limited", "NSE", 1925.50, "INR", 1.10, 5200000, 798000.0, "₹7.98L Cr"),
        ("AMZN", "Amazon.com, Inc.", "US", 188.40, "USD", -1.25, 29400000, 1960000.0, "$1.96T"),
        ("GOOGL", "Alphabet Inc.", "US", 168.90, "USD", 0.90, 22100000, 2110000.0, "$2.11T"),
    ]
    cards = []
    for tkr, name, mkt, prc, cur, chg, vol, mc, fmc in benchmark_specs:
        cards.append(
            TrendingStockCard(
                ticker=tkr,
                company_name=name,
                market=mkt,
                current_price=prc,
                currency=cur,
                daily_change_pct=chg,
                daily_volume=vol,
                market_cap=mc,
                formatted_market_cap=fmc,
            )
        )
    return cards


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_popular_stocks(market_filter: str = "ALL", max_items: int = 8) -> tuple[list[TrendingStockCard], bool, str]:
    """Caches popular stocks and market movers for fast Streamlit rendering."""
    return fetch_daily_popular_stocks(market_filter=market_filter, max_items=max_items)


@st.cache_data(ttl=300, show_spinner=False)
def compute_stock_comparison_metrics(ticker_symbol: str, use_live: bool = True) -> Optional[dict[str, Any]]:
    """Fetches and computes core valuation & FSA metrics for the 2-stock head-to-head comparator."""
    try:
        clean_ticker = ticker_symbol.strip()
        if not clean_ticker:
            return None
        resolved = resolve_ticker_query(clean_ticker)
        state = execute_pipeline(resolved, mode="deterministic", use_live=use_live)
        if not state or not state.snapshot or not state.fsa_summary or not state.valuation_output:
            return None
        snap = state.snapshot
        fsa = state.fsa_summary
        val = state.valuation_output
        sym = "₹" if snap.currency == "INR" else "$"
        return {
            "ticker": snap.ticker,
            "company_name": snap.company_name,
            "currency": snap.currency,
            "currency_symbol": sym,
            "current_price": snap.current_price,
            "target_price": val.probability_weighted_intrinsic_value,
            "implied_upside_pct": val.implied_upside_pct,
            "recommendation": val.recommendation,
            "dupont_roe": fsa.reconstructed_roe,
            "operating_margin": fsa.ebit_margin,
            "base_wacc": val.base_wacc,
            "reverse_dcf_growth": val.reverse_dcf_implied_revenue_cagr,
            "altman_z": fsa.altman_z_score,
            "altman_zone": fsa.altman_zone,
        }
    except Exception as exc:
        logger.warning(f"Comparator metrics computation failed for {ticker_symbol}: {exc}")
        return None


def render_command_center_home(on_select_ticker_cb: Any) -> None:
    """Renders the executive institutional Command Center landing page."""
    # 1. Hero Institutional Briefing Banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
                    border: 1px solid rgba(56, 189, 248, 0.3);
                    border-radius: 12px;
                    padding: 26px 30px;
                    margin-bottom: 22px;
                    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
                    backdrop-filter: blur(12px);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid #0284c7; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                    Institutional Equity Research • 2027 CFA Level 1
                </span>
                <span style="color: #94A3B8; font-size: 0.82rem; font-family: monospace;">
                    Dual-Mode Runtime Invariance
                </span>
            </div>
            <h1 style="color: #F8FAFC; font-size: 2.2rem; font-weight: 800; margin: 0 0 10px 0; letter-spacing: -0.02em;">
                EquiSight <span style="color: #38BDF8; font-weight: 400;">Command Center</span>
            </h1>
            <p style="color: #94A3B8; font-size: 1.0rem; line-height: 1.55; margin: 0; max-width: 950px;">
                Autonomous 2027 CFA Level 1 Neuro-Symbolic Equity Research Terminal strictly separating 100% deterministic CFA mathematics from qualitative multi-agent SEC/MD&amp;A dialectical debate and RAG evidence synthesis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4-Column Architecture Highlight Grid
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        st.markdown(
            """
            <div class="arch-card">
                <div class="arch-title">🧮 100% Deterministic CFA Math Engine</div>
                <div class="arch-desc">Zero LLM arithmetic; 5-Way DuPont, 2-Stage FCFF DCF, Altman Z-Score, and Working Capital CCC.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_a2:
        st.markdown(
            """
            <div class="arch-card">
                <div class="arch-title">⚖️ Dialectical Bull vs. Bear Filing RAG</div>
                <div class="arch-desc">Multi-agent debate grounded in <code>[FILING-CHUNK-X]</code> citations from authentic regulatory disclosures.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_a3:
        st.markdown(
            """
            <div class="arch-card">
                <div class="arch-title">🎯 SciPy brentq Reverse-DCF &amp; 10k MC</div>
                <div class="arch-desc">Market-implied growth expectations gap &amp; 10,000 vectorized NumPy Monte Carlo simulation paths.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_a4:
        st.markdown(
            """
            <div class="arch-card">
                <div class="arch-title">🛡️ 3-Layer Zero-Hallucination Audit Gate</div>
                <div class="arch-desc">Automated verification of math identities, schema completeness, and numerical faithfulness.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # 2. Central Unified Smart Search Bar
    st.markdown("### 🔍 Institutional Equity Search & Ingestion")

    def _on_home_search_enter() -> None:
        q = st.session_state.get("home_search_query_input", "").strip()
        if q:
            resolved = resolve_ticker_query(q)
            on_select_ticker_cb(resolved)

    search_c1, search_c2 = st.columns([4, 1])
    with search_c1:
        st.text_input(
            "🔍 Search Any Company Name or Ticker — e.g., Tata Consultancy, Infosys, Reliance, Nvidia, Apple, Amazon, TCS, INFY",
            value="",
            key="home_search_query_input",
            on_change=_on_home_search_enter,
            placeholder="Search company name or ticker (e.g. Tata Consultancy, Infosys, Reliance, Nvidia, Apple, Amazon, TCS, INFY)...",
        )
    with search_c2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.button(
            "🚀 Ingest & Deep-Dive",
            key="btn_home_search_submit",
            on_click=_on_home_search_enter,
            use_container_width=True,
            type="primary",
        )

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    # 3. Live Daily Market Movers Grid — Directly Clickable Ticker Cards
    st.markdown("### 📡 Live Daily Market Movers")
    mkt_col1, mkt_col2 = st.columns([3, 1])
    with mkt_col1:
        mkt_filter_selection = st.radio(
            "Market Filter",
            options=["🌍 All Markets", "🇮🇳 NSE India", "🇺🇸 US Markets"],
            horizontal=True,
            key="home_mkt_filter_radio",
            label_visibility="collapsed",
        )

    filter_code_map = {
        "🌍 All Markets": "ALL",
        "🇮🇳 NSE India": "NSE",
        "🇺🇸 US Markets": "US",
    }
    mkt_code = filter_code_map.get(mkt_filter_selection, "ALL")

    pulse_cards, is_pulse_live, pulse_msg = get_cached_popular_stocks(market_filter=mkt_code, max_items=8)

    if is_pulse_live and pulse_cards:
        st.caption(f"📡 Live Daily Market Movers — Ranked by Today's Volume & Price Action ({len(pulse_cards)} active movers)")
        display_cards = pulse_cards
    else:
        st.warning(f"⚠️ {pulse_msg}")
        display_cards = get_benchmark_mover_cards()
        if mkt_code == "NSE":
            display_cards = [c for c in display_cards if c.market == "NSE"]
        elif mkt_code == "US":
            display_cards = [c for c in display_cards if c.market == "US"]

    # Render each TrendingStockCard in a 4-column grid as directly clickable button (NO separate Load button)
    cols_per_row = 4
    for r_idx in range(0, len(display_cards), cols_per_row):
        row_cards = display_cards[r_idx : r_idx + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, card in zip(cols, row_cards):
            with col:
                st.markdown('<div class="mover-card-btn">', unsafe_allow_html=True)
                tag = getattr(card, "activity_tag", "MOVER")
                sym = getattr(card, "currency_symbol", "₹" if card.currency == "INR" else "$")
                chg = getattr(card, "day_change_pct", card.daily_change_pct)
                vol = getattr(card, "day_volume", card.daily_volume)
                mcap = getattr(card, "market_cap_formatted", card.formatted_market_cap)
                comp_display = card.company_name[:25] + ("…" if len(card.company_name) > 25 else "")

                card_label = (
                    f"{card.ticker}  •  {tag}\n"
                    f"{comp_display}\n"
                    f"{sym}{card.current_price:,.2f}  ({chg:+.2f}%)\n"
                    f"MCap: {mcap}  |  Vol: {vol:,}"
                ).replace("$", r"\$")

                st.button(
                    card_label,
                    key=f"mover_card_{card.ticker}",
                    on_click=on_select_ticker_cb,
                    args=(card.ticker,),
                    use_container_width=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

    # 4. Interactive 2-Stock Head-to-Head Valuation Comparator
    st.markdown("### ⚖️ Interactive 2-Stock Head-to-Head Valuation Comparator")
    st.caption("Compare live pricing, DCF targets, 5-Way DuPont drivers, WACC, and Altman Z-score solvency side-by-side.")

    comp_universe = ["NVDA", "TCS.NS", "VARUNBEV.NS", "ZOMATO.NS", "INFY.NS", "AAPL", "AMZN", "GOOGL"]
    c_col1, c_col2 = st.columns(2)
    with c_col1:
        stock_a = st.selectbox("Stock A", options=comp_universe, index=0, key="comparator_stock_a")
    with c_col2:
        stock_b = st.selectbox("Stock B", options=comp_universe, index=1, key="comparator_stock_b")

    data_a = compute_stock_comparison_metrics(stock_a)
    data_b = compute_stock_comparison_metrics(stock_b)

    if data_a and data_b:
        col_name_a = f"Stock A: {stock_a} ({data_a['company_name'][:18]})"
        col_name_b = f"Stock B: {stock_b} ({data_b['company_name'][:18]})"
        comp_rows = [
            {"Valuation & FSA Metric": "1. Live Traded Price", col_name_a: f"{data_a['currency_symbol']} {data_a['current_price']:,.2f}", col_name_b: f"{data_b['currency_symbol']} {data_b['current_price']:,.2f}"},
            {"Valuation & FSA Metric": "2. 12M DCF Intrinsic Target", col_name_a: f"{data_a['currency_symbol']} {data_a['target_price']:,.2f}", col_name_b: f"{data_b['currency_symbol']} {data_b['target_price']:,.2f}"},
            {"Valuation & FSA Metric": "3. Implied Upside %", col_name_a: f"{data_a['implied_upside_pct'] * 100:+.2f}%", col_name_b: f"{data_b['implied_upside_pct'] * 100:+.2f}%"},
            {"Valuation & FSA Metric": "4. CFA Recommendation", col_name_a: data_a['recommendation'], col_name_b: data_b['recommendation']},
            {"Valuation & FSA Metric": "5. 5-Way DuPont ROE", col_name_a: f"{data_a['dupont_roe'] * 100:.2f}%", col_name_b: f"{data_b['dupont_roe'] * 100:.2f}%"},
            {"Valuation & FSA Metric": "6. Operating Margin (EBIT)", col_name_a: f"{data_a['operating_margin'] * 100:.2f}%", col_name_b: f"{data_b['operating_margin'] * 100:.2f}%"},
            {"Valuation & FSA Metric": "7. Base WACC", col_name_a: f"{data_a['base_wacc'] * 100:.2f}%", col_name_b: f"{data_b['base_wacc'] * 100:.2f}%"},
            {"Valuation & FSA Metric": "8. Reverse-DCF Implied Growth", col_name_a: f"{data_a['reverse_dcf_growth'] * 100:.2f}%", col_name_b: f"{data_b['reverse_dcf_growth'] * 100:.2f}%"},
            {"Valuation & FSA Metric": "9. Altman Z-Score & Zone", col_name_a: f"{data_a['altman_z']:.2f} [{data_a['altman_zone']}]", col_name_b: f"{data_b['altman_z']:.2f} [{data_b['altman_zone']}]"},
        ]
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            st.button(f"📊 Deep-Dive into {stock_a}", key=f"btn_dd_a_{stock_a}", on_click=on_select_ticker_cb, args=(stock_a,), use_container_width=True)
        with btn_c2:
            st.button(f"📊 Deep-Dive into {stock_b}", key=f"btn_dd_b_{stock_b}", on_click=on_select_ticker_cb, args=(stock_b,), use_container_width=True)
    else:
        st.info("Select two valid equities to compare institutional valuation & financial metrics.")


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

    # Ensure active view exists in session state before widget instantiation
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = VIEW_HOME
    if "active_ticker_query" not in st.session_state:
        st.session_state["active_ticker_query"] = "VARUNBEV.NS"
    if "unified_ticker_search_input" not in st.session_state:
        st.session_state["unified_ticker_search_input"] = st.session_state["active_ticker_query"]

    # Top-level View Selector in Sidebar
    st.sidebar.markdown("### 🧭 Terminal Navigation")
    active_view = st.sidebar.radio(
        "Navigation View",
        options=[VIEW_HOME, VIEW_TERMINAL],
        key="active_view",
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")

    # Callback to update search query and active ticker before widget instantiation
    def _set_active_search_ticker(ticker_sym: str) -> None:
        st.session_state["unified_ticker_search_input"] = ticker_sym
        st.session_state["active_ticker_query"] = ticker_sym
        st.session_state["active_view"] = VIEW_TERMINAL

    def _on_sidebar_search_change() -> None:
        val = st.session_state.get("unified_ticker_search_input", "").strip()
        if val:
            st.session_state["active_ticker_query"] = val
            st.session_state["active_view"] = VIEW_TERMINAL

    def _on_run_pipeline_click() -> None:
        st.session_state["active_view"] = VIEW_TERMINAL

    # Unified Smart Search Bar & Quick-Launch Coverage
    st.sidebar.markdown("### 🔍 Equity Search")
    search_query = st.sidebar.text_input(
        "🔍 Search Company Name or Ticker",
        value=st.session_state.get("active_ticker_query", "VARUNBEV.NS"),
        key="unified_ticker_search_input",
        on_change=_on_sidebar_search_change,
        help="Search by company name (e.g. Tata Consultancy, Infosys, Reliance, Amazon, Alphabet, Nvidia) or symbol (e.g. TCS, INFY, NVDA, GOOG, AAPL)",
    )
    st.session_state["active_ticker_query"] = search_query

    # Sleek Quick-Launch Ticker Bar
    st.sidebar.caption("⚡ Quick-Launch Coverage:")
    st.sidebar.markdown('<div class="quick-pill-container">', unsafe_allow_html=True)
    ql_cols_1 = st.sidebar.columns(4)
    quick_tickers_1 = [("NVDA", "NVDA"), ("TCS", "TCS.NS"), ("INFY", "INFY.NS"), ("VARUNBEV", "VARUNBEV.NS")]
    for col, (label, sym) in zip(ql_cols_1, quick_tickers_1):
        with col:
            st.markdown('<div class="quick-pill-btn">', unsafe_allow_html=True)
            st.button(
                label,
                key=f"btn_ql_{label}",
                on_click=_set_active_search_ticker,
                args=(sym,),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

    ql_cols_2 = st.sidebar.columns(4)
    quick_tickers_2 = [("ZOMATO", "ZOMATO.NS"), ("AMZN", "AMZN"), ("GOOG", "GOOG"), ("AAPL", "AAPL")]
    for col, (label, sym) in zip(ql_cols_2, quick_tickers_2):
        with col:
            st.markdown('<div class="quick-pill-btn">', unsafe_allow_html=True)
            st.button(
                label,
                key=f"btn_ql_{label}",
                on_click=_set_active_search_ticker,
                args=(sym,),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Resolve query via resolve_ticker_query
    raw_query = search_query.strip()
    if raw_query:
        try:
            resolved_sym = resolve_ticker_query(raw_query)
            active_ticker = resolved_sym
            if resolved_sym.upper() != raw_query.upper():
                st.sidebar.caption(f"🔍 Resolved: **{raw_query}** ➔ **`{resolved_sym}`**")
        except Exception:
            active_ticker = raw_query.upper()
    else:
        active_ticker = ""

    # Runtime Mode
    active_mode = st.sidebar.radio(
        "Execution Mode",
        options=["deterministic", "llm"],
        key="active_mode_radio",
        help="Deterministic executes offline audited fixtures; LLM runs live CrewAI debate.",
    )

    # Always use pure live market data by default across EquiSight
    use_live_data = True

    # Live Daily Market Movers Pulse Expander in sidebar
    with st.sidebar.expander("🔥 Live Daily Market Movers", expanded=False):
        mkt_filter = st.selectbox("Market Filter", ["ALL", "US", "NSE"], key="pulse_filter_select")
        pulse_cards, is_pulse_live, pulse_msg = fetch_daily_popular_stocks(market_filter=mkt_filter, max_items=6)
        if is_pulse_live:
            st.caption(pulse_msg)
            for c in pulse_cards:
                chg_color = "#3fb950" if c.daily_change_pct >= 0 else "#f85149"
                sign = "+" if c.daily_change_pct >= 0 else ""
                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    st.markdown(
                        f"""
                        <div style="background-color: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 4px 8px; margin-bottom: 4px;">
                            <span style="font-weight: 700; color: #f0f6fc;">{c.ticker}</span>
                            <span style="font-size: 0.75rem; color: #8b949e; margin-left: 4px;">{c.market}</span>
                            <span style="float: right; color: {chg_color}; font-weight: 700; font-size: 0.8rem;">{sign}{c.daily_change_pct:.2f}%</span><br>
                            <span style="font-size: 0.75rem; color: #8b949e;">{c.currency} {c.current_price:,.2f} | {c.formatted_market_cap}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    st.button(
                        "Load",
                        key=f"pulse_btn_{c.ticker}",
                        on_click=_set_active_search_ticker,
                        args=(c.ticker,),
                        use_container_width=True,
                    )
        else:
            st.warning(pulse_msg)

    # API key check for LLM mode
    has_api_key = bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    user_api_key_entered = False
    if active_mode == "llm":
        with st.sidebar.expander("🔑 LLM API Key Configuration", expanded=not has_api_key):
            user_api_key = st.text_input(
                "Gemini / Groq / OpenAI API Key",
                type="password",
                key="user_api_key_input",
                help="Enter API key to execute live CrewAI dialectical debate.",
            )
            if user_api_key:
                clean_key = user_api_key.strip()
                if clean_key.startswith("gsk_"):
                    os.environ["GROQ_API_KEY"] = clean_key
                elif clean_key.startswith("sk-"):
                    os.environ["OPENAI_API_KEY"] = clean_key
                else:
                    os.environ["GROQ_API_KEY"] = clean_key
                    os.environ["GEMINI_API_KEY"] = clean_key
                user_api_key_entered = True
                has_api_key = True
        if not has_api_key:
            st.sidebar.warning(
                "⚠️ No LLM API key detected (`GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENAI_API_KEY`). "
                "Running in LLM mode without an API key will raise an LLMExecutionError per zero silent fallback policy. "
                "Configure an API key above or switch back to Deterministic mode."
            )

    run_clicked = st.sidebar.button("Run Coverage Pipeline", type="primary", use_container_width=True, on_click=_on_run_pipeline_click)

    # Dispatch: If Command Center (Home) is selected, render Home Page and return
    if active_view == VIEW_HOME:
        render_command_center_home(on_select_ticker_cb=_set_active_search_ticker)
        return

    # Deep-Dive Terminal View
    if not active_ticker:
        st.sidebar.info("💡 Enter a company name or ticker symbol (e.g. Infosys, TCS, NVDA, GOOG, AAPL) to generate coverage.")
        st.info("👈 Enter a company name or ticker symbol in the sidebar to generate coverage.")
        return

    current_selection_key = f"{active_ticker}::{active_mode}::{use_live_data}"

    if run_clicked or user_api_key_entered or st.session_state.get("last_executed_key") != current_selection_key:
        with st.spinner(f"Ingesting {active_ticker} ({active_mode}) and executing research engines..."):
            pipeline_result = execute_pipeline(active_ticker, active_mode, use_live_data)
            st.session_state["pipeline_result"] = pipeline_result
            st.session_state["last_executed_key"] = current_selection_key

    state = st.session_state.get("pipeline_result")
    pipe_err = getattr(execute_pipeline, "last_error", None)

    if not state or not state.snapshot:
        if pipe_err:
            if "LLMExecutionError" in pipe_err or "groq" in pipe_err.lower() or "llm" in pipe_err.lower():
                st.error(f"❌ **LLM Execution Failure Diagnostic:** {pipe_err}")
            elif "DataUnavailableError" in pipe_err or "unavailable" in pipe_err.lower():
                st.error(f"❌ **Market Gateway Data Unavailable for '{active_ticker}':** {pipe_err}")
            else:
                st.error(f"❌ **Pipeline Execution Diagnostic for '{active_ticker}':** {pipe_err}")
        else:
            st.error(f"❌ Failed to load research data for '{active_ticker}'. Please verify ticker.")
        st.info(
            "💡 **Institutional Data Integrity Guidance:**\n"
            "- **Benchmark Fixtures:** `VARUNBEV.NS`, `TCS.NS`, `ZOMATO.NS`, `NVDA` run 100% offline with audited financial statements.\n"
            "- **Indian Equities:** Use `.NS` suffix (e.g. `INFY.NS`, `RELIANCE.NS`, `HDFCBANK.NS`, `ITC.NS`) or bare names like `Infosys`, `Reliance`, `TCS`.\n"
            "- **US Equities:** Use standard exchange tickers (e.g. `AAPL`, `MSFT`, `NVDA`, `AMZN`, `GOOGL`).\n"
            "- **Zero-Synthetic Policy:** EquiSight never fabricates fake numbers if market data is unavailable."
        )
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
        key=f"wacc_slider_{active_ticker}",
        help="Adjust base discount rate in basis points.",
    )

    base_tgr = float(assumptions.base.terminal_growth_rate)
    clamped_base_tgr = float(np.clip(base_tgr, 0.010, 0.065))
    tgr_override = st.sidebar.slider(
        "Terminal Growth Rate (g)",
        min_value=0.010,
        max_value=0.065,
        value=clamped_base_tgr,
        step=0.0001,
        format="%.4f",
        key=f"tgr_slider_{active_ticker}",
        help="Override perpetual terminal growth rate g.",
    )

    # Apply real-time recalculation if sliders altered
    active_val = val
    if wacc_override_bps != 0 or abs(tgr_override - clamped_base_tgr) > 5e-5:
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

    # Header Banner & 3-Pill Provenance Bar
    formatted_mc = format_market_cap(snapshot.market_cap, snapshot.currency)
    is_live_data = (snapshot.data_source == "LIVE_YFINANCE")
    source_pill = "📡 Data Source: Live Yahoo Finance" if is_live_data else "📦 Offline Fallback Snapshot"

    if state.mode == "llm":
        llm_model = getattr(state, "llm_model_used", "") or "Groq"
        llm_lat = getattr(state, "llm_latency_ms", 0.0) or 0.0
        engine_pill = f"🤖 Narrative Engine: Live Groq LLM ({llm_model} • {llm_lat:.0f} ms) | Math: 100% Deterministic"
    else:
        engine_pill = "⚙️ Engine: 100% Deterministic CFA Synthesizer"

    st.markdown(
        f"""
        <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                <div>
                    <h1 style="margin: 0; padding: 0; font-size: 2.1rem; color: #f0f6fc;">
                        {snapshot.company_name} <span style="color: #8b949e; font-size: 1.4rem;">({snapshot.ticker})</span>
                    </h1>
                    <p style="margin: 4px 0 0 0; color: #8b949e; font-size: 0.95rem;">
                        Sector: <b style="color: #c9d1d9;">{snapshot.sector}</b> | 
                        Industry: <b style="color: #c9d1d9;">{snapshot.industry}</b> | 
                        Denomination: <b style="color: #c9d1d9;">{snapshot.unit_label}</b>
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.85rem; color: #8b949e;">Dual-Mode Engine:</span> 
                    <code style="background-color: #21262d; color: #58a6ff; padding: 3px 8px; border-radius: 4px;">{state.mode.upper()}</code>
                </div>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
                <div class="provenance-pill" style="color: {'#3fb950' if is_live_data else '#d29922'};">
                    {source_pill}
                </div>
                <div class="provenance-pill" style="color: #a78bfa;">
                    {engine_pill}
                </div>
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
        col1, col2, col3, col4 = st.columns(4)
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
            formatted_mc = format_market_cap(snapshot.market_cap, snapshot.currency)
            formatted_ev = format_market_cap(snapshot.enterprise_value, snapshot.currency)
            anchors_df = pd.DataFrame([
                {"Metric": "Market Capitalization", "Value": f"{formatted_mc} ({curr_sym} {snapshot.market_cap:,.2f})"},
                {"Metric": "Enterprise Value (EV)", "Value": f"{formatted_ev} ({curr_sym} {snapshot.enterprise_value:,.2f})"},
                {"Metric": "Shares Outstanding", "Value": f"{snapshot.shares_outstanding:,.2f} {snapshot.unit_label.split()[-1]}"},
                {"Metric": "Data Source", "Value": snapshot.data_source},
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

        # 2027 CFA Steady-State Convergence & Terminal Parameters
        st.markdown("#### 4. 2027 CFA Level 1 Steady-State Convergence & Terminal Parameters")
        base_tv = active_val.base_dcf.terminal_value
        base_pv_tv = active_val.base_dcf.pv_of_terminal_value
        base_ev = active_val.base_dcf.enterprise_value
        tv_pct_ev = (base_pv_tv / base_ev * 100) if base_ev > 0 else 0.0
        terminal_g = active_val.base_dcf.terminal_growth_rate
        terminal_wacc = active_val.base_dcf.wacc
        last_ebit = max(1.0, active_val.base_dcf.projected_ebit[-1])
        implied_tv_ebit = base_tv / last_ebit

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Base Terminal Value", f"{curr_sym} {base_tv:,.1f}")
        c2.metric("PV of Terminal Value", f"{curr_sym} {base_pv_tv:,.1f}", f"{tv_pct_ev:.1f}% of EV")
        c3.metric("Terminal Gordon g", f"{terminal_g * 100:.2f}%", f"WACC: {terminal_wacc * 100:.2f}%")
        c4.metric("Implied Terminal Multiple", f"{implied_tv_ebit:.1f}x EBIT", "Steady-State Convergence")

        st.caption(
            "📐 **CFA Steady-State Invariance**: Terminal cash flow models convergence where maintenance CapEx tapers "
            "toward D&A plus steady growth (DA + g × CapEx × 1.5) and economic reinvestment rate is bounded by "
            "steady-state ROIC [g / max(ROIC, WACC + 2%)], eliminating peak cyclical CapEx distortion."
        )

        # 5x5 Sensitivity Heatmap
        st.markdown("#### 5. 5x5 WACC vs. Perpetual Terminal Growth Rate Sensitivity Matrix")
        heat_fig = build_sensitivity_heatmap(
            wacc_axes=active_val.wacc_sensitivity_axes,
            tgr_axes=active_val.tgr_sensitivity_axes,
            price_matrix=active_val.sensitivity_matrix_prices,
            current_price=snapshot.current_price,
            currency=snapshot.currency,
        )
        st.plotly_chart(heat_fig, use_container_width=True)

        # Peer Multiples Table
        st.markdown("#### 6. Relative Comparable Company Valuation Multiples")
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

        if state.mode == "llm":
            model_name = getattr(state, "llm_model_used", "") or "Groq"
            tab5_badge = f"""
            <div style="background-color: rgba(167, 139, 250, 0.15); border: 1px solid #8b5cf6; border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; display: inline-flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.1rem;">🤖</span>
                <span style="color: #c4b5fd; font-weight: 600; font-size: 0.9rem;">
                    Dialectical Debate & Commentary Generated by: <b>Live Groq LLM ({model_name})</b>
                </span>
            </div>
            """
        else:
            tab5_badge = f"""
            <div style="background-color: rgba(56, 189, 248, 0.15); border: 1px solid #0284c7; border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; display: inline-flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.1rem;">⚙️</span>
                <span style="color: #38bdf8; font-weight: 600; font-size: 0.9rem;">
                    Dialectical Debate & Commentary Generated by: <b>Deterministic Rule Synthesizer</b>
                </span>
            </div>
            """
        st.markdown(tab5_badge, unsafe_allow_html=True)

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
                "status": getattr(state, "status", "SUCCESS"),
                "repair_attempts": getattr(state, "repair_attempts", 0),
                "llm_model_used": getattr(state, "llm_model_used", ""),
                "llm_latency_ms": getattr(state, "llm_latency_ms", 0.0),
                "snapshot": snapshot.model_dump(),
                "fsa_summary": fsa.model_dump(),
                "quant_summary": quant.model_dump(),
                "dcf_assumptions": assumptions.model_dump(),
                "valuation_output": active_val.model_dump(),
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
