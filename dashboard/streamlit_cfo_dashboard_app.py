"""
Streamlit CFO Dashboard for Manufacturing Dataset
Author: FP&A / BI Dashboard Template

How to run locally:
    pip install -r requirements.txt
    streamlit run streamlit_cfo_dashboard_app.py

Expected dataset columns:
Date, Region, Product, Revenue, Cost, Profit, Units Sold, Customer Satisfaction,
Manufacturing Cost, Labor Hours, Machine Downtime (hours), Inventory Levels,
Order Lead Time (days), Return Rate (%), On-Time Delivery (%)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------- Page configuration -----------------------------
st.set_page_config(
    page_title="Manufacturing CFO AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_FILE = "manufacturing_company_dataset(3).xlsx"
DATE_COL = "Date"
BLUE = "#2563EB"
CYAN = "#22D3EE"
NAVY = "#0F172A"
BG = "#F8FAFC"

# ----------------------------- Styling -----------------------------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 42%, #DBEAFE 100%);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F172A 0%, #1D4ED8 55%, #38BDF8 100%);
        }}
        [data-testid="stSidebar"] * {{ color: white !important; }}
        .hero {{
            padding: 26px 30px;
            border-radius: 24px;
            background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 55%, #38BDF8 100%);
            color: white;
            box-shadow: 0 20px 50px rgba(30, 64, 175, 0.25);
            margin-bottom: 20px;
        }}
        .hero h1 {{ margin: 0; font-size: 2.35rem; letter-spacing: -0.04em; }}
        .hero p {{ margin: 8px 0 0 0; color: #DBEAFE; font-size: 1.02rem; }}
        .metric-card {{
            padding: 20px 18px;
            border-radius: 20px;
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
            min-height: 126px;
        }}
        .metric-label {{ color: #64748B; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.10em; font-weight: 800; }}
        .metric-value {{ color: #0F172A; font-size: 1.78rem; font-weight: 900; margin-top: 8px; }}
        .metric-delta-pos {{ color: #059669; font-weight: 800; font-size: 0.88rem; }}
        .metric-delta-neg {{ color: #DC2626; font-weight: 800; font-size: 0.88rem; }}
        .glass {{
            padding: 18px;
            border-radius: 20px;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.06);
            margin-bottom: 16px;
        }}
        h2, h3 {{ color: #0F172A; letter-spacing: -0.02em; }}
        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.78);
            padding: 16px;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------- Data model -----------------------------
@dataclass(frozen=True)
class KpiSet:
    revenue: float
    gross_profit: float
    ebitda: float
    net_margin: float
    inventory_turns: float
    working_capital: float


def find_default_file() -> Path | None:
    candidates = [Path(DEFAULT_FILE), Path("data") / DEFAULT_FILE, Path("/mnt/data") / DEFAULT_FILE]
    return next((p for p in candidates if p.exists()), None)


@st.cache_data(show_spinner=False)
def load_data_from_bytes(file_bytes: bytes, name: str) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_bytes)
    elif suffix == ".csv":
        df = pd.read_csv(file_bytes)
    else:
        raise ValueError("Upload an Excel or CSV file.")
    return clean_data(df)


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    return clean_data(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if DATE_COL not in df.columns:
        raise ValueError(f"Dataset must include a '{DATE_COL}' column.")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

    numeric_cols = [
        "Revenue", "Cost", "Profit", "Units Sold", "Customer Satisfaction",
        "Manufacturing Cost", "Labor Hours", "Machine Downtime (hours)",
        "Inventory Levels", "Order Lead Time (days)", "Return Rate (%)", "On-Time Delivery (%)",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Month"] = df[DATE_COL].dt.to_period("M").dt.to_timestamp()
    df["Quarter"] = df[DATE_COL].dt.to_period("Q").astype(str)
    df["Gross Profit"] = df["Revenue"] - df["Cost"]
    df["EBITDA"] = df["Gross Profit"] - df["Manufacturing Cost"]
    df["Net Margin"] = np.where(df["Revenue"] != 0, df["Profit"] / df["Revenue"], 0)
    df["Gross Margin"] = np.where(df["Revenue"] != 0, df["Gross Profit"] / df["Revenue"], 0)
    df["EBITDA Margin"] = np.where(df["Revenue"] != 0, df["EBITDA"] / df["Revenue"], 0)
    df["Unit Price"] = np.where(df["Units Sold"] != 0, df["Revenue"] / df["Units Sold"], 0)
    df["Unit Cost"] = np.where(df["Units Sold"] != 0, df["Cost"] / df["Units Sold"], 0)
    df["Inventory Value"] = df["Inventory Levels"] * df["Unit Cost"].replace([np.inf, -np.inf], 0)
    df["Working Capital"] = df["Inventory Value"]  # Dataset proxy; AR/AP not provided.
    df["Inventory Turns"] = np.where(df["Inventory Value"] > 0, df["Cost"] / df["Inventory Value"], 0)
    return df.replace([np.inf, -np.inf], 0)


def aggregate_kpis(df: pd.DataFrame) -> KpiSet:
    revenue = df["Revenue"].sum()
    gross_profit = df["Gross Profit"].sum()
    ebitda = df["EBITDA"].sum()
    profit = df["Profit"].sum()
    net_margin = profit / revenue if revenue else 0
    inv_turns = df["Cost"].sum() / df["Inventory Value"].mean() if df["Inventory Value"].mean() else 0
    working_capital = df["Working Capital"].mean()
    return KpiSet(revenue, gross_profit, ebitda, net_margin, inv_turns, working_capital)


def format_money(v: float) -> str:
    return f"${v/1_000_000:.2f}M" if abs(v) >= 1_000_000 else f"${v/1_000:.1f}K"


def format_pct(v: float) -> str:
    return f"{v:.1%}"


def delta_text(current: float, prior: float, invert: bool = False) -> Tuple[str, str]:
    if prior == 0:
        return "N/A", "metric-delta-pos"
    delta = (current - prior) / abs(prior)
    positive = delta >= 0
    if invert:
        positive = not positive
    return f"{delta:+.1%} vs prior", "metric-delta-pos" if positive else "metric-delta-neg"


def kpi_card(label: str, value: str, delta: str = "", delta_class: str = "metric-delta-pos") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="{delta_class}">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def base_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin=dict(l=20, r=20, t=60, b=35),
        font=dict(family="Inter, Arial", color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Month", as_index=False).agg(
        Revenue=("Revenue", "sum"),
        Cost=("Cost", "sum"),
        Profit=("Profit", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        EBITDA=("EBITDA", "sum"),
        Units=("Units Sold", "sum"),
        Inventory=("Inventory Levels", "mean"),
        Downtime=("Machine Downtime (hours)", "mean"),
        Lead_Time=("Order Lead Time (days)", "mean"),
        Return_Rate=("Return Rate (%)", "mean"),
        OTD=("On-Time Delivery (%)", "mean"),
    )


def ai_commentary(df: pd.DataFrame) -> List[str]:
    m = monthly(df)
    k = aggregate_kpis(df)
    latest = m.iloc[-1]
    prior = m.iloc[-2] if len(m) > 1 else latest
    rev_growth = (latest["Revenue"] - prior["Revenue"]) / abs(prior["Revenue"]) if prior["Revenue"] else 0
    margin = k.ebitda / k.revenue if k.revenue else 0
    top_region = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False).index[0]
    top_product = df.groupby("Product")["Gross Profit"].sum().sort_values(ascending=False).index[0]
    worst_return = df.groupby("Product")["Return Rate (%)"].mean().sort_values(ascending=False).index[0]
    return [
        f"Revenue in the latest month moved {rev_growth:+.1%} versus the prior month, indicating {'momentum' if rev_growth >= 0 else 'softness'} in recent demand.",
        f"EBITDA margin is {margin:.1%}; prioritize manufacturing-cost reduction where product gross margin is already healthy.",
        f"{top_region} is the largest revenue region, while {top_product} contributes the strongest gross profit pool.",
        f"{worst_return} has the highest average return rate; quality and after-sales diagnostics should be reviewed.",
        f"Working capital is proxied by inventory value because AR/AP are not present in the source dataset.",
    ]

# ----------------------------- Shared UI -----------------------------
def header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Manufacturing CFO AI Dashboard</h1>
            <p>Executive FP&A cockpit for revenue, profitability, operations, forecasting, scenarios, and automated insights.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("CFO Controls")
    uploaded = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"])
    st.sidebar.caption("Filters apply across all dashboard pages.")

    min_date, max_date = df[DATE_COL].min().date(), df[DATE_COL].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    regions = st.sidebar.multiselect("Region", sorted(df["Region"].dropna().unique()), default=sorted(df["Region"].dropna().unique()))
    products = st.sidebar.multiselect("Product", sorted(df["Product"].dropna().unique()), default=sorted(df["Product"].dropna().unique()))

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start, end = pd.to_datetime(min_date), pd.to_datetime(max_date)

    out = df[(df[DATE_COL] >= start) & (df[DATE_COL] <= end)]
    out = out[out["Region"].isin(regions) & out["Product"].isin(products)]
    return out


def kpi_strip(df: pd.DataFrame) -> None:
    k = aggregate_kpis(df)
    m = monthly(df)
    prior_df = df[df["Month"] < df["Month"].max()]
    prior = aggregate_kpis(prior_df) if not prior_df.empty else k
    vals = [
        ("Revenue", format_money(k.revenue), *delta_text(k.revenue, prior.revenue)),
        ("Gross Profit", format_money(k.gross_profit), *delta_text(k.gross_profit, prior.gross_profit)),
        ("EBITDA", format_money(k.ebitda), *delta_text(k.ebitda, prior.ebitda)),
        ("Net Margin", format_pct(k.net_margin), *delta_text(k.net_margin, prior.net_margin)),
        ("Inventory Turns", f"{k.inventory_turns:.2f}x", *delta_text(k.inventory_turns, prior.inventory_turns)),
        ("Working Capital", format_money(k.working_capital), *delta_text(k.working_capital, prior.working_capital, invert=True)),
    ]
    cols = st.columns(6)
    for col, item in zip(cols, vals):
        with col:
            kpi_card(*item)

# ----------------------------- Pages -----------------------------
def executive_summary(df: pd.DataFrame) -> None:
    kpi_strip(df)
    c1, c2 = st.columns([2, 1])
    m = monthly(df)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["Month"], y=m["Revenue"], mode="lines+markers", name="Revenue"))
        fig.add_trace(go.Scatter(x=m["Month"], y=m["EBITDA"], mode="lines+markers", name="EBITDA"))
        st.plotly_chart(base_layout(fig, 430).update_layout(title="Revenue and EBITDA Trend"), use_container_width=True)
    with c2:
        st.markdown('<div class="glass"><h3>AI Executive Commentary</h3>', unsafe_allow_html=True)
        for insight in ai_commentary(df):
            st.markdown(f"• {insight}")
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        bridge = [df["Revenue"].sum(), -df["Cost"].sum(), -df["Manufacturing Cost"].sum(), df["EBITDA"].sum()]
        fig = go.Figure(go.Waterfall(
            x=["Revenue", "COGS", "Mfg Cost", "EBITDA"], y=bridge,
            measure=["absolute", "relative", "relative", "total"], connector={"line": {"color": "#94A3B8"}}
        ))
        st.plotly_chart(base_layout(fig).update_layout(title="Profit Waterfall"), use_container_width=True)
    with c4:
        by_region = df.groupby("Region", as_index=False)[["Revenue", "Gross Profit", "EBITDA"]].sum()
        fig = px.bar(by_region, x="Region", y=["Revenue", "Gross Profit", "EBITDA"], barmode="group", title="Regional Performance")
        st.plotly_chart(base_layout(fig), use_container_width=True)


def revenue_analysis(df: pd.DataFrame) -> None:
    m = monthly(df)
    fig = px.area(m, x="Month", y="Revenue", title="Revenue Trend")
    st.plotly_chart(base_layout(fig, 440), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.treemap(df, path=["Region", "Product"], values="Revenue", color="Gross Margin", title="Revenue Driver Tree: Region → Product")
        st.plotly_chart(base_layout(fig), use_container_width=True)
    with c2:
        mix = df.groupby(["Product", "Region"], as_index=False)["Revenue"].sum()
        fig = px.bar(mix, x="Product", y="Revenue", color="Region", title="Product / Region Revenue Mix")
        st.plotly_chart(base_layout(fig), use_container_width=True)


def profitability_analysis(df: pd.DataFrame) -> None:
    m = monthly(df)
    m["Gross Margin"] = np.where(m["Revenue"] > 0, m["Gross_Profit"] / m["Revenue"], 0)
    m["EBITDA Margin"] = np.where(m["Revenue"] > 0, m["EBITDA"] / m["Revenue"], 0)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(m, x="Month", y=["Gross Margin", "EBITDA Margin"], markers=True, title="Margin Trend")
        st.plotly_chart(base_layout(fig), use_container_width=True)
    with c2:
        by_product = df.groupby("Product", as_index=False).agg(Revenue=("Revenue", "sum"), Gross_Profit=("Gross Profit", "sum"), EBITDA=("EBITDA", "sum"))
        fig = px.scatter(by_product, x="Revenue", y="EBITDA", size="Gross_Profit", text="Product", title="Profitability Bubble Map")
        st.plotly_chart(base_layout(fig), use_container_width=True)
    fig = go.Figure(go.Waterfall(
        x=["Revenue", "COGS", "Gross Profit", "Mfg Cost", "EBITDA", "Reported Profit"],
        y=[df["Revenue"].sum(), -df["Cost"].sum(), df["Gross Profit"].sum(), -df["Manufacturing Cost"].sum(), df["EBITDA"].sum(), df["Profit"].sum()],
        measure=["absolute", "relative", "total", "relative", "total", "total"]
    ))
    st.plotly_chart(base_layout(fig, 430).update_layout(title="Full Profit Bridge"), use_container_width=True)


def operational_kpis(df: pd.DataFrame) -> None:
    ops = df.groupby("Month", as_index=False).agg(
        Inventory=("Inventory Levels", "mean"), Downtime=("Machine Downtime (hours)", "mean"),
        Lead_Time=("Order Lead Time (days)", "mean"), Return_Rate=("Return Rate (%)", "mean"), OTD=("On-Time Delivery (%)", "mean"),
        Satisfaction=("Customer Satisfaction", "mean"), Labor=("Labor Hours", "mean"), Units=("Units Sold", "sum")
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(ops, x="Month", y=["Inventory", "Downtime", "Lead_Time"], markers=True, title="Operational Trend KPIs")
        st.plotly_chart(base_layout(fig), use_container_width=True)
    with c2:
        heat = df.pivot_table(index="Region", columns="Product", values="On-Time Delivery (%)", aggfunc="mean")
        fig = px.imshow(heat, text_auto=".1f", aspect="auto", title="On-Time Delivery Heatmap")
        st.plotly_chart(base_layout(fig), use_container_width=True)
    fig = px.scatter(df, x="Machine Downtime (hours)", y="Manufacturing Cost", size="Units Sold", color="Region", hover_data=["Product"], title="Downtime vs Manufacturing Cost Driver View")
    st.plotly_chart(base_layout(fig, 430), use_container_width=True)


def forecasting(df: pd.DataFrame) -> None:
    st.markdown("### Forecast Simulation")
    horizon = st.slider("Forecast horizon (months)", 3, 24, 12)
    growth_adj = st.slider("Revenue growth adjustment", -0.25, 0.50, 0.05, 0.01)
    margin_adj = st.slider("EBITDA margin improvement", -0.10, 0.20, 0.02, 0.01)
    m = monthly(df)
    series = m["Revenue"].values
    x = np.arange(len(series))
    coef = np.polyfit(x, series, 1) if len(series) > 1 else [0, series[-1]]
    future_x = np.arange(len(series), len(series) + horizon)
    last_month = m["Month"].max()
    future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    forecast_rev = np.maximum(np.polyval(coef, future_x) * (1 + growth_adj), 0)
    ebitda_margin = df["EBITDA"].sum() / df["Revenue"].sum() if df["Revenue"].sum() else 0
    forecast_ebitda = forecast_rev * (ebitda_margin + margin_adj)
    f = pd.DataFrame({"Month": future_months, "Revenue Forecast": forecast_rev, "EBITDA Forecast": forecast_ebitda})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m["Month"], y=m["Revenue"], mode="lines+markers", name="Actual Revenue"))
    fig.add_trace(go.Scatter(x=f["Month"], y=f["Revenue Forecast"], mode="lines+markers", name="Forecast Revenue"))
    fig.add_trace(go.Scatter(x=f["Month"], y=f["EBITDA Forecast"], mode="lines+markers", name="Forecast EBITDA"))
    st.plotly_chart(base_layout(fig, 460).update_layout(title="Revenue and EBITDA Forecast"), use_container_width=True)
    st.dataframe(f.style.format({"Revenue Forecast": "${:,.0f}", "EBITDA Forecast": "${:,.0f}"}), use_container_width=True)


def scenario_planning(df: pd.DataFrame) -> None:
    st.markdown("### CFO Scenario Lab")
    c1, c2, c3, c4 = st.columns(4)
    revenue_change = c1.slider("Revenue change", -0.30, 0.50, 0.10, 0.01)
    cogs_change = c2.slider("COGS change", -0.20, 0.30, 0.03, 0.01)
    mfg_change = c3.slider("Manufacturing cost change", -0.25, 0.25, -0.05, 0.01)
    inventory_change = c4.slider("Inventory change", -0.30, 0.40, -0.05, 0.01)
    base = aggregate_kpis(df)
    scenario_revenue = base.revenue * (1 + revenue_change)
    scenario_cost = df["Cost"].sum() * (1 + cogs_change)
    scenario_mfg = df["Manufacturing Cost"].sum() * (1 + mfg_change)
    scenario_gp = scenario_revenue - scenario_cost
    scenario_ebitda = scenario_gp - scenario_mfg
    scenario_wc = base.working_capital * (1 + inventory_change)
    scen = pd.DataFrame({
        "KPI": ["Revenue", "Gross Profit", "EBITDA", "Working Capital"],
        "Base": [base.revenue, base.gross_profit, base.ebitda, base.working_capital],
        "Scenario": [scenario_revenue, scenario_gp, scenario_ebitda, scenario_wc],
    })
    scen["Variance"] = scen["Scenario"] - scen["Base"]
    fig = px.bar(scen, x="KPI", y=["Base", "Scenario"], barmode="group", title="Base vs Scenario")
    st.plotly_chart(base_layout(fig), use_container_width=True)
    fig = go.Figure(go.Waterfall(x=scen["KPI"], y=scen["Variance"], measure=["relative"] * len(scen)))
    st.plotly_chart(base_layout(fig).update_layout(title="Scenario Variance Bridge"), use_container_width=True)


def variance_analysis(df: pd.DataFrame) -> None:
    m = monthly(df)
    if len(m) < 2:
        st.warning("Need at least two months for variance analysis.")
        return
    current = m.iloc[-1]
    prior = m.iloc[-2]
    variances = pd.DataFrame({
        "Driver": ["Revenue", "Cost", "Manufacturing Cost", "EBITDA"],
        "Variance": [
            current["Revenue"] - prior["Revenue"],
            -(current["Cost"] - prior["Cost"]),
            -(current["EBITDA"] - (current["Revenue"] - current["Cost"]) - (prior["EBITDA"] - (prior["Revenue"] - prior["Cost"]))),
            current["EBITDA"] - prior["EBITDA"],
        ],
    })
    fig = go.Figure(go.Waterfall(x=variances["Driver"], y=variances["Variance"], measure=["relative", "relative", "relative", "total"]))
    st.plotly_chart(base_layout(fig, 430).update_layout(title="Latest Month EBITDA Variance Bridge"), use_container_width=True)
    by_product = df.groupby(["Month", "Product"], as_index=False)["Revenue"].sum()
    latest_months = by_product[by_product["Month"].isin(m["Month"].tail(2))]
    fig = px.bar(latest_months, x="Product", y="Revenue", color=latest_months["Month"].astype(str), barmode="group", title="Product Revenue Variance")
    st.plotly_chart(base_layout(fig), use_container_width=True)


def ai_insights(df: pd.DataFrame) -> None:
    st.markdown("### AI-Generated FP&A Insights")
    insights = ai_commentary(df)
    for i, item in enumerate(insights, 1):
        st.markdown(f"<div class='glass'><b>Insight {i}</b><br>{item}</div>", unsafe_allow_html=True)
    corr_cols = ["Revenue", "Gross Profit", "EBITDA", "Units Sold", "Manufacturing Cost", "Labor Hours", "Machine Downtime (hours)", "Inventory Levels", "Order Lead Time (days)", "Return Rate (%)", "On-Time Delivery (%)"]
    corr = df[corr_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Financial / Operational Correlation Heatmap")
    st.plotly_chart(base_layout(fig, 620), use_container_width=True)


PAGES = {
    "Executive Summary": executive_summary,
    "Revenue Analysis": revenue_analysis,
    "Profitability Analysis": profitability_analysis,
    "Operational KPIs": operational_kpis,
    "Forecasting": forecasting,
    "Scenario Planning": scenario_planning,
    "Variance Analysis": variance_analysis,
    "AI Insights": ai_insights,
}


def main() -> None:
    inject_css()
    default_path = find_default_file()
    if default_path is None:
        st.sidebar.info("Upload your dataset to begin.")
        uploaded = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"], key="first_upload")
        if uploaded is None:
            st.stop()
        df = load_data_from_bytes(uploaded, uploaded.name)
    else:
        df = load_data_from_path(str(default_path))
        uploaded = st.sidebar.file_uploader("Replace dataset", type=["xlsx", "xls", "csv"])
        if uploaded is not None:
            df = load_data_from_bytes(uploaded, uploaded.name)

    header()
    filtered = sidebar_filters(df)
    if filtered.empty:
        st.warning("No data matches the selected filters.")
        st.stop()

    page = st.sidebar.radio("Dashboard Page", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Rows selected: {len(filtered):,} / {len(df):,}")
    st.sidebar.caption(f"Data period: {filtered[DATE_COL].min().date()} to {filtered[DATE_COL].max().date()}")
    PAGES[page](filtered)


if __name__ == "__main__":
    main()
