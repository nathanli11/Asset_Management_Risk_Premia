"""
pages/analysis.py — Page 2 : Analyse du Portefeuille Backtesté
"""
import sys
import os

_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR   = os.path.dirname(_PAGES_DIR)
_SRC_DIR   = os.path.dirname(_APP_DIR)
for _p in [_APP_DIR, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import traceback
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State, callback, no_update, dash_table
import dash_bootstrap_components as dbc

from utils import (
    build_reporting_engine, deserialize_result, get_as_of_date,
    fmt_pct, fmt_ratio, fmt_eur, metric_color_class, config_summary,
)

dash.register_page(__name__, path="/analysis", name="Portfolio Analysis", title="Portfolio Analysis | Amundi")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _graph(fig, height=380, cid=""):
    kwargs = dict(
        figure=fig,
        config={"displayModeBar": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        style={"height": f"{height}px"},
    )
    if cid:
        kwargs["id"] = f"graph-{cid}"
    return dcc.Graph(**kwargs)


def _card(title, body, extra=""):
    return html.Div([
        html.Div(html.Span(title, className="card-title"), className="card-header"),
        html.Div(body, className="card-body"),
    ], className=f"card {extra}")


def _empty_state(msg="Aucune donnée disponible."):
    return html.Div([
        html.Div("📭", style={"fontSize": "32px", "opacity": ".4"}),
        html.Div(msg, className="text-muted text-small mt-8"),
    ], className="preview-placeholder")


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return go.Figure()


# ── KPI cards helper ──────────────────────────────────────────────────────────

_KPI_DEFS = [
    ("Rendement cumulé", "Rendement cumulé (période)", fmt_pct, True),
    ("CAGR (période)",   "CAGR (période)",             fmt_pct, True),
    ("Sharpe ratio",     "Sharpe ratio (période)",      fmt_ratio, True),
    ("Sortino ratio",    "Sortino ratio (période)",     fmt_ratio, True),
    ("Max Drawdown",     "Max Drawdown (période)",      fmt_pct,  False),
    ("Volatilité",       "Volatilité (période)",        lambda v: fmt_pct(v, sign=False), None),
    ("Info Ratio",       "Information Ratio (période)", fmt_ratio, True),
    ("PnL (€)",          "PnL (€)",                    fmt_eur,  True),
]


def _make_kpi_cards(metrics: dict):
    cards = []
    for label, key, fmt_fn, pos_good in _KPI_DEFS:
        val = metrics.get(key)
        fmt_val = fmt_fn(val)
        if pos_good is None:
            css = "kpi-neu"
        else:
            cls = metric_color_class(val, pos_good)
            css = "kpi-pos" if "pos" in cls else ("kpi-neg" if "neg" in cls else "kpi-neu")
        cards.append(
            html.Div([
                html.Div(label, className="kpi-label"),
                html.Div(fmt_val, className="kpi-value"),
            ], className=f"kpi-card {css}")
        )
    return html.Div(cards, className="kpi-grid", id="ana-kpi-grid")


# ── Layout ───────────────────────────────────────────────────────────────────

layout = html.Div([
    # ── En-tête avec sélecteur ────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("Portfolio Analysis", className="page-title"),
            html.Div("Reporting complet de la stratégie sélectionnée",
                     className="page-subtitle"),
        ]),
        html.Div([
            html.Span("Portefeuille :", className="selector-label"),
            dcc.Dropdown(
                id="ana-portfolio-selector",
                placeholder="Sélectionnez un backtest sauvegardé…",
                clearable=False,
                className="portfolio-selector",
                style={"fontSize": "13px", "minWidth": "320px"},
            ),
        ], className="flex flex-center gap-12"),
    ], className="page-header"),

    # ── KPI cards (zone dynamique) ────────────────────────────────────
    html.Div(id="ana-kpi-cards"),

    # ── Onglets ───────────────────────────────────────────────────────
    html.Div([
        dbc.Tabs([
            dbc.Tab(label="Vue d'ensemble",  tab_id="tab-overview"),
            dbc.Tab(label="Performance",     tab_id="tab-performance"),
            dbc.Tab(label="Allocation",      tab_id="tab-allocation"),
            dbc.Tab(label="Attribution",     tab_id="tab-attribution"),
            dbc.Tab(label="Portefeuille",    tab_id="tab-holdings"),
        ], id="ana-tabs", active_tab="tab-overview"),

        # Sélecteur de date de rebalancement (visible pour certains onglets)
        html.Div([
            html.Span("Date de rebalancement :", className="selector-label"),
            dcc.Dropdown(
                id="ana-rebal-date",
                placeholder="Sélectionnez une date…",
                clearable=False,
                style={"fontSize": "12px", "minWidth": "200px"},
            ),
            html.Div(id="ana-rebal-date-wrap", style={"display": "none"}),
        ], id="ana-date-selector-row",
           className="flex flex-center gap-12 mt-12",
           style={"display": "none"}),

        # Contenu de l'onglet actif
        dcc.Loading(
            type="circle", color="#00A3AD",
            children=html.Div(id="ana-tab-content", className="tab-content-area"),
        ),
    ], className="card card-body"),

], className="page-wrap")


# ── Callback : options du sélecteur de portefeuille ──────────────────────────

@callback(
    Output("ana-portfolio-selector", "options"),
    Input("backtest-store", "data"),
)
def update_portfolio_options(store_data):
    if not store_data:
        return []
    return [
        {"label": f"{label}  —  {config_summary(data.get('config', {}))}", "value": label}
        for label, data in store_data.items()
    ]


# ── Callback : dates de rebalancement ────────────────────────────────────────

@callback(
    Output("ana-rebal-date", "options"),
    Output("ana-rebal-date", "value"),
    Input("ana-portfolio-selector", "value"),
    State("backtest-store", "data"),
)
def update_rebal_dates(portfolio_id, store_data):
    if not portfolio_id or not store_data or portfolio_id not in store_data:
        return [], None
    data = store_data[portfolio_id]
    dates = sorted(data.get("rebalancing_dates", []))
    options = [{"label": d[:10], "value": d} for d in dates]
    return options, dates[-1] if dates else None


# ── Callback : visibilité du sélecteur de date ────────────────────────────────

@callback(
    Output("ana-date-selector-row", "style"),
    Input("ana-tabs", "active_tab"),
)
def toggle_date_selector(active_tab):
    needs_date = active_tab in ("tab-allocation", "tab-holdings")
    return {} if needs_date else {"display": "none"}


# ── Callback : KPI cards ──────────────────────────────────────────────────────

@callback(
    Output("ana-kpi-cards", "children"),
    Input("ana-portfolio-selector", "value"),
    State("backtest-store", "data"),
)
def update_kpi_cards(portfolio_id, store_data):
    if not portfolio_id or not store_data or portfolio_id not in store_data:
        return html.Div()

    try:
        data = store_data[portfolio_id]
        re   = build_reporting_engine(data)
        result = deserialize_result(data)
        as_of  = result.daily_returns.index.max()
        m = re.compute_metrics(as_of)
        return _make_kpi_cards(m)
    except Exception:
        traceback.print_exc()
        return html.Div()


# ── Callback : Contenu de l'onglet ────────────────────────────────────────────

@callback(
    Output("ana-tab-content", "children"),
    Input("ana-portfolio-selector", "value"),
    Input("ana-tabs",               "active_tab"),
    Input("ana-rebal-date",         "value"),
    State("backtest-store", "data"),
)
def render_tab(portfolio_id, active_tab, rebal_date, store_data):
    if not portfolio_id or not store_data or portfolio_id not in store_data:
        return _empty_state("Sélectionnez un portefeuille dans le menu ci-dessus pour afficher l'analyse.")

    try:
        data   = store_data[portfolio_id]
        re     = build_reporting_engine(data)
        result = deserialize_result(data)

        if active_tab == "tab-overview":
            return _tab_overview(re)
        elif active_tab == "tab-performance":
            return _tab_performance(re, result)
        elif active_tab == "tab-allocation":
            return _tab_allocation(re, rebal_date)
        elif active_tab == "tab-attribution":
            return _tab_attribution(re)
        elif active_tab == "tab-holdings":
            return _tab_holdings(re, rebal_date)
        return html.Div()
    except Exception:
        traceback.print_exc()
        return html.Div([
            html.Div("Une erreur est survenue lors du chargement des données.", className="status-error"),
        ])


# ── Tab renderers ─────────────────────────────────────────────────────────────

def _tab_overview(re):
    fig_cum = re.plot_cumulative_returns_vs_all_benchmarks()
    fig_cum.update_layout(margin=dict(t=40, b=20, l=10, r=10), height=360,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    fig_dd = re.plot_drawdowns()
    fig_dd.update_layout(margin=dict(t=40, b=20, l=10, r=10), height=280)

    return html.Div([
        html.Div([
            html.Div(_graph(fig_cum, height=360), className="card"),
        ], className="mb-16"),
        html.Div(_graph(fig_dd, height=280), className="card"),
    ])


def _tab_performance(re, result):
    fig_cal  = re.plot_calendar_returns_heatmap()
    fig_vol  = re.plot_historical_volatility()
    fig_corr = re.plot_historical_correlation()
    fig_pnl  = re.plot_pnl()

    for f in [fig_cal, fig_vol, fig_corr, fig_pnl]:
        f.update_layout(margin=dict(t=40, b=20, l=10, r=10), height=300)

    # Metrics table overview
    as_of = result.daily_returns.index.max()
    m = re.compute_metrics(as_of)
    metric_rows = [
        ("Rendement annualisé",      fmt_pct(m.get("Rendement annualisé (période)"))),
        ("Rendement cumulé YTD",     fmt_pct(m.get("Rendement cumulé YTD"))),
        ("Rendement cumulé 1 an",    fmt_pct(m.get("Rendement cumulé 1 an"))),
        ("Rendement cumulé 2 ans",   fmt_pct(m.get("Rendement cumulé 2 ans"))),
        ("Sharpe 1 an",              fmt_ratio(m.get("Sharpe ratio 1 an"))),
        ("Sortino 1 an",             fmt_ratio(m.get("Sortino ratio 1 an"))),
        ("VaR 95% (période)",        fmt_pct(m.get("VaR 95% (période)"), sign=False)),
        ("CVaR 95% (période)",       fmt_pct(m.get("CVaR 95% (période)"), sign=False)),
        ("Tracking Error",           fmt_pct(m.get("Tracking Error (période)"), sign=False)),
        ("Information Ratio",        fmt_ratio(m.get("Information Ratio (période)"))),
        ("Corrélation benchmark",    fmt_ratio(m.get("Corrélation avec benchmark (période)"))),
        ("TC total (%)",             fmt_pct(m.get("TC total (%)"), sign=False)),
        ("TC total (€)",             fmt_eur(m.get("TC total (€)"))),
        ("Benchmark cumulé (période)", fmt_pct(m.get("Benchmark - Rendement cumulé (période)"))),
        ("Benchmark annualisé",      fmt_pct(m.get("Benchmark - Rendement annualisé (période)"))),
    ]
    tbl_rows = [html.Tr([html.Td(k, className="text-muted fw-600"), html.Td(v)]) for k, v in metric_rows]
    tbl = html.Table([
        html.Tbody(tbl_rows),
    ], className="library-table")

    return html.Div([
        html.Div([
            html.Div(_graph(fig_cal, height=260), className="card"),
        ], className="mb-16"),
        html.Div([
            html.Div(_graph(fig_vol, height=300), className="card"),
            html.Div(_graph(fig_corr, height=300), className="card"),
        ], className="chart-grid-2 mb-16"),
        html.Div([
            html.Div(_graph(fig_pnl, height=280), className="card"),
            _card("Métriques détaillées", tbl),
        ], className="chart-grid-2-1"),
    ])


def _tab_allocation(re, rebal_date):
    if not rebal_date:
        return _empty_state("Sélectionnez une date de rebalancement.")
    rd = pd.Timestamp(rebal_date)

    # Composition charts
    figs_comp = re.plot_composition_barcharts(rd)
    # Allocation vs benchmark
    figs_vs = re.plot_group_allocation_vs_benchmark_barcharts(rd)
    # Return impact
    figs_impact = re.plot_group_return_impact_barcharts(rd)

    def _tab_group(label_map, figs_dict, height=340):
        tabs = []
        for dim, lbl in label_map.items():
            fig = figs_dict.get(dim, go.Figure())
            fig.update_layout(margin=dict(t=40, b=20, l=10, r=10), height=height)
            tabs.append(dbc.Tab(label=lbl, tab_id=f"dim-{dim}",
                                children=html.Div(_graph(fig, height=height), className="mt-12")))
        return dbc.Tabs(tabs, active_tab=f"dim-{list(label_map.keys())[0]}")

    dims = {"Sector": "Secteur", "Country": "Pays", "Industry": "Industrie", "Currency": "Devise"}

    return html.Div([
        html.Div([
            _card("Composition du portefeuille",
                  _tab_group(dims, figs_comp, 340)),
        ], className="mb-16"),
        html.Div([
            _card("Portefeuille vs Benchmark (MSCI World)",
                  _tab_group(dims, figs_vs, 340)),
        ], className="mb-16"),
        html.Div([
            _card("Impact sur le rendement par groupe",
                  _tab_group(dims, figs_impact, 340)),
        ]),
    ])


def _tab_attribution(re):
    dims = {"Sector": "Secteur", "Country": "Pays", "Industry": "Industrie"}
    tabs = []
    for dim, lbl in dims.items():
        try:
            df = re.get_group_yearly_attribution_report(group_by=dim)
            if df.empty:
                content = _empty_state("Données insuffisantes pour l'attribution.")
            else:
                value_cols = [c for c in df.columns if c != "Analyse"]
                tbl = dash_table.DataTable(
                    data=df.to_dict("records"),
                    columns=[
                        {"name": "Analyse", "id": "Analyse"} if c == "Analyse"
                        else {"name": str(c), "id": str(c), "type": "numeric",
                              "format": dash_table.Format.Format(
                                  precision=2, scheme=dash_table.Format.Scheme.percentage
                              )}
                        for c in df.columns
                    ],
                    style_table={"overflowX": "auto", "borderRadius": "6px"},
                    style_header={
                        "backgroundColor": "#0E2A47",
                        "color": "rgba(255,255,255,.85)",
                        "fontWeight": "600",
                        "fontSize": "11px",
                        "textTransform": "uppercase",
                        "letterSpacing": ".5px",
                    },
                    style_cell={
                        "fontFamily": "Inter, sans-serif",
                        "fontSize": "12px",
                        "padding": "8px 12px",
                        "border": "1px solid #D9E2EC",
                        "textAlign": "right",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": "Analyse"}, "textAlign": "left", "fontWeight": "500",
                         "color": "#6B7FA3", "minWidth": "260px"},
                    ],
                    style_data_conditional=[
                        {"if": {"filter_query": "{Analyse} contains 'TOTAL'"},
                         "backgroundColor": "#FFF8E1", "fontWeight": "700"},
                        {"if": {"filter_query": "{Analyse} = ''"},
                         "backgroundColor": "transparent", "height": "8px"},
                    ] + [
                        {"if": {"column_id": c, "filter_query": f"{{{c}}} > 0"},
                         "color": "#00A650"}
                        for c in [str(x) for x in value_cols]
                    ] + [
                        {"if": {"column_id": c, "filter_query": f"{{{c}}} < 0"},
                         "color": "#E30613"}
                        for c in [str(x) for x in value_cols]
                    ],
                    page_size=40,
                    sort_action="native",
                )
                content = html.Div(tbl, className="mt-12 data-table-wrap")
        except Exception:
            traceback.print_exc()
            content = _empty_state("Erreur lors du calcul de l'attribution.")

        tabs.append(dbc.Tab(label=lbl, tab_id=f"attr-{dim}", children=content))

    return html.Div([
        html.Div("Attribution Brinson-Fachler annuelle par groupe", className="section-label mb-8"),
        html.Div("Effets Allocation, Sélection, Interaction et Actif par année.", className="text-muted text-small mb-12"),
        dbc.Tabs(tabs, active_tab="attr-Sector"),
    ])


def _tab_holdings(re, rebal_date):
    if not rebal_date:
        return _empty_state("Sélectionnez une date de rebalancement.")
    rd = pd.Timestamp(rebal_date)

    try:
        comp = re.get_portfolio_composition(rd)
        top10 = re.get_top_10_weights(rd)
    except Exception:
        traceback.print_exc()
        return _empty_state("Erreur lors du chargement des données de composition.")

    if comp.empty:
        return _empty_state("Aucune donnée de composition à cette date.")

    comp_disp = comp.copy()
    comp_disp["Weight"] = (comp_disp["Weight"] * 100).round(4)
    comp_disp["Price"]  = comp_disp["Price"].round(2)
    if "Abs Weight" in comp_disp.columns:
        comp_disp = comp_disp.drop(columns=["Abs Weight"])

    tbl_full = dash_table.DataTable(
        data=comp_disp.to_dict("records"),
        columns=[{"name": c, "id": c} for c in comp_disp.columns],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#0E2A47",
            "color": "rgba(255,255,255,.85)",
            "fontWeight": "600",
            "fontSize": "11px",
        },
        style_cell={
            "fontFamily": "Inter, sans-serif",
            "fontSize": "12px",
            "padding": "7px 12px",
            "border": "1px solid #D9E2EC",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#F7F9FC"},
        ],
        page_size=25,
        sort_action="native",
        filter_action="native",
    )

    # Top 10 chart
    if not top10.empty:
        top10_disp = top10.head(10).copy()
        weight_col = "Abs Weight" if "Abs Weight" in top10_disp.columns else "Weight"
        tickers = top10_disp["Ticker"].tolist()
        weights = (top10_disp[weight_col] * 100).tolist()
        fig_top = go.Figure(go.Bar(
            y=tickers[::-1], x=weights[::-1],
            orientation="h",
            marker_color="#00A3AD",
            text=[f"{w:.2f}%" for w in weights[::-1]],
            textposition="outside",
        ))
        fig_top.update_layout(
            title="Top 10 positions par poids",
            xaxis_title="Poids (%)",
            margin=dict(t=40, b=20, l=10, r=60),
            height=340,
            plot_bgcolor="#F7F9FC",
            paper_bgcolor="white",
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=12),
        )
        top10_section = html.Div(_graph(fig_top, height=340), className="card mb-16")
    else:
        top10_section = html.Div()

    nb = len(comp_disp)
    total_w = comp_disp["Weight"].sum()
    header = html.Div([
        html.Span(f"{nb} positions", className="tag"),
        html.Span(f"  Poids total : {total_w:.2f}%",
                  className="text-muted text-small ml-8"),
    ], className="flex flex-center gap-8 mb-12")

    return html.Div([
        top10_section,
        _card("Composition complète du portefeuille", [
            header,
            html.Div(tbl_full, className="data-table-wrap"),
        ]),
    ])
