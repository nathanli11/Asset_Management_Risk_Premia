"""
pages/comparison.py — Page 3 : Comparaison de Portefeuilles (jusqu'à 4)
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
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback, no_update, dash_table
import dash_bootstrap_components as dbc

from utils import (
    build_reporting_engine, deserialize_result, get_as_of_date,
    fmt_pct, fmt_ratio, fmt_eur, metric_color_class,
    PORTFOLIO_COLORS, config_summary,
)

dash.register_page(__name__, path="/comparison", name="Portfolio Comparison", title="Portfolio Comparison | Amundi")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _empty_state(msg="Sélectionnez au moins 2 portefeuilles pour comparer."):
    return html.Div([
        html.Div("⚖", style={"fontSize": "40px", "opacity": ".3"}),
        html.Div(msg, className="text-muted text-small mt-8"),
    ], className="preview-placeholder", style={"minHeight": "250px"})


def _card(title, body, extra=""):
    return html.Div([
        html.Div(html.Span(title, className="card-title"), className="card-header"),
        html.Div(body, className="card-body"),
    ], className=f"card {extra}")


def _graph(fig, height=360):
    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        style={"height": f"{height}px"},
    )


# ── Métriques à comparer ─────────────────────────────────────────────────────

_COMPARE_METRICS = [
    ("Rendement cumulé (période)",       fmt_pct,  True,  "Performance"),
    ("CAGR (période)",                   fmt_pct,  True,  "Performance"),
    ("Rendement annualisé (période)",    fmt_pct,  True,  "Performance"),
    ("Rendement cumulé YTD",             fmt_pct,  True,  "Performance"),
    ("Rendement cumulé 1 an",            fmt_pct,  True,  "Performance"),
    ("Rendement cumulé 2 ans",           fmt_pct,  True,  "Performance"),
    ("Sharpe ratio (période)",           fmt_ratio, True,  "Ratios"),
    ("Sharpe ratio 1 an",                fmt_ratio, True,  "Ratios"),
    ("Sortino ratio (période)",          fmt_ratio, True,  "Ratios"),
    ("Information Ratio (période)",      fmt_ratio, True,  "Ratios"),
    ("Calmar ratio (période)",           fmt_ratio, True,  "Ratios"),
    ("Volatilité (période)",             lambda v: fmt_pct(v, sign=False), False, "Risque"),
    ("Volatilité 1 an",                  lambda v: fmt_pct(v, sign=False), False, "Risque"),
    ("Max Drawdown (période)",           fmt_pct,  False, "Risque"),
    ("Max Drawdown 1 an",                fmt_pct,  False, "Risque"),
    ("VaR 95% (période)",                lambda v: fmt_pct(v, sign=False), False, "Risque"),
    ("CVaR 95% (période)",               lambda v: fmt_pct(v, sign=False), False, "Risque"),
    ("Tracking Error (période)",         lambda v: fmt_pct(v, sign=False), None,  "Relatif"),
    ("Information Ratio (période)",      fmt_ratio, True,  "Relatif"),
    ("Corrélation avec benchmark (période)", fmt_ratio, None, "Relatif"),
    ("Benchmark - Rendement cumulé (période)", fmt_pct, True, "Benchmark"),
    ("TC total (%)",                     lambda v: fmt_pct(v, sign=False), False, "Coûts"),
    ("TC total (€)",                     fmt_eur,  False, "Coûts"),
    ("PnL (€)",                          fmt_eur,  True,  "Performance"),
]


# ── Layout ───────────────────────────────────────────────────────────────────

def _slot(n: int, color: str):
    return html.Div([
        html.Div([
            html.Div(className="compare-dot", style={"background": color}),
            html.Span(f"Portefeuille {n}", className="compare-slot-label"),
        ], className="compare-slot-header"),
        dcc.Dropdown(
            id=f"cmp-port-{n}",
            placeholder=f"Portefeuille {n}…",
            clearable=True,
            style={"fontSize": "12px"},
        ),
    ], className="compare-slot")


layout = html.Div([
    # ── En-tête ───────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("Portfolio Comparison", className="page-title"),
            html.Div("Comparez jusqu'à 4 portefeuilles sur les mêmes métriques de reporting",
                     className="page-subtitle"),
        ]),
    ], className="page-header"),

    # ── Sélecteurs ───────────────────────────────────────────────────
    html.Div([
        _slot(1, PORTFOLIO_COLORS[0]),
        _slot(2, PORTFOLIO_COLORS[1]),
        _slot(3, PORTFOLIO_COLORS[2]),
        _slot(4, PORTFOLIO_COLORS[3]),
    ], className="compare-selectors"),

    # ── Contenu de la comparaison ─────────────────────────────────────
    dcc.Loading(
        type="circle", color="#00A3AD",
        children=html.Div(id="cmp-content"),
    ),

], className="page-wrap")


# ── Callback : options des sélecteurs ─────────────────────────────────────────

@callback(
    Output("cmp-port-1", "options"),
    Output("cmp-port-2", "options"),
    Output("cmp-port-3", "options"),
    Output("cmp-port-4", "options"),
    Input("backtest-store", "data"),
)
def update_comparison_options(store_data):
    if not store_data:
        opts = []
        return opts, opts, opts, opts
    opts = [
        {"label": f"{label}  —  {config_summary(data.get('config', {}))}", "value": label}
        for label, data in store_data.items()
    ]
    return opts, opts, opts, opts


# ── Callback : contenu de la comparaison ─────────────────────────────────────

@callback(
    Output("cmp-content", "children"),
    Input("cmp-port-1", "value"),
    Input("cmp-port-2", "value"),
    Input("cmp-port-3", "value"),
    Input("cmp-port-4", "value"),
    State("backtest-store", "data"),
)
def update_comparison(p1, p2, p3, p4, store_data):
    selected = [p for p in [p1, p2, p3, p4] if p and store_data and p in store_data]

    if len(selected) < 2:
        return _empty_state()

    try:
        colors_used = PORTFOLIO_COLORS[:len(selected)]
        datas  = [store_data[p] for p in selected]
        engines = []
        results = []
        for d in datas:
            engines.append(build_reporting_engine(d))
            results.append(deserialize_result(d))

        return html.Div([
            _section_metrics(selected, engines, results, colors_used),
            _section_nav(selected, results, colors_used),
            _section_drawdowns(selected, results, colors_used),
            _section_performance_bars(selected, engines, results, colors_used),
            _section_sector_allocation(selected, engines, results, colors_used),
        ])
    except Exception:
        traceback.print_exc()
        return html.Div("Une erreur est survenue.", className="status-error")


# ── Sections de comparaison ───────────────────────────────────────────────────

def _section_metrics(labels, engines, results, colors):
    """Tableau de comparaison des métriques."""
    metrics_list = []
    for engine, result in zip(engines, results):
        as_of = result.daily_returns.index.max()
        m = engine.compute_metrics(as_of)
        metrics_list.append(m)

    # Build header
    header_cells = [html.Th("Métrique")]
    for i, label in enumerate(labels):
        header_cells.append(html.Th(
            html.Div([
                html.Div(className="compare-dot",
                         style={"background": colors[i], "display": "inline-block", "marginRight": "6px"}),
                html.Span(label, style={"fontSize": "11px"}),
            ], className="flex flex-center gap-4"),
        ))

    rows = []
    prev_group = None
    for metric_key, fmt_fn, pos_good, group in _COMPARE_METRICS:
        # Group separator row
        if group != prev_group:
            rows.append(html.Tr([
                html.Td(group.upper(),
                        colSpan=len(labels) + 1,
                        style={
                            "background": "#F2F5FA",
                            "color": "#6B7FA3",
                            "fontSize": "10px",
                            "fontWeight": "700",
                            "letterSpacing": "1px",
                            "padding": "6px 14px",
                        }),
            ]))
            prev_group = group

        vals = [m.get(metric_key) for m in metrics_list]

        # Find best / worst for numeric values
        numeric = []
        for v in vals:
            try:
                numeric.append(float(v))
            except (TypeError, ValueError):
                numeric.append(None)

        valid_nums = [x for x in numeric if x is not None]
        best_idx  = None
        worst_idx = None
        if len(valid_nums) >= 2 and pos_good is not None:
            if pos_good:
                best_val  = max(valid_nums)
                worst_val = min(valid_nums)
            else:
                best_val  = min(valid_nums)
                worst_val = max(valid_nums)
            best_idx  = numeric.index(best_val)  if best_val  in numeric else None
            worst_idx = numeric.index(worst_val) if worst_val in numeric else None
            if best_val == worst_val:
                best_idx = worst_idx = None

        cells = [html.Td(metric_key, style={"textAlign": "left", "fontWeight": "500",
                                             "color": "#6B7FA3", "fontSize": "11px",
                                             "padding": "7px 14px", "whiteSpace": "nowrap"})]
        for i, (v, n) in enumerate(zip(vals, numeric)):
            fmt_val = fmt_fn(v)
            css = ""
            if i == best_idx:
                css = "cell-best"
            elif i == worst_idx:
                css = "cell-worst"
            elif n is not None and pos_good is not None:
                if pos_good:
                    css = "cell-pos" if n > 0 else ("cell-neg" if n < 0 else "")
                else:
                    css = "cell-neg" if n > 0 else ("cell-pos" if n < 0 else "")
            cells.append(html.Td(fmt_val, className=css))

        rows.append(html.Tr(cells))

    tbl = html.Div([
        html.Table([
            html.Thead(html.Tr(header_cells)),
            html.Tbody(rows),
        ], className="metrics-compare-table"),
    ], style={"overflowX": "auto", "overflowY": "auto", "maxHeight": "520px"})

    return _card("Tableau de Comparaison des Métriques", tbl, "mb-16")


def _section_nav(labels, results, colors):
    """Rendements cumulés normalisés à 100."""
    fig = go.Figure()
    for i, (label, result, color) in enumerate(zip(labels, results, colors)):
        nav = result.nav
        if nav.empty:
            continue
        nav_norm = nav / float(result.initial_capital) * 100.0
        fig.add_trace(go.Scatter(
            x=nav_norm.index, y=nav_norm.values,
            name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"{label}<br>%{{x|%d/%m/%Y}}: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Rendements cumulés (base 100)",
        xaxis_title="Date", yaxis_title="Valeur (base 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        plot_bgcolor="#F7F9FC",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=50, b=30, l=10, r=10),
        height=400,
        hovermode="x unified",
    )
    return html.Div(_card("Rendements Cumulés Comparés", _graph(fig, height=400)), className="mb-16")


def _section_drawdowns(labels, results, colors):
    """Courbes de drawdown comparées."""
    fig = go.Figure()
    for label, result, color in zip(labels, results, colors):
        dr = result.daily_returns
        if dr.empty:
            continue
        cum  = (1 + dr).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak * 100
        # Convert hex to rgba for fill
        if color.startswith("#") and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            fill_color = f"rgba({r},{g},{b},0.08)"
        else:
            fill_color = color

        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=label,
            line=dict(color=color, width=1.8),
            fill="tozeroy",
            fillcolor=fill_color,
            hovertemplate=f"{label}<br>%{{x|%d/%m/%Y}}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="Drawdowns Comparés (%)",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        plot_bgcolor="#F7F9FC",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=50, b=30, l=10, r=10),
        height=320,
        hovermode="x unified",
    )
    return html.Div(_card("Drawdowns", _graph(fig, height=320)), className="mb-16")


def _section_performance_bars(labels, engines, results, colors):
    """Barres de comparaison sur métriques clés."""
    key_metrics = [
        ("CAGR (période)",            fmt_pct,  "CAGR"),
        ("Sharpe ratio (période)",    fmt_ratio, "Sharpe"),
        ("Max Drawdown (période)",    fmt_pct,  "Max DD"),
        ("Volatilité (période)",      lambda v: fmt_pct(v, sign=False), "Volatilité"),
        ("Information Ratio (période)", fmt_ratio, "Info Ratio"),
        ("Tracking Error (période)",  lambda v: fmt_pct(v, sign=False), "Tracking Err."),
    ]

    n = len(key_metrics)
    fig = make_subplots(rows=2, cols=3, subplot_titles=[k[2] for k in key_metrics])

    for idx, (mkey, fmt_fn, short) in enumerate(key_metrics):
        row = idx // 3 + 1
        col = idx %  3 + 1
        vals = []
        for engine, result in zip(engines, results):
            as_of = result.daily_returns.index.max()
            m = engine.compute_metrics(as_of)
            try:
                vals.append(float(m.get(mkey, 0) or 0))
            except (TypeError, ValueError):
                vals.append(0.0)

        fig.add_trace(
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors[:len(labels)],
                text=[fmt_fn(v) for v in vals],
                textposition="outside",
                showlegend=False,
            ),
            row=row, col=col,
        )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="#F7F9FC",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        margin=dict(t=60, b=30, l=10, r=10),
        height=480,
    )
    return html.Div(
        _card("Comparaison des Métriques Clés", _graph(fig, height=480)),
        className="mb-16"
    )


def _section_sector_allocation(labels, engines, results, colors):
    """Comparaison de l'allocation sectorielle sur la dernière date commune."""
    all_sector_dfs = []
    for engine, result, label in zip(engines, results, labels):
        rebal_dates = sorted(result.weights.keys())
        if not rebal_dates:
            continue
        last_date = rebal_dates[-1]
        try:
            comp = engine.result.weights[last_date]
            # Get sector mapping via reporting engine
            loader = engine.loader
            sector_map = loader.get_sector_mapping(last_date)
            df_sec = pd.DataFrame({"Weight": comp, "Sector": sector_map})
            df_sec = df_sec.groupby("Sector")["Weight"].sum().reset_index()
            df_sec["Portfolio"] = label
            all_sector_dfs.append(df_sec)
        except Exception:
            pass

    if not all_sector_dfs:
        return html.Div()

    combined = pd.concat(all_sector_dfs, ignore_index=True)
    all_sectors = sorted(combined["Sector"].dropna().unique().tolist())

    fig = go.Figure()
    for label, color in zip(labels, colors):
        port_df = combined[combined["Portfolio"] == label]
        port_df = port_df.set_index("Sector")["Weight"].reindex(all_sectors).fillna(0.0)
        fig.add_trace(go.Bar(
            name=label,
            x=all_sectors,
            y=port_df.values * 100,
            marker_color=color,
        ))

    fig.update_layout(
        title="Allocation sectorielle comparée (%)",
        barmode="group",
        xaxis=dict(tickangle=-35, gridcolor="#D9E2EC"),
        yaxis=dict(title="Poids (%)", gridcolor="#D9E2EC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        plot_bgcolor="#F7F9FC",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        margin=dict(t=60, b=80, l=10, r=10),
        height=420,
    )
    return _card("Allocation Sectorielle Comparée", _graph(fig, height=420))
