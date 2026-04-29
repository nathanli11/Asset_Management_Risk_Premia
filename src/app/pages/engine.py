"""
pages/engine.py — Page 1 : Moteur de Backtesting
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
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

from utils import (
    serialize_result, fmt_pct, fmt_ratio, fmt_eur,
    metric_color_class, SIGNAL_LABELS_FR, METHOD_LABELS_FR,
    SELECTION_LABELS_FR, QUANTILE_LABELS_FR, config_summary,
)

dash.register_page(__name__, path="/", name="Backtesting Engine", title="Backtesting Engine | Amundi")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _card(title, children, extra_class=""):
    return html.Div([
        html.Div(html.Span(title, className="card-title"), className="card-header"),
        html.Div(children, className="card-body"),
    ], className=f"card {extra_class}")


def _section(label, children):
    return html.Div([
        html.Div(label, className="form-section-title"),
        *children,
    ], className="form-section")


def _radio(cid, options, value):
    return dcc.RadioItems(
        id=cid,
        options=[{"label": v, "value": k} for k, v in options.items()],
        value=value,
        className="radio-group",
        inputClassName="radio-input",
        labelClassName="radio-label",
    )


# ── Layout ───────────────────────────────────────────────────────────────────

layout = html.Div([
    # ── En-tête ───────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("Backtesting Engine", className="page-title"),
            html.Div("Configurez et lancez votre stratégie momentum long-only sur MSCI World",
                     className="page-subtitle"),
        ]),
    ], className="page-header"),

    # ── Corps principal ───────────────────────────────────────────────
    html.Div([

        # ── Panneau de configuration ──────────────────────────────────
        html.Div([
            html.Div("Configuration de la stratégie", className="config-panel-header"),
            html.Div([

                _section("Signal", [
                    _radio("eng-signal", SIGNAL_LABELS_FR, "MOMENTUM_12_1"),
                ]),

                _section("Méthode d'allocation", [
                    _radio("eng-method", METHOD_LABELS_FR, "RISK_PARITY"),
                ]),

                _section("Sélection de l'univers", [
                    _radio("eng-selection", SELECTION_LABELS_FR, "BEST_IN_UNIVERSE"),
                    html.Div("Quantile", className="form-label mt-8"),
                    dcc.Dropdown(
                        id="eng-quantile",
                        options=[{"label": v, "value": k} for k, v in QUANTILE_LABELS_FR.items()],
                        value="QUINTILE",
                        clearable=False,
                        style={"fontSize": "12px"},
                    ),
                ]),

                _section("Paramètres", [
                    html.Div("Fréquence de rebalancement", className="form-label"),
                    _radio("eng-freq", {"monthly": "Mensuel", "quarterly": "Trimestriel"}, "monthly"),

                    html.Div(className="mt-8"),
                    dcc.Checklist(
                        id="eng-ucits",
                        options=[{"label": " Appliquer contrainte UCITS 5/10/40", "value": "yes"}],
                        value=["yes"],
                        className="check-group",
                    ),

                    html.Div(className="mt-8"),
                    html.Div([
                        html.Div([
                            html.Label("Coûts TC (bps)", className="form-label"),
                            dcc.Input(id="eng-tc-bps", type="number", value=5.0,
                                      min=0, max=100, step=0.5,
                                      className="dash-input"),
                        ], className="form-field"),
                        html.Div([
                            html.Label("Lookback cov. (j)", className="form-label"),
                            dcc.Input(id="eng-lookback", type="number", value=252,
                                      min=60, max=1260, step=10,
                                      className="dash-input"),
                        ], className="form-field"),
                    ], className="form-row"),

                    html.Label("Capital initial (€)", className="form-label"),
                    dcc.Input(id="eng-capital", type="number", value=1_000_000,
                              min=10_000, step=10_000, className="dash-input w-full"),
                ]),

                _section("Période d'analyse", [
                    html.Div([
                        html.Div([
                            html.Label("Date de début", className="form-label"),
                            dcc.DatePickerSingle(
                                id="eng-start-date",
                                date="2010-01-04",
                                display_format="DD/MM/YYYY",
                                first_day_of_week=1,
                                className="w-full",
                            ),
                        ], className="form-field"),
                        html.Div([
                            html.Label("Date de fin", className="form-label"),
                            dcc.DatePickerSingle(
                                id="eng-end-date",
                                date="2024-12-31",
                                display_format="DD/MM/YYYY",
                                first_day_of_week=1,
                                className="w-full",
                            ),
                        ], className="form-field"),
                    ], className="form-row"),
                ]),

                # Nom personnalisé
                _section("Nom de la stratégie", [
                    dcc.Input(
                        id="eng-strategy-name",
                        type="text",
                        placeholder="Laissez vide pour nommage auto",
                        className="dash-input w-full",
                    ),
                ]),

                # Bouton Lancer
                html.Div([
                    html.Button(
                        "▶  Lancer le Backtest",
                        id="eng-run-btn",
                        className="btn-run",
                        n_clicks=0,
                    ),
                    html.Div(id="eng-run-status", className="mt-8"),
                ], className="form-section"),

            ], className="config-body"),
        ], className="config-panel"),

        # ── Panneau de résultats ──────────────────────────────────────
        html.Div([
            dcc.Loading(
                id="eng-loading",
                type="circle",
                color="#00A3AD",
                children=html.Div(id="eng-preview-area", children=[
                    html.Div([
                        html.Div("⚙", className="preview-placeholder-icon"),
                        html.Div("Configurez les paramètres et lancez le backtest",
                                 className="preview-placeholder-text"),
                        html.Div("Les résultats s'afficheront ici",
                                 className="text-muted text-small mt-8"),
                    ], className="preview-placeholder"),
                ]),
            ),
        ], className="preview-panel"),

    ], className="engine-layout"),

    # ── Bibliothèque des backtests ────────────────────────────────────
    html.Div([
        _card("Bibliothèque de Backtests", [
            html.Div(id="eng-library-content", children=[
                html.Div("Aucun backtest sauvegardé.",
                         className="text-muted text-small"),
            ]),
        ]),
    ], className="library-section"),

    # Store temporaire résultat courant
    dcc.Store(id="eng-current-result", storage_type="memory", data=None),
], className="page-wrap")


# ── Callback : Lancer le backtest ─────────────────────────────────────────────

@callback(
    Output("eng-preview-area",   "children"),
    Output("eng-run-status",     "children"),
    Output("eng-current-result", "data"),
    Input("eng-run-btn", "n_clicks"),
    State("eng-signal",         "value"),
    State("eng-method",         "value"),
    State("eng-selection",      "value"),
    State("eng-quantile",       "value"),
    State("eng-freq",           "value"),
    State("eng-ucits",          "value"),
    State("eng-tc-bps",         "value"),
    State("eng-capital",        "value"),
    State("eng-start-date",     "date"),
    State("eng-end-date",       "date"),
    State("eng-lookback",       "value"),
    State("eng-strategy-name",  "value"),
    prevent_initial_call=True,
)
def run_backtest(
    n_clicks, signal, method, selection, quantile,
    freq, ucits, tc_bps, capital, start_date, end_date, lookback, strat_name,
):
    if not n_clicks:
        return no_update, no_update, no_update

    try:
        from backtesting import (
            BacktestConfig, BacktestEngine, DataLoader,
            SignalType, AllocationMethod, SelectionMode, QuantileMode,
        )

        apply_ucits = bool(ucits and "yes" in ucits)
        config = BacktestConfig(
            signal_type         = SignalType[signal],
            allocation_method   = AllocationMethod[method],
            selection_mode      = SelectionMode[selection],
            quantile_mode       = QuantileMode[quantile],
            rebalancing_freq    = freq,
            apply_ucits         = apply_ucits,
            transaction_cost_bps= float(tc_bps or 5.0),
            initial_capital     = float(capital or 1_000_000),
            start_date          = start_date,
            end_date            = end_date,
            cov_lookback_days   = int(lookback or 252),
        )

        loader = DataLoader()
        engine = BacktestEngine(config, loader)
        result = engine.run()

        # Override label if user provided a name
        if strat_name and strat_name.strip():
            result.strategy_label = strat_name.strip()

        serialized = serialize_result(result, config)

        # Build preview
        preview = _build_preview(serialized, loader, result)
        status  = html.Div("✓ Backtest terminé avec succès.", className="status-success")
        return preview, status, serialized

    except Exception as exc:
        tb = traceback.format_exc()
        err = html.Div([
            html.Div("✗ Erreur lors du backtest :", className="fw-600"),
            html.Div(str(exc), className="mt-8 text-small"),
        ], className="status-error")
        print(tb)
        return no_update, err, no_update


def _build_preview(serialized, loader, result):
    """Construit le panneau de prévisualisation après un backtest."""
    from backtesting.reporting import ReportingEngine

    re = ReportingEngine(result, loader)
    as_of = result.daily_returns.index.max()
    m = re.compute_metrics(as_of)

    kpis = [
        ("Rendement cumulé", fmt_pct(m.get("Rendement cumulé (période)")), metric_color_class(m.get("Rendement cumulé (période)"))),
        ("CAGR",             fmt_pct(m.get("CAGR (période)")),             metric_color_class(m.get("CAGR (période)"))),
        ("Sharpe",           fmt_ratio(m.get("Sharpe ratio (période)")),   metric_color_class(m.get("Sharpe ratio (période)"))),
        ("Max Drawdown",     fmt_pct(m.get("Max Drawdown (période)")),     metric_color_class(m.get("Max Drawdown (période)"), False)),
        ("Volatilité",       fmt_pct(m.get("Volatilité (période)"), sign=False), "kpi-neu"),
        ("PnL",              fmt_eur(m.get("PnL (€)")),                    metric_color_class(m.get("PnL (€)"))),
    ]

    kpi_cards = html.Div([
        html.Div([
            html.Div(lbl, className="kpi-label"),
            html.Div(val, className="kpi-value"),
        ], className=f"kpi-card {cls.replace('metric-', 'kpi-')}" if 'pos' in cls or 'neg' in cls else "kpi-card kpi-neu")
        for lbl, val, cls in kpis
    ], className="kpi-grid mb-16")

    fig = re.plot_cumulative_returns()
    fig.update_layout(
        margin=dict(t=36, b=20, l=10, r=10),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    chart = html.Div([
        dcc.Graph(
            figure=fig,
            config={"displayModeBar": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
            style={"height": "320px"},
        ),
    ], className="card")

    save_section = _build_save_section(serialized["strategy_label"])

    return html.Div([kpi_cards, chart, save_section])


def _build_save_section(strategy_label):
    return html.Div([
        html.Div([
            html.Div([
                html.Div("Nom affiché dans la bibliothèque", className="text-muted text-small mb-8"),
                dcc.Input(
                    id="eng-save-name",
                    type="text",
                    value=strategy_label,
                    placeholder="Nom de la stratégie",
                    className="dash-input",
                    style={"width": "320px"},
                ),
            ]),
            html.Button(
                "💾  Sauvegarder dans la bibliothèque",
                id="eng-save-btn",
                className="btn-save",
                n_clicks=0,
            ),
        ], className="flex flex-center gap-12"),
        html.Div(id="eng-save-status", className="mt-8"),
    ], className="card card-body mt-16")


# ── Callback : Sauvegarder ────────────────────────────────────────────────────

@callback(
    Output("backtest-store",  "data"),
    Output("eng-save-status", "children"),
    Input("eng-save-btn",     "n_clicks"),
    State("eng-current-result", "data"),
    State("eng-save-name",      "value"),
    State("backtest-store",     "data"),
    prevent_initial_call=True,
)
def save_to_library(n_clicks, result_data, save_name, current_store):
    if not n_clicks or result_data is None:
        return no_update, no_update

    label = (save_name or "").strip() or result_data.get("strategy_label", "Strategy")
    result_data = {**result_data, "strategy_label": label}

    new_store = {**(current_store or {}), label: result_data}
    msg = html.Div(f'✓ Stratégie "{label}" sauvegardée.', className="status-success")
    return new_store, msg


# ── Callback : Bibliothèque ───────────────────────────────────────────────────

@callback(
    Output("eng-library-content", "children"),
    Input("backtest-store", "data"),
)
def update_library(store_data):
    if not store_data:
        return html.Div("Aucun backtest sauvegardé. Lancez et sauvegardez une stratégie pour commencer.",
                        className="text-muted text-small")

    rows = []
    for label, data in store_data.items():
        cfg = data.get("config", {})
        nav = data.get("nav", {})
        dr  = data.get("daily_returns", {})

        if nav and dr:
            nav_series = {pd.Timestamp(k): float(v) for k, v in nav.items()}
            dr_series  = {pd.Timestamp(k): float(v) for k, v in dr.items()}
            import numpy as np
            dr_vals = list(dr_series.values())
            cum_ret = (1 + pd.Series(dr_vals)).prod() - 1
            vol = pd.Series(dr_vals).std() * (252 ** 0.5)
            cum_str = fmt_pct(cum_ret)
            vol_str = fmt_pct(vol, sign=False)
        else:
            cum_str = vol_str = "--"

        start = cfg.get("start_date", "?")[:10]
        end   = cfg.get("end_date",   "?")[:10]
        tc    = cfg.get("transaction_cost_bps", "?")
        cap   = f"€{float(cfg.get('initial_capital', 0)):,.0f}"
        ucits = "✓" if cfg.get("apply_ucits") else "✗"

        rows.append(html.Tr([
            html.Td(html.Span(label, className="fw-600"), style={"maxWidth": "220px", "overflow": "hidden", "textOverflow": "ellipsis"}),
            html.Td(html.Span(SIGNAL_LABELS_FR.get(cfg.get("signal_type",""), "?"), className="tag")),
            html.Td(METHOD_LABELS_FR.get(cfg.get("allocation_method",""), "?")),
            html.Td(f"{start} → {end}"),
            html.Td(cap),
            html.Td(f"{tc} bps"),
            html.Td(ucits),
            html.Td(cum_str, className="metric-pos" if "+" in cum_str else "metric-neg" if cum_str.startswith("-") else ""),
            html.Td(vol_str),
            html.Td(
                html.Button("✕", id={"type": "eng-delete-btn", "index": label},
                            className="btn-danger", n_clicks=0)
            ),
        ]))

    table = html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("Stratégie"),
                html.Th("Signal"),
                html.Th("Allocation"),
                html.Th("Période"),
                html.Th("Capital"),
                html.Th("TC"),
                html.Th("UCITS"),
                html.Th("Rend. cum."),
                html.Th("Vol."),
                html.Th(""),
            ])),
            html.Tbody(rows),
        ], className="library-table"),
    ], className="library-table-wrap")
    return table


# ── Callback : Supprimer un backtest ──────────────────────────────────────────

@callback(
    Output("backtest-store", "data", allow_duplicate=True),
    Input({"type": "eng-delete-btn", "index": dash.ALL}, "n_clicks"),
    State("backtest-store", "data"),
    prevent_initial_call=True,
)
def delete_from_library(n_clicks_list, store_data):
    if not store_data:
        return no_update

    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return no_update

    triggered_id = ctx.triggered[0]["prop_id"]
    import json
    try:
        id_part = triggered_id.split(".")[0]
        id_dict = json.loads(id_part)
        label   = id_dict.get("index")
        if label and label in store_data:
            new_store = {k: v for k, v in store_data.items() if k != label}
            return new_store
    except Exception:
        pass
    return no_update
