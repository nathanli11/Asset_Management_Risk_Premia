"""
app.py — Point d'entrée principal de l'application Amundi Backtesting
Usage : python app.py  (depuis src/app/)
"""
import sys
import os

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_APP_DIR)
for _p in [_APP_DIR, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ── Application Dash ──────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=os.path.join(_APP_DIR, "pages"),
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"charset": "utf-8"},
    ],
    title="Amundi Backtesting",
)
server = app.server  # WSGI hook


# ── Composants sidebar ────────────────────────────────────────────────────────

def _nav_link(href: str, icon: str, label: str, nav_id: str):
    return dcc.Link(
        html.Div([
            html.Span(icon,  className="nav-icon"),
            html.Span(label, className="nav-label"),
        ], className="nav-inner"),
        href=href,
        className="nav-item",
        id=nav_id,
    )


def sidebar():
    return html.Aside([
        # ── Logo ──────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("A", className="logo-letter"),
            ], className="logo-badge"),
            html.Div([
                html.Div("AMUNDI", className="brand-name"),
                html.Div("Backtesting Platform", className="brand-sub"),
            ], className="brand-text"),
        ], className="sidebar-brand"),

        # ── Navigation ────────────────────────────────────────────────
        html.Nav([
            _nav_link("/",           "⚙",  "Backtesting Engine",   "nav-engine"),
            _nav_link("/analysis",   "📊", "Portfolio Analysis",   "nav-analysis"),
            _nav_link("/comparison", "⚖",  "Portfolio Comparison", "nav-compare"),
        ], className="sidebar-nav"),

        # ── Pied de page ──────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span(id="lib-count", className="lib-badge", children="0"),
                html.Span(" backtest(s)", className="lib-label"),
            ], className="lib-info"),
            html.Div("sauvegardés en session", className="lib-sub"),
        ], className="sidebar-footer"),
    ], className="sidebar", id="sidebar")


# ── Layout principal ─────────────────────────────────────────────────────────

app.layout = html.Div([
    # Stores globaux
    dcc.Store(id="backtest-store",   storage_type="session", data={}),
    dcc.Store(id="run-result-store", storage_type="memory",  data=None),
    dcc.Location(id="url", refresh=False),

    # Shell
    html.Div([
        sidebar(),
        html.Main([
            dash.page_container,
        ], className="main-area", id="main-area"),
    ], className="app-shell"),
], id="app-root")


# ── Callbacks globaux ─────────────────────────────────────────────────────────

@callback(
    Output("lib-count", "children"),
    Input("backtest-store", "data"),
)
def _update_lib_count(store):
    return str(len(store or {}))


# Active nav link highlighting via URL
@callback(
    Output("nav-engine",   "className"),
    Output("nav-analysis", "className"),
    Output("nav-compare",  "className"),
    Input("url", "pathname"),
)
def _highlight_nav(pathname):
    base = "nav-item"
    active = "nav-item active"
    exact_match = pathname == "/" or pathname is None
    return (
        active if exact_match else base,
        active if pathname == "/analysis" else base,
        active if pathname == "/comparison" else base,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050, host="127.0.0.1")
