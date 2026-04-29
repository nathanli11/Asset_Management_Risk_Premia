"""
utils.py — Utilitaires partagés pour l'application de backtesting Amundi
"""
import sys
import os
import numpy as np
import pandas as pd

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_APP_DIR)
for _p in [_APP_DIR, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Couleurs portefeuilles (comparaison) ─────────────────────────────────────

PORTFOLIO_COLORS = ["#00A3AD", "#E30613", "#0E2A47", "#F5A623"]

SIGNAL_LABELS_FR = {
    "MOMENTUM_12_1":                  "Momentum 12-1",
    "IDIOSYNCRATIC_12_1":             "Momentum Idiosyncratique 12-1",
    "MOMENTUM_5Y_MEAN_REVERTING":     "Momentum 5Y (Mean-Reverting)",
    "IDIOSYNCRATIC_5Y_MEAN_REVERTING": "Momentum Idiosyncratique 5Y",
}
METHOD_LABELS_FR = {
    "RISK_PARITY":   "Risk Parity (ERC)",
    "MIN_VARIANCE":  "Minimum Variance",
    "SIGNAL_WEIGHT": "Signal Weight",
}
SELECTION_LABELS_FR = {
    "BEST_IN_UNIVERSE": "Meilleur univers",
    "BEST_IN_CLASS":    "Meilleur classe",
}
QUANTILE_LABELS_FR = {
    "DECILE":   "Décile (Top 10 %)",
    "QUINTILE": "Quintile (Top 20 %)",
    "QUARTILE": "Quartile (Top 25 %)",
}


# ── Sérialisation ─────────────────────────────────────────────────────────────

def _f(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def serialize_result(result, config=None) -> dict:
    """Convertit un BacktestResult en dict JSON-serialisable."""
    cfg = config if config is not None else result.config
    return {
        "strategy_label": result.strategy_label,
        "method":         result.method,
        "signal":         result.signal,
        "initial_capital": _f(result.initial_capital),
        "nav":            {str(k): _f(v) for k, v in result.nav.items()},
        "daily_returns":  {str(k): _f(v) for k, v in result.daily_returns.items()},
        "weights": {
            str(k): {str(tk): _f(w) for tk, w in series.items()}
            for k, series in result.weights.items()
        },
        "transaction_costs_eur": {str(k): _f(v) for k, v in result.transaction_costs_eur.items()},
        "rebalancing_dates": [str(d) for d in result.rebalancing_dates],
        "config": {
            "signal_type":         cfg.signal_type.name,
            "allocation_method":   cfg.allocation_method.name,
            "selection_mode":      cfg.selection_mode.name,
            "quantile_mode":       cfg.quantile_mode.name,
            "rebalancing_freq":    cfg.rebalancing_freq,
            "apply_ucits":         bool(cfg.apply_ucits),
            "transaction_cost_bps": float(cfg.transaction_cost_bps),
            "initial_capital":     float(cfg.initial_capital),
            "start_date":          cfg.start_date,
            "end_date":            cfg.end_date,
            "cov_lookback_days":   int(cfg.cov_lookback_days),
        },
    }


def deserialize_result(data: dict):
    """Reconstruit un BacktestResult depuis un dict JSON."""
    from backtesting import (
        BacktestResult, BacktestConfig,
        SignalType, AllocationMethod, SelectionMode, QuantileMode,
    )

    nav = pd.Series({pd.Timestamp(k): float(v) for k, v in data["nav"].items()}).sort_index()
    dr  = pd.Series({pd.Timestamp(k): float(v) for k, v in data["daily_returns"].items()}).sort_index()
    weights = {
        pd.Timestamp(k): pd.Series({str(tk): float(w) for tk, w in v.items()})
        for k, v in data["weights"].items()
    }
    tc    = {pd.Timestamp(k): float(v) for k, v in data["transaction_costs_eur"].items()}
    rebal = [pd.Timestamp(d) for d in data["rebalancing_dates"]]

    c = data["config"]
    config = BacktestConfig(
        signal_type         = SignalType[c["signal_type"]],
        allocation_method   = AllocationMethod[c["allocation_method"]],
        selection_mode      = SelectionMode[c["selection_mode"]],
        quantile_mode       = QuantileMode[c["quantile_mode"]],
        rebalancing_freq    = c["rebalancing_freq"],
        apply_ucits         = bool(c["apply_ucits"]),
        transaction_cost_bps= float(c["transaction_cost_bps"]),
        initial_capital     = float(c["initial_capital"]),
        start_date          = c["start_date"],
        end_date            = c["end_date"],
        cov_lookback_days   = int(c["cov_lookback_days"]),
    )
    return BacktestResult(
        method            = data["method"],
        signal            = data["signal"],
        strategy_label    = data["strategy_label"],
        daily_returns     = dr,
        nav               = nav,
        weights           = weights,
        transaction_costs_eur = tc,
        initial_capital   = float(data["initial_capital"]),
        config            = config,
        rebalancing_dates = rebal,
    )


def get_data_loader():
    """Instancie et retourne un DataLoader."""
    from backtesting import DataLoader
    return DataLoader()


def build_reporting_engine(data: dict):
    """Construit un ReportingEngine depuis un résultat sérialisé."""
    from backtesting.reporting import ReportingEngine
    result = deserialize_result(data)
    loader = get_data_loader()
    return ReportingEngine(result, loader)


def get_last_rebal_date(result_data: dict):
    """Retourne la dernière date de rebalancement sous forme de pd.Timestamp."""
    dates = [pd.Timestamp(d) for d in result_data.get("rebalancing_dates", [])]
    return max(dates) if dates else None


def get_as_of_date(result_data: dict):
    """Retourne la dernière date de daily_returns (date de référence pour les métriques)."""
    dates = [pd.Timestamp(k) for k in result_data.get("daily_returns", {})]
    return max(dates) if dates else None


# ── Formatage ─────────────────────────────────────────────────────────────────

def fmt_pct(v, d: int = 2, sign: bool = True) -> str:
    if v is None or v == "--":
        return "--"
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return "--"
        fmt = f"+{x*100:.{d}f}%" if sign and x > 0 else f"{x*100:.{d}f}%"
        return fmt
    except (TypeError, ValueError):
        return str(v)


def fmt_ratio(v, d: int = 2) -> str:
    if v is None or v == "--":
        return "--"
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return "--"
        return f"{x:.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def fmt_eur(v) -> str:
    if v is None or v == "--":
        return "--"
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return "--"
        sign = "+" if x > 0 else ""
        return f"{sign}€{x:,.0f}"
    except (TypeError, ValueError):
        return str(v)


def metric_color_class(v, positive_is_good: bool = True) -> str:
    """Retourne 'metric-pos', 'metric-neg' ou 'metric-neutral'."""
    if v is None or v == "--":
        return "metric-neutral"
    try:
        x = float(v)
        if np.isnan(x):
            return "metric-neutral"
        if positive_is_good:
            return "metric-pos" if x > 0 else ("metric-neg" if x < 0 else "metric-neutral")
        else:
            return "metric-neg" if x > 0 else ("metric-pos" if x < 0 else "metric-neutral")
    except (TypeError, ValueError):
        return "metric-neutral"


# ── Résumé config portefeuille ───────────────────────────────────────────────

def config_summary(cfg_dict: dict) -> str:
    sig  = SIGNAL_LABELS_FR.get(cfg_dict.get("signal_type", ""), "?")
    meth = METHOD_LABELS_FR.get(cfg_dict.get("allocation_method", ""), "?")
    sel  = SELECTION_LABELS_FR.get(cfg_dict.get("selection_mode", ""), "?")
    qt   = QUANTILE_LABELS_FR.get(cfg_dict.get("quantile_mode", ""), "?")
    freq = "Mensuel" if cfg_dict.get("rebalancing_freq") == "monthly" else "Trimestriel"
    ucits = "UCITS" if cfg_dict.get("apply_ucits") else "No-UCITS"
    return f"{sig} · {meth} · {sel} · {qt} · {freq} · {ucits}"
