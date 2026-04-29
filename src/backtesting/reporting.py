

# ------------------------------------------------------------------
# Librairies
# ------------------------------------------------------------------

# externes
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Présentation
# ------------------------------------------------------------------

"""
Reporting institutionnel — Stratégie Momentum Long-Only MSCI World
-----------------------------------------------------------------
1. Performance   : rendements cumulés, annualisés, CAGR sur multiples horizons
2. Risque        : Sharpe, Sortino, Tracking Error, Information Ratio, Calmar,
                   VaR, CVaR, Volatilité, Max Drawdown
3. Benchmark     : MSCI World EUR — corrélations, rendements comparés
4. Coûts         : TC cumulés en % et en €
5. Visualisations: graphiques Plotly (cumul returns, drawdowns, vol, corrél.)
6. Composition   : barcharts secteur/pays/devise/industrie, top 10 poids,
                   top 10 contributions à la performance et au risque

Benchmark : MSCI World EUR (rendements mensuels interpolés en journalier)
Taux sans risque : fixe à 1,03 % par an (moyenne 15 ans)
"""

# Libellés méthodes d'allocation
METHODS_LABELS = {
    "risk_parity":   "RiskParity",
    "min_variance":  "MinVariance",
    "signal_weight": "SignalWeight",
    # Libellés courts (depuis BacktestResult.method)
    "ERC":    "RiskParity",
    "MinVar": "MinVariance",
    "SigW":   "SignalWeight",
}

# Constantes d'annualisation et seuils
ANNUALIZATION = 252
CONF_LEVEL = 0.05        # 95 % VaR / CVaR
_NA = "--"               # Valeur affichée quand données insuffisantes

# Taux sans risque fixe : 1,03 % par an (≈ moyenne 15 ans)
RF_ANNUAL: float = 0.0103
RF_DAILY: float = RF_ANNUAL / 252

# Palette institutionnelle type Amundi
AMUNDI_DARK = "#101828"
AMUNDI_NAVY = "#0E2A47"
AMUNDI_TEAL = "#00A3AD"
AMUNDI_RED = "#E30613"
AMUNDI_LIGHT_BG = "#F7F9FC"
AMUNDI_GRID = "#D9E2EC"
AMUNDI_TEXT = "#1A1A1A"


class ReportingEngine:
    """
    Calcule les métriques de performances, de risque et relatives au benchmark,
    génère les visualisations Plotly et exporte les résultats.

    Benchmark : MSCI World EUR (via DataLoader.get_ester_returns_series)
    Taux sans risque : RF_DAILY (fixe, 1,03 % / 252 par jour)

    Paramètres
    ----------
    result      : BacktestResult — résultats du moteur de backtesting.
    data_loader : DataLoader    — accès aux prix et métadonnées.
    """

    # Racine stockage
    _STOCKAGE_ROOT = (
        Path(__file__).resolve().parent.parent.parent / "data" / "storage"
    )

    def __init__(self, result, data_loader):
        self.result = result
        self.loader = data_loader

        # Séries journalières de référence
        self._dr = result.daily_returns          # Rendements journaliers portefeuille
        self._nav = result.nav                    # NAV quotidienne

        # Série de taux sans risque journalier (fixe)
        if not self._dr.empty:
            self._rf_dr = pd.Series(
                RF_DAILY,
                index=self._dr.index,
                name="risk_free",
            )
        else:
            self._rf_dr = pd.Series(dtype=float)

        # Benchmark MSCI World : rendements journaliers interpolés depuis mensuel
        if not self._dr.empty:
            self._ester_dr = self.loader.get_ester_returns_series(
                self._dr.index[0], self._dr.index[-1]
            ).reindex(self._dr.index).fillna(0.0)
        else:
            self._ester_dr = pd.Series(dtype=float)

        # Niveaux cumulés du benchmark (reconstruit depuis les rendements journaliers)
        self._estron_levels = self._build_benchmark_levels()

    # ------------------------------------------------------------------
    # Calcul des métriques
    # ------------------------------------------------------------------

    def compute_metrics(self, as_of_date: pd.Timestamp) -> dict:
        """
        Calcule l'ensemble des métriques de performance à chaque date de rebalancement.
        Retourne un dictionnaire {metric_name: value}.
        """
        dr = self._dr[self._dr.index <= as_of_date]

        # Benchmark aligné
        benchmark = self._ester_dr.reindex(dr.index).fillna(0.0)

        # Taux sans risque aligné
        rf = self._rf_dr.reindex(dr.index).fillna(RF_DAILY)

        if dr.empty:
            return self._empty_metrics()

        nav_slice = self._nav[self._nav.index <= as_of_date]
        start_nav = float(self.result.initial_capital)
        end_nav = float(nav_slice.iloc[-1]) if not nav_slice.empty else start_nav

        m = {}

        # ---- A - Performance ----
        m["Rendement cumulé (période)"] = self._cum_return(dr)
        m["Rendement cumulé 1 an"] = self._cum_return_nav_window(as_of_date, months=12)
        m["Rendement cumulé 2 ans"] = self._cum_return_nav_window(as_of_date, months=24)
        m["Rendement cumulé YTD"] = self._cum_return_ytd(dr, as_of_date)
        m["Rendement cumulé MTD"] = self._cum_return_mtd(dr, as_of_date)

        m["Rendement annualisé (période)"] = self._ann_return(dr)
        m["CAGR (période)"] = self._cagr(start_nav, end_nav, dr)
        m["CAGR 1 an"] = self._cagr_window(dr, as_of_date, months=12)
        m["CAGR 2 ans"] = self._cagr_window(dr, as_of_date, months=24)
        m["CAGR YTD"] = self._cagr_ytd(dr, as_of_date)
        m["CAGR MTD"] = self._cagr_mtd(dr, as_of_date)

        # ---- B - Risque & Performance ajustée (RF fixe pour Sharpe/Sortino) ----
        m["Sharpe ratio (période)"] = self._sharpe(dr, rf)
        m["Sharpe ratio 1 an"] = self._sharpe_window(dr, rf, as_of_date, months=12)
        m["Sortino ratio (période)"] = self._sortino(dr, rf)
        m["Sortino ratio 1 an"] = self._sortino_window(dr, rf, as_of_date, months=12)

        # TE, IR, Corrélation : vs benchmark MSCI World
        m["Tracking Error (période)"] = self._tracking_error(dr, benchmark)
        m["Tracking Error 1 an"] = self._tracking_error_window(dr, benchmark, as_of_date, months=12)
        m["Tracking Error 3 ans"] = self._tracking_error_window(dr, benchmark, as_of_date, months=36)

        m["Information Ratio (période)"] = self._info_ratio(dr, benchmark)
        m["Information Ratio 1 an"] = self._info_ratio_window(dr, benchmark, as_of_date, months=12)
        m["Information Ratio 2 ans"] = self._info_ratio_window(dr, benchmark, as_of_date, months=24)

        m["Calmar ratio (période)"] = self._calmar(dr)

        m["VaR 95% (période)"] = self._var(dr)
        m["VaR 95% 1 an"] = self._var_window(dr, as_of_date, months=12)
        m["CVaR 95% (période)"] = self._cvar(dr)
        m["CVaR 95% 1 an"] = self._cvar_window(dr, as_of_date, months=12)

        m["Volatilité (période)"] = self._volatility(dr)
        m["Volatilité 1 an"] = self._vol_window(dr, as_of_date, months=12)
        m["Volatilité 2 ans"] = self._vol_window(dr, as_of_date, months=24)

        m["Max Drawdown (période)"] = self._max_drawdown(dr)
        m["Max Drawdown 1 an"] = self._mdd_window(dr, as_of_date, months=12)
        m["Max Drawdown 2 ans"] = self._mdd_window(dr, as_of_date, months=24)
        m["Max Drawdown YTD"] = self._mdd_ytd(dr, as_of_date)
        m["Max Drawdown MTD"] = self._mdd_mtd(dr, as_of_date)

        # ---- C - Benchmark MSCI World ----
        m["Corrélation avec benchmark (période)"] = self._corr(dr, benchmark)
        m["Benchmark - Rendement cumulé (période)"] = self._benchmark_cum_return_period(as_of_date)
        m["Benchmark - Rendement cumulé 1 an"] = self._benchmark_cum_return_window(as_of_date, months=12)
        m["Benchmark - Rendement cumulé 2 ans"] = self._benchmark_cum_return_window(as_of_date, months=24)
        m["Benchmark - Rendement cumulé YTD"] = self._benchmark_cum_return_period_type(as_of_date, "YTD")
        m["Benchmark - Rendement cumulé MTD"] = self._benchmark_cum_return_period_type(as_of_date, "MTD")
        m["Benchmark - Rendement annualisé (période)"] = self._benchmark_ann_return_period(as_of_date)

        # ---- D - Coûts de transaction ----
        tc_eur_all = self.result.transaction_costs_eur
        tc_eur_total = sum(v for k, v in tc_eur_all.items() if k <= as_of_date)
        tc_pct_total = tc_eur_total / self.result.initial_capital
        m["TC total (€)"] = tc_eur_total
        m["TC total (%)"] = tc_pct_total

        # ---- E - PnL ----
        m["PnL (€)"] = end_nav - start_nav

        return m

    def compute_all_metrics(self) -> pd.DataFrame:
        """Retourne un DataFrame (index=Date, colonnes=métriques)."""
        rebal_dates = sorted(self.result.weights.keys())
        rows = {}
        for date in rebal_dates:
            rows[date] = self.compute_metrics(date)
        df = pd.DataFrame(rows).T
        df.index.name = "Date"
        return df

    # ------------------------------------------------------------------
    # Construction des niveaux cumulés du benchmark MSCI World
    # ------------------------------------------------------------------

    def _build_benchmark_levels(self) -> pd.Series:
        """
        Reconstruit une série de niveaux cumulés à partir des rendements
        journaliers MSCI World (interpolés depuis les rendements mensuels).
        Utilisé pour les calculs de rendements benchmark (cum, annualisé, etc.).
        """
        if self._ester_dr.empty:
            return pd.Series(dtype=float)
        cumulative = (1.0 + self._ester_dr).cumprod()
        # Base 100
        return cumulative * 100.0

    # ------------------------------------------------------------------
    # Fonctions internes de calcul des métriques
    # ------------------------------------------------------------------

    def _slice(self, dr: pd.Series, start: pd.Timestamp) -> pd.Series:
        available = dr.index[dr.index >= start]
        if available.empty:
            return pd.Series(dtype=float)
        return dr.loc[available[0]:]

    def _closest_before(self, dr: pd.Series, target: pd.Timestamp) -> pd.Timestamp:
        candidates = dr.index[dr.index <= target]
        return candidates[-1] if not candidates.empty else None

    def _window(self, dr: pd.Series, as_of: pd.Timestamp, months: int) -> pd.Series:
        start_target = as_of - pd.DateOffset(months=months)
        start = self._closest_before(dr, start_target)
        if start is None:
            return pd.Series(dtype=float)
        return dr.loc[start:as_of]

    def _benchmark_bounds_window(self, as_of: pd.Timestamp, months: int):
        """Bornes (start, end, level_start, level_end) du benchmark sur une fenêtre glissante."""
        levels = self._estron_levels
        if levels.empty:
            return None, None, None, None
        as_of = pd.Timestamp(as_of)
        end_candidates = levels.index[levels.index <= as_of]
        start_target = as_of - pd.DateOffset(months=months)
        start_candidates = levels.index[levels.index <= start_target]
        if end_candidates.empty or start_candidates.empty:
            return None, None, None, None
        s = start_candidates[-1]
        e = end_candidates[-1]
        return s, e, float(levels.loc[s]), float(levels.loc[e])

    def _benchmark_daily_returns(self) -> pd.Series:
        """Rendements journaliers MSCI World (interpolés depuis mensuel)."""
        return self._ester_dr.copy()

    def _benchmark_bounds_period(self, as_of: pd.Timestamp):
        levels = self._estron_levels
        if levels.empty:
            return None, None, None, None
        as_of = pd.Timestamp(as_of)
        eligible = levels.index[levels.index <= as_of]
        if eligible.empty:
            return None, None, None, None
        s = eligible[0]
        e = eligible[-1]
        return s, e, float(levels.loc[s]), float(levels.loc[e])

    def _benchmark_bounds_period_type(self, as_of: pd.Timestamp, period: str):
        levels = self._estron_levels
        if levels.empty:
            return None, None, None, None
        as_of = pd.Timestamp(as_of)
        idx = levels.index
        if period == "YTD":
            prev_mask = idx.year == (as_of.year - 1)
            curr_mask = (idx.year == as_of.year) & (idx <= as_of)
        elif period == "MTD":
            prev_month = as_of.to_period("M") - 1
            prev_mask = (idx.year == prev_month.year) & (idx.month == prev_month.month)
            curr_mask = (idx.year == as_of.year) & (idx.month == as_of.month) & (idx <= as_of)
        else:
            return None, None, None, None
        prev_idx = idx[prev_mask]
        curr_idx = idx[curr_mask]
        if prev_idx.empty or curr_idx.empty:
            return None, None, None, None
        s = prev_idx[-1]
        e = curr_idx[-1]
        return s, e, float(levels.loc[s]), float(levels.loc[e])

    def _benchmark_cum_return_period(self, as_of: pd.Timestamp) -> object:
        s, e, _, _ = self._benchmark_bounds_period(as_of)
        if s is None or e is None:
            return _NA
        dr = self._benchmark_daily_returns()
        sub = dr[(dr.index > s) & (dr.index <= e)]
        if len(sub) < 1:
            return _NA
        return float((1.0 + sub).prod() - 1.0)

    def _benchmark_cum_return_window(self, as_of: pd.Timestamp, months: int) -> object:
        s, e, _, _ = self._benchmark_bounds_window(as_of, months)
        if s is None or e is None:
            return _NA
        dr = self._benchmark_daily_returns()
        sub = dr[(dr.index > s) & (dr.index <= e)]
        if len(sub) < 1:
            return _NA
        return float((1.0 + sub).prod() - 1.0)

    def _benchmark_cum_return_period_type(self, as_of: pd.Timestamp, period: str) -> object:
        s, e, _, _ = self._benchmark_bounds_period_type(as_of, period)
        if s is None or e is None:
            return _NA
        dr = self._benchmark_daily_returns()
        sub = dr[(dr.index > s) & (dr.index <= e)]
        if len(sub) < 1:
            return _NA
        return float((1.0 + sub).prod() - 1.0)

    def _benchmark_ann_return_period(self, as_of: pd.Timestamp) -> object:
        s, e, _, _ = self._benchmark_bounds_period(as_of)
        if s is None or e is None:
            return _NA
        dr = self._benchmark_daily_returns()
        sub = dr[(dr.index > s) & (dr.index <= e)]
        if len(sub) < 2:
            return _NA
        return float(sub.mean() * ANNUALIZATION)

    def _period_nav_bounds(self, as_of: pd.Timestamp, period: str):
        nav = self._nav.dropna().sort_index()
        if nav.empty:
            return None, None, None, None
        as_of = pd.Timestamp(as_of)
        idx = nav.index
        if period == "YTD":
            prev_mask = idx.year == (as_of.year - 1)
            curr_mask = (idx.year == as_of.year) & (idx <= as_of)
        elif period == "MTD":
            prev_month_cal = as_of.to_period("M") - 1
            prev_mask = (idx.year == prev_month_cal.year) & (idx.month == prev_month_cal.month)
            curr_mask = (idx.year == as_of.year) & (idx.month == as_of.month) & (idx <= as_of)
        else:
            return None, None, None, None
        prev_idx = idx[prev_mask]
        curr_idx = idx[curr_mask]
        if prev_idx.empty or curr_idx.empty:
            return None, None, None, None
        start_date = prev_idx[-1]
        end_date = curr_idx[-1]
        return start_date, end_date, float(nav.loc[start_date]), float(nav.loc[end_date])

    def _ytd_start(self, as_of: pd.Timestamp) -> pd.Timestamp:
        s, _, _, _ = self._period_nav_bounds(as_of, "YTD")
        return s

    def _mtd_start(self, as_of: pd.Timestamp) -> pd.Timestamp:
        s, _, _, _ = self._period_nav_bounds(as_of, "MTD")
        return s

    # ---- Rendements cumulés ----

    def _cum_return(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        return float((1 + dr).prod() - 1)

    def _cum_return_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._cum_return(sub) if len(sub) >= 2 else _NA

    def _cum_return_nav_window(self, as_of: pd.Timestamp, months: int) -> object:
        nav = self._nav.dropna().sort_index()
        if nav.empty:
            return _NA
        start_target = pd.Timestamp(as_of) - pd.DateOffset(months=months)
        start_candidates = nav.index[nav.index <= start_target]
        end_candidates = nav.index[nav.index <= pd.Timestamp(as_of)]
        if start_candidates.empty or end_candidates.empty:
            return _NA
        nav_start = float(nav.loc[start_candidates[-1]])
        nav_end = float(nav.loc[end_candidates[-1]])
        return float(nav_end / nav_start - 1.0)

    def _cum_return_ytd(self, dr, as_of) -> object:
        _, _, nav_s, nav_e = self._period_nav_bounds(as_of, "YTD")
        if nav_s is None or nav_e is None:
            return _NA
        return float(nav_e / nav_s - 1.0)

    def _cum_return_mtd(self, dr, as_of) -> object:
        _, _, nav_s, nav_e = self._period_nav_bounds(as_of, "MTD")
        if nav_s is None or nav_e is None:
            return _NA
        return float(nav_e / nav_s - 1.0)

    # ---- Rendements annualisés ----

    def _ann_return(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        cumulative = (1 + dr).prod()
        n_years = len(dr) / ANNUALIZATION
        return float(cumulative ** (1 / n_years) - 1)

    # ---- CAGR ----

    def _cagr(self, start_nav, end_nav, dr) -> object:
        if len(dr) < 2:
            return _NA
        n_years = len(dr) / ANNUALIZATION
        if n_years <= 0 or start_nav <= 0:
            return _NA
        return float((end_nav / start_nav) ** (1 / n_years) - 1)

    def _cagr_from_returns(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        cum = float((1 + dr).prod())
        n_years = len(dr) / ANNUALIZATION
        if n_years <= 0:
            return _NA
        return float(cum ** (1 / n_years) - 1)

    def _cagr_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._cagr_from_returns(sub)

    def _cagr_ytd(self, dr, as_of) -> object:
        cr = self._cum_return_ytd(dr, as_of)
        if cr == _NA:
            return _NA
        s, e, _, _ = self._period_nav_bounds(as_of, "YTD")
        if s is None or e is None:
            return _NA
        n_years = (e - s).days / 365.25
        if n_years <= 0:
            return _NA
        return float((1 + float(cr)) ** (1 / n_years) - 1)

    def _cagr_mtd(self, dr, as_of) -> object:
        cr = self._cum_return_mtd(dr, as_of)
        if cr == _NA:
            return _NA
        s, e, _, _ = self._period_nav_bounds(as_of, "MTD")
        if s is None or e is None:
            return _NA
        n_years = (e - s).days / 365.25
        if n_years <= 0:
            return _NA
        return float((1 + float(cr)) ** (1 / n_years) - 1)

    # ---- Sharpe (excess return vs taux sans risque fixe) ----

    def _sharpe(self, dr: pd.Series, rf: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        excess = dr - rf.reindex(dr.index).fillna(RF_DAILY)
        vol = excess.std()
        if vol < 1e-10:
            return _NA
        return float(excess.mean() / vol * np.sqrt(ANNUALIZATION))

    def _sharpe_window(self, dr, rf, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        if len(sub) < 2:
            return _NA
        return self._sharpe(sub, rf)

    # ---- Sortino (excess return vs taux sans risque fixe) ----

    def _sortino(self, dr: pd.Series, rf: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        excess = dr - rf.reindex(dr.index).fillna(RF_DAILY)
        downside = excess[excess < 0].std()
        if downside < 1e-10:
            return _NA
        return float(excess.mean() / downside * np.sqrt(ANNUALIZATION))

    def _sortino_window(self, dr, rf, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        if len(sub) < 2:
            return _NA
        return self._sortino(sub, rf)

    # ---- Tracking Error (vs MSCI World) ----

    def _tracking_error(self, dr: pd.Series, benchmark: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        active = dr - benchmark.reindex(dr.index).fillna(0)
        return float(active.std() * np.sqrt(ANNUALIZATION))

    def _tracking_error_window(self, dr, benchmark, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        if len(sub) < 2:
            return _NA
        return self._tracking_error(sub, benchmark)

    # ---- Information Ratio (vs MSCI World) ----

    def _info_ratio(self, dr: pd.Series, benchmark: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        active = dr - benchmark.reindex(dr.index).fillna(0)
        te = active.std()
        if te < 1e-10:
            return _NA
        return float(active.mean() / te * np.sqrt(ANNUALIZATION))

    def _info_ratio_window(self, dr, benchmark, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        if len(sub) < 2:
            return _NA
        return self._info_ratio(sub, benchmark)

    # ---- Calmar ----

    def _calmar(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        ann = self._ann_return(dr)
        if ann == _NA:
            return _NA
        mdd = self._max_drawdown(dr)
        if mdd == _NA or abs(float(mdd)) < 1e-10:
            return _NA
        return float(ann) / abs(float(mdd))

    # ---- VaR / CVaR ----

    def _var(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        return float(-np.percentile(dr.dropna(), CONF_LEVEL * 100))

    def _var_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._var(sub)

    def _cvar(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        threshold = np.percentile(dr.dropna(), CONF_LEVEL * 100)
        tail = dr[dr <= threshold]
        return float(-tail.mean()) if len(tail) > 0 else _NA

    def _cvar_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._cvar(sub)

    # ---- Volatilité ----

    def _volatility(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        return float(dr.std() * np.sqrt(ANNUALIZATION))

    def _vol_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._volatility(sub)

    # ---- Max Drawdown ----

    def _max_drawdown(self, dr: pd.Series) -> object:
        if len(dr) < 2:
            return _NA
        cum = (1 + dr).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())

    def _mdd_window(self, dr, as_of, months) -> object:
        sub = self._window(dr, as_of, months)
        return self._max_drawdown(sub)

    def _mdd_ytd(self, dr, as_of) -> object:
        s, e, _, _ = self._period_nav_bounds(as_of, "YTD")
        if s is None or e is None:
            return _NA
        sub = dr[(dr.index > s) & (dr.index <= e)]
        return self._max_drawdown(sub)

    def _mdd_mtd(self, dr, as_of) -> object:
        s, e, _, _ = self._period_nav_bounds(as_of, "MTD")
        if s is None or e is None:
            return _NA
        sub = dr[(dr.index > s) & (dr.index <= e)]
        return self._max_drawdown(sub)

    # ---- Corrélation (vs MSCI World) ----

    def _corr(self, dr: pd.Series, benchmark: pd.Series) -> object:
        df = pd.concat([dr, benchmark], axis=1, join="inner").dropna()
        if df.shape[0] < 30:
            return _NA
        if df.iloc[:, 1].std() <= 1e-12:
            return _NA
        return float(df.iloc[:, 0].corr(df.iloc[:, 1]))

    def _empty_metrics(self) -> dict:
        return {
            k: _NA
            for k in [
                "Rendement cumulé (période)", "Rendement cumulé 1 an",
                "Rendement cumulé 2 ans", "Rendement cumulé YTD",
                "Rendement cumulé MTD", "Rendement annualisé (période)",
                "CAGR (période)", "CAGR 1 an", "CAGR 2 ans", "CAGR YTD", "CAGR MTD",
                "Sharpe ratio (période)", "Sharpe ratio 1 an",
                "Sortino ratio (période)", "Sortino ratio 1 an",
                "Tracking Error (période)", "Tracking Error 1 an",
                "Tracking Error 3 ans", "Information Ratio (période)",
                "Information Ratio 1 an", "Information Ratio 2 ans",
                "Calmar ratio (période)", "VaR 95% (période)", "VaR 95% 1 an",
                "CVaR 95% (période)", "CVaR 95% 1 an", "Volatilité (période)",
                "Volatilité 1 an", "Volatilité 2 ans", "Max Drawdown (période)",
                "Max Drawdown 1 an", "Max Drawdown 2 ans", "Max Drawdown YTD",
                "Max Drawdown MTD", "Corrélation avec benchmark (période)",
                "Benchmark - Rendement cumulé (période)",
                "Benchmark - Rendement cumulé 1 an",
                "Benchmark - Rendement cumulé 2 ans",
                "Benchmark - Rendement cumulé YTD",
                "Benchmark - Rendement cumulé MTD",
                "Benchmark - Rendement annualisé (période)",
                "TC total (€)", "TC total (%)", "PnL (€)",
            ]
        }

    # ------------------------------------------------------------------
    # Analyses de composition du portefeuille
    # ------------------------------------------------------------------

    def get_portfolio_composition(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Retourne la composition détaillée du portefeuille à une date donnée.
        Colonnes : Ticker, Name, Country, Currency, Sector, Industry, Price, Weight.
        """
        if date not in self.result.weights:
            rebal_dates = sorted(self.result.weights.keys())
            eligible = [d for d in rebal_dates if d <= date]
            if not eligible:
                return pd.DataFrame()
            date = eligible[-1]

        weights = self.result.weights[date]
        info = self.loader.informations
        prices = self.loader.get_price_at(date, weights.index.tolist())

        rows = []
        for ticker, w in weights.items():
            row = {
                "Ticker": ticker,
                "Weight": w,
                "Price": float(prices.get(ticker, np.nan)),
            }
            if ticker in info.index:
                row["Name"] = info.loc[ticker, "Name"]
                row["Country"] = info.loc[ticker, "Country"]
                row["Currency"] = info.loc[ticker, "Currency"]
                row["Sector"] = info.loc[ticker, "Sector"]
                row["Industry"] = info.loc[ticker, "Industry"]
            else:
                row.update({"Name": "", "Country": "", "Currency": "",
                            "Sector": "", "Industry": ""})
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df[["Ticker", "Name", "Country", "Currency", "Sector", "Industry",
                 "Price", "Weight"]]
        return df.sort_values("Weight", ascending=False).reset_index(drop=True)

    def get_top_10_weights(self, date: pd.Timestamp) -> pd.DataFrame:
        """Top 10 positions par poids absolu."""
        comp = self.get_portfolio_composition(date)
        if comp.empty:
            return pd.DataFrame()
        comp["Abs Weight"] = comp["Weight"].abs()
        return comp.nlargest(10, "Abs Weight")[
            ["Ticker", "Name", "Sector", "Country", "Weight"]
        ].reset_index(drop=True)

    def get_top_10_return_contribution(self, date: pd.Timestamp):
        """
        Retourne deux tableaux :
        1) Top 5 contributeurs positifs à la performance
        2) Top 5 contributeurs négatifs
        """
        rebal_dates = sorted(self.result.weights.keys())
        if date not in self.result.weights:
            eligible = [d for d in rebal_dates if d <= date]
            if not eligible:
                return pd.DataFrame(), pd.DataFrame()
            date = eligible[-1]
        if date not in rebal_dates:
            return pd.DataFrame(), pd.DataFrame()
        idx = rebal_dates.index(date)
        if idx == 0:
            return pd.DataFrame(), pd.DataFrame()
        prev_date = rebal_dates[idx - 1]
        prev_weights = self.result.weights[prev_date]
        all_tickers = prev_weights.index.tolist()
        daily_ret = self.loader.get_daily_returns(prev_date, date, all_tickers)
        if daily_ret.empty:
            return pd.DataFrame(), pd.DataFrame()
        period_ret = (1 + daily_ret.fillna(0)).prod() - 1
        ctr = prev_weights * period_ret.reindex(prev_weights.index).fillna(0)
        info = self.loader.informations
        df = pd.DataFrame({"Ticker": ctr.index, "Contribution": ctr.values})
        if info is not None and not info.empty:
            df = df.merge(info[["Name", "Sector"]], left_on="Ticker",
                         right_index=True, how="left")
        else:
            df["Name"] = ""
            df["Sector"] = ""
        top_pos = (df[df["Contribution"] > 0]
                   .sort_values("Contribution", ascending=False)
                   .head(5).reset_index(drop=True))
        top_neg = (df[df["Contribution"] < 0]
                   .sort_values("Contribution", ascending=True)
                   .head(5).reset_index(drop=True))
        return top_pos, top_neg

    def get_top_10_risk_contribution(self, date: pd.Timestamp):
        """
        Retourne deux tableaux :
        1) Top 5 contributeurs positifs au risque
        2) Top 5 contributeurs négatifs au risque
        """
        if date not in self.result.weights:
            rebal_dates = sorted(self.result.weights.keys())
            eligible = [d for d in rebal_dates if d <= date]
            if not eligible:
                return pd.DataFrame(), pd.DataFrame()
            date = eligible[-1]
        weights = self.result.weights[date]
        tickers = weights.index.tolist()
        start_lookback = date - pd.DateOffset(days=365)
        daily_ret = self.loader.get_daily_returns(start_lookback, date, tickers)
        if daily_ret.empty or daily_ret.shape[0] < 20:
            return pd.DataFrame(), pd.DataFrame()
        w = weights.reindex(daily_ret.columns).fillna(0).values
        cov = daily_ret.fillna(0).cov().values * ANNUALIZATION
        portfolio_var = float(w @ cov @ w)
        if portfolio_var < 1e-12:
            return pd.DataFrame(), pd.DataFrame()
        sigma_p = float(np.sqrt(portfolio_var))
        rc = w * (cov @ w) / sigma_p
        df = pd.DataFrame({
            "Ticker": daily_ret.columns,
            "Risk Contribution": rc,
            "Weight": [float(weights.get(t, 0.0)) for t in daily_ret.columns],
        })
        info = self.loader.informations
        if info is not None and not info.empty:
            df = df.merge(info[["Name", "Sector"]], left_on="Ticker",
                         right_index=True, how="left")
        else:
            df["Name"] = ""
            df["Sector"] = ""
        top_pos = (df[df["Risk Contribution"] > 0]
                   .sort_values("Risk Contribution", ascending=False)
                   .head(5).reset_index(drop=True))
        top_neg = (df[df["Risk Contribution"] < 0]
                   .sort_values("Risk Contribution", ascending=True)
                   .head(5).reset_index(drop=True))
        return top_pos, top_neg

    def _normalize_group_dimension(self, group_by: str) -> str:
        """Normalise la dimension d'agrégation demandée."""
        mapping = {
            "sector": "Sector",
            "country": "Country",
            "industry": "Industry",
            "currency": "Currency",
        }
        key = str(group_by).strip().lower()
        if key not in mapping:
            raise ValueError(
                "group_by doit être l'une des valeurs suivantes : "
                "Sector, Country, Industry, Currency"
            )
        return mapping[key]

    def _resolve_rebalancing_date(self, date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Retourne la date de rebalancement effective la plus proche <= date."""
        rebal_dates = sorted(self.result.weights.keys())
        if not rebal_dates:
            return None
        date = pd.Timestamp(date)
        if date in self.result.weights:
            return date
        eligible = [d for d in rebal_dates if d <= date]
        return eligible[-1] if eligible else None

    def get_group_return_impact(
        self,
        date: pd.Timestamp,
        group_by: str = "Sector",
    ) -> pd.DataFrame:
        """
        Agrège, à une date de rebalancement, l'impact total par dimension
        sur le rendement du portefeuille et le rendement propre du groupe
        sur la période de détention suivante.

        Retourne un DataFrame avec :
        - {group_by}
        - Weight
        - Group Return
        - Total Impact
        - Rebalancing Date
        - Period End
        """
        group_col = self._normalize_group_dimension(group_by)
        rebal_date = self._resolve_rebalancing_date(date)
        if rebal_date is None:
            return pd.DataFrame()

        rebal_dates = sorted(self.result.weights.keys())
        try:
            idx = rebal_dates.index(rebal_date)
        except ValueError:
            return pd.DataFrame()

        if idx + 1 < len(rebal_dates):
            period_end = rebal_dates[idx + 1]
        else:
            period_end = self._dr.index.max() if not self._dr.empty else rebal_date

        weights = self.result.weights.get(rebal_date, pd.Series(dtype=float))
        if weights.empty or period_end is None:
            return pd.DataFrame()

        daily_ret = self.loader.get_daily_returns(rebal_date, period_end, weights.index.tolist())
        if daily_ret.empty:
            return pd.DataFrame()

        ticker_returns = (1.0 + daily_ret.fillna(0.0)).prod() - 1.0
        comp = self.get_portfolio_composition(rebal_date)
        if comp.empty:
            return pd.DataFrame()

        comp = comp[["Ticker", "Weight", group_col]].copy()
        comp[group_col] = (
            comp[group_col]
            .fillna("")
            .replace("", "Non renseigné")
        )
        comp["Ticker Return"] = comp["Ticker"].map(ticker_returns).fillna(0.0)
        comp["Total Impact"] = comp["Weight"] * comp["Ticker Return"]

        group_weights = comp.groupby(group_col)["Weight"].sum()
        group_impacts = comp.groupby(group_col)["Total Impact"].sum()
        group_returns = group_impacts.div(group_weights.where(group_weights.abs() > 1e-12))

        grouped = pd.DataFrame({
            group_col: group_weights.index,
            "Weight": group_weights.values,
            "Group Return": group_returns.fillna(0.0).values,
            "Total Impact": group_impacts.values,
        })
        grouped["Rebalancing Date"] = pd.Timestamp(rebal_date)
        grouped["Period End"] = pd.Timestamp(period_end)

        return grouped.sort_values("Total Impact", ascending=False).reset_index(drop=True)

    def get_group_allocation_vs_benchmark(
        self,
        date: pd.Timestamp,
        group_by: str = "Sector",
    ) -> pd.DataFrame:
        """
        Compare les poids agrégés du portefeuille et du benchmark MSCI World
        par dimension de classification à une date de rebalancement.

        Retourne un DataFrame avec :
        - {group_by}
        - Portfolio Weight
        - Benchmark Weight
        - Active Weight
        - Rebalancing Date
        """
        group_col = self._normalize_group_dimension(group_by)
        rebal_date = self._resolve_rebalancing_date(date)
        if rebal_date is None:
            return pd.DataFrame()

        portfolio_comp = self.get_portfolio_composition(rebal_date)
        benchmark_comp = self.loader.get_component_info(rebal_date)
        if portfolio_comp.empty or benchmark_comp.empty:
            return pd.DataFrame()

        benchmark_col_map = {
            "Sector": "sector_name",
            "Country": "country_name",
            "Industry": "indsutry_name",
            "Currency": "currency_code",
        }
        benchmark_group_col = benchmark_col_map[group_col]
        if benchmark_group_col not in benchmark_comp.columns:
            return pd.DataFrame()

        portfolio_grouped = portfolio_comp[["Weight", group_col]].copy()
        portfolio_grouped[group_col] = (
            portfolio_grouped[group_col]
            .fillna("")
            .replace("", "Non renseigne")
        )
        portfolio_weights = portfolio_grouped.groupby(group_col)["Weight"].sum()

        benchmark_grouped = benchmark_comp[[benchmark_group_col, "weight_msci_world"]].copy()
        benchmark_grouped[benchmark_group_col] = (
            benchmark_grouped[benchmark_group_col]
            .fillna("")
            .replace("", "Non renseigne")
        )
        benchmark_weights = benchmark_grouped.groupby(benchmark_group_col)["weight_msci_world"].sum()
        benchmark_weights.index.name = group_col

        all_groups = portfolio_weights.index.union(benchmark_weights.index)
        comparison = pd.DataFrame({
            group_col: all_groups,
            "Portfolio Weight": portfolio_weights.reindex(all_groups).fillna(0.0).values,
            "Benchmark Weight": benchmark_weights.reindex(all_groups).fillna(0.0).values,
        })
        comparison["Active Weight"] = (
            comparison["Portfolio Weight"] - comparison["Benchmark Weight"]
        )
        comparison["Rebalancing Date"] = pd.Timestamp(rebal_date)

        comparison["Sort Key"] = comparison[
            ["Portfolio Weight", "Benchmark Weight"]
        ].max(axis=1)
        comparison = comparison.sort_values(
            ["Sort Key", group_col], ascending=[False, True]
        ).drop(columns="Sort Key")

        return comparison.reset_index(drop=True)

    def _get_benchmark_group_column(self, group_by: str) -> str:
        """Retourne le nom de colonne benchmark correspondant à la dimension demandée."""
        group_col = self._normalize_group_dimension(group_by)
        benchmark_col_map = {
            "Sector": "sector_name",
            "Country": "country_name",
            "Industry": "indsutry_name",
            "Currency": "currency_code",
        }
        return benchmark_col_map[group_col]

    def _get_rebalancing_period_end(self, rebal_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Retourne la prochaine date de rebalancement, sinon la dernière date de NAV."""
        rebal_date = self._resolve_rebalancing_date(rebal_date)
        if rebal_date is None:
            return None
        rebal_dates = sorted(self.result.weights.keys())
        try:
            idx = rebal_dates.index(rebal_date)
        except ValueError:
            return None
        if idx + 1 < len(rebal_dates):
            return pd.Timestamp(rebal_dates[idx + 1])
        if not self._nav.empty:
            return pd.Timestamp(self._nav.index.max())
        return pd.Timestamp(rebal_date)

    def _compute_ticker_period_returns(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        tickers: list,
    ) -> pd.Series:
        """
        Calcule les rendements cumulés par ticker entre deux dates
        à partir des prix observés au plus proche <= date.
        """
        if not tickers:
            return pd.Series(dtype=float)

        p_start = pd.Series(self.loader.get_price_at(start_date, tickers), dtype=float)
        p_end = pd.Series(self.loader.get_price_at(end_date, tickers), dtype=float)
        valid = p_start.notna() & p_end.notna() & (p_start > 0)

        returns = pd.Series(0.0, index=tickers, dtype=float)
        if valid.any():
            returns.loc[valid] = p_end.loc[valid] / p_start.loc[valid] - 1.0
        return returns

    def _compute_group_weighted_returns(
        self,
        df: pd.DataFrame,
        group_col: str,
        weight_col: str,
        return_col: str,
    ) -> pd.DataFrame:
        """
        Agrège par groupe les poids, contributions pondérées et rendements moyens pondérés.
        """
        if df.empty:
            return pd.DataFrame(columns=[group_col, "Weight", "Weighted Contribution", "Return"])

        work = df[[group_col, weight_col, return_col]].copy()
        work[group_col] = work[group_col].fillna("").replace("", "Non renseigné")
        work["Weighted Contribution"] = work[weight_col] * work[return_col]

        grouped = work.groupby(group_col, dropna=False).agg(
            Weight=(weight_col, "sum"),
            Weighted_Contribution=("Weighted Contribution", "sum"),
        )
        grouped["Return"] = grouped["Weighted_Contribution"].div(
            grouped["Weight"].where(grouped["Weight"].abs() > 1e-12)
        ).fillna(0.0)
        grouped = grouped.rename(columns={"Weighted_Contribution": "Weighted Contribution"})
        grouped = grouped.reset_index()
        return grouped

    def compute_group_period_attribution(
        self,
        date: pd.Timestamp,
        group_by: str = "Sector",
    ) -> pd.DataFrame:
        """
        Calcule l'attribution active par groupe sur une période de détention
        entre une date de rebalancement et la suivante.

        Méthode : Brinson-Fachler
        - Allocation = (Wp - Wb) * (Rb - Rb_total)
        - Sélection  = Wb * (Rp - Rb)
        - Interaction = (Wp - Wb) * (Rp - Rb)
        - Actif = Allocation + Sélection
        """
        group_col = self._normalize_group_dimension(group_by)
        benchmark_group_col = self._get_benchmark_group_column(group_col)

        rebal_date = self._resolve_rebalancing_date(date)
        period_end = self._get_rebalancing_period_end(rebal_date)
        if rebal_date is None or period_end is None:
            return pd.DataFrame()

        portfolio_comp = self.get_portfolio_composition(rebal_date)
        benchmark_comp = self.loader.get_component_info(rebal_date)
        if portfolio_comp.empty or benchmark_comp.empty:
            return pd.DataFrame()
        if benchmark_group_col not in benchmark_comp.columns or "weight_msci_world" not in benchmark_comp.columns:
            return pd.DataFrame()

        portfolio_comp = portfolio_comp[["Ticker", "Weight", group_col]].copy()
        benchmark_comp = benchmark_comp[["ticker", "weight_msci_world", benchmark_group_col]].copy()

        port_total = float(portfolio_comp["Weight"].sum())
        if abs(port_total) > 1e-12:
            portfolio_comp["Weight"] = portfolio_comp["Weight"] / port_total

        bench_total = float(benchmark_comp["weight_msci_world"].sum())
        if abs(bench_total) > 1e-12:
            benchmark_comp["weight_msci_world"] = benchmark_comp["weight_msci_world"] / bench_total

        all_tickers = sorted(
            set(portfolio_comp["Ticker"].dropna().tolist())
            | set(benchmark_comp["ticker"].dropna().tolist())
        )
        ticker_returns = self._compute_ticker_period_returns(rebal_date, period_end, all_tickers)
        if ticker_returns.empty:
            return pd.DataFrame()

        portfolio_comp["Ticker Return"] = portfolio_comp["Ticker"].map(ticker_returns).fillna(0.0)
        benchmark_comp["Ticker Return"] = benchmark_comp["ticker"].map(ticker_returns).fillna(0.0)

        portfolio_grouped = self._compute_group_weighted_returns(
            portfolio_comp,
            group_col=group_col,
            weight_col="Weight",
            return_col="Ticker Return",
        ).rename(columns={
            "Weight": "Portfolio Weight",
            "Weighted Contribution": "Portfolio Contribution",
            "Return": "Portfolio Return",
        })

        benchmark_grouped = self._compute_group_weighted_returns(
            benchmark_comp.rename(columns={benchmark_group_col: group_col}),
            group_col=group_col,
            weight_col="weight_msci_world",
            return_col="Ticker Return",
        ).rename(columns={
            "Weight": "Benchmark Weight",
            "Weighted Contribution": "Benchmark Contribution",
            "Return": "Benchmark Return",
        })

        benchmark_total_return = float(
            (benchmark_comp["weight_msci_world"] * benchmark_comp["Ticker Return"]).sum()
        )

        all_groups = portfolio_grouped[group_col].astype(str).tolist()
        all_groups = pd.Index(all_groups).union(benchmark_grouped[group_col].astype(str))

        attribution = pd.DataFrame({group_col: all_groups})
        attribution = attribution.merge(portfolio_grouped, on=group_col, how="left")
        attribution = attribution.merge(benchmark_grouped, on=group_col, how="left")

        for col in [
            "Portfolio Weight",
            "Portfolio Contribution",
            "Portfolio Return",
            "Benchmark Weight",
            "Benchmark Contribution",
            "Benchmark Return",
        ]:
            attribution[col] = attribution[col].fillna(0.0)

        zero_port_mask = attribution["Portfolio Weight"].abs() <= 1e-12
        attribution.loc[zero_port_mask, "Portfolio Return"] = attribution.loc[
            zero_port_mask, "Benchmark Return"
        ]

        attribution["Benchmark Total Return"] = benchmark_total_return
        attribution["Allocation Effect"] = (
            (attribution["Portfolio Weight"] - attribution["Benchmark Weight"])
            * (attribution["Benchmark Return"] - attribution["Benchmark Total Return"])
        )
        attribution["Selection Effect"] = (
            attribution["Benchmark Weight"]
            * (attribution["Portfolio Return"] - attribution["Benchmark Return"])
        )
        attribution["Interaction Effect"] = (
            (attribution["Portfolio Weight"] - attribution["Benchmark Weight"])
            * (attribution["Portfolio Return"] - attribution["Benchmark Return"])
        )
        attribution["Active Effect"] = (
            attribution["Allocation Effect"] + attribution["Selection Effect"]
        )
        attribution["Rebalancing Date"] = pd.Timestamp(rebal_date)
        attribution["Period End"] = pd.Timestamp(period_end)
        attribution["Year"] = int(pd.Timestamp(period_end).year)

        return attribution.sort_values("Active Effect", ascending=False).reset_index(drop=True)

    def compute_group_yearly_attribution(
        self,
        group_by: str = "Sector",
    ) -> pd.DataFrame:
        """
        Concatène les attributions par période sur toute la période de backtesting
        et les rattache à l'année de fin de période.
        """
        frames = []
        for rebal_date in sorted(self.result.weights.keys()):
            period_attr = self.compute_group_period_attribution(rebal_date, group_by=group_by)
            if not period_attr.empty:
                frames.append(period_attr)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_group_yearly_attribution_tables(
        self,
        group_by: str = "Sector",
    ) -> dict:
        """
        Retourne un dictionnaire de tableaux annuels par effet :
        Allocation, Selection, Interaction, Active.
        """
        group_col = self._normalize_group_dimension(group_by)
        yearly = self.compute_group_yearly_attribution(group_by=group_col)
        if yearly.empty:
            return {}

        effects_map = {
            "Allocation": "Allocation Effect",
            "Selection": "Selection Effect",
            "Interaction": "Interaction Effect",
            "Active": "Active Effect",
        }

        tables = {}
        year_columns = sorted(yearly["Year"].dropna().astype(int).unique().tolist())

        for label, effect_col in effects_map.items():
            pivot = yearly.pivot_table(
                index=group_col,
                columns="Year",
                values=effect_col,
                aggfunc="sum",
                fill_value=0.0,
            )
            pivot = pivot.reindex(columns=year_columns, fill_value=0.0)
            pivot["Total"] = pivot.sum(axis=1)
            pivot = pivot.sort_values("Total", ascending=False)
            total_row = pd.DataFrame([pivot.sum(axis=0)], index=[f"TOTAL Effet {label}"])
            pivot = pd.concat([pivot, total_row], axis=0)
            tables[label] = pivot

        return tables

    def get_group_yearly_attribution_report(
        self,
        group_by: str = "Sector",
    ) -> pd.DataFrame:
        """
        Construit un tableau consolidé dans l'esprit d'un rapport d'attribution,
        avec une section par effet et un total par année.
        """
        tables = self.get_group_yearly_attribution_tables(group_by=group_by)
        if not tables:
            return pd.DataFrame()

        group_col = self._normalize_group_dimension(group_by)
        effect_labels = [
            ("Allocation", "Effet Allocation"),
            ("Selection", "Effet Sélection"),
            ("Interaction", "Effet Interaction"),
            ("Active", "Effet Actif (Alloc + Sélec)"),
        ]

        rows = []
        all_columns = None

        for key, row_prefix in effect_labels:
            table = tables.get(key)
            if table is None or table.empty:
                continue
            if all_columns is None:
                all_columns = table.columns.tolist()

            detail_table = table.drop(index=f"TOTAL Effet {key}", errors="ignore")
            for group_name, values in detail_table.iterrows():
                row_label = f"{row_prefix} - {group_name}"
                rows.append([row_label] + values.tolist())

            total_label = "TOTAL " + row_prefix
            total_series = table.loc[f"TOTAL Effet {key}"] if f"TOTAL Effet {key}" in table.index else detail_table.sum(axis=0)
            rows.append([total_label] + total_series.tolist())
            rows.append([""] + [np.nan] * len(all_columns))

        if not rows or all_columns is None:
            return pd.DataFrame()

        report = pd.DataFrame(rows, columns=["Analyse"] + all_columns)
        if not report.empty and report.iloc[-1, 0] == "":
            report = report.iloc[:-1].copy()

        rename_map = {col: str(col) for col in report.columns if isinstance(col, (int, np.integer))}
        report = report.rename(columns=rename_map)
        return report

    def style_group_yearly_attribution_report(
        self,
        group_by: str = "Sector",
    ):
        """
        Retourne une version stylée du tableau d'attribution annuel.
        Compatible avec pandas Styler pour Streamlit / notebooks.
        """
        report = self.get_group_yearly_attribution_report(group_by=group_by)
        if report.empty:
            return report

        value_columns = [c for c in report.columns if c != "Analyse"]

        def _cell_style(value):
            if pd.isna(value):
                return ""
            if value > 0:
                return "background-color: #DFF3E3; color: #0B6E27;"
            if value < 0:
                return "background-color: #F9D9DE; color: #A61B29;"
            return "background-color: #F5F7FA; color: #475467;"

        def _row_style(row):
            label = str(row.iloc[0])
            if not label:
                return ["background-color: white;"] * len(row)
            if label.startswith("TOTAL "):
                return ["background-color: #FFF7BF; font-weight: 700;"] * len(row)
            return [""] * len(row)

        return (
            report.style
            .format({col: "{:.2%}" for col in value_columns}, na_rep="")
            .applymap(_cell_style, subset=value_columns)
            .apply(_row_style, axis=1)
        )

    # ------------------------------------------------------------------
    # Visualisations Plotly
    # ------------------------------------------------------------------

    def plot_cumulative_returns(self) -> go.Figure:
        """Courbes des rendements cumulés : portefeuille vs MSCI World."""
        dr = self._dr
        if dr.empty:
            return go.Figure()
        nav_norm = (self._nav / self.result.initial_capital) * 100
        bench_cum = (1 + self._ester_dr.reindex(dr.index).fillna(0)).cumprod() * 100
        strategy_label = getattr(self.result, "strategy_label",
                                 METHODS_LABELS.get(self.result.method, self.result.method))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nav_norm.index, y=nav_norm.values,
            name=f"Portefeuille ({strategy_label})",
            line=dict(color=AMUNDI_TEAL, width=2)))
        fig.add_trace(go.Scatter(
            x=bench_cum.index, y=bench_cum.values,
            name="MSCI World EUR (benchmark)",
            line=dict(color=AMUNDI_NAVY, width=1.5, dash="dash")))
        fig.update_layout(
            title="Rendements cumulés : Portefeuille vs MSCI World EUR",
            xaxis_title="Date", yaxis_title="Valeur normalisée (base 100)",
            legend=dict(x=0.01, y=0.99, font=dict(color=AMUNDI_TEXT)),
            paper_bgcolor="white", plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            template="plotly_white")
        return fig

    def _load_multi_benchmark_levels(self) -> pd.DataFrame:
        """
        Charge les niveaux des benchmarks complémentaires depuis
        data/storage/benchmark_returns.parquet.
        """
        cached = getattr(self, "_multi_benchmark_levels_cache", None)
        if cached is not None:
            return cached.copy()

        path = self._STOCKAGE_ROOT / "benchmark_returns.parquet"
        if not path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        benchmark_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not benchmark_cols:
            return pd.DataFrame()

        levels = df[benchmark_cols].copy().dropna(how="all")
        self._multi_benchmark_levels_cache = levels.copy()
        return levels

    def get_multi_benchmark_base100(self) -> pd.DataFrame:
        """
        Retourne les 3 benchmarks du fichier benchmark_returns.parquet
        normalisés en base 100 sur leur première observation disponible.
        """
        levels = self._load_multi_benchmark_levels()
        if levels.empty:
            return pd.DataFrame()

        base100 = pd.DataFrame(index=levels.index)
        for col in levels.columns:
            series = levels[col].dropna()
            if series.empty:
                continue
            first_value = float(series.iloc[0])
            if abs(first_value) <= 1e-12:
                continue
            base100[col] = levels[col] / first_value * 100.0

        return base100.dropna(how="all")

    def plot_cumulative_returns_vs_all_benchmarks(self) -> go.Figure:
        """
        Courbes base 100 du portefeuille versus les 3 benchmarks
        présents dans data/storage/benchmark_returns.parquet.
        """
        if self._nav.empty:
            return go.Figure()

        portfolio_base100 = (self._nav / self.result.initial_capital) * 100.0
        benchmarks_base100 = self.get_multi_benchmark_base100()

        fig = go.Figure()
        strategy_label = getattr(
            self.result,
            "strategy_label",
            METHODS_LABELS.get(self.result.method, self.result.method),
        )

        fig.add_trace(go.Scatter(
            x=portfolio_base100.index,
            y=portfolio_base100.values,
            name=f"Portefeuille ({strategy_label})",
            line=dict(color=AMUNDI_TEAL, width=2.5),
        ))

        benchmark_colors = [
            AMUNDI_NAVY,
            AMUNDI_RED,
            "#7A5AF8",
            "#FDB022",
            "#344054",
        ]

        for i, col in enumerate(benchmarks_base100.columns):
            fig.add_trace(go.Scatter(
                x=benchmarks_base100.index,
                y=benchmarks_base100[col].values,
                name=col,
                line=dict(
                    color=benchmark_colors[i % len(benchmark_colors)],
                    width=1.8,
                    dash="dash",
                ),
            ))

        fig.update_layout(
            title="Rendements cumulés base 100 : Portefeuille vs 3 benchmarks",
            xaxis_title="Date",
            yaxis_title="Valeur normalisée (base 100)",
            legend=dict(x=0.01, y=0.99, font=dict(color=AMUNDI_TEXT)),
            paper_bgcolor="white",
            plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            template="plotly_white",
        )
        return fig

    def plot_drawdowns(self) -> go.Figure:
        """Courbe des drawdowns du portefeuille sur toute la période."""
        dr = self._dr
        if dr.empty:
            return go.Figure()
        cum = (1 + dr).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values * 100, fill="tozeroy",
            name="Drawdown (%)",
            line=dict(color=AMUNDI_RED),
            fillcolor="rgba(227, 6, 19, 0.18)"))
        fig.update_layout(
            title="Drawdowns du portefeuille",
            xaxis_title="Date", yaxis_title="Drawdown (%)",
            paper_bgcolor="white", plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            template="plotly_white")
        return fig

    def plot_historical_volatility(self) -> go.Figure:
        """Volatilité historique annualisée du portefeuille."""
        dr = self._dr.dropna().sort_index()
        if dr.empty:
            return go.Figure()
        vol_hist = dr.expanding(min_periods=2).std() * np.sqrt(ANNUALIZATION) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vol_hist.index, y=vol_hist.values,
            name="Volatilité historique (%)",
            line=dict(color=AMUNDI_NAVY, width=2)))
        fig.update_layout(
            title="Volatilité historique annualisée",
            xaxis_title="Date", yaxis_title="Volatilité (%)",
            paper_bgcolor="white", plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            template="plotly_white")
        return fig

    def plot_historical_correlation(self) -> go.Figure:
        """Corrélation historique entre le portefeuille et le MSCI World EUR."""
        dr = self._dr.dropna().sort_index()
        if dr.empty:
            return go.Figure()
        bench = self._ester_dr.dropna().sort_index()
        df = pd.concat([dr.rename("portfolio"), bench.rename("benchmark")],
                       axis=1, join="inner").dropna()
        if df.shape[0] < 30:
            return go.Figure()
        corr_hist = df["portfolio"].expanding(min_periods=30).corr(df["benchmark"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=corr_hist.index, y=corr_hist.values,
            name="Corrélation historique",
            line=dict(color=AMUNDI_TEAL, width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color=AMUNDI_GRID)
        fig.update_layout(
            title="Corrélation historique Portefeuille vs MSCI World EUR",
            xaxis_title="Date", yaxis_title="Corrélation",
            paper_bgcolor="white", plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            yaxis=dict(range=[-1, 1], gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            template="plotly_white")
        return fig

    def plot_composition_barcharts(self, date: pd.Timestamp) -> dict:
        """Bar charts de composition : Sector, Country, Industry, Currency."""
        comp = self.get_portfolio_composition(date)
        if comp.empty:
            return {}
        figures = {}
        for dim in ["Sector", "Country", "Industry", "Currency"]:
            grouped = comp.groupby(dim)["Weight"].sum().sort_values()
            fig = go.Figure(go.Bar(
                x=grouped.values * 100, y=grouped.index,
                orientation="h",
                marker_color=[AMUNDI_TEAL if v >= 0 else AMUNDI_RED
                              for v in grouped.values]))
            fig.update_layout(
                title=f"Poids par {dim} — {date.date()}",
                xaxis_title="Poids (%)", yaxis_title=dim,
                paper_bgcolor="white", plot_bgcolor=AMUNDI_LIGHT_BG,
                font=dict(color=AMUNDI_TEXT),
                xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
                yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
                template="plotly_white")
            figures[dim] = fig
        return figures

    def plot_group_return_impact(
        self,
        date: pd.Timestamp,
        group_by: str = "Sector",
    ) -> go.Figure:
        """
        Trace, pour une date de rebalancement, l'impact total par groupe
        sur l'axe de gauche et le rendement du groupe sur l'axe de droite.
        """
        group_col = self._normalize_group_dimension(group_by)
        grouped = self.get_group_return_impact(date, group_by=group_col)
        if grouped.empty:
            return go.Figure()

        rebal_date = pd.Timestamp(grouped["Rebalancing Date"].iloc[0])
        period_end = pd.Timestamp(grouped["Period End"].iloc[0])

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=grouped[group_col],
                y=grouped["Total Impact"] * 100.0,
                name="Impact total sur le rendement portefeuille",
                marker_color=[
                    AMUNDI_TEAL if v >= 0 else AMUNDI_RED
                    for v in grouped["Total Impact"]
                ],
                customdata=np.column_stack([
                    grouped["Weight"].values * 100.0,
                    grouped["Group Return"].values * 100.0,
                ]),
                hovertemplate=(
                    f"{group_col}: %{{x}}<br>"
                    "Impact total: %{y:.2f}%<br>"
                    "Poids: %{customdata[0]:.2f}%<br>"
                    "Return groupe: %{customdata[1]:.2f}%"
                    "<extra></extra>"
                ),
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=grouped[group_col],
                y=grouped["Group Return"] * 100.0,
                name=f"Return {group_col}",
                mode="lines+markers",
                line=dict(color=AMUNDI_NAVY, width=2),
                marker=dict(color=AMUNDI_NAVY, size=8),
                hovertemplate=(
                    f"{group_col}: %{{x}}<br>"
                    "Return groupe: %{y:.2f}%"
                    "<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=(
                f"Impact sur le rendement portefeuille et return par {group_col} "
                f"— {rebal_date.date()} à {period_end.date()}"
            ),
            xaxis_title=group_col,
            paper_bgcolor="white",
            plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            legend=dict(x=0.01, y=0.99, font=dict(color=AMUNDI_TEXT)),
            xaxis=dict(
                gridcolor=AMUNDI_GRID,
                zerolinecolor=AMUNDI_GRID,
                tickangle=-35,
            ),
            template="plotly_white",
        )
        fig.update_yaxes(
            title_text="Impact total sur le rendement portefeuille (%)",
            gridcolor=AMUNDI_GRID,
            zerolinecolor=AMUNDI_GRID,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text=f"Return {group_col} (%)",
            gridcolor=AMUNDI_GRID,
            zerolinecolor=AMUNDI_GRID,
            secondary_y=True,
        )

        return fig

    def plot_group_return_impact_barcharts(self, date: pd.Timestamp) -> dict:
        """Retourne les graphiques d'impact/return pour Sector, Country, Industry, Currency."""
        figures = {}
        for dim in ["Sector", "Country", "Industry", "Currency"]:
            figures[dim] = self.plot_group_return_impact(date, group_by=dim)
        return figures

    def plot_group_allocation_vs_benchmark(
        self,
        date: pd.Timestamp,
        group_by: str = "Sector",
    ) -> go.Figure:
        """
        Trace un bar chart groupé comparant l'allocation du portefeuille
        à celle du benchmark MSCI World par dimension.
        """
        group_col = self._normalize_group_dimension(group_by)
        comparison = self.get_group_allocation_vs_benchmark(date, group_by=group_col)
        if comparison.empty:
            return go.Figure()

        rebal_date = pd.Timestamp(comparison["Rebalancing Date"].iloc[0])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison[group_col],
            y=comparison["Portfolio Weight"],
            name="Portefeuille",
            marker_color="#636EFA",
            hovertemplate=(
                f"{group_col}: %{{x}}<br>"
                "Poids portefeuille: %{y:.2%}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Bar(
            x=comparison[group_col],
            y=comparison["Benchmark Weight"],
            name="MSCI World",
            marker_color="#EF553B",
            hovertemplate=(
                f"{group_col}: %{{x}}<br>"
                "Poids benchmark: %{y:.2%}<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=f"Poids par {group_col} : Portefeuille vs MSCI World — {rebal_date.date()}",
            xaxis_title=group_col,
            yaxis_title="Poids",
            barmode="group",
            paper_bgcolor="white",
            plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            legend=dict(x=0.82, y=0.98, font=dict(color=AMUNDI_TEXT)),
            xaxis=dict(
                gridcolor=AMUNDI_GRID,
                zerolinecolor=AMUNDI_GRID,
                tickangle=-45,
                categoryorder="array",
                categoryarray=comparison[group_col].tolist(),
            ),
            yaxis=dict(
                gridcolor=AMUNDI_GRID,
                zerolinecolor=AMUNDI_GRID,
                tickformat=".0%",
            ),
            template="plotly_white",
        )
        return fig

    def plot_group_allocation_vs_benchmark_barcharts(self, date: pd.Timestamp) -> dict:
        """
        Retourne les bar charts Portefeuille vs MSCI World
        pour Sector, Country, Industry et Currency.
        """
        figures = {}
        for dim in ["Sector", "Country", "Industry", "Currency"]:
            figures[dim] = self.plot_group_allocation_vs_benchmark(date, group_by=dim)
        return figures

    def plot_pnl(self) -> go.Figure:
        """PnL cumulé du portefeuille."""
        dr = self._dr
        if dr.empty:
            return go.Figure()
        pnl = self._nav - self.result.initial_capital
        strategy_label = getattr(self.result, "strategy_label",
                                 METHODS_LABELS.get(self.result.method, self.result.method))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pnl.index, y=pnl.values,
            name="PnL cumulé (€)",
            line=dict(color=AMUNDI_TEAL, width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 163, 173, 0.12)"))
        fig.add_hline(y=0, line_dash="dash", line_color=AMUNDI_GRID, line_width=1)
        fig.update_layout(
            title=f"Portfolio PnL — {strategy_label} (€)",
            xaxis_title="Date", yaxis_title="PnL cumulé (€)",
            paper_bgcolor="white", plot_bgcolor=AMUNDI_LIGHT_BG,
            font=dict(color=AMUNDI_TEXT),
            xaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            template="plotly_white")
        return fig

    def _calendar_returns_from_levels(self, levels: pd.Series) -> pd.Series:
        if levels.empty:
            return pd.Series(dtype=float)
        s = levels.dropna().sort_index()
        if s.empty:
            return pd.Series(dtype=float)
        year_end = s.groupby(s.index.year).last()
        calendar = year_end / year_end.shift(1) - 1.0
        calendar.index = calendar.index.astype(int)
        return calendar

    def compute_calendar_returns(self) -> pd.DataFrame:
        """Calcule les calendar returns annuels du portefeuille."""
        portfolio_calendar = self._calendar_returns_from_levels(self._nav)
        years = sorted(portfolio_calendar.index.tolist())
        calendar_returns = pd.DataFrame(index=["Portfolio"], columns=years, dtype=float)
        if not years:
            return calendar_returns
        calendar_returns.loc["Portfolio", portfolio_calendar.index] = portfolio_calendar.values
        return calendar_returns

    def plot_calendar_returns_heatmap(self) -> go.Figure:
        """Heatmap des rendements annuels calendrier."""
        calendar_returns = self.compute_calendar_returns()
        if calendar_returns.empty or calendar_returns.shape[1] == 0:
            return go.Figure()
        calendar_returns = calendar_returns.dropna(axis=1, how="all")
        if calendar_returns.shape[1] == 0:
            return go.Figure()
        z = calendar_returns.values * 100.0
        zmax = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
        if zmax == 0:
            zmax = 1.0
        text = np.empty_like(z, dtype=object)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                text[i, j] = "" if pd.isna(z[i, j]) else f"{z[i, j]:.2f}%"
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=calendar_returns.columns.tolist(),
            y=calendar_returns.index.tolist(),
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12, "color": AMUNDI_DARK},
            colorscale=[[0.0, AMUNDI_RED], [0.5, "#FFFFFF"], [1.0, AMUNDI_TEAL]],
            zmin=-zmax, zmax=zmax,
            colorbar=dict(
                title=dict(text="Return (%)", font=dict(color=AMUNDI_TEXT)),
                tickfont=dict(color=AMUNDI_TEXT)),
            hovertemplate="%{y} | %{x}: %{z:.2f}%<extra></extra>"))
        fig.update_layout(
            title="Calendar Returns (Annual) – Portfolio",
            xaxis_title="Year",
            xaxis=dict(tickmode="array",
                       tickvals=calendar_returns.columns.tolist(),
                       ticktext=[str(y) for y in calendar_returns.columns.tolist()],
                       gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID),
            paper_bgcolor="white", plot_bgcolor=AMUNDI_LIGHT_BG,
            template="plotly_white",
            font=dict(family="Arial", size=12, color=AMUNDI_TEXT),
            yaxis=dict(gridcolor=AMUNDI_GRID, zerolinecolor=AMUNDI_GRID))
        return fig

    # ------------------------------------------------------------------
    # Exports — convention de nommage scalable
    # ------------------------------------------------------------------

    def _get_output_dir(self) -> Path:
        """
        Crée et retourne le sous-dossier de stockage pour cette stratégie.

        Convention : {strategy_label}_{start}_{end}/
        Où strategy_label = {signal}_{method}_{selection}_{quantile}_{freq}_{ucits}
        """
        strategy_label = getattr(
            self.result, "strategy_label",
            METHODS_LABELS.get(self.result.method, self.result.method),
        )
        rebal = sorted(self.result.weights.keys())
        start_str = rebal[0].strftime("%Y%m%d") if rebal else "NA"
        end_str = rebal[-1].strftime("%Y%m%d") if rebal else "NA"
        folder_name = f"{strategy_label}_{start_str}_{end_str}"
        out_dir = self._STOCKAGE_ROOT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def export_bbu_csv(self, output_dir: Optional[Path] = None) -> Path:
        """Exporte le fichier CSV BBU (Date, Ticker, Weights)."""
        if output_dir is None:
            output_dir = self._get_output_dir()
        strategy_label = getattr(self.result, "strategy_label", self.result.method)
        rows = []
        for date, weights in self.result.weights.items():
            for ticker, w in weights.items():
                rows.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Weights": round(float(w), 6),
                })
        df = pd.DataFrame(rows)[["Date", "Ticker", "Weights"]]
        path = output_dir / f"bbu_{strategy_label}.csv"
        df.to_csv(path, index=False)
        print(f"[Report] BBU CSV exporté : {path}")
        return path

    def export_detailed_parquet(self, output_dir: Optional[Path] = None) -> Path:
        """
        Exporte la composition enrichie en Parquet.
        Colonnes : Date, Ticker, Name, Country, Currency, Sector, Industry, Price, Weight.
        """
        if output_dir is None:
            output_dir = self._get_output_dir()
        strategy_label = getattr(self.result, "strategy_label", self.result.method)
        all_rows = []
        for date in sorted(self.result.weights.keys()):
            comp = self.get_portfolio_composition(date)
            if not comp.empty:
                comp.insert(0, "Date", date.strftime("%Y-%m-%d"))
                all_rows.append(comp)
        if not all_rows:
            return output_dir / "empty.parquet"
        df = pd.concat(all_rows, ignore_index=True)
        path = output_dir / f"composition_detaillee_{strategy_label}.parquet"
        df.to_parquet(path, index=False)
        print(f"[Report] Composition détaillée exportée : {path}")
        return path

    def export_metrics_parquet(
        self, metrics_df: pd.DataFrame, output_dir: Optional[Path] = None
    ) -> Path:
        """Exporte le tableau des métriques de performance en Parquet."""
        if output_dir is None:
            output_dir = self._get_output_dir()
        strategy_label = getattr(self.result, "strategy_label", self.result.method)
        df = metrics_df.copy().replace(_NA, np.nan)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        path = output_dir / f"metriques_{strategy_label}.parquet"
        df.to_parquet(path)
        print(f"[Report] Métriques exportées : {path}")
        return path

    def export_nav_parquet(self, output_dir: Optional[Path] = None) -> Path:
        """Exporte la série de NAV journalière en Parquet."""
        if output_dir is None:
            output_dir = self._get_output_dir()
        strategy_label = getattr(self.result, "strategy_label", self.result.method)
        nav_df = self._nav.rename("NAV").to_frame()
        nav_df.index.name = "Date"
        path = output_dir / f"nav_{strategy_label}.parquet"
        nav_df.to_parquet(path)
        print(f"[Report] NAV exportée : {path}")
        return path

    # ------------------------------------------------------------------
    # Run complet du reporting
    # ------------------------------------------------------------------

    def run_full_report(self, output_dir: Optional[Path] = None) -> str:
        """
        Génère l'ensemble du reporting et sauvegarde tous les fichiers :
          - CSV BBU
          - Parquet composition enrichie
          - Parquet métriques de performance
          - Parquet NAV

        Retourne le chemin du dossier de sortie.
        """
        if output_dir is None:
            output_dir = self._get_output_dir()

        strategy_label = getattr(self.result, "strategy_label", self.result.method)
        print(f"\n[ReportingEngine] Génération du reporting pour : {strategy_label}")
        print(f"[ReportingEngine] Dossier de sortie : {output_dir}")

        metrics_df = self.compute_all_metrics()
        self.export_metrics_parquet(metrics_df, output_dir)
        self.export_bbu_csv(output_dir)
        self.export_detailed_parquet(output_dir)
        self.export_nav_parquet(output_dir)

        print(f"[ReportingEngine] Reporting complet généré dans : {output_dir}\n")
        return str(output_dir)
