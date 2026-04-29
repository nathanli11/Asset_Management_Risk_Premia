"""
signals.py
----------
Calcul des signaux momentum pour la stratégie long-only sur le MSCI World.

Signaux implémentés :
  1. Momentum 12-1 : rendement cumulé sur 12 mois, skip 1 mois
  2. Momentum idiosyncratique 12-1 : résidu de régression sur benchmark, 12-1
  3. Momentum 5 ans (mean reverting) : rendement 5 ans, signe inversé (contrarian)
  4. Momentum idiosyncratique 5 ans (mean reverting) : résidu 5 ans, signe inversé

Tous les signaux sont :
  - Calculés uniquement avec les données disponibles à la date de rebalancement
  - Standardisés (z-score) en cross-section intra-sectorielle si un mapping sectoriel
    est fourni, sinon en cross-section globale

Sélection des titres :
  - Best-in-universe : classement global
  - Best-in-class    : classement intra-sectoriel
  - Granularité : décile / quintile / quartile
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Minimum de mois d'historique requis pour considérer un signal comme valide
_MIN_MONTHS_REQUIRED = 6


class SignalType(Enum):
    MOMENTUM_12_1 = "momentum_12_1"
    IDIOSYNCRATIC_12_1 = "idiosyncratic_12_1"
    MOMENTUM_5Y_MEAN_REVERTING = "momentum_5y_mean_reverting"
    IDIOSYNCRATIC_5Y_MEAN_REVERTING = "idiosyncratic_5y_mean_reverting"


class SelectionMode(Enum):
    BEST_IN_UNIVERSE = "best_in_universe"
    BEST_IN_CLASS = "best_in_class"


class QuantileMode(Enum):
    DECILE = "decile"     # top 10 %
    QUINTILE = "quintile" # top 20 %
    QUARTILE = "quartile" # top 25 %


# Seuil quantile pour chaque mode (percentile inférieur du top N%)
_QUANTILE_THRESHOLD: Dict[QuantileMode, float] = {
    QuantileMode.DECILE: 0.90,
    QuantileMode.QUINTILE: 0.80,
    QuantileMode.QUARTILE: 0.75,
}

# Libellés courts pour les noms de fichiers / dossiers
SIGNAL_LABELS: Dict[SignalType, str] = {
    SignalType.MOMENTUM_12_1: "Mom12_1",
    SignalType.IDIOSYNCRATIC_12_1: "IdioMom12_1",
    SignalType.MOMENTUM_5Y_MEAN_REVERTING: "Mom5Y_MR",
    SignalType.IDIOSYNCRATIC_5Y_MEAN_REVERTING: "IdioMom5Y_MR",
}

SELECTION_LABELS: Dict[SelectionMode, str] = {
    SelectionMode.BEST_IN_UNIVERSE: "BIU",
    SelectionMode.BEST_IN_CLASS: "BIC",
}

QUANTILE_LABELS: Dict[QuantileMode, str] = {
    QuantileMode.DECILE: "D10",
    QuantileMode.QUINTILE: "Q20",
    QuantileMode.QUARTILE: "Q25",
}


class SignalCalculator:
    """
    Calcule les signaux momentum pour un univers de tickers à une date donnée.

    Paramètres
    ----------
    prices          : DataFrame de prix EUR journaliers (index=Date, cols=tickers)
    benchmark_returns : Série des rendements mensuels MSCI World (index=date de rebal)
    """

    _LOOKBACK_12_1_TOTAL = 12   # mois de lookback pour momentum 12-1
    _LOOKBACK_SKIP = 1           # skip du dernier mois (mean reversion court terme)
    _LOOKBACK_5Y_MONTHS = 60    # mois de lookback pour momentum 5 ans

    def __init__(self, prices: pd.DataFrame, benchmark_returns: pd.Series):
        self._prices = prices.sort_index()
        self._benchmark_returns = benchmark_returns.sort_index()
        self._monthly_prices: Optional[pd.DataFrame] = None

    @property
    def monthly_prices(self) -> pd.DataFrame:
        if self._monthly_prices is None:
            self._monthly_prices = self._prices.resample("ME").last()
        return self._monthly_prices

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def compute(
        self,
        signal_type: SignalType,
        rebal_date: pd.Timestamp,
        universe: List[str],
        sector_mapping: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Calcule et retourne le signal normalisé (z-score) pour l'univers donné.

        Paramètres
        ----------
        signal_type    : type de signal à calculer
        rebal_date     : date de rebalancement (fin de mois)
        universe       : liste des tickers dans l'univers à cette date
        sector_mapping : pd.Series {ticker: sector}, pour standardisation intra-sectorielle

        Retourne
        --------
        pd.Series {ticker: score_zscore} – NaN pour tickers sans historique suffisant
        """
        rebal_date = pd.Timestamp(rebal_date)
        valid_tickers = [t for t in universe if t in self._prices.columns]

        if not valid_tickers:
            return pd.Series(dtype=float)

        # Calcul du signal brut
        if signal_type == SignalType.MOMENTUM_12_1:
            raw = self._momentum_12_1(rebal_date, valid_tickers)
        elif signal_type == SignalType.IDIOSYNCRATIC_12_1:
            raw = self._idiosyncratic_12_1(rebal_date, valid_tickers)
        elif signal_type == SignalType.MOMENTUM_5Y_MEAN_REVERTING:
            raw = self._momentum_5y(rebal_date, valid_tickers, mean_reverting=True)
        elif signal_type == SignalType.IDIOSYNCRATIC_5Y_MEAN_REVERTING:
            raw = self._idiosyncratic_5y(rebal_date, valid_tickers, mean_reverting=True)
        else:
            raise ValueError(f"Type de signal inconnu : {signal_type}")

        if raw.empty:
            return raw

        # Standardisation intra-sectorielle ou globale
        if sector_mapping is not None and not sector_mapping.empty:
            return self._standardize_within_sectors(raw, sector_mapping)
        return self._zscore(raw)

    def select_universe(
        self,
        signal: pd.Series,
        quantile_mode: QuantileMode,
        selection_mode: SelectionMode,
        sector_mapping: Optional[pd.Series] = None,
        min_stocks: int = 20,
    ) -> List[str]:
        """
        Sélectionne les tickers dans le top N% selon le signal.

        Paramètres
        ----------
        signal         : pd.Series {ticker: score}
        quantile_mode  : granularité (décile / quintile / quartile)
        selection_mode : best_in_universe ou best_in_class
        sector_mapping : requis pour best_in_class
        min_stocks     : nombre minimum de titres retenus (fallback si trop peu)

        Retourne
        --------
        List[str] : tickers sélectionnés
        """
        signal = signal.dropna()
        if signal.empty:
            return []

        threshold = _QUANTILE_THRESHOLD[quantile_mode]

        if selection_mode == SelectionMode.BEST_IN_UNIVERSE:
            cutoff = signal.quantile(threshold)
            selected = signal[signal >= cutoff].index.tolist()

        elif selection_mode == SelectionMode.BEST_IN_CLASS:
            if sector_mapping is None or sector_mapping.empty:
                logger.warning(
                    "Best-in-class requiert un sector_mapping ; fallback best-in-universe"
                )
                cutoff = signal.quantile(threshold)
                selected = signal[signal >= cutoff].index.tolist()
            else:
                selected = []
                for sector in sector_mapping.unique():
                    tickers_sector = sector_mapping[sector_mapping == sector].index
                    sub = signal.reindex(tickers_sector).dropna()
                    if sub.empty:
                        continue
                    cutoff = sub.quantile(threshold)
                    selected.extend(sub[sub >= cutoff].index.tolist())
        else:
            raise ValueError(f"Mode de sélection inconnu : {selection_mode}")

        # Fallback : si trop peu de titres, prendre les min_stocks meilleurs
        if len(selected) < min_stocks and len(signal) >= min_stocks:
            selected = signal.nlargest(min_stocks).index.tolist()

        return list(set(selected))

    # ------------------------------------------------------------------
    # Calcul des signaux bruts
    # ------------------------------------------------------------------

    def _momentum_12_1(
        self, rebal_date: pd.Timestamp, tickers: List[str]
    ) -> pd.Series:
        """
        Momentum(12-1) : P(t-1) / P(t-12) - 1
        t-1  = fin du mois précédant la date de rebalancement
        t-12 = fin du mois il y a 12 mois depuis rebal_date (soit t-13 mois avant t)
        """
        t1 = rebal_date - pd.DateOffset(months=self._LOOKBACK_SKIP)
        t12 = rebal_date - pd.DateOffset(months=self._LOOKBACK_12_1_TOTAL)

        p1 = self._price_at_month_end(t1, tickers)
        p12 = self._price_at_month_end(t12, tickers)

        result: Dict[str, float] = {}
        for t in tickers:
            v1, v12 = p1.get(t), p12.get(t)
            if v1 is not None and v12 is not None and v12 > 1e-8:
                result[t] = v1 / v12 - 1.0
        return pd.Series(result)

    def _idiosyncratic_12_1(
        self, rebal_date: pd.Timestamp, tickers: List[str]
    ) -> pd.Series:
        """
        Momentum idiosyncratique (12-1) :
          1. Calcule les rendements mensuels des actions sur [t-12, t-1]
          2. Régression OLS de chaque action sur le benchmark
          3. Accumulation des résidus → signal idiosyncratique
        """
        t_start = rebal_date - pd.DateOffset(months=self._LOOKBACK_12_1_TOTAL)
        t_end = rebal_date - pd.DateOffset(months=self._LOOKBACK_SKIP)

        stock_ret = self._monthly_returns(t_start, t_end, tickers)
        bench_ret = self._benchmark_monthly_returns(t_start, t_end)
        return self._idiosyncratic_from_regression(stock_ret, bench_ret)

    def _momentum_5y(
        self,
        rebal_date: pd.Timestamp,
        tickers: List[str],
        mean_reverting: bool = True,
    ) -> pd.Series:
        """
        Momentum 5 ans : P(t-1) / P(t-60) - 1
        Si mean_reverting=True : signe inversé (contrarian long terme).
        La littérature académique montre que le momentum 3-5 ans tend à se retourner.
        """
        t1 = rebal_date - pd.DateOffset(months=self._LOOKBACK_SKIP)
        t60 = rebal_date - pd.DateOffset(months=self._LOOKBACK_5Y_MONTHS)

        p1 = self._price_at_month_end(t1, tickers)
        p60 = self._price_at_month_end(t60, tickers)

        result: Dict[str, float] = {}
        for t in tickers:
            v1, v60 = p1.get(t), p60.get(t)
            if v1 is not None and v60 is not None and v60 > 1e-8:
                ret = v1 / v60 - 1.0
                result[t] = -ret if mean_reverting else ret
        return pd.Series(result)

    def _idiosyncratic_5y(
        self,
        rebal_date: pd.Timestamp,
        tickers: List[str],
        mean_reverting: bool = True,
    ) -> pd.Series:
        """
        Momentum idiosyncratique 5 ans :
        Régression OLS sur benchmark sur la fenêtre [t-60, t-1].
        Si mean_reverting=True : signe inversé.
        """
        t_start = rebal_date - pd.DateOffset(months=self._LOOKBACK_5Y_MONTHS)
        t_end = rebal_date - pd.DateOffset(months=self._LOOKBACK_SKIP)

        stock_ret = self._monthly_returns(t_start, t_end, tickers)
        bench_ret = self._benchmark_monthly_returns(t_start, t_end)
        signal = self._idiosyncratic_from_regression(stock_ret, bench_ret)

        if mean_reverting:
            signal = -signal
        return signal

    # ------------------------------------------------------------------
    # Helpers : données
    # ------------------------------------------------------------------

    def _price_at_month_end(
        self, target: pd.Timestamp, tickers: List[str]
    ) -> Dict[str, float]:
        """
        Prix de clôture au dernier jour de bourse ≤ target.
        Utilise les prix journaliers pour ne pas manquer de données.
        """
        target = pd.Timestamp(target)
        available = self._prices.index[self._prices.index <= target]
        if available.empty:
            return {}
        row = self._prices.loc[available[-1]]
        return {
            t: float(row[t])
            for t in tickers
            if t in row.index and pd.notna(row[t]) and float(row[t]) > 0
        }

    def _monthly_returns(
        self, start: pd.Timestamp, end: pd.Timestamp, tickers: List[str]
    ) -> pd.DataFrame:
        """Rendements mensuels simples sur [start, end]."""
        valid = [t for t in tickers if t in self.monthly_prices.columns]
        mp = self.monthly_prices
        sub = mp.loc[
            (mp.index >= start) & (mp.index <= end), valid
        ]
        if sub.shape[0] < 2:
            return pd.DataFrame()
        return sub.pct_change().iloc[1:].dropna(how="all")

    def _benchmark_monthly_returns(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """Rendements mensuels du benchmark sur [start, end]."""
        b = self._benchmark_returns
        return b[(b.index >= start) & (b.index <= end)]

    # ------------------------------------------------------------------
    # Helpers : régression idiosyncratique
    # ------------------------------------------------------------------

    def _idiosyncratic_from_regression(
        self,
        stock_monthly_ret: pd.DataFrame,
        bench_monthly_ret: pd.Series,
    ) -> pd.Series:
        """
        Pour chaque ticker :
          1. Aligne les rendements mensuels avec le benchmark
          2. OLS : r_stock = alpha + beta * r_bench + epsilon
          3. Accumule les résidus : produit de (1 + epsilon_t) - 1
        Retourne le signal idiosyncratique accumulé.
        """
        if stock_monthly_ret.empty or bench_monthly_ret.empty:
            return pd.Series(dtype=float)

        common_idx = stock_monthly_ret.index.intersection(bench_monthly_ret.index)
        if len(common_idx) < _MIN_MONTHS_REQUIRED:
            return pd.Series(dtype=float)

        x_full = bench_monthly_ret.loc[common_idx].values
        aligned = stock_monthly_ret.loc[common_idx]

        result: Dict[str, float] = {}
        for ticker in aligned.columns:
            col = aligned[ticker]
            mask = col.notna()
            if mask.sum() < _MIN_MONTHS_REQUIRED:
                continue
            y = col[mask].values
            x = x_full[mask.values]
            try:
                slope, intercept, _, _, _ = stats.linregress(x, y)
                residuals = y - (intercept + slope * x)
                # Signal = rendement idiosyncratique cumulé
                result[ticker] = float((1.0 + pd.Series(residuals)).prod() - 1.0)
            except Exception:
                continue

        return pd.Series(result)

    # ------------------------------------------------------------------
    # Helpers : standardisation
    # ------------------------------------------------------------------

    def _standardize_within_sectors(
        self, signal: pd.Series, sector_mapping: pd.Series
    ) -> pd.Series:
        """
        Z-score intra-sectoriel.
        Les tickers sans secteur reçoivent un z-score global.
        """
        out = signal.copy().astype(float)
        sectors_in_signal = sector_mapping.reindex(signal.index).dropna().unique()

        for sector in sectors_in_signal:
            tickers_in_sector = sector_mapping[sector_mapping == sector].index
            sub = signal.reindex(tickers_in_sector).dropna()
            if len(sub) < 2:
                continue
            std = sub.std()
            if std < 1e-10:
                continue
            out.loc[sub.index] = (sub - sub.mean()) / std

        # Tickers sans secteur : z-score global
        no_sector = signal.index.difference(sector_mapping.reindex(signal.index).dropna().index)
        if len(no_sector) > 0:
            sub = signal.reindex(no_sector).dropna()
            if len(sub) >= 2:
                std = sub.std()
                if std >= 1e-10:
                    out.loc[sub.index] = (sub - sub.mean()) / std

        return out

    def _zscore(self, signal: pd.Series) -> pd.Series:
        """Z-score en cross-section globale."""
        s = signal.dropna()
        if len(s) < 2:
            return signal.astype(float)
        std = s.std()
        if std < 1e-10:
            return signal.astype(float)
        result = (s - s.mean()) / std
        return result.reindex(signal.index).astype(float)
