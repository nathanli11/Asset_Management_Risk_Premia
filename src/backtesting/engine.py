"""
engine.py
---------
Moteur de backtesting pour la stratégie momentum long-only sur le MSCI World.

Architecture :
  BacktestConfig   : paramètres de la stratégie (signal, allocation, sélection, etc.)
  BacktestResult   : résultats du backtest (NAV, poids, coûts, rendements journaliers)
  BacktestEngine   : orchestre l'ensemble du backtesting

Garanties :
  - Pas de look-ahead bias : signaux calculés uniquement avec données ≤ date de rebalancement
  - Pas de survivorship bias : univers reconstruit dynamiquement à chaque rebalancement
  - Coûts de transaction paramétrables (défaut : 5 bps sur le turnover)
  - Rebalancement mensuel ou trimestriel
  - Contrainte UCITS 5/10/40 via optimisation sous contrainte (cf. allocation.py)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_loader import DataLoader, RF_ANNUAL
from .signals import (
    SignalCalculator,
    SignalType,
    SelectionMode,
    QuantileMode,
    SIGNAL_LABELS,
    SELECTION_LABELS,
    QUANTILE_LABELS,
)
from .allocation import AllocationEngine, AllocationMethod, METHOD_LABELS

logger = logging.getLogger(__name__)

_STORAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "storage"


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """
    Configuration complète d'une stratégie de backtesting.

    Paramètres
    ----------
    signal_type       : type de signal momentum (cf. SignalType)
    allocation_method : méthode de pondération (cf. AllocationMethod)
    selection_mode    : best_in_universe ou best_in_class (cf. SelectionMode)
    quantile_mode     : granularité de sélection (décile / quintile / quartile)
    rebalancing_freq  : "monthly" ou "quarterly"
    apply_ucits       : appliquer la contrainte UCITS 5/10/40
    transaction_cost_bps : frais de transaction en points de base (défaut : 5 bps)
    initial_capital   : capital initial en euros (défaut : 1 000 000 €)
    start_date        : date de début du backtesting
    end_date          : date de fin du backtesting
    cov_lookback_days : fenêtre de lookback pour la covariance (ERC, MinVar)
    """
    signal_type: SignalType = SignalType.MOMENTUM_12_1
    allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY
    selection_mode: SelectionMode = SelectionMode.BEST_IN_UNIVERSE
    quantile_mode: QuantileMode = QuantileMode.QUINTILE
    rebalancing_freq: str = "monthly"
    apply_ucits: bool = True
    transaction_cost_bps: float = 5.0
    initial_capital: float = 1_000_000.0
    start_date: str = "2007-04-20"
    end_date: str = "2026-01-30"
    cov_lookback_days: int = 252

    def strategy_name(self) -> str:
        """Nom unique et auto-descriptif de la stratégie (convention de nommage scalable)."""
        signal_lbl = SIGNAL_LABELS[self.signal_type]
        method_lbl = METHOD_LABELS[self.allocation_method]
        sel_lbl = SELECTION_LABELS[self.selection_mode]
        quant_lbl = QUANTILE_LABELS[self.quantile_mode]
        ucits_lbl = "UCITS" if self.apply_ucits else "NoUCITS"
        freq_lbl = "M" if self.rebalancing_freq == "monthly" else "Q"
        return f"{signal_lbl}_{method_lbl}_{sel_lbl}_{quant_lbl}_{freq_lbl}_{ucits_lbl}"


@dataclass
class BacktestResult:
    """
    Résultats complets d'un backtesting.

    Compatible avec l'interface ReportingEngine.

    Attributs
    ---------
    method              : str (ex. "ERC") — utilisé par ReportingEngine pour les labels
    signal              : str (ex. "Mom12_1")
    strategy_label      : str (nom complet de la stratégie)
    daily_returns       : pd.Series (rendements journaliers du portefeuille)
    nav                 : pd.Series (valeur liquidative quotidienne)
    weights             : dict {date: pd.Series} — poids à chaque rebalancement
    transaction_costs_eur : dict {date: float} — coûts de transaction en €
    initial_capital     : float
    config              : BacktestConfig
    rebalancing_dates   : liste des dates de rebalancement effectuées
    """
    method: str
    signal: str
    strategy_label: str
    daily_returns: pd.Series
    nav: pd.Series
    weights: Dict[pd.Timestamp, pd.Series]
    transaction_costs_eur: Dict[pd.Timestamp, float]
    initial_capital: float
    config: BacktestConfig
    rebalancing_dates: List[pd.Timestamp] = field(default_factory=list)


# ------------------------------------------------------------------
# Moteur de backtesting
# ------------------------------------------------------------------

class BacktestEngine:
    """
    Orchestre le backtesting de la stratégie momentum long-only.

    Workflow à chaque date de rebalancement :
      1. Charger l'univers dynamique (composition MSCI World)
      2. Calculer le signal (momentum) sur l'univers
      3. Sélectionner le sous-ensemble (top décile / quintile / quartile)
      4. Calculer les poids (ERC / MinVar / SigW)
      5. Appliquer la contrainte UCITS 5/10/40
      6. Calculer le turnover et les coûts de transaction
      7. Calculer les rendements journaliers entre deux rebalancements

    Paramètres
    ----------
    config      : BacktestConfig
    data_loader : DataLoader
    """

    def __init__(self, config: BacktestConfig, data_loader: DataLoader):
        self.config = config
        self.loader = data_loader
        self._alloc_engine = AllocationEngine(
            cov_lookback_days=config.cov_lookback_days,
            min_obs=min(60, config.cov_lookback_days // 4),
        )

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """
        Lance le backtesting complet et retourne un BacktestResult.
        """
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)

        logger.info(
            "Démarrage backtest : %s | %s → %s",
            self.config.strategy_name(), start.date(), end.date(),
        )

        rebal_dates = self._get_rebalancing_dates(start, end)
        if not rebal_dates:
            raise ValueError("Aucune date de rebalancement trouvée dans la période.")

        # Chargement des données de prix et du benchmark (une seule fois)
        prices = self.loader.get_prices()
        benchmark_returns = self.loader.get_benchmark_returns()

        # Initialisation du calculateur de signaux
        signal_calc = SignalCalculator(prices, benchmark_returns)

        weights_dict: Dict[pd.Timestamp, pd.Series] = {}
        tc_eur_dict: Dict[pd.Timestamp, float] = {}
        prev_weights: Optional[pd.Series] = None
        current_nav = self.config.initial_capital

        for rebal_date in rebal_dates:
            logger.debug("Rebalancement : %s", rebal_date.date())

            # 1. Univers dynamique
            universe = self.loader.get_universe(rebal_date)
            if not universe:
                logger.warning("Univers vide à %s, skip", rebal_date.date())
                continue

            # 2. Mapping sectoriel
            sector_mapping = self.loader.get_sector_mapping(rebal_date)

            # 3. Signal
            signal = signal_calc.compute(
                self.config.signal_type,
                rebal_date,
                universe,
                sector_mapping,
            )

            # 4. Sélection des titres
            selected = signal_calc.select_universe(
                signal,
                self.config.quantile_mode,
                self.config.selection_mode,
                sector_mapping,
            )
            if not selected:
                logger.warning("Aucun titre sélectionné à %s, skip", rebal_date.date())
                continue

            # 5. Rendements historiques pour ERC / MinVar
            returns_for_alloc = self._get_returns_for_allocation(
                rebal_date, selected, prices
            )

            # 6. Allocation
            signal_scores = signal.reindex(selected) if signal is not None else None
            new_weights = self._alloc_engine.compute(
                method=self.config.allocation_method,
                tickers=selected,
                returns=returns_for_alloc,
                signal_scores=signal_scores,
                apply_ucits=self.config.apply_ucits,
            )

            # 7. Calcul du turnover et des coûts de transaction
            drift_weights = self._compute_drift_weights(
                prev_weights, prices, rebal_date
            )
            turnover = self._compute_turnover(new_weights, drift_weights)
            tc_fraction = turnover * self.config.transaction_cost_bps / 10_000.0
            tc_eur = tc_fraction * current_nav
            tc_eur_dict[rebal_date] = tc_eur

            # Mise à jour de la NAV après déduction des frais
            current_nav *= 1.0 - tc_fraction

            # Stockage des poids
            weights_dict[rebal_date] = new_weights
            prev_weights = new_weights

        if not weights_dict:
            raise ValueError("Aucun poids calculé : vérifier les données et la configuration.")

        # 8. Calcul des rendements journaliers et de la NAV
        daily_returns, nav_series = self._compute_nav_series(
            weights_dict, prices, self.config.initial_capital, tc_eur_dict, start, end
        )

        strategy_label = self.config.strategy_name()
        result = BacktestResult(
            method=METHOD_LABELS[self.config.allocation_method],
            signal=SIGNAL_LABELS[self.config.signal_type],
            strategy_label=strategy_label,
            daily_returns=daily_returns,
            nav=nav_series,
            weights=weights_dict,
            transaction_costs_eur=tc_eur_dict,
            initial_capital=self.config.initial_capital,
            config=self.config,
            rebalancing_dates=list(weights_dict.keys()),
        )

        logger.info(
            "Backtest terminé : %d rebalancements, NAV finale = %.0f €",
            len(weights_dict),
            nav_series.iloc[-1] if not nav_series.empty else 0.0,
        )
        return result

    # ------------------------------------------------------------------
    # Dates de rebalancement
    # ------------------------------------------------------------------

    def _get_rebalancing_dates(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> List[pd.Timestamp]:
        """
        Retourne les dates de rebalancement dans [start, end].
        - monthly  : toutes les dates disponibles dans les fichiers de composition
        - quarterly: uniquement les fins de trimestre (mars, juin, sept, déc)
        """
        all_dates = self.loader.get_all_rebalancing_dates()
        filtered = [d for d in all_dates if start <= d <= end]

        if self.config.rebalancing_freq == "quarterly":
            # Conserver uniquement les fins de trimestre (mois 3, 6, 9, 12)
            filtered = [d for d in filtered if d.month in (3, 6, 9, 12)]

        return filtered

    # ------------------------------------------------------------------
    # Calcul des rendements pour l'allocation
    # ------------------------------------------------------------------

    def _get_returns_for_allocation(
        self,
        rebal_date: pd.Timestamp,
        tickers: List[str],
        prices: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les rendements journaliers sur la fenêtre de lookback
        pour l'estimation de la covariance (utilisée par ERC et MinVar).
        Pas de look-ahead : uniquement des données ≤ rebal_date.
        """
        if self.config.allocation_method not in (
            AllocationMethod.RISK_PARITY,
            AllocationMethod.MIN_VARIANCE,
        ):
            return None

        lookback_start = rebal_date - pd.DateOffset(
            days=int(self.config.cov_lookback_days * 1.5)
        )
        available = [t for t in tickers if t in prices.columns]
        if not available:
            return None

        sub = prices.loc[
            (prices.index >= lookback_start) & (prices.index <= rebal_date),
            available
        ]
        if len(sub) < 2:
            return None

        return sub.pct_change().iloc[1:].fillna(0.0)

    # ------------------------------------------------------------------
    # Turnover et coûts de transaction
    # ------------------------------------------------------------------

    def _compute_drift_weights(
        self,
        prev_weights: Optional[pd.Series],
        prices: pd.DataFrame,
        rebal_date: pd.Timestamp,
    ) -> Optional[pd.Series]:
        """
        Calcule les poids « driftés » du portefeuille juste avant le rebalancement
        (poids après variation des prix depuis le dernier rebalancement, avant rebalancement).
        """
        if prev_weights is None or prev_weights.empty:
            return None

        tickers = prev_weights.index.tolist()

        # Prix à la date de rebalancement précédente et à la date courante
        rebal_dates_all = self.loader.get_all_rebalancing_dates()
        prev_rebal_dates = [d for d in rebal_dates_all if d < rebal_date]
        if not prev_rebal_dates:
            return prev_weights

        prev_rebal = prev_rebal_dates[-1]

        p_prev = pd.Series(self.loader.get_price_at(prev_rebal, tickers))
        p_curr = pd.Series(self.loader.get_price_at(rebal_date, tickers))

        valid = [t for t in tickers if pd.notna(p_prev.get(t)) and pd.notna(p_curr.get(t))
                 and float(p_prev.get(t, 0)) > 0]

        if not valid:
            return prev_weights

        ret = pd.Series({t: float(p_curr[t]) / float(p_prev[t]) - 1.0 for t in valid})
        # Drift des poids : proportionnel à (1 + rendement)
        w_drift = prev_weights.reindex(valid).fillna(0.0) * (1.0 + ret)
        total = w_drift.sum()
        if total > 1e-8:
            w_drift = w_drift / total
        return w_drift

    @staticmethod
    def _compute_turnover(
        new_weights: pd.Series,
        drift_weights: Optional[pd.Series],
    ) -> float:
        """
        Turnover = Σ|w_new_i - w_drift_i| / 2
        Si pas de poids précédents : turnover = 100% (premier investissement).
        """
        if drift_weights is None or drift_weights.empty:
            return 1.0  # Premier investissement : 100% de turnover

        all_tickers = new_weights.index.union(drift_weights.index)
        w_new = new_weights.reindex(all_tickers).fillna(0.0)
        w_drift = drift_weights.reindex(all_tickers).fillna(0.0)
        return float((w_new - w_drift).abs().sum() / 2.0)

    # ------------------------------------------------------------------
    # Calcul de la NAV journalière
    # ------------------------------------------------------------------

    def _compute_nav_series(
        self,
        weights_dict: Dict[pd.Timestamp, pd.Series],
        prices: pd.DataFrame,
        initial_capital: float,
        tc_eur_dict: Dict[pd.Timestamp, float],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        """
        Calcule la série de rendements journaliers et la NAV quotidienne.

        Entre deux dates de rebalancement, les poids restent constants.
        Le rendement journalier du portefeuille est :
          r_t = Σ w_i * r_i,t
        où w_i sont les poids du dernier rebalancement.

        La NAV intègre les coûts de transaction appliqués à chaque rebalancement.
        """
        rebal_dates_sorted = sorted(weights_dict.keys())
        trading_days = prices.index[
            (prices.index >= start) & (prices.index <= end)
        ]

        if trading_days.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Construire les rendements journaliers par ticker
        # Pour éviter de tout charger en mémoire, on traite par période
        nav_values: Dict[pd.Timestamp, float] = {}
        daily_ret_values: Dict[pd.Timestamp, float] = {}

        nav = initial_capital
        current_weights: Optional[pd.Series] = None

        # Index des dates de rebalancement pour recherche rapide
        rebal_set = set(rebal_dates_sorted)

        # Précalcul des rendements journaliers (pour tous les tickers nécessaires)
        all_tickers_needed = list(
            set(t for w in weights_dict.values() for t in w.index)
        )
        available_tickers = [t for t in all_tickers_needed if t in prices.columns]

        # Récupère les prix pour la plage complète
        # On calcule les rendements journaliers une seule fois
        prices_sub = prices.loc[
            (prices.index >= start - pd.DateOffset(days=5)) & (prices.index <= end),
            available_tickers,
        ]
        daily_returns_all = prices_sub.pct_change()

        # Remplacer les NaN par 0 uniquement pour le calcul du rendement de portefeuille
        daily_returns_all = daily_returns_all.fillna(0.0)

        prev_day: Optional[pd.Timestamp] = None

        for day in trading_days:
            # Mise à jour des poids si on est à une date de rebalancement
            if day in rebal_set and day in weights_dict:
                # Appliquer les coûts de transaction sur la NAV avant de mettre à jour
                tc = tc_eur_dict.get(day, 0.0)
                if tc > 0:
                    nav -= tc
                current_weights = weights_dict[day]

            if current_weights is None:
                # Avant le premier rebalancement : portefeuille non investi
                daily_ret_values[day] = 0.0
                nav_values[day] = nav
                prev_day = day
                continue

            # Rendement journalier du portefeuille
            if day in daily_returns_all.index:
                day_ret = daily_returns_all.loc[day]
                # Aligner les tickers du portefeuille avec les rendements disponibles
                tickers_in_w = current_weights.index.tolist()
                w_aligned = current_weights.reindex(
                    [t for t in tickers_in_w if t in day_ret.index]
                ).fillna(0.0)
                r_aligned = day_ret.reindex(w_aligned.index).fillna(0.0)
                port_ret = float((w_aligned * r_aligned).sum())
            else:
                port_ret = 0.0

            daily_ret_values[day] = port_ret
            nav *= 1.0 + port_ret
            nav_values[day] = nav
            prev_day = day

        daily_returns_series = pd.Series(daily_ret_values, name="portfolio_return")
        nav_series = pd.Series(nav_values, name="NAV")

        return daily_returns_series, nav_series

    # ------------------------------------------------------------------
    # Export des résultats
    # ------------------------------------------------------------------

    def export_results(self, result: BacktestResult) -> Path:
        """
        Crée le sous-dossier de stockage pour la stratégie et
        exporte la composition enrichie (CSV) ainsi que les poids (CSV BBU).

        Convention de nommage :
          {signal}_{method}_{selection}_{quantile}_{freq}_{ucits}_{start}_{end}/

        Retourne le chemin du dossier créé.
        """
        rebal_dates = sorted(result.weights.keys())
        start_str = rebal_dates[0].strftime("%Y%m%d") if rebal_dates else "NA"
        end_str = rebal_dates[-1].strftime("%Y%m%d") if rebal_dates else "NA"

        folder_name = (
            f"{result.strategy_label}"
            f"_{start_str}"
            f"_{end_str}"
        )
        out_dir = _STORAGE_ROOT / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- CSV BBU : Date, Ticker, Weights ---
        bbu_rows = []
        for date, weights in result.weights.items():
            for ticker, w in weights.items():
                bbu_rows.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Weights": round(float(w), 6),
                })
        bbu_df = pd.DataFrame(bbu_rows)[["Date", "Ticker", "Weights"]]
        bbu_path = out_dir / f"bbu_{result.strategy_label}.csv"
        bbu_df.to_csv(bbu_path, index=False)
        logger.info("BBU CSV exporté : %s", bbu_path)

        # --- CSV composition enrichie ---
        comp_rows = []
        info = self.loader.informations
        for date, weights in sorted(result.weights.items()):
            prices_at_date = self.loader.get_price_at(date, weights.index.tolist())
            for ticker, w in weights.items():
                row: dict = {
                    "Date": date.strftime("%Y-%m-%d"),
                    "Ticker": ticker,
                    "Weight": round(float(w), 6),
                    "Price": round(float(prices_at_date.get(ticker, float("nan"))), 4),
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
                comp_rows.append(row)

        comp_df = pd.DataFrame(comp_rows)
        col_order = [
            "Date", "Ticker", "Name", "Country", "Currency",
            "Sector", "Industry", "Price", "Weight",
        ]
        comp_df = comp_df[[c for c in col_order if c in comp_df.columns]]
        comp_path = out_dir / f"composition_{result.strategy_label}.csv"
        comp_df.to_csv(comp_path, index=False)
        logger.info("Composition enrichie exportée : %s", comp_path)

        return out_dir
