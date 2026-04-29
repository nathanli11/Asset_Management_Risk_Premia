"""
data_loader.py
--------------
Couche d'accès aux données pour le backtesting de la stratégie momentum long-only.

Charge et met en cache :
  - Prix EUR journaliers (tous composants historiques MSCI World)
  - Compositions mensuelles dynamiques (1 fichier par date de rebalancement)
  - Rendements mensuels du benchmark MSCI World EUR
  - Métadonnées des tickers (nom, secteur, pays, devise, industrie)

Interface compatible avec ReportingEngine :
  - loader.informations                         → pd.DataFrame indexé par ticker
  - loader.get_price_at(date, tickers)          → dict {ticker: prix}
  - loader.get_daily_returns(start, end, tickers) → pd.DataFrame
  - loader.get_ester_returns_series(start, end) → pd.Series (MSCI World daily returns)
  - loader.get_benchmark_daily_returns_series(start, end) → pd.Series (MSCI World)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_ROOT = _ROOT / "data"
_STORAGE_ROOT = _DATA_ROOT / "storage"

RF_ANNUAL: float = 0.0103   # taux sans risque fixe : 1,03 % par an
RF_DAILY: float = RF_ANNUAL / 252


class DataLoader:
    """
    Centralise le chargement et la mise en cache de toutes les données
    nécessaires au backtesting et au reporting.

    Paramètres
    ----------
    prices_path      : chemin vers le fichier de prix EUR journaliers
    benchmark_path   : chemin vers le fichier des rendements MSCI World mensuels
    components_dir   : dossier contenant les fichiers de composition mensuelle
    info_path        : chemin vers le fichier d'informations descriptives
    weights_path     : chemin vers le fichier des poids mensuels MSCI World
    """

    def __init__(
        self,
        prices_path: Optional[Path] = None,
        benchmark_path: Optional[Path] = None,
        components_dir: Optional[Path] = None,
        info_path: Optional[Path] = None,
        weights_path: Optional[Path] = None,
    ):
        self._prices_path = prices_path or (
            _STORAGE_ROOT / "historical_prices_eur_components_msci_world.parquet"
        )
        self._benchmark_path = benchmark_path or (
            _STORAGE_ROOT / "benchmark" / "historical_benchmark_returns_msci_world.parquet"
        )
        self._components_dir = components_dir or (
            _STORAGE_ROOT / "monthly_components_msci_world"
        )
        self._info_path = info_path or (
            _DATA_ROOT / "informations_historical_components_msci_world.parquet"
        )
        self._weights_path = weights_path or (
            _STORAGE_ROOT / "historical_weights_components_msci_world_by_month.parquet"
        )

        # Caches internes
        self._prices_cache: Optional[pd.DataFrame] = None
        self._monthly_prices_cache: Optional[pd.DataFrame] = None
        self._benchmark_cache: Optional[pd.Series] = None
        self._components_cache: Dict[pd.Timestamp, List[str]] = {}
        self._component_df_cache: Dict[pd.Timestamp, Optional[pd.DataFrame]] = {}
        self._info_cache: Optional[pd.DataFrame] = None
        self._rebal_dates_cache: Optional[List[pd.Timestamp]] = None

    # ------------------------------------------------------------------
    # Propriétés publiques
    # ------------------------------------------------------------------

    @property
    def informations(self) -> pd.DataFrame:
        """Métadonnées des tickers, indexées par ticker."""
        if self._info_cache is None:
            self._info_cache = self._load_informations()
        return self._info_cache

    # ------------------------------------------------------------------
    # Méthodes publiques – données de marché
    # ------------------------------------------------------------------

    def get_prices(self) -> pd.DataFrame:
        """
        Retourne le DataFrame des prix EUR journaliers.
        Index = Date, colonnes = tickers.
        """
        if self._prices_cache is None:
            self._prices_cache = self._load_prices()
        return self._prices_cache

    def get_monthly_prices(self) -> pd.DataFrame:
        """
        Retourne les prix mensuels (dernier jour de chaque mois).
        Utilisé pour le calcul des signaux.
        """
        if self._monthly_prices_cache is None:
            self._monthly_prices_cache = self.get_prices().resample("ME").last()
        return self._monthly_prices_cache

    def get_benchmark_returns(self) -> pd.Series:
        """
        Retourne la série de rendements mensuels MSCI World EUR.
        Index = dates de rebalancement, valeurs = rendements simples.
        """
        if self._benchmark_cache is None:
            self._benchmark_cache = self._load_benchmark()
        return self._benchmark_cache

    def get_universe(self, rebal_date: pd.Timestamp) -> List[str]:
        """
        Retourne la liste des tickers présents dans le MSCI World
        à une date de rebalancement donnée.
        Pas de biais de survivorship : univers dynamique strict.
        """
        rebal_date = pd.Timestamp(rebal_date)
        if rebal_date not in self._components_cache:
            self._components_cache[rebal_date] = self._load_universe(rebal_date)
        return self._components_cache[rebal_date]

    def get_sector_mapping(self, rebal_date: pd.Timestamp) -> pd.Series:
        """
        Retourne pd.Series {ticker: sector_name} pour l'univers à rebal_date.
        """
        rebal_date = pd.Timestamp(rebal_date)
        df = self._get_component_df(rebal_date)
        if df is None or df.empty:
            return pd.Series(dtype=str)
        mapping = df.set_index("ticker")["sector_name"]
        return mapping[~mapping.index.duplicated(keep="first")]

    def get_component_info(self, rebal_date: pd.Timestamp) -> pd.DataFrame:
        """Retourne le DataFrame complet de composition pour une date donnée."""
        df = self._get_component_df(rebal_date)
        return df if df is not None else pd.DataFrame()

    def get_price_at(self, date: pd.Timestamp, tickers: List[str]) -> dict:
        """
        Retourne dict {ticker: prix} au jour de bourse le plus proche ≤ date.
        Compatible avec l'interface ReportingEngine.
        """
        prices = self.get_prices()
        date = pd.Timestamp(date)
        available = prices.index[prices.index <= date]
        if available.empty:
            return {t: np.nan for t in tickers}
        row = prices.loc[available[-1]]
        return {
            t: float(row[t]) if t in row.index and pd.notna(row[t]) else np.nan
            for t in tickers
        }

    def get_daily_returns(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        tickers: List[str],
    ) -> pd.DataFrame:
        """
        Retourne un DataFrame de rendements journaliers simples.
        Index = Date (de start à end inclus), colonnes = tickers.
        Compatible avec l'interface ReportingEngine.
        """
        prices = self.get_prices()
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        available = [t for t in tickers if t in prices.columns]
        if not available:
            return pd.DataFrame()
        # Inclure un jour avant start pour calculer le premier rendement
        pre_start = prices.index[prices.index < start]
        if pre_start.empty:
            slice_start = start
        else:
            slice_start = pre_start[-1]
        sub = prices.loc[
            (prices.index >= slice_start) & (prices.index <= end), available
        ]
        rets = sub.pct_change().iloc[1:]
        return rets[(rets.index >= start) & (rets.index <= end)]

    def get_ester_returns_series(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """
        Interface compatible ReportingEngine.
        Retourne une série de rendements journaliers MSCI World EUR,
        interpolés à partir des rendements mensuels.

        Utilisé comme BENCHMARK dans les calculs de TE, IR et corrélation.
        Pour Sharpe/Sortino, le taux sans risque fixe (RF_DAILY) est utilisé
        directement dans ReportingEngine.
        """
        return self.get_benchmark_daily_returns_series(start, end)

    def get_benchmark_daily_returns_series(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """
        Rendements journaliers MSCI World EUR par interpolation des rendements mensuels.

        Méthode : chaque journée dans un mois reçoit un rendement équivalent
        à la racine n-ième du rendement mensuel (n = nombre de jours de bourse du mois).
        """
        benchmark = self.get_benchmark_returns()
        prices = self.get_prices()
        start, end = pd.Timestamp(start), pd.Timestamp(end)

        trading_days = prices.index[(prices.index >= start) & (prices.index <= end)]
        if trading_days.empty:
            return pd.Series(dtype=float)

        daily_values: Dict[pd.Timestamp, float] = {}
        for td in trading_days:
            candidates = benchmark.index[benchmark.index <= td]
            if candidates.empty:
                daily_values[td] = 0.0
                continue
            monthly_dt = candidates[-1]
            monthly_ret = float(benchmark.loc[monthly_dt])
            # Compter les jours de bourse dans ce mois
            month_start = monthly_dt.replace(day=1)
            month_days = prices.index[
                (prices.index >= month_start) & (prices.index <= monthly_dt)
            ]
            n = max(len(month_days), 1)
            daily_values[td] = (1.0 + monthly_ret) ** (1.0 / n) - 1.0

        return pd.Series(daily_values)

    def get_all_rebalancing_dates(self) -> List[pd.Timestamp]:
        """
        Retourne toutes les dates de rebalancement disponibles
        (extraites des fichiers de composition mensuelle), triées.
        """
        if self._rebal_dates_cache is None:
            files = sorted(
                self._components_dir.glob("msci_world_components_*.parquet")
            )
            dates = []
            for f in files:
                try:
                    date_str = f.stem.replace("msci_world_components_", "")
                    dates.append(pd.Timestamp(date_str))
                except Exception:
                    continue
            self._rebal_dates_cache = sorted(dates)
        return self._rebal_dates_cache

    # ------------------------------------------------------------------
    # Loaders privés
    # ------------------------------------------------------------------

    def _load_prices(self) -> pd.DataFrame:
        if not self._prices_path.exists():
            raise FileNotFoundError(f"Fichier de prix introuvable : {self._prices_path}")
        df = pd.read_parquet(self._prices_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        logger.info("Prix chargés : %s", df.shape)
        return df

    def _load_benchmark(self) -> pd.Series:
        if not self._benchmark_path.exists():
            raise FileNotFoundError(f"Fichier benchmark introuvable : {self._benchmark_path}")
        df = pd.read_parquet(self._benchmark_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        s = df["msci_world_returns"].dropna()
        logger.info("Benchmark chargé : %d observations", len(s))
        return s

    def _load_informations(self) -> pd.DataFrame:
        path = self._info_path
        if not path.exists():
            alt = _STORAGE_ROOT / "informations_historical_components_msci_world.parquet"
            if alt.exists():
                path = alt
            else:
                logger.warning("Fichier informations introuvable")
                return pd.DataFrame()
        df = pd.read_parquet(path)
        # Normalisation des noms de colonnes (corrige la faute indsutry_name)
        rename_map = {
            "ticker": "ticker",
            "company_name": "Name",
            "sector_name": "Sector",
            "indsutry_name": "Industry",
            "currency_code": "Currency",
            "country_name": "Country",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "ticker" in df.columns:
            df = df.set_index("ticker")
        df = df[~df.index.duplicated(keep="first")]
        logger.info("Informations chargées : %d tickers", len(df))
        return df

    def _load_universe(self, rebal_date: pd.Timestamp) -> List[str]:
        df = self._get_component_df(rebal_date)
        if df is None or df.empty:
            logger.warning("Univers vide pour %s", rebal_date)
            return []
        tickers = df["ticker"].dropna().unique().tolist()
        return tickers

    def _get_component_df(self, rebal_date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Charge et met en cache le fichier de composition pour une date."""
        rebal_date = pd.Timestamp(rebal_date)
        if rebal_date in self._component_df_cache:
            return self._component_df_cache[rebal_date]

        date_str = rebal_date.strftime("%Y-%m-%d")
        path = self._components_dir / f"msci_world_components_{date_str}.parquet"

        if not path.exists():
            # Cherche la date disponible la plus proche (inférieure ou égale)
            files = sorted(self._components_dir.glob("msci_world_components_*.parquet"))
            candidates = [
                f for f in files
                if f.stem.split("_")[-1] <= date_str
            ]
            if not candidates:
                self._component_df_cache[rebal_date] = None
                return None
            path = candidates[-1]
            logger.debug(
                "Composition pour %s non trouvée ; utilisation de %s",
                date_str, path.name,
            )

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.error("Erreur lecture %s : %s", path, e)
            self._component_df_cache[rebal_date] = None
            return None

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        self._component_df_cache[rebal_date] = df
        return df
