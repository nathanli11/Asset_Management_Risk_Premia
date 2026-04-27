import pandas as pd
from pathlib import Path
from .utils import get_rebalancing_dates, calculate_pct_change
from ..data.utils import ParquetSaver


class HistoricalPricesFilter:
    """Filtre les prix historiques en EUR pour les dates de rebalancing."""
    
    def __init__(self, historical_prices_path: str | Path, rebalancing_dates_path: str | Path):
        self.historical_prices_path = Path(historical_prices_path)
        self.rebalancing_dates_path = Path(rebalancing_dates_path)
        self.rebalancing_dates = get_rebalancing_dates(rebalancing_dates_path)
    
    def filter_prices(self) -> pd.DataFrame:
        """Filtre les prix historiques pour les dates de rebalancing."""
        df = pd.read_parquet(self.historical_prices_path)
        filtered_df = df[df['date'].isin(self.rebalancing_dates)].copy()
        return filtered_df
    
    def verify_dates(self, filtered_df: pd.DataFrame) -> bool:
        """Vérifie que les dates filtrées correspondent aux dates de rebalancing."""
        filtered_dates = sorted(filtered_df['date'].unique().tolist())
        
        dates_match = len(filtered_dates) == len(self.rebalancing_dates)
        dates_identical = filtered_dates == self.rebalancing_dates
        
        print(f"Nombre de dates rebalancing: {len(self.rebalancing_dates)}")
        print(f"Nombre de dates filtrées: {len(filtered_dates)}")
        print(f"Dates identiques: {dates_identical}")
        
        return dates_match and dates_identical
    
    def save_filtered_prices(self, filtered_df: pd.DataFrame):
        """Sauvegarde le DataFrame filtré en parquet."""
        saver = ParquetSaver(subfolder="storage/benchmark")
        saver.save_parquet(filtered_df, "historical_eur_prices_for_benchmark_components_msci_world.parquet")


class HistoricalReturnsCalculator:
    """Calcule les retours simples à partir des prix historiques."""
    
    def __init__(self, prices_path: str | Path = None):
        if prices_path is None:
            prices_path = Path.cwd() / 'data' / 'storage' / 'benchmark' / 'historical_eur_prices_for_benchmark_components_msci_world.parquet'
        self.prices_path = Path(prices_path)
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calcule les retours simples à partir des prix."""
        df = pd.read_parquet(self.prices_path)
        date_col = df['date']
        prices_only = df.drop(columns=['date'])
        returns_only = calculate_pct_change(prices_only)
        returns_df = returns_only.copy()
        returns_df['date'] = date_col
        return returns_df
    
    def save_returns(self, returns_df: pd.DataFrame):
        """Sauvegarde le DataFrame des retours en parquet."""
        saver = ParquetSaver(subfolder="storage/benchmark")
        saver.save_parquet(returns_df, "historical_eur_returns_for_benchmark_components_msci_world.parquet")
        


