import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark.benchmark import HistoricalPricesFilter, HistoricalReturnsCalculator


def main():
    project_root = Path.cwd()
    
    prices_path = project_root / 'data' / 'storage' / 'historical_prices_eur_components_msci_world.parquet'
    dates_path = project_root / 'data' / 'storage' / 'historical_weights_components_msci_world_by_month.parquet'
    
    # Création du fichier des prix historiques à chaque date de rebalancement du portefeuille
    filter_obj = HistoricalPricesFilter(prices_path, dates_path)
    
    filtered_df = filter_obj.filter_prices()
    print(f"\nDimensions du DataFrame filtré: {filtered_df.shape}")
    
    # verification que les dates filtrées correspondent bien aux dates de rebalancement
    is_valid = filter_obj.verify_dates(filtered_df)
    print(f"\nVérification des dates: {is_valid}\n")
    
    # sauvegarde du DataFrame filtré
    filter_obj.save_filtered_prices(filtered_df)
    
    # Calcul des rendements simples à partir des prix filtrés
    returns_calculator = HistoricalReturnsCalculator()
    returns_df = returns_calculator.calculate_returns()
    print(f"\nDimensions du DataFrame des retours: {returns_df.shape}")
    
    # Sauvegarde du DataFrame des retours
    returns_calculator.save_returns(returns_df)

if __name__ == "__main__":
    main()
