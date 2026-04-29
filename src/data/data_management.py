import pandas as pd
from pathlib import Path

import sys
# Ajouter la racine du projet au path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.data.utils import ParquetSaver

def enrich_historical_components(df: pd.DataFrame) -> pd.DataFrame:
    df["exchange_rate"] = df["currency_code"] + "EUR Curncy"
    return df

def convert_prices_to_eur(prices_df : pd.DataFrame, info_df : pd.DataFrame, currencies_df : pd.DataFrame) -> pd.DataFrame:

    prices_eur = prices_df.copy()

    for idx, row in info_df.iterrows():
        ticker = row["ticker"]
        exchange_rate_col = row["exchange_rate"]

        if ticker in prices_eur.columns and exchange_rate_col in currencies_df.columns:
            prices_eur[ticker] = prices_df[ticker] * currencies_df[exchange_rate_col]
    
    return prices_eur

    
class MSCIWorldComponentsByDate:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_path = self.project_root / "data"
        self.storage_path = self.data_path / "storage"

    def create_components_files_by_date(self):
        # Lire les données
        weights_df = pd.read_parquet(self.data_path / "historical_weights_components_msci_world_by_month.parquet")
        info_df = pd.read_parquet(self.storage_path / "informations_historical_components_msci_world.parquet")
        
        # Convertir la colonne date en datetime
        weights_df['date'] = pd.to_datetime(weights_df['date'], format='%Y-%m-%d')

        # Créer le dossier de destination
        monthly_components_path = self.storage_path / "monthly_components_msci_world"
        monthly_components_path.mkdir(parents=True, exist_ok=True)

        # Pour chaque date
        for idx, row in weights_df.iterrows():
            date = row['date']
            # Utiliser la ligne directement pour les poids
            date_weights = row
            
            # Identifier les tickers avec poids non nul
            components_data = []
            for ticker in weights_df.columns:
                if ticker != 'date':
                    weight = date_weights[ticker]
                    if pd.notna(weight) and weight > 0:
                        components_data.append({
                            'date': date,
                            'ticker': ticker,
                            'weight_msci_world': weight
                        })
            
            if components_data:
                # Créer DataFrame pour cette date
                components_df = pd.DataFrame(components_data)
                
                # Merger avec les informations
                components_df = components_df.merge(info_df, on='ticker', how='left')
                
                # Sauvegarder
                date = pd.to_datetime(date)
                date_str = date.strftime('%Y-%m-%d')
                filename = f"msci_world_components_{date_str}.parquet"
                saver = ParquetSaver(subfolder=Path("storage") / "monthly_components_msci_world")
                saver.save_parquet(components_df, filename)
                
                print(f"✓ Fichier créé : {filename} ({len(components_df)} composants)")
