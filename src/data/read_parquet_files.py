import pandas as pd
from pathlib import Path


class ParquetLoader:
    def __init__(self):
        # Chemin du fichier courant
        self.current_file = Path(__file__).resolve()

        # Racine du projet
        self.project_path_initial = self.current_file.parents[2]

        # Dossier data
        self.data_path = self.project_path_initial / "data"

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """
        Charger un fichier parquet à partir du dossier data
        """
        file_path = self.data_path / filename

        df = pd.read_parquet(file_path)

        print(f"✓ {filename} chargé avec succès")

        return df


if __name__ == "__main__":
    loader = ParquetLoader()

    # Exemple de lecture
    #df_prices = loader.load_parquet("historical_prices_components_msci_world.parquet")
    #df_weights = loader.load_parquet("historical_weights_components_msci_world_by_month.parquet")
    #df_fx = loader.load_parquet("historical_currencies_eur_components_msci_world.parquet")
    df_info = loader.load_parquet("informations_historical_components_msci_world.parquet")
    print(df_info.head())