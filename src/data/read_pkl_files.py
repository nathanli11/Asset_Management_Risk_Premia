import pandas as pd
from pathlib import Path

"""
## Objectif : Ce script lit un fichier .pkl contenant des données de prix, le convertit en DataFrame, puis l'enregistre au format CSV
"""
class PricePklReader:
    def __init__(self):
        # Chemin vers le fichier courant (lecture_pkl.py)
        self.current_file = Path(__file__).resolve()

        # Racine du projet (AM_Arthur_Le_Net_M2272)
        self.project_path_initial = self.current_file.parents[2]

        # Chemins des fichiers
        self.input_path = self.project_path_initial / "data" / "prices.pkl"
        self.output_path = self.project_path_initial / "data" / "historical_weights_components_msci_world_by_month.csv"

    def convert(self):
        df = pd.read_pickle(self.input_path)
        df.to_csv(self.output_path, index=False)
        # print(f"Fichier converti : {self.output_path}")

if __name__ == "__main__":
    prices_date = PricePklReader()
    prices_date.convert()
