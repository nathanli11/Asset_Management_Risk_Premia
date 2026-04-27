import pandas as pd
from pathlib import Path


class ParquetSaver:
    def __init__(self, subfolder=None):
        # Chemin du fichier courant
        self.current_file = Path(__file__).resolve()

        # Racine du projet (fixe)
        self.project_path_initial = self.current_file.parents[2]

        # Base data path
        base_path = self.project_path_initial / "data"

        # Data path flexible (str ou Path)
        if subfolder:
            self.data_path = base_path / Path(subfolder)
        else:
            self.data_path = base_path

        # Création automatique du dossier
        self.data_path.mkdir(parents=True, exist_ok=True)

    def save_parquet(self, df: pd.DataFrame, filename: str):
        """
        Sauvegarde un DataFrame en parquet dans le data_path
        """
        file_path = self.data_path / filename

        df.to_parquet(file_path, index=False)

        print(f"✓ Fichier sauvegardé : {file_path}")

"""
Exemple d'utilisation : 

# Exemple 1 : data/
saver1 = ParquetSaver()

# Exemple data / storage / benchmark
saver2 = ParquetSaver(subfolder=Path("storage") / "benchmark")

# Test
df = pd.DataFrame({"A": [1, 2, 3]})

saver2.save_parquet(df, "test.parquet")

"""

