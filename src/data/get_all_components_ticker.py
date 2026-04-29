import pandas as pd
from pathlib import Path

"""
# Objectif : ce script sert uniquement à extraire tous les tickers présents dans le fichier parquet des prix historiques des composants du MSCI World, et à les sauvegarder dans un fichier Excel
pour interroger bloomberg afin de réccupérer les informations nécessaires à la conception du portefeuille (secteur, pays, devise, etc.)

# Chemin du fichier parquet
parquet_path = Path(
    "/Users/arthurlenet/Desktop/M2 272/coursS2/AssetManagement/Projet/AM_Arthur_Le_Net_M2272/data/historical_prices_components_msci_world.parquet"
)

# Chemin de sortie Excel
xlsx_path = parquet_path.parent / "ticker_components.xlsx"

# Charger uniquement la structure du fichier
df = pd.read_parquet(parquet_path)

# Récupérer tous les tickers depuis les noms de colonnes
tickers = list(df.columns)

# Supprimer les doublons sans changer l'ordre
tickers_unique = list(dict.fromkeys(tickers))

# Créer un DataFrame propre
tickers_df = pd.DataFrame({
    "ticker": tickers_unique
})

# Exporter en Excel
tickers_df.to_excel(xlsx_path, index=False)

print(f"Nombre de colonnes initiales : {len(tickers)}")
print(f"Nombre de tickers uniques : {len(tickers_unique)}")
print(f"Fichier Excel créé : {xlsx_path}")

"""