import pandas as pd
from pathlib import Path


def get_rebalancing_dates(parquet_file_path: str | Path) -> list[str]:
    """
    Extrait les dates uniques du fichier parquet et retourne une liste triée.
    
    Args:
        parquet_file_path: Chemin du fichier parquet contenant les dates
        
    Returns:
        Liste des dates uniques triées en ordre chronologique
    """
    df = pd.read_parquet(parquet_file_path)
    rebalancing_dates = sorted(df['date'].unique().tolist())
    return rebalancing_dates


def calculate_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les pourcentages de changement (pct_change) du DataFrame.
    
    Args:
        df: DataFrame avec les prix ou valeurs
        
    Returns:
        DataFrame avec les pourcentages de changement (première ligne = NaN)
    """
    return df.pct_change()

