import pandas as pd
from pathlib import Path


class ParquetConverter:
    def __init__(self):
        self.current_file = Path(__file__).resolve()
        self.project_path_initial = self.current_file.parents[2]

        self.csv_files = [
            self.project_path_initial / "data" / "historical_prices_components_msci_world.csv",
            self.project_path_initial / "data" / "historical_weights_components_msci_world_by_month.csv",
            self.project_path_initial / "data" / "historical_currencies_eur_components_msci_world.csv",
        ]

        self.xlsx_files = [
            self.project_path_initial / "data" / "informations_historical_components_msci_world.xlsx",
            self.project_path_initial / "data" / "bloomberg_benchmark.xlsx",
        ]

    def convert_csv_to_parquet(self):
        for csv_file in self.csv_files:
            df = pd.read_csv(csv_file)
            parquet_file = csv_file.with_suffix(".parquet")
            df.to_parquet(parquet_file, index=False)
            print(f"✓ {csv_file.name} → {parquet_file.name}")

    def convert_xlsx_to_parquet(self):
        for xlsx_file in self.xlsx_files:
            df = pd.read_excel(xlsx_file, engine="openpyxl")
            parquet_file = xlsx_file.with_suffix(".parquet")
            df.to_parquet(parquet_file, index=False)
            print(f"✓ {xlsx_file.name} → {parquet_file.name}")


if __name__ == "__main__":
    converter = ParquetConverter()
    converter.convert_csv_to_parquet()
    converter.convert_xlsx_to_parquet()