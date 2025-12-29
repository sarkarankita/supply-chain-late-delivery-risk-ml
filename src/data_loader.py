# Data Ingestion

from pathlib import Path
import pandas as pd


class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_main_dataset(self) -> pd.DataFrame:
        file_path = self.data_dir / "DataCoSupplyChainDataset.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        return pd.read_csv(file_path, encoding="latin1")

    def load_data_dictionary(self) -> pd.DataFrame:
        file_path = self.data_dir / "DescriptionDataCoSupplyChain.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        return pd.read_csv(file_path, encoding="latin1")
    
class pre_data_Loader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_main_dataset(self) -> pd.DataFrame:
        file_path = self.data_dir / "X_preprocessed.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        return pd.read_csv(file_path, encoding="latin1")

    def load_data_dictionary(self) -> pd.DataFrame:
        file_path = self.data_dir / "X_preprocessed.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found")

        return pd.read_csv(file_path, encoding="latin1")
    

