"""Загрузка датасета, split train/val/test."""
from __future__ import annotations

from typing import NamedTuple
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

#from typing import Any



class SplitResult(NamedTuple):
    """60% train, 20% val, 20% test. x_cv = train+val для CV."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    x_cv: pd.DataFrame
    y_cv: pd.Series
    feature_names: tuple[str, ...]

class PrepareDataset:
    """ Падрыхтуем дадзеныя да карыстанняя """

    def __init__(self, data_dir: str | Path) -> None:
        # 1. Захоўваем шлях да папкі (гарантуем, што гэта Path)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Захоўваем імя файла і поўны шлях да яго
        self.file_name = None
        self.file_path = None
        
        # 3. Нарыхтоўкі для даных і таргет-слупка (запоўняцца пазней)
        self.df = None
        self.target_column = None #"Revenue" # Можна адразу прапісаць дэфолтны таргет для гэтага датасэта
        self.preprocessor = None

    def load_csv(self, *, file_name: str) -> pd.DataFrame:
        """Загрузка даных з CSV файла, вызначанага ў __init__"""

        self.file_name = file_name
        self.file_path = self.data_dir / self.file_name
        
        # Чытаем даныя і захоўваем іх у self, каб яны былі даступныя паўсюль

        try:
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл не знойдзены: {self.file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Файл пусты: {self.file_path}")
        return self.df

        # self.df = pd.read_csv(self.file_path)
        
        # print(f"[LoadDataset] {self.file_name}: {self.df.shape[0]} радкоў x {self.df.shape[1]} слупкоў")
                
        # return self.df
