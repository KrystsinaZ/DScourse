import pandas as pd
from sklearn.datasets import fetch_openml

# Модуль загрузкі дадзеных

def load_from_csv(file_path: str, target_column: str = None):
    """Загрузка дадзеных з CSV файла з магчымасцю падзелу."""
    df = pd.read_csv(file_path)
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    return df

def load_from_json(file_path: str, target_column: str = None):
    # Загружаем JSON адразу як табліцу
    df = pd.read_json("data.json")
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    return df


def load_from_openml(name_or_id, target_column: str = None, version: int = 1):
    """Загрузка дадзеных з OpenML па назве або ID."""
    dataset = fetch_openml(
        name=name_or_id if isinstance(name_or_id, str) else None,
        data_id=name_or_id if isinstance(name_or_id, int) else None,
        version=version,
        as_frame=True,
        parser="auto",
    )
    df = dataset.frame

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    return df
