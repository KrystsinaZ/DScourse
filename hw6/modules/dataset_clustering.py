from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from typing import Any, NamedTuple

class Result(NamedTuple):
    x_scaled: np.ndarray       # Чыстая матрыца для K-Means
    x_raw: pd.DataFrame        # Лікавыя прыкметы да маштабавання
    metadata: pd.DataFrame     # Тэкставыя палі (ID, назвы песень, выканаўцы)
    feature_names_out: tuple[str, ...] # Выходныя назвы прыкмет (пасля OHE)

class PrepareClusteringDataset:
    
    def __init__(self, data_dir: str | Path, random_state: int = 42) -> None:
        # 1. Захоўваем шлях да папкі (гарантуем, што гэта Path)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Захоўваем імя файла і поўны шлях да яго
        self.file_name = None
        self.file_path = None
        
        # 3. Нарыхтоўкі для даных (запоўняцца пазней)
        self.df = None
        self.random_state = random_state
           
        # 4. Ствараем дынамічныя атрыбуты для кожнага экзэмпляра класа
        self.continuous_cols: list[str] = []
        self.binary_cols: list[str] = []
        self.categorical_cols: list[str] = [] # Тэкставыя катэгорыі (калі ёсць для OHE)
        self.metadata_cols: list[str] = []    # Палі ідэнтыфікатараў (ID, track_name, artists)
                
        
    def load_csv(self, *, file_name: str) -> pd.DataFrame:
        """Загрузка даных з CSV файла, вызначанага ў __init__"""

        self.file_name = file_name
        self.file_path = self.data_dir / self.file_name
        
        # Чытаем даныя і захоўваем іх у self, каб яны былі даступныя паўсюль
        self.df = pd.read_csv(self.file_path)
        
        # print(f"[LoadDataset] {self.file_name}: {self.df.shape[0]} радкоў x {self.df.shape[1]} слупкоў")
                
        return self.df
  

    def report_missing_and_duplicates(self, frame: pd.DataFrame) -> None:
        """
        Друкуе поўную справаздачу аб прапушчаных значэннях і дублікатах у датафрэйме.
        """
        total_rows = frame.shape[0]
        total_cols = frame.shape[1]
        
        print(f"[Dataset Report] Памер дадзеных: {total_rows} радкоў, {total_cols} прыкмет.")
           
        # 1. Праверка на паўторы (дублікаты)
        duplicate_count = int(frame.duplicated().sum())
        duplicate_pct = (duplicate_count / total_rows) * 100
        print(f"Паўтаральныя радкі: {duplicate_count} ({duplicate_pct:.2f}%)")
        print("-" * 50)
        
        # 2. Праверка на прапушчаныя значэнні па кожнай калонцы
        missing_counts = frame.isna().sum()
        # Пакідаем толькі тыя калонкі, дзе ёсць хоць адзін пропуск
        missing_data = missing_counts[missing_counts > 0]
        
        if missing_data.empty:
            print("Прапушчаныя значэнні адсутнічаюць ва ўсіх калонках! ")
        else:
            print("Калонкі з прапушчанымі значэннямі:")
            print(f"{'Прыкмета':<30} | {'Колькасць':<10} | {'Адсотак':<10}")
            print("-" * 50)
            for col, count in missing_data.items():
                pct = (count / total_rows) * 100
                print(f"{col:<30} | {count:<10} | {pct:.2f}%")
                
        # 3. Аўтаматычна запаўняем атрыбуты класа пасля аналізу
        numeric_cols = frame.select_dtypes(include=['number']).columns
        
        # Часовыя спісы для збору калонак
        found_continuous: list[str] = []
        found_binary: list[str] = []
        
        for col in numeric_cols:
            unique_count = int(frame[col].nunique())
            if unique_count <= 2:
                found_binary.append(str(col))
            elif unique_count > 10: 
                found_continuous.append(str(col))
                
        # Прысвойваем сабраныя спісы ў атрыбуты аб'екта
        self.continuous_cols = found_continuous
        self.binary_cols = found_binary
        
        print(f"Вызначана BINARY_COLS:     {len(self.binary_cols)} прыкмет.")
        print(f"Вызначана CONTINUOUS_COLS: {len(self.continuous_cols)} прыкмет.")

        # Давайце яўна вызначым, што будзе з'яўляцца метададзенымі:
        # Аддзяляем тэкст, які НЕ ідзе ў навучанне (напрыклад, назва трэка ці ID)
        self.metadata_cols = list(frame.select_dtypes(include=['object', 'category']).columns)
        
        # Напрыклад, калі сярод катэгорый ёсць кароткія рэчы (кшталту explicit ці genre), 
        # іх адпраўляем у OHE, а цяжкія (track_name) — у метададзеныя:
        # Для прастаты: усе лікавыя — у мадэль, усе тэкставыя — у метададзеныя.

        
    def prepare_for_clustering(self, frame: pd.DataFrame) -> Result:
        """
        Выконвае ачыстку, трансфармацыю прыкмет (OHE + Scaling) 
        і аддзяляе метададзеныя ад навучальнага сэту.
        """
        # 1. Ачыстка: выклікаем унутранае выдаленне пропускаў ці дублікатаў, калі трэба
        df_clean = frame.drop_duplicates().dropna().reset_index(drop=True)
        
        # 2. Выдзяляем метададзеныя (ідэнтыфікатары), якія не павінны трапіць у K-Means
       
        # Бярэм слупкі для мадэлі
        features_to_model = self.continuous_cols + self.binary_cols
        x_raw = df_clean[features_to_model].copy()
        metadata = df_clean[self.metadata_cols].copy() if self.metadata_cols else pd.DataFrame(index=df_clean.index)
        
        # 3. Ствараем працэсар прыкмет (OHE + Scaling)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.continuous_cols),
                # Для track_genre і explicit выкарыстоўваем шчыльны OHE
                ("cat", OneHotEncoder(drop="first", sparse_output=False), self.categorical_cols)
            ],
            remainder="passthrough"
        )
        
        # 4. Трансфармацыя
        x_scaled = np.asarray(preprocessor.fit_transform(x_raw), dtype=np.float64)

        # 5. Збор імёнаў прыкмет на выхадзе
        if hasattr(preprocessor, "get_feature_names_out"):
            raw_names = preprocessor.get_feature_names_out()
            feature_names_out = tuple(str(name) for name in raw_names)
        else:
            feature_names_out = tuple(x_raw.columns)

        return Result(
            x_scaled=x_scaled,
            x_raw=x_raw,
            metadata=metadata,
            feature_names_out=feature_names_out
        )

        
    # def prepare_regression(
    #     self,
    #     frame: pd.DataFrame,
    #     *,
    #     target_name: str = "PageValues",
    #     drop_cols: list[str] | None = None,
    #     test_size: float = 0.2,
    #     val_size: float = 0.2,
    #     random_state: int = 42,
    #     stratify_zero_inflated: bool = True,
    # ) -> SplitResult:
    #     """
    #     Падзел 60/20/20 + CV-hold-out, што і ў PrepareDataset.prepare(),
    #     але без stratify=target (мэта бесперапынная) і з выдаленнем Revenue.
    #     """
    #     self.target_name = target_name
    #     drop_cols = self.LEAKAGE_COLS_DEFAULT if drop_cols is None else drop_cols

    #     cols_to_drop = [target_name] + [c for c in drop_cols if c in frame.columns and c != target_name]
    #     features = frame.drop(columns=cols_to_drop)
    #     target = frame[target_name]

    #     share_zero = 100.0 * (target == 0).mean()
    #     print(
    #         f"[PrepareRegressionDataset] Радкоў: {len(frame)}, таргет = {target_name}"
    #         f"\n  Нулявых значэнняў: {share_zero:.2f}%, "
    #         f"сярэдняе={target.mean():.3f}, медыяна={target.median():.3f}, "
    #         f"max={target.max():.3f}"
    #     )
    #     if drop_cols:
    #         print(f"[PrepareRegressionDataset] Выдалены з прыкмет (абарона ад уцёку): {cols_to_drop[1:]}")

    #     names = tuple(str(col) for col in features.columns)

    #     strat_labels = _zero_inflated_bins(target) if stratify_zero_inflated else None

    #     x_cv, x_test, y_cv, y_test = train_test_split(
    #         features, target,
    #         test_size=test_size,
    #         random_state=random_state,
    #         stratify=strat_labels,
    #     )

    #     relative_val = val_size / (1.0 - test_size)
    #     strat_labels_cv = _zero_inflated_bins(y_cv) if stratify_zero_inflated else None

    #     x_train, x_val, y_train, y_val = train_test_split(
    #         x_cv, y_cv,
    #         test_size=relative_val,
    #         random_state=random_state,
    #         stratify=strat_labels_cv,
    #     )

    #     x_train, x_val, x_test = x_train.copy(), x_val.copy(), x_test.copy()

    #     print(
    #         f"[PrepareRegressionDataset] Падзел static_3way (60/20/20): "
    #         f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}"
    #     )
    #     print(
    #         f"[PrepareRegressionDataset] Падзел пад CV + hold-out: "
    #         f"CV={len(y_cv)}, test={len(y_test)}"
    #     )

    #     return SplitResult(
    #         x_train=x_train, x_val=x_val, x_test=x_test,
    #         y_train=y_train, y_val=y_val, y_test=y_test,
    #         x_cv=x_cv, y_cv=y_cv,
    #         feature_names=names,
    #     )

    # def create_preprocessor_regression(self, categories_dict: dict[str, list]) -> ColumnTransformer:
    #     """
    #     Той жа ColumnTransformer, што і create_preprocessor() у бацькоўскім класе,
    #     але PageValues тут ужо не сярод прыкмет (ён — таргет).
    #     """
    #     true_numeric_cols = [
    #         "Administrative", "Administrative_Duration",
    #         "Informational", "Informational_Duration",
    #         "ProductRelated", "ProductRelated_Duration",
    #         "BounceRates", "ExitRates", "SpecialDay",
    #         # PageValues выдалены: ён цяпер таргет, а не прыкмета
    #     ]
    #     numeric_categorical_cols = ["OperatingSystems", "Browser", "Region", "TrafficType"]
    #     ohe_cols = ["VisitorType"]
    #     visitors_type = ["New_Visitor", "Other", "Returning_Visitor"]
    #     months_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    #     self.preprocessor = ColumnTransformer(
    #         transformers=[
    #             ("num", StandardScaler(), true_numeric_cols),
    #             ("cat_num_ohe", OneHotEncoder(
    #                 categories=[categories_dict[col] for col in numeric_categorical_cols],
    #                 drop="first", sparse_output=False, handle_unknown="ignore",
    #             ), numeric_categorical_cols),
    #             ("cat_str_ohe", OneHotEncoder( # "error"
    #                 categories=[visitors_type],
    #                 drop="first", sparse_output=False, handle_unknown="ignore",
    #             ), ohe_cols),
    #             ("cat_ord", OrdinalEncoder( #use_encoded_value
    #                 categories=[months_order], handle_unknown="error",
    #             ), ["Month"]),
    #         ],
    #         remainder="passthrough",  # Weekend і г.д. застаюцца як ёсць
    #     )
    #     return self.preprocessor


    # def generate_diagnostic_table(self, frame: pd.DataFrame, target_name: str = "PageValues") -> None:
    #     """
    #     Разлічвае статыстычныя паказчыкі і выводзіць тэкст табліцы ў фармаце Markdown.
    #     """
    #     # 1. Дадзеныя для аналізу
    #     target = frame[target_name]
    #     # Выключаем таргет і выцечкі (LEAKAGE_COLS_DEFAULT), пакідаем толькі прыкметы
    #     leakage_cols = getattr(self, 'LEAKAGE_COLS_DEFAULT', ["Revenue"])
    #     cols_to_drop = [target_name] + [c for c in leakage_cols if c in frame.columns]
    #     features = frame.drop(columns=cols_to_drop)
        
    #     # Спіс толькі лікавых прыкмет для skewness, kurtosis і карэляцыі
    #     numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

    #     # 2. Разлік паказчыкаў
    #     # Доля нулявых значэнняў
    #     share_zero = (target == 0).mean() * 100

    #     # Skewness і Kurtosis таргета
    #     skew = target.skew()
    #     kurt = target.kurtosis()

    #     # Максімальная карэляцыя прыкмет з таргетам
    #     correlations = features[numeric_cols].corrwith(target).abs()
    #     max_corr_col = correlations.idxmax()
    #     max_corr_val = correlations.max()

    #     # Мультыкалінеарнасць (матрыца карэляцыі прыкмет)
    #     feat_corr = features[numeric_cols].corr()
        
    #     # Атрыманне значэнняў для канкрэтных пар
    #     r_bounce_exit = feat_corr.loc["BounceRates", "ExitRates"] if "BounceRates" in feat_corr.index else 0
    #     r_product_dur = feat_corr.loc["ProductRelated", "ProductRelated_Duration"] if "ProductRelated" in feat_corr.index else 0

    #     # Кардынальнасць катэгорый
    #     card_traffic = frame["TrafficType"].nunique() if "TrafficType" in frame.columns else 0
    #     card_browser = frame["Browser"].nunique() if "Browser" in frame.columns else 0

    #     # 3. Вывад гатовай Markdown-табліцы
    #     print("| Уласцівасць | Значэнне |")
    #     print("| :--- | :--- | :--- |")
    #     print(f"| Доля нулявых {target_name} | {share_zero:.2f} % |")
    #     print(f"| Skewness / Kurtosis | {skew:.2f} / {kurt:.1f} |")
    #     print(f"| Max корреляцыя з прыкметамі | {max_corr_val:.2f} ({max_corr_col}) |")
    #     print(f"| BounceRates ↔ ExitRates | r = {r_bounce_exit:.2f} |")
    #     print(f"| ProductRelated ↔ ProductRelated_Duration | r = {r_product_dur:.2f} |")
    #     print(f"| Кардынальнасць TrafficType / Browser | {card_traffic} / {card_browser} катэгорый |")
