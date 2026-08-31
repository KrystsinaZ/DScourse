"""Загрузка датасета, split train/val/test."""
from __future__ import annotations

import pandas as pd

from typing import NamedTuple

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

#from typing import Any
import pandas as pd


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
        self.df = pd.read_csv(self.file_path)
        
        print(f"[LoadDataset] {self.file_name}: {self.df.shape[0]} радкоў x {self.df.shape[1]} слупкоў")
                
        return self.df

  
    def prepare(self,
            frame: pd.DataFrame,
            *,
            target_name: str,
            # model: str = 'cv_holdout',
            test_size: float = 0.2,
            val_size: float = 0.2,
            random_state: int = 42,
        ) -> SplitResult:

        """
            падзел даных метадамі
            'static_3way: train/val/test' = (60/20/20):
            'cv_holdout' (Крос-валідацыя + hold-out тэст)
            даныя, падзеленыя па 'cv_holdout' патрабуюцца для ацэнкі learning_curve, таму неалежна ад абранага метада падрыхтуем і такі падзел 

        Падыход 'static_3way' (train/val/test) (60/20/20):
            Пры статычным падзеле існуе высокая рызыка няўдалага разрэзу даных. У валідацыйны
        набор (20%) можа патрапіць анамальна малая колькасць мэтавых падзей (напрыклад, усяго 5%
        замест 15.5%). У выніку мадэль пакажа нізкую якасць або няўстойлівы F1/PR-AUC
        проста таму, што ёй «не пашанцавала» з канкрэтнай выпадковай выбаркай.
        
        Падыход 'cv_holdout' (Крос-валідацыя + hold-out тэст):
            Cтандарт для незбалансаваных таблічных дадзеных. Мы адразаем 20% даных у "сейф"
        (фінальны hold-out тэст для выніковай справаздачы). На астатніх 80% запускаецца
        StratifiedKFold(5). Пры гэтым кожны радок з навучальнага набору паспее пабываць і ў ролі навучання,
        і ў ролі валідацыі. Вынік усярэджваецца па 5 фолдах, што цалкам выключае фактар
        выпадковага «шанцу/немагчымасці» і гарантуе аб'ектыўную ацэнку PR-AUC.
        """
        self.target_name = target_name

        features = frame.drop(columns=[target_name])
        target = frame[target_name]     
        # падлік, калі там лікі 1 і 0 або True і False
        n_targ_1 = int((target == 1).sum()) 
        n_targ_0 = len(target) - n_targ_1
            
        # Працэнт адзінак
        share_1 = 100.0 * n_targ_1 / len(frame)
           
        print(
            f"[PrepareDataset] Колькасць радкоў у даных {len(frame)}, з іх"
            f"\n(Купля: {n_targ_1} ({self.target_name}==1), што склала ({share_1:.3f}%), "
            f"\n(Адмова: {n_targ_0} ({self.target_name}==0)"
        )

                     
        names = tuple(str(col) for col in features.columns)

        x_cv, x_test, y_cv, y_test = train_test_split(
                features,
                target,
                test_size=test_size,
                random_state=random_state,
                stratify=target,
            )
        
        
        relative_val = val_size / (1.0 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_cv,
            y_cv,
            test_size=relative_val,
            random_state=random_state,
            stratify=y_cv,
            )

        # Абарона ад зменаў першапачатковага dataFrame і супадзення спасылак 
        x_train = x_train.copy()
        x_val = x_val.copy()
        x_test = x_test.copy()

        print(
            f"[PrepareDataset] Падзел static_3way (60/20/20): "
            f"train={len(y_train)}, val={len(y_val)}, "
            f"test={len(y_test)}"
            f"\n               Размеркаванне ({self.target_name}==1) па выбарках: train = {int(y_train.sum())}, "
            f"val={int(y_val.sum())}, test={int(y_test.sum())}"
        )
        print(
            f"[PrepareDataset] Падзел пад СV + hold out: "
            f"CV={len(y_cv)}, test={len(y_test)}"
            f"\n               Размеркаванне ({self.target_name}==1) па выбарках: СV = {int(y_train.sum())+int(y_val.sum())}, "
            f"test={int(y_test.sum())}"
        )        

        # print("[PrepareDataset] Даныя з test не прымаюць удзел у CV і падборы характарыстык")
        print("[PrepareDataset] Даныя не масштабаваліся на этапе падзелу выбаркі")

        return SplitResult(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            x_cv=x_cv,
            y_cv=y_cv,
            feature_names=names,
        )

    # def create_preprocessor(self) -> ColumnTransformer:
    #     """
    #     Аўтаматычна стварае і вяртае ColumnTransformer, настроены
    #     пад спецыфіку датасэта online_shoppers_intention.
    #     """
    #     # 1. Спіс лічбавых прыкмет для маштабавання
    #     true_numeric_cols = [
    #         "Administrative", "Administrative_Duration", 
    #         "Informational", "Informational_Duration", 
    #         "ProductRelated", "ProductRelated_Duration",
    #         "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    #     ]

    #     # 2. Спіс лічбавых катэгорый (інтэрпрэтуем як катэгорыі, а не лікі)
    #     numeric_categorical_cols = ["OperatingSystems", "Browser", "Region", "TrafficType"]

    #     # 3. Спіс тэкставых катэгорый для One-Hot
    #     ohe_cols = ["VisitorType"]

    #     # 4. Спіс і парадак месяцаў для OrdinalEncoder (улічваем 'June')
    #     months_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    #     # Збіраем ColumnTransformer
    #     self.preprocessor = ColumnTransformer(
    #         transformers=[
    #             ('num', StandardScaler(), true_numeric_cols),
    #             ('cat_num_ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), numeric_categorical_cols),
    #             ('cat_str_ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ohe_cols),
    #             ('cat_ord', OrdinalEncoder(categories=[months_order]), ["Month"])
    #         ],
    #         remainder='passthrough'  # Пакідаем бінарныя прыкметы (напрыклад, Weekend) як ёсць
    #     )

    #     # print("[PrepareDataset] ColumnTransformer паспяхова створаны.")
    #     return self.preprocessor      

    def create_preprocessor(self, categories_dict: dict[str, list]) -> ColumnTransformer:
        """
        Аўтаматычна стварае і вяртае ColumnTransformer, настроены
        пад спецыфіку датасэта online_shoppers_intention з выкарыстаннем метададзеных.
        """
        # 1. Спіс лічбавых прыкмет для маштабавання
        true_numeric_cols = [
            "Administrative", "Administrative_Duration", 
            "Informational", "Informational_Duration", 
            "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay"
        ]
        # true_numeric_cols = [
        #     "Administrative", "Administrative_Duration", 
        #     "Informational", "Informational_Duration", 
        #     "ProductRelated", "Avg_Time_Per_Product",
        #     "ExitRates", "PageValues", "SpecialDay"
        # ]        


        # 2. Спіс лічбавых катэгорый (інтэрпрэтуем як катэгорыі, а не лікі)
        numeric_categorical_cols = ["OperatingSystems", "Browser", "Region", "TrafficType"]

        # 3. Спіс тэкставых катэгорый для One-Hot
        ohe_cols = ["VisitorType"] 
        visitors_type = ['New_Visitor', 'Other', 'Returning_Visitor']
        # 4. Спіс і парадак месяцаў для OrdinalEncoder (фіксаваны)
        months_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Збіраем ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                # Маштабаванне лічбаў
                ('num', StandardScaler(), true_numeric_cols),
                
                # Кадзіраванне лічбавых катэгорый (выцягваем спісы са слоўніка)
                ('cat_num_ohe', OneHotEncoder(
                    categories=[categories_dict[col] for col in numeric_categorical_cols],
                    drop='first', 
                    sparse_output=False, 
                    handle_unknown='error'
                ), numeric_categorical_cols),
                
                # Кадзіраванне тэкставых катэгорый (заварочваем спіс у [] для вытрымкі структуры)
                # ('cat_str_ohe', OneHotEncoder(
                #     categories=[categories_dict["VisitorType"]], # Бярэм спіс значэнняў па ключы-радку
                #     drop='first', 
                #     sparse_output=False, 
                #     handle_unknown='error'
                # ), ["VisitorType"]),
                
                ('cat_str_ohe', OneHotEncoder(
                    categories=[visitors_type],
                    drop='first', 
                    sparse_output=False, 
                    handle_unknown='error'
                ), ohe_cols),

                # Кадзіраванне месяцаў
                ('cat_ord', OrdinalEncoder(
                    categories=[months_order],
                    handle_unknown='error'
                ), ["Month"])
            ],
            remainder='passthrough'  # Пакідаем бінарныя прыкметы (напрыклад, Weekend) як ёсць
        )

        return self.preprocessor

    def create_preprocessor2(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Аўтаматычна стварае і вяртае ColumnTransformer, настроены
        пад спецыфіку датасэта online_shoppers_intention з яўным знаёмствам з катэгорыямі.
        """
        # 1. Спіс лічбавых прыкмет для маштабавання
        true_numeric_cols = [
            "Administrative", "Administrative_Duration", 
            "Informational", "Informational_Duration", 
            "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay"
        ]

        # 2. Спіс лічбавых катэгорый (інтэрпрэтуем як катэгорыі, а не лікі)
        numeric_categorical_cols = ["OperatingSystems", "Browser", "Region", "TrafficType"]

        # 3. Спіс тэкставых катэгорый для One-Hot
        ohe_cols = ["VisitorType"] 
        visitors_type = ['New_Visitor', 'Other', 'Returning_Visitor']

        # 4. Спіс і парадак месяцаў для OrdinalEncoder (улічваем 'June')
        months_order = ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # === ЗБІРАЕМ УНІКАЛЬНЫЯ КАТЭГОРЫІ ДА ФІТУ ===
        # === СТРОГАЕ САРТАВАННЕ КАТЭГОРЫЙ ===
        # # Для лічбавых катэгорый сартаванне СТРОГА АБАВЯЗКОВАЕ!
        # unique_num_categories = [
        #     sorted(list(df[col].dropna().unique())) for col in numeric_categorical_cols
        # ]
        # # Тэкставыя катэгорыі сартаваць не абавязкова, але для прыгажосці і парадку таксама можна
        # unique_str_categories = [
        #     sorted(list(df[col].dropna().unique())) for col in ohe_cols
        # ]
        
        # print(f"unique_num_categories = {unique_num_categories}")
        # print(f"unique_str_categories = {unique_str_categories}")

        # =======================================================================

        # Збіраем ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), true_numeric_cols),
                
                # Замяняем handle_unknown='ignore' на поўнае знаёмства
                ('cat_num_ohe', OneHotEncoder(), numeric_categorical_cols),
                
#               ('cat_num_ohe', OneHotEncoder(
#                     categories=unique_num_categories,
#                     drop='first', 
#                     sparse_output=False, 
#                     handle_unknown='error'
#                ), numeric_categorical_cols),
                
                ('cat_str_ohe', OneHotEncoder(
                    categories=[visitors_type],
                    drop='first', 
                    sparse_output=False, 
                    handle_unknown='error'
                ), ohe_cols),
                
                ('cat_ord', OrdinalEncoder(
                    categories=[months_order],
                    handle_unknown='error' # June больш ніколі не выкліча памылак
                ), ["Month"])
            ],
            remainder='passthrough'  # Пакідаем бінарныя прыкметы (напрыклад, Weekend) як ёсць
        )

        return self.preprocessor

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Трансфармацыя прыкмет для онлайн-шопінгу:
        - Стварае новую метрыку Avg_Time_Per_Product (уніканне мультыкалінеарнасці з ProductRelated_Duration).
        - Выдаляе лішнія і высокаскарэляваныя прыкметы (Bounce Rates, ProductRelated_Duration).
        """
        # Ствараем копію, каб не змяняць арыгінальны датафрэйм наўпрост (добры тон распрацоўкі)
        df = df.copy()
        
        # 1. Стварэнне адноснай метрыкі
        # Дадаем 1e-5, каб пазбегнуць ZeroDivisionError, калі ProductRelated == 0
        df["Avg_Time_Per_Product"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1e-5)
        
        # 2. Спіс калонак для выдалення (высокая карэляцыя 0.91 і 0.84)
        columns_to_drop = ["BounceRates", "ProductRelated_Duration"]
        
        # Выдаляем бяспечна (толькі калі яны ёсць у датафрэйме)
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        return df
