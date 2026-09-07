"""
PrepareRegressionDataset — пашырэнне PrepareDataset (dataset.py) для
рэгрэсійнай задачы з таргетам PageValues.

Не змяняе і не дублюе PrepareDataset: наслядоўвае ад яго і дадае
толькі тое, што сапраўды адрозніваецца для рэгрэсіі:

  1. Revenue выдаляецца з прыкмет (не толькі PageValues, які і так
     сыходзіць у таргет праз базавы .prepare()). Revenue моцна залежыць
     ад PageValues у зыходных даных — пакінуць яго ў X азначала б
     "адваротны" ўцёк інфармацыі пра ўласны таргет.
  2. Падзел робіцца БЕЗ stratify=target (мэта цяпер бесперапынная), але
     з апцыянальным стратыфікаваным падзелам па бінах нуль/квантылі
     (той жа механізм, што і ў RegressorBench.make_zero_inflated_bins),
     каб 78% нулявых сесій PageValues не патрапілі няроўна ў train/val/test.
  3. create_preprocessor_regression() — той жа ColumnTransformer, што і
     ў бацькоўскім класе, толькі без калонкі PageValues (яна больш не
     прыкмета) і без Revenue.

Патрабуе modules/dataset.py (PrepareDataset).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from modules.dataset import PrepareDataset, SplitResult


def _zero_inflated_bins(y: pd.Series, n_quantile_bins: int = 4) -> np.ndarray:
    """Псеўда-клас для стратыфікаванага падзелу: 0 = нулявыя сесіі, 1..n = квантылі ненулявых."""
    y_arr = np.asarray(y)
    bins = np.zeros(len(y_arr), dtype=int)
    nonzero_mask = y_arr > 0
    if nonzero_mask.sum() > 0:
        nonzero_vals = y_arr[nonzero_mask]
        q_bins = pd.qcut(nonzero_vals, q=n_quantile_bins, labels=False, duplicates="drop")
        bins[nonzero_mask] = q_bins.astype(int) + 1
    return bins


class PrepareRegressionDataset(PrepareDataset):
    """Пасьпадкоўвае load_csv() і структуру SplitResult ад PrepareDataset без змен."""

    LEAKAGE_COLS_DEFAULT = ["Revenue"]
    # Revenue выдаляецца па змаўчанні з прыкмет рэгрэсіі.
    # Калі наўмысна хочацца праверыць эфект уцёку, перадайце drop_cols=[] у prepare_regression().

    def prepare_regression(
        self,
        frame: pd.DataFrame,
        *,
        target_name: str = "PageValues",
        drop_cols: list[str] | None = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        stratify_zero_inflated: bool = True,
    ) -> SplitResult:
        """
        Падзел 60/20/20 + CV-hold-out, што і ў PrepareDataset.prepare(),
        але без stratify=target (мэта бесперапынная) і з выдаленнем Revenue.
        """
        self.target_name = target_name
        drop_cols = self.LEAKAGE_COLS_DEFAULT if drop_cols is None else drop_cols

        cols_to_drop = [target_name] + [c for c in drop_cols if c in frame.columns and c != target_name]
        features = frame.drop(columns=cols_to_drop)
        target = frame[target_name]

        share_zero = 100.0 * (target == 0).mean()
        print(
            f"[PrepareRegressionDataset] Радкоў: {len(frame)}, таргет = {target_name}"
            f"\n  Нулявых значэнняў: {share_zero:.2f}%, "
            f"сярэдняе={target.mean():.3f}, медыяна={target.median():.3f}, "
            f"max={target.max():.3f}"
        )
        if drop_cols:
            print(f"[PrepareRegressionDataset] Выдалены з прыкмет (абарона ад уцёку): {cols_to_drop[1:]}")

        names = tuple(str(col) for col in features.columns)

        strat_labels = _zero_inflated_bins(target) if stratify_zero_inflated else None

        x_cv, x_test, y_cv, y_test = train_test_split(
            features, target,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_labels,
        )

        relative_val = val_size / (1.0 - test_size)
        strat_labels_cv = _zero_inflated_bins(y_cv) if stratify_zero_inflated else None

        x_train, x_val, y_train, y_val = train_test_split(
            x_cv, y_cv,
            test_size=relative_val,
            random_state=random_state,
            stratify=strat_labels_cv,
        )

        x_train, x_val, x_test = x_train.copy(), x_val.copy(), x_test.copy()

        print(
            f"[PrepareRegressionDataset] Падзел static_3way (60/20/20): "
            f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}"
        )
        print(
            f"[PrepareRegressionDataset] Падзел пад CV + hold-out: "
            f"CV={len(y_cv)}, test={len(y_test)}"
        )

        # Карысная праверка на выпадак, калі ўсе прыкметы зніклі:
        if not names:
            raise ValueError("Няма прыкмет пасля выдалення таргета і ўцечак. Праверце логіку фільтрацыі калонак!")

        return SplitResult(
            x_train=x_train, x_val=x_val, x_test=x_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            x_cv=x_cv, y_cv=y_cv,
            feature_names=names,
        )

    def create_preprocessor_regression(self, categories_dict: dict[str, list]) -> ColumnTransformer:
        """
        Той жа ColumnTransformer, што і create_preprocessor() у бацькоўскім класе,
        але PageValues тут ужо не сярод прыкмет (ён — таргет).
        """
        true_numeric_cols = [
            "Administrative", "Administrative_Duration",
            "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "SpecialDay",
            # PageValues выдалены: ён цяпер таргет, а не прыкмета
        ]
        numeric_categorical_cols = ["OperatingSystems", "Browser", "Region", "TrafficType"]
        ohe_cols = ["VisitorType"]
        visitors_type = ["New_Visitor", "Other", "Returning_Visitor"]
        months_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), true_numeric_cols),
                ("cat_num_ohe", OneHotEncoder(
                    categories=[categories_dict[col] for col in numeric_categorical_cols],
                    drop="first", sparse_output=False, handle_unknown="ignore",
                ), numeric_categorical_cols),
                ("cat_str_ohe", OneHotEncoder( # "error"
                    categories=[visitors_type],
                    drop="first", sparse_output=False, handle_unknown="ignore",
                ), ohe_cols),
                ("cat_ord", OrdinalEncoder( #use_encoded_value
                    categories=[months_order], handle_unknown="error",
                ), ["Month"]),
            ],
            remainder="passthrough",  # Weekend і г.д. застаюцца як ёсць
        )
        return self.preprocessor

    def generate_diagnostic_table(self, frame: pd.DataFrame, target_name: str = "PageValues") -> None:
        """
        Разлічвае статыстычныя паказчыкі і выводзіць тэкст табліцы ў фармаце Markdown.
        """
        # 1. Дадзеныя для аналізу
        target = frame[target_name]
        # Выключаем таргет і выцечкі (LEAKAGE_COLS_DEFAULT), пакідаем толькі прыкметы
        leakage_cols = getattr(self, 'LEAKAGE_COLS_DEFAULT', ["Revenue"])
        cols_to_drop = [target_name] + [c for c in leakage_cols if c in frame.columns]
        features = frame.drop(columns=cols_to_drop)
        
        # Спіс толькі лікавых прыкмет для skewness, kurtosis і карэляцыі
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()

        # 2. Разлік паказчыкаў
        # Доля нулявых значэнняў
        share_zero = (target == 0).mean() * 100

        # Skewness і Kurtosis таргета
        skew = target.skew()
        kurt = target.kurtosis()

        # Максімальная карэляцыя прыкмет з таргетам
        correlations = features[numeric_cols].corrwith(target).abs()
        max_corr_col = correlations.idxmax()
        max_corr_val = correlations.max()

        # Мультыкалінеарнасць (матрыца карэляцыі прыкмет)
        feat_corr = features[numeric_cols].corr()
        
        # Атрыманне значэнняў для канкрэтных пар
        r_bounce_exit = feat_corr.loc["BounceRates", "ExitRates"] if "BounceRates" in feat_corr.index else 0
        r_product_dur = feat_corr.loc["ProductRelated", "ProductRelated_Duration"] if "ProductRelated" in feat_corr.index else 0

        # Кардынальнасць катэгорый
        card_traffic = frame["TrafficType"].nunique() if "TrafficType" in frame.columns else 0
        card_browser = frame["Browser"].nunique() if "Browser" in frame.columns else 0

        # 3. Вывад гатовай Markdown-табліцы
        print("| Уласцівасць | Значэнне |")
        print("| :--- | :--- |")
        print(f"| Доля нулявых {target_name} | {share_zero:.2f} % |")
        print(f"| Асіметрыя / Эксцэс | {skew:.2f} / {kurt:.1f} |")
        print(f"| Max корреляцыя з прыкметамі | {max_corr_val:.2f} ({max_corr_col}) |")
        print(f"| BounceRates ↔ ExitRates | r = {r_bounce_exit:.2f} |")
        print(f"| ProductRelated ↔ ProductRelated_Duration | r = {r_product_dur:.2f} |")
        print(f"| Кардынальнасць TrafficType / Browser | {card_traffic} / {card_browser} катэгорый |")
