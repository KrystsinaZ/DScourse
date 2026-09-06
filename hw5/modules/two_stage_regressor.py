"""
TwoStageRegressor — асобны sklearn-сумяшчальны wrapper для zero-inflated
таргету PageValues (78% нулявых значэнняў, гл. аналіз у пачатку задачы).

Ідэя (класічны падыход да zero-inflated рэгрэсіі, часам называецца
"hurdle model" / "two-part model"):

    Stage 1 (Gate):      класіфікатар прадказвае P(PageValues > 0 | X)
    Stage 2 (Regressor): рэгрэсар навучаны ТОЛЬКІ на радках з PageValues > 0
    Combine:             фінальны прагноз = gate(X) * regressor(X)

Гэта НЕ мадыфікуе і не выклікае RegressorBench напрамую — гэта самастойны
эстыматар, які сумяшчальны з sklearn API (fit/predict), таму яго можна
падаць у Pipeline і ў RegressorBench.cross_validate_newV() гэтак жа, як
LGBMRegressor ці Ridge.

Два рэжымы камбінавання (параметр combine):
  "gated" (па змаўчанні, РЭКАМЕНДУЕЦЦА) — цвёрды парог: калі P(nonzero) <
        threshold, прагноз = 0, іначай = прагноз рэгрэсара. Эмпірычная
        дыягностыка (гл. справаздачу, раздзел 5) паказала, што гэта важны
        выбар, а не дробязь: пры адпаведна абраным (unbalanced) класіфікатары
        "gated" дае найлепшы MAE сярод усіх пратэставаных метадаў.
  "probability" — множым НЕПАЗЕРАРВАНУЮ імавернасць P(nonzero) на прагноз
        рэгрэсара. Тэарэтычна больш плаўны прагноз без "рэзкага" абнулення,
        АЛЕ на практыцы P(nonzero) амаль ніколі не роўны дакладна нулю —
        таму гэты рэжым сістэматычна дае малы, але ненулявы прагноз амаль
        КОЖНАЙ сесіі (у нашай дыягностыцы: false-positive rate 99.8% на
        сапраўды нулявых сесіях), што рэзка пагаршае MAE. Пакінуты як
        опцыя, але НЕ рэкамендуецца для моцна zero-inflated таргетаў тыпу
        PageValues без дадатковай каліброўкі імавернасцей.
"""
from __future__ import annotations

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted


class TwoStageRegressor(BaseEstimator, RegressorMixin):
    """Hurdle-мадэль: класіфікатар-варотнік (nonzero) + рэгрэсар на ненулявой частцы."""

    def __init__(
        self,
        classifier=None,
        regressor=None,
        combine: str = "gated",
        threshold: float = 0.5,
        clip_negative: bool = True,
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.combine = combine
        self.threshold = threshold
        self.clip_negative = clip_negative

    def fit(self, X, y):
        if self.classifier is None or self.regressor is None:
            raise ValueError(
                "[TwoStageRegressor] Патрэбны і classifier, і regressor. "
                "Прыклад: TwoStageRegressor(classifier=LGBMClassifier(...), regressor=LGBMRegressor(...))"
            )
        if self.combine not in ("probability", "gated"):
            raise ValueError(f"[TwoStageRegressor] Невядомы combine='{self.combine}'. Абярыце 'probability' ці 'gated'.")

        y = np.asarray(y).ravel()
        is_nonzero = (y > 0).astype(int)

        # --- Stage 1: варотнік P(PageValues > 0) на ЎСІХ радках ---
        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X, is_nonzero)

        # --- Stage 2: рэгрэсія ТОЛЬКІ на радках дзе PageValues > 0 ---
        nonzero_mask = y > 0
        n_nonzero = int(nonzero_mask.sum())
        if n_nonzero < 2:
            raise ValueError("[TwoStageRegressor] Занадта мала ненулявых назіранняў для навучання рэгрэсара.")

        self.regressor_ = clone(self.regressor)
        X_nonzero = X[nonzero_mask] if not hasattr(X, "iloc") else X.iloc[nonzero_mask]
        y_nonzero = y[nonzero_mask]
        self.regressor_.fit(X_nonzero, y_nonzero)

        self.share_nonzero_train_ = float(nonzero_mask.mean())
        return self

    def predict_gate_proba(self, X) -> np.ndarray:
        """P(PageValues > 0 | X) — карысна асобна для дыягностыкі якасці варотніка."""
        check_is_fitted(self, "classifier_")
        return np.asarray(self.classifier_.predict_proba(X))[:, 1]

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ["classifier_", "regressor_"])

        proba_nonzero = self.predict_gate_proba(X)
        reg_preds = np.asarray(self.regressor_.predict(X)).ravel()

        if self.clip_negative:
            reg_preds = np.clip(reg_preds, 0.0, None)

        if self.combine == "gated":
            gate = (proba_nonzero >= self.threshold).astype(float)
            final_preds = gate * reg_preds
        else:  # "probability"
            final_preds = proba_nonzero * reg_preds

        return final_preds
