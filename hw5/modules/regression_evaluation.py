"""
RegressionEvaluator — аналаг ModelEvaluator для рэгрэсійнай задачы.

RMSE — галоўная метрыка выбару (аналаг PR-AUC у класіфікацыі),
бо яна найбольш адчувальная да вялікага хваста PageValues (max=362).
MAE дадаецца побач, каб паказаць "тыповую" памылку без уплыву выкідаў.
R2 дадаецца для інтэрпрэтацыі (% растлумачанай варыяцыі).

MAPE наўмысна НЕ выкарыстоўваецца: пры 78% нулявых значэнняў PageValues
MAPE проста не вызначаны (дзяленне на нуль) для большасці назіранняў.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import ttest_rel
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error

# Выкарыстоўваецца ТОЛЬКІ калі pick_best() выклікаецца без явнага complexity-мапы.
# Пры рабоце з RegressorBench перадавайце bench.get_complexity_map() у pick_best(),
# каб і восьмы (і любы новы) метад карэктна ўдзельнічаў у tie-break без праўкі
# гэтага файла — гл. заўвагу ў docstring pick_best().
DEFAULT_COMPLEXITY = {
    "Dummy": 0,
    "Ridge": 1,
    "DecisionTree": 2,
    "ExtraTrees": 3,
    "LightGBM": 4,
    "XGBoost": 4,
    "CatBoost": 4,
}


class RegressionEvaluator:
    """RMSE галоўная. R2 і MAE — для інтэрпрэтацыі і ўстойлівасці да выкідаў."""

    def metrics(self, y_true: Any, y_pred: Any) -> dict[str, Any]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return {
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "medae": float(median_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def evaluate_model_performance(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        m = self.metrics(y_true, y_pred)
        return pd.DataFrame({
            "Мадэль": [model_name],
            "RMSE": [m["rmse"]],
            "MAE": [m["mae"]],
            "MedAE": [m["medae"]],
            "R2": [m["r2"]],
        })

    def compile_performance_report(
        self,
        y_true: pd.Series,
        preds_dict: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Універсальны менеджэр справаздач (аналаг ModelEvaluator.compile_performance_report)."""
        all_reports = []
        for name, preds in preds_dict.items():
            row = self.evaluate_model_performance(name, y_true, preds)
            all_reports.append(row)

        if not all_reports:
            return pd.DataFrame()

        df_final = pd.concat(all_reports, ignore_index=True)
        return df_final.sort_values(by="RMSE", ascending=True)

    def pick_best(
        self,
        fold_rmse: dict[str, np.ndarray],
        *,
        alpha: float = 0.05,
        complexity: dict[str, int] | None = None,
    ) -> str:
        """
        Прынцып Оккама, аналаг ModelEvaluator.pick_best:
        лепшая мадэль па RMSE (менш=лепш). Калі парны t-тэст паміж
        топ-1 і топ-2 незначны — бярэ больш простую мадэль.

        complexity: мапа "імя мадэлі -> ранг складанасці". Перадайце сюды
        bench.get_complexity_map(), каб дынамічна дададзеныя мадэлі (напрыклад,
        восьмы шлях TwoStageRegressor праз add_two_stage_model()) удзельнічалі
        ў tie-break карэктна. Калі не перададзена, выкарыстоўваецца
        DEFAULT_COMPLEXITY (толькі 7 базавых метадаў гэтага модуля).
        """
        complexity_map = complexity if complexity is not None else DEFAULT_COMPLEXITY

        means = {
            name: float(scores.mean())
            for name, scores in fold_rmse.items()
            if name != "Dummy"
        }

        if not means:
            print("[Evaluator] Рэальных мадэляў для выбару не знойдзена. Бяру Dummy.")
            return "Dummy"

        # Сартуем па ЎЗРАСТАННІ RMSE (менш = лепш), у адрозненне ад PR-AUC
        ranked = sorted(means, key=lambda k: means[k])

        top = ranked[0]
        if len(ranked) < 2:
            print(f"[Evaluator] Навучана ўсяго адна мадэль. Бяру яе: {top}")
            return top

        second = ranked[1]

        stat, p_value = ttest_rel(fold_rmse[top], fold_rmse[second])
        df = len(fold_rmse[top]) - 1
        t_critical = float(stats.t.ppf(1 - alpha / 2, df))

        print(f"\n[Evaluator] Парны t-тэст Сцюдэнта па фолдах (RMSE): {top} vs {second}")
        print(f"t-статыстыка={stat:.3f}, p-value={p_value:.4f}, t_critical={t_critical:.3f}")

        p_val_float = float(p_value) if not np.isnan(p_value) else 1.0
        significant = p_val_float < alpha

        if significant:
            print(f"[Evaluator] Розніца значная (p<{alpha}), бяром лідара: {top}")
            return top

        chosen = min((top, second), key=lambda name: complexity_map.get(name, 10))
        print(
            f"[Evaluator] У распрацоўку: {chosen}. "
            f"(p-value ({p_val_float:.4f}) >= {alpha}) Розніца НЕ значная паміж {top} і {second}"
        )
        return chosen
