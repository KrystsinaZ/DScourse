"""Метрики""" """, CV-сводка, paired t-test и подбор порога."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import ttest_rel
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_NAMES = ("legit", "fraud")
COMPLEXITY = {
    "Dummy": 0,
    "LogisticRegression": 1,
    "DecisionTree": 2,
    "ExtraTrees": 3,
    "LightGBM": 4,
    "XGBoost": 4,
}


class ModelEvaluator:
    """PR-AUC главная. Accuracy при таком дисбалансе врёт."""

    def metrics(
        self,
        y_true: Any,
        y_pred: Any,
        y_proba: Any | None = None,
    ) -> dict[str, Any]:
        """Precision/Recall/F1/F2 + ROC-AUC и PR-AUC."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        result: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "f2": float(
                fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            ),
            "roc_auc": float("nan"),
            "avg_precision": float("nan"),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "report": classification_report(
                y_true,
                y_pred,
                target_names=TARGET_NAMES,
                digits=4,
                zero_division=0,
            ),
        }
        if y_proba is not None:
            result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            result["avg_precision"] = float(
                average_precision_score(y_true, y_proba)
            )
        return result



    def evaluate_model_performance(
        self,
        model_name: str,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_probs: pd.Series
    ) -> pd.DataFrame:
        """
        Вылічае метрыкі і вяртае вынік радком DataFrame.
        Optimized for imbalanced binary classification (pos_label=True).
        """
        # Calculate all explicit metrics for the target class (True)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=True)
        f1 = f1_score(y_true, y_pred, pos_label=True)
        f2 = fbeta_score(y_true, y_pred, beta=2, pos_label=True)
        
        roc_auc = roc_auc_score(y_true, y_probs)
        pr_auc_cv = average_precision_score(y_true, y_probs)

        # Structure data into a dictionary
        metrics_data = {
            "Мадэль": [model_name],
            "PR-AUC": [pr_auc_cv],
            "ROC-AUC": [roc_auc],
            "Precision": [precision],
            "Recall": [recall],
            "F1": [f1],
            "F2": [f2],
            "Accuracy": [accuracy],
        }

        # Return as a clean single-row table
        return pd.DataFrame(metrics_data)

    def compile_performance_report(self, y_true: pd.Series, probas_dict: dict[str, np.ndarray], threshold: float = 0.20) -> pd.DataFrame:
        """
        Універсальны менеджэр справаздач: бярэ слоўнік імавернасцей (OOF або Test) 
        і збірае супольную табліцу ўсіх метрык, адсартаваную па PR-AUC.
        """
        
        all_reports = []
        
        for m_name, y_probs in probas_dict.items():

            y_pred = pd.Series((y_probs >= threshold).astype(int), index=y_true.index)
            y_probs_series = pd.Series(y_probs, index=y_true.index)
            
            model_df = self.evaluate_model_performance(
                model_name=m_name,
                y_true=y_true,
                y_pred=y_pred,
                y_probs=y_probs_series
            )
            all_reports.append(model_df)
            
        if not all_reports:
            return pd.DataFrame()
            
        df_final = pd.concat(all_reports, ignore_index=True)

        return df_final.sort_values(by="PR-AUC", ascending=False)

    # def print_report(
    #     self,
    #     y_true: Any,
    #     y_pred: Any,
    #     *,
    #     title: str,
    #     y_proba: Any | None = None,
    #     threshold: float | None = None,
    # ) -> dict[str, Any]:
    #     """Печатаю метрики. Accuracy не для выбора модели."""
    #     result = self.metrics(y_true, y_pred, y_proba)
    #     print(f"\n[{title}]")
    #     if threshold is not None:
    #         print(f"Threshold:     {threshold:.3f} (не дефолтные 0.5)")
    #     print(
    #         f"Accuracy:      {result['accuracy']:.4f} "
    #         "(accuracy при дисбалансе врёт)"
    #     )
    #     print(f"Precision:     {result['precision']:.4f}")
    #     print(f"Recall:        {result['recall']:.4f}")
    #     print(f"F1:            {result['f1']:.4f}")
    #     print(f"F2:            {result['f2']:.4f}")
    #     print(f"ROC-AUC:       {result['roc_auc']:.4f}")
    #     print(f"PR-AUC:        {result['avg_precision']:.4f}")
    #     print(result["report"])
    #     return result

    # def best_threshold(
    #     self,
    #     y_true: Any,
    #     y_proba: Any,
    #     *,
    #     beta: float = 1.0,
    # ) -> tuple[float, float]:
    #     """Порог на val по F-beta. 0.5 тут плохой дефолт."""
    #     y_true = np.asarray(y_true)
    #     y_proba = np.asarray(y_proba)
    #     thresholds = np.linspace(0.05, 0.95, 91)
    #     best_thr = 0.5
    #     best_score = -1.0
    #     for thr in thresholds:
    #         pred = (y_proba >= thr).astype(int)
    #         score = float(
    #             fbeta_score(y_true, pred, beta=beta, zero_division=0)
    #         )
    #         if score > best_score:
    #             best_score = score
    #             best_thr = float(thr)
    #     print(
    #         f"[Evaluator] Лучший порог по F{beta:g} "
    #         f"на val: {best_thr:.3f} (F={best_score:.4f})"
    #     )
    #     return best_thr, best_score

    def cv_table(
        self,
        fold_pr: dict[str, np.ndarray],
        fold_roc: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        
        rows = []
        for name, pr_scores in fold_pr.items():
            roc_scores = fold_roc[name]
            rows.append(
                {
                    "model": name,
                    "pr_auc_mean": round(float(pr_scores.mean()), 4),
                    "pr_auc_std": round(float(pr_scores.std()), 4),
                    "roc_auc_mean": round(float(roc_scores.mean()), 4),
                    "roc_auc_std": round(float(roc_scores.std()), 4),
                }
            )
        table = pd.DataFrame(rows).sort_values(
            "pr_auc_mean",
            ascending=False,
        )
        # print("\nCV StratifiedKFold(5), метрыка выбару = PR-AUC:")
        # print(table.to_string(index=False))
        return table
        
    def pick_best(
        self,
        fold_pr: dict[str, np.ndarray],
        *,
        alpha: float = 0.05,
    ) -> str:
        """
        Прынцып Оккама: Не трэба пладзіць сутнасці без неабходнасці
        Лепшая мадэль па PR-AUC. Калі t-test незначны — бярэ больш простую.
        """
        from scipy.stats import ttest_rel  # Забяспечваем імпарт тэсту
        # 1. Вызначаем шкалу складанасці мадэляў (чым менш лічба, тым прасцей мадэль)
        # Лагістычная рэгрэсія і Dummy — самыя простыя, бустынгі — самыя складаныя
        complexity_scale = {
            "Dummy": 0,
            "LogisticRegression": 1,
            "LinearRegression": 1,
            "KNN": 2,
            "SVM": 3,
            "DecisionTree": 2,
            "ExtraTrees": 3,
            "RandomForest": 4,
            "LightGBM": 5,
            "XGBoost": 6,
            "CatBoost": 6
        }
        # Фільтрацыя і разлік сярэдніх
        means = {
            name: float(scores.mean())
            for name, scores in fold_pr.items()
            if name != "Dummy"
        }

        # Калі рэальных мадэляў наогул няма (быў толькі Dummy)
        if not means:
            print("[Evaluator] Рэальных мадэляў для выбару не знойдзена. Бяру Dummy.")
            return "Dummy"
            
        # Ранжыраванне 
        # ranked = sorted(means, key=means.get, reverse=True)
        ranked = sorted(means, key=lambda k: float(means[k]), reverse=True)
        
        top = ranked[0]
        # Калі навучана ўсяго адна рэальная мадэль, парны тэст зрабіць нельга
        if len(ranked) < 2:
            print(f"[Evaluator] Навучана ўсяго адна мадэль. Бяру яе: {top}")
            return top
        
        second = ranked[1]

        # Парны t-тэст (ttest_rel):Правяраецца, ці з'яўляецца перавага Топ-1 над Топ-2 статыстычна значнай.
        stat, p_value = ttest_rel(fold_pr[top], fold_pr[second])
        
        
        # stats.t.ppf(1 - 0.05 / 2, df=4)
        # Аўтаматычны разлік: калі фолдаў стане 10, df стане 9, і t_critical пералічыцца сам
        df = len(fold_pr[top]) - 1
        t_critical = float(stats.t.ppf(1 - alpha / 2, df))

        print(f"\n[Evaluator] Парны t-тэст Сцюдэнта па фолдах: {top} vs {second}")
        print(f"t-статыстыка={stat:.3f}, p-value={p_value:.4f}")
        print(f"Парогавае значэнне t-статыстыкі = {t_critical:.3f} - гэта мінімальны парог для t, які трэба перасягнуць, каб перастаць лічыць розніцу выпадковай")
        # t-крытэрый Сцюдэнта паказвае, наколькі далёка сярэднія вынікі дзвюх мадэляў разышліся паміж сабой, з улікам варыяцыі метрык па фолдах.
          
                
        # Бяспечная праверка на значнасць (абарона ад NaN)
        p_val_float = float(p_value) if not np.isnan(p_value) else 1.0
        significant = p_val_float < alpha
        
        if significant:
            print(f"[Evaluator] Розніца значная (p<{alpha}), бяром лідара: {top}")
            return top

        # Выбар больш простай мадэлі праз кастомны слоўнік
        # Калі мадэлі няма ў нашым спісе складанасці, па змаўчанні лічым яе складанай (дзесятка)
        chosen = min(
            (top, second),
            key=lambda name: complexity_scale.get(name, 10),
        )

        print(
            f"[Evaluator] У распрацоўку: {chosen}. \n (p-value ({p_val_float:.4f}) >= {alpha}) Розніца НЕ значная паміж метадамі (t-статыстыка < {t_critical:.3f} )"
            
        )
        return chosen


    # def overfit_table(
    #     self,
    #     *,
    #     y_train: Any,
    #     p_train: Any,
    #     y_val: Any,
    #     p_val: Any,
    #     y_test: Any | None = None,
    #     p_test: Any | None = None,
    #     threshold: float = 0.5,
    #     gap_limit: float = 0.12,
    # ) -> pd.DataFrame:
    #     """Train vs val. Если на train космос, а на val нет — переобучение."""
    #     parts: list[tuple[str, Any, Any]] = [
    #         ("train", y_train, p_train),
    #         ("val", y_val, p_val),
    #     ]
    #     if y_test is not None and p_test is not None:
    #         parts.append(("test", y_test, p_test))

    #     rows = []
    #     metrics_by_split: dict[str, dict[str, Any]] = {}
    #     for split_name, y_true, y_proba in parts:
    #         pred = (np.asarray(y_proba) >= threshold).astype(int)
    #         scored = self.metrics(y_true, pred, y_proba)
    #         metrics_by_split[split_name] = scored
    #         rows.append(
    #             {
    #                 "split": split_name,
    #                 "pr_auc": round(scored["avg_precision"], 4),
    #                 "roc_auc": round(scored["roc_auc"], 4),
    #                 "f1": round(scored["f1"], 4),
    #                 "recall": round(scored["recall"], 4),
    #                 "precision": round(scored["precision"], 4),
    #             }
    #         )
    #     table = pd.DataFrame(rows)
    #     train_pr = metrics_by_split["train"]["avg_precision"]
    #     val_pr = metrics_by_split["val"]["avg_precision"]
    #     gap = float(train_pr - val_pr)
    #     print("\n[Overfit] train / val / test (порог с val, test не для выбора):")
    #     print(table.to_string(index=False))
    #     print(f"[Overfit] gap PR-AUC(train-val)={gap:.4f}")
    #     if gap >= gap_limit:
    #         print(
    #             f"[Overfit] Похоже на переобучение: gap ≥ {gap_limit}. "
    #             "Смотри learning curves."
    #         )
    #     else:
    #         print(
    #             f"[Overfit] Gap < {gap_limit}, на валидации не разваливается."
    #         )
    #     if "test" in metrics_by_split:
    #         val_test = abs(
    #             metrics_by_split["val"]["avg_precision"]
    #             - metrics_by_split["test"]["avg_precision"]
    #         )
    #         print(
    #             f"[Overfit] |PR-AUC(val-test)|={val_test:.4f} "
    #             "(если близко — на тесте не подгонялся)"
    #         )
    #     return table

    # def to_row(
    #     self,
    #     name: str,
    #     result: dict[str, Any],
    #     *,
    #     pr_auc_cv: str = "",
    # ) -> pd.DataFrame:
    #     """Строка в финальную таблицу. Test один раз."""
    #     return pd.DataFrame(
    #         [
    #             {
    #                 "model": name,
    #                 "pr_auc_cv": pr_auc_cv,
    #                 "pr_auc_test": round(result["avg_precision"], 4),
    #                 "roc_auc": round(result["roc_auc"], 4),
    #                 "precision": round(result["precision"], 4),
    #                 "recall": round(result["recall"], 4),
    #                 "f1": round(result["f1"], 4),
    #                 "f2": round(result["f2"], 4),
    #                 "accuracy": round(result["accuracy"], 4),
    #             }
    #         ]
    #     )


    def optimize_threshold_for_fbeta(
        self, 
        y_true: pd.Series, 
        oof_probas: np.ndarray, 
        model_name: str, 
        beta: float = 2.0
    ) -> tuple[float, float]:
        """
        Архітэктурны аптымізатар: шукае лепшы бізнес-парог для максімізацыі F-beta score.
        Шчыльны пошук (крок 0.01) на OOF прагнозах з матэматычнай устойлівасцю.
        """
        print(f"\n[Threshold Optimizer] Шукаем лепшы працоўны парог па F{beta:g} на OOF для {model_name}...")
        
        y_true_np = np.asarray(y_true)
        oof_probas = np.asarray(oof_probas)
        
        # Ствараем шчыльную сетку парогаў ад 0.05 да 0.95 з крокам 0.01 (роўна 91 кропка)
        thresholds = np.linspace(0.05, 0.95, 91)
        
        best_threshold = 0.5
        best_score = -1.0
        
        for t in thresholds:
            # Хуткае бінарнае пераўтварэнне ў памяці
            y_pred_t = (oof_probas >= t).astype(int)
            
            # Універсальны разлік для любой бэты з абаронай ад дзялення на нуль
            current_score = float(
                fbeta_score(y_true_np, y_pred_t, beta=beta, zero_division=0)
            )
            
            if current_score > best_score:
                best_score = current_score
                best_threshold = float(t)
                
        print(f"🎯 ЗНОЙДЗЕНЫ АПТЫМАЛЬНЫ ПАРОГ: {best_threshold:.3f} (Валідацыйны F{beta:g}: {best_score:.4f})")
        
        # Вяртаем картэж: і парог, і атрыманую метрыку
        return best_threshold, best_score


    
    def optimize_threshold_for_fbeta_2(self, y_true: pd.Series, oof_probas: np.ndarray, model_name: str, beta: float = 2.0) -> float:
        """
        Архітэктурны аптымізатар: шукае лепшы бізнес-парог для максімізацыі F-beta score.
        Строга на Out-of-Fold (OOF) прагнозах валідацыі. Нуль лішкавасці ў main.py!
        """
        print(f"\n[Threshold Optimizer] Шукаем лепшы працоўны парог па F{beta:.0f} на OOF для {model_name}...")
        
        best_threshold = 0.5
        best_f = 0.0
        
        # Перабіраем парогі ад 0.05 да 0.95 з крокам 0.05
        # Выкарыстоўваем адну кампактную лакальную зменную oof_probas для хуткасці памяці!
        for t in np.arange(0.05, 0.95, 0.05):
            y_pred_t = pd.Series((oof_probas >= t).astype(int), index=y_true.index)
            
            # Выклікаем функцыю metrics і забіраем ключ f2 (або f1)
            raw_res = self.metrics(y_true=y_true, y_pred=y_pred_t)
            current_f = raw_res["f2"] if beta == 2.0 else raw_res["f1"]
            
            if current_f > best_f:
                best_f = current_f
                best_threshold = float(t)
                
        print(f"🎯 ЗНОЙДЗЕНЫ АПТЫМАЛЬНЫ ПАРОГ: {best_threshold:.2f} (Валідацыйны F{beta:.0f}: {best_f:.4f})")
        return best_threshold


    # def compile_performance_report(self, y_true: pd.Series, probas_dict: dict[str, np.ndarray], threshold: float = 0.20) -> pd.DataFrame:
    #     """Збірае супольную табліцу ўсіх метрык, выклікаючы вашу функцыю metrics."""
        
    #     all_reports = []
        
    #     for m_name, y_probs in probas_dict.items():
    #         y_pred = pd.Series((y_probs >= threshold).astype(int), index=y_true.index)
            
    #         # Выклікаем вашу родную функцыю metrics
    #         raw_metrics = self.metrics(y_true=y_true, y_pred=y_pred, y_proba=y_probs)
            
    #         metrics_data = {
    #             "Мадэль": m_name,
    #             "PR-AUC": round(raw_metrics["avg_precision"], 4),
    #             "ROC-AUC": round(raw_metrics["roc_auc"], 4),
    #             "Precision": round(raw_metrics["precision"], 4),
    #             "Recall": round(raw_metrics["recall"], 4),
    #             "F1": round(raw_metrics["f1"], 4),
    #             "F2": round(raw_metrics["f2"], 4),
    #             "Accuracy": round(raw_metrics["accuracy"], 4),
    #         }
    #         all_reports.append(metrics_data)
            
    #     if not all_reports:
    #         return pd.DataFrame()
            
    #     df_final = pd.DataFrame(all_reports)
    #     return df_final.sort_values(by="PR-AUC", ascending=False)
