from __future__ import annotations

from pathlib import Path
import pandas as pd
import seaborn as sns

from dython.nominal import associations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from typing import Any #, dict

# ---- Palette ----
BG        = '#F3F5F8'
PANEL     = '#FFFFFF'
INK       = '#22303F'
INK_SOFT  = '#7A8AA0'
SLATE     = '#4A5C73'
BLUE      = "#4682B4"
TEAL      = '#4C8C8C'
AMBER     = '#C99A3E'
CORAL     = '#C46B5E'
CORAL_SOFT= "#E9967A" 
LAVENDER  = '#8C7FB0'
GRID      = '#E3E8EE'

class DataVisualizer:
    """
    Графікі Matplotlib / Seaborn
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")
        

    def _save(self, fig: Figure, name: str) -> Path:
        path = self.output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        # print(f"[Visualizer] Захавана: {path}")
        return path

    def style_ax(self, ax, title=None, xlabel=None, ylabel=None):
        if title: ax.set_title(title, color=INK, pad=12, fontsize=13)
        if xlabel: ax.set_xlabel(xlabel, color=INK_SOFT)
        if ylabel: ax.set_ylabel(ylabel, color=INK_SOFT)
        ax.spines['left'].set_color(GRID)
        ax.spines['bottom'].set_color(GRID)
        return ax

    def plot_correlation_matrix(
        self, 
        x_train: pd.DataFrame, 
        filename: str = "correlation_matrix.png"
    ) -> Path:
        """
        Будуе сумесную матрыцу карэляцый (Лікі + Катэгорыі) праз бібліятэку Dython.
        Выкарыстоўвае Pearson's r, Cramér's V і Correlation Ratio аўтаматычна.
        """
               
        # 1. Вызначаем спісы прыкмет
        true_numeric_cols = [
            "Administrative", "Administrative_Duration", 
            "Informational", "Informational_Duration", 
            "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay"
        ]
        numeric_categorical_cols = ["OperatingSystems", "Browser", "Region", "TrafficType"]
        ohe_cols = ["VisitorType"]
        ordinal_cols = ["Month"]

        nominal_features = numeric_categorical_cols + ohe_cols + ordinal_cols

        # 2. Перастрахоўка: бярэм толькі наяўныя прыкметы
        available_cols = [col for col in (true_numeric_cols + nominal_features) if col in x_train.columns]
        df_to_plot = x_train[available_cols].copy()

        nominal_features_present = [col for col in nominal_features if col in df_to_plot.columns]
        for col in nominal_features_present:
            df_to_plot[col] = df_to_plot[col].astype(str)

        # 3. ЛІЧЫМ КАРЭЛЯЦЫЮ ПРАЗ DYTHON (БЕЗ МАЛЯВАННЯ - plot=False)
        complete_correlation = associations(
            dataset=df_to_plot,
            nominal_columns=nominal_features_present,
            plot=False,  # <--- ВАЖНА: проста лічым матрыцу ў памяці, не малюем унутры dython
            compute_only=False,
            clustering=False
        )
        
        # Дастаем чыстую матрыцу каэфіцыентаў
        corr_matrix = complete_correlation['corr']

        # 4. СТВАРАЕМ ФІГУРУ САМАСТОЙНА (ТОЛЬКІ АДЗІН РАЗ)
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)

        # Часова змяншаем памер шрыфтоў
        sns.set_context("notebook", font_scale=0.7)

        # 5. МАЛЮЕМ ПРАЗ SEABORN У НАШ AX
        sns.heatmap(
            corr_matrix, 
            ax=ax, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            cbar=True,
            # annotate_over_colors=True
        )
        
        # Скідаем кантэкст шрыфтоў
        sns.set_context("notebook", font_scale=1.0)

        # 6. Стылізацыя восі
        ax.set_title("Сумесная матрыца карэляцый (Лічбавыя + Катэгарыяльныя прыкметы)", fontsize=12, color=INK, pad=15)
        ax.tick_params(axis='x', colors=INK, labelsize=9)
        ax.tick_params(axis='y', colors=INK, labelsize=9)
        
        # Акуратны паварот подпісаў
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()

        # 7. Бяспечнае захаванне праз ваш унутраны метад
        output_path = self._save(fig, filename)
        plt.close(fig)  # Гарантавана зачыняем фігуру
        
        return output_path

    # ================================================================== #
    # РЭГРЭСІЙНЫЯ ГРАФІКІ (PageValues)
    # ================================================================== #

    def plot_regression_metric_comparison(
        self,
        fold_scores: dict,
        *,
        metric_name: str = "RMSE",
        lower_is_better: bool = True,
        filename: str = "regression_metric_comparison.png",
    ) -> Path:
        """
        Бар-чарт па адной метрыцы рэгрэсіі (RMSE/MAE/R2) для ЎСІХ мадэляў
        бэнчмарку з std-планкамі памылак па фолдах. Мадэлі сартуюцца ад
        лепшай да горшай (з улікам lower_is_better). Выклікаецца асобна
        для RMSE, MAE і R2 — гл. Мал. 1-3 у справаздачы.
        """
        names = list(fold_scores.keys())
        means = np.array([np.mean(fold_scores[n]) for n in names])
        stds = np.array([np.std(fold_scores[n]) for n in names])

        order = np.argsort(means) if lower_is_better else np.argsort(-means)
        names = [names[i] for i in order]
        means = means[order]
        stds = stds[order]

        brand_colors = [BLUE, TEAL, AMBER, CORAL, LAVENDER, SLATE, INK_SOFT, CORAL_SOFT]
        colors = [brand_colors[i % len(brand_colors)] for i in range(len(names))]

        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
        ax.set_facecolor(PANEL)

        bars = ax.bar(
            names, means, yerr=stds, capsize=5,
            color=colors, edgecolor=GRID, linewidth=1, zorder=3,
            error_kw={"ecolor": SLATE, "elinewidth": 1.5, "capthick": 1.5},
        )

        if metric_name.upper() == "R2":
            ax.axhline(0, color=INK_SOFT, linestyle="--", linewidth=1, zorder=2)

        ax.grid(axis="y", color=GRID, linestyle="-", linewidth=1, zorder=0)

        direction = "менш = лепш" if lower_is_better else "больш = лепш"
        self.style_ax(
            ax,
            title=f"{metric_name} па метадах, StratifiedKFold(5) на бінах нуль/квантылі PageValues",
            ylabel=f"{metric_name} ({direction})",
        )

        ax.tick_params(axis="x", colors=INK, rotation=25, labelsize=9)
        ax.tick_params(axis="y", colors=INK_SOFT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, v in zip(bars, means):
            offset = abs(v) * 0.02 + 0.05
            va = "bottom" if v >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + offset if v >= 0 else v - offset,
                f"{v:.3f}" if metric_name.upper() == "R2" else f"{v:.2f}",
                ha="center", va=va, color=INK, fontsize=9, fontweight="bold", zorder=4,
            )

        return self._save(fig, filename)

    def plot_actual_vs_predicted(
        self,
        y_true,
        y_pred,
        model_name: str,
        *,
        filename: str = "actual_vs_predicted.png",
    ) -> Path:
        """Дыяграма рассеяння: сапраўдны vs прадказаны рэгрэсійны таргет (напр. PageValues)."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=BG)
        ax.set_facecolor(PANEL)

        ax.scatter(y_true, y_pred, alpha=0.35, s=18, color=BLUE, edgecolor=GRID, linewidth=0.3, zorder=3)

        top = float(max(y_true.max(), y_pred.max()) * 1.02) if len(y_true) else 1.0
        ax.plot([0, top], [0, top], color=CORAL, linestyle="--", linewidth=1.5, zorder=2,
                label="Ідэальны прагноз (y=x)")

        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.grid(True, color=GRID, linestyle="-", linewidth=0.6, zorder=0)

        self.style_ax(
            ax,
            title=f"{model_name}: сапраўдны vs прадказаны таргет",
            xlabel="Сапраўднае значэнне",
            ylabel="Прадказанае значэнне",
        )
        ax.tick_params(colors=INK_SOFT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend = ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID)
        if legend:
            plt.setp(legend.get_texts(), color=INK)

        return self._save(fig, filename)

    def plot_zero_prediction_diagnostic(
        self,
        y_true,
        oof_predictions: dict,
        model_names: list,
        *,
        filename: str = "zero_prediction_diagnostic.png",
    ) -> Path:
        """
        Параўнанне размеркавання прагнозаў НА САПРАЎДЫ НУЛЯВЫХ радках (y_true==0)
        для 2+ мадэляў. Дыягностыка "ілжывых станоўчых" прагнозаў пры
        zero-inflated таргеце.
        """
        y_true = np.asarray(y_true)
        zero_mask = y_true == 0
        n_models = len(model_names)
        if n_models == 0:
            raise ValueError("[Visualizer] model_names пусты — няма чаго параўноўваць.")

        palette = [TEAL, CORAL_SOFT, BLUE, AMBER, LAVENDER, SLATE]

        fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), facecolor=BG)
        if n_models == 1:
            axes = [axes]

        for ax, name, color in zip(axes, model_names, palette):
            data = np.asarray(oof_predictions[name])[zero_mask]
            false_positive_rate = float((data > 0.5).mean()) if len(data) else 0.0

            ax.set_facecolor(PANEL)
            ax.hist(data, bins=40, color=color, edgecolor=INK, linewidth=0.4, zorder=3)
            ax.axvline(0, color=INK, linestyle="--", linewidth=1)

            ax.set_title(
                f"{name}:\nпрагнозы на сапраўды нулявых сесіях\n"
                f"(false-positive rate = {false_positive_rate:.1%})",
                color=INK, fontsize=10,
            )
            ax.set_xlabel("Прадказанае значэнне", color=INK_SOFT)
            ax.grid(True, color=GRID, linestyle="-", linewidth=0.5, zorder=0)
            ax.tick_params(colors=INK_SOFT)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.patch.set_facecolor(BG)
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_regression_feature_importance(
        self,
        model_pipeline,
        *,
        model_name: str,
        filename: str = "regression_feature_importance.png",
        top_k: int = 15,
    ) -> Path:
        """
        Важнасць прыкмет для фінальнай рэгрэсійнай мадэлі. У адрозненне ад
        plot_final_feature_importance() (які чакае named_steps "preprocessor"/
        "model" з класіфікацыйнага пайплайна), гэты метад бярэ крокі ПАЗІЦЫЙНА
        (model_pipeline[0] / model_pipeline[-1]), таму падыходзіць для
        RegressorBench.fitted[name] з крокамі "prep"/"reg".
        """
        preprocessor = model_pipeline[0]
        reg = model_pipeline[-1]

        if preprocessor is not None:
            try:
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                feature_names = np.array([f"Feature_{i}" for i in range(reg.n_features_in_)])
        else:
            feature_names = np.array([f"Feature_{i}" for i in range(reg.n_features_in_)])

        if hasattr(reg, "feature_importances_"):
            importances = reg.feature_importances_
            metric_title = "Важнасць (Gain / Gini)"
        elif hasattr(reg, "coef_"):
            coef = reg.coef_
            importances = np.abs(np.ravel(coef))
            metric_title = "Абсалютная вага каэфіцыента"
        else:
            print(f"[Visualizer] Мадэль {model_name} не падтрымлівае feature_importances_ ці coef_. Графік адменены.")
            return Path()

        if len(feature_names) != len(importances):
            min_len = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_len]
            importances = importances[:min_len]

        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}) \
            .sort_values(by="Importance", ascending=False).head(top_k)
        df_imp["Feature"] = (
            df_imp["Feature"].astype(str)
            .str.replace("num__", "", regex=False)
            .str.replace("cat_num_ohe__", "", regex=False)
            .str.replace("cat_str_ohe__", "", regex=False)
            .str.replace("cat_ord__", "", regex=False)
        )

        fig, ax = plt.subplots(figsize=(10, 2.5 + top_k * 0.35), facecolor=BG)
        ax.set_facecolor(PANEL)

        sns.barplot(data=df_imp, x="Importance", y="Feature", ax=ax, color=BLUE, edgecolor=INK, linewidth=0.8)

        self.style_ax(
            ax=ax,
            title=f"Топ-{top_k} важных прыкмет рэгрэсійнай мадэлі {model_name}",
            xlabel=metric_title,
            ylabel="Прыкметы",
        )
        ax.grid(True, axis="x", linestyle="--", color=GRID, alpha=1.0, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", colors=INK, labelsize=10)

        plt.tight_layout()
        return self._save(fig, filename)
