from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from typing import Any #, dict
import matplotlib.pyplot as plt

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


    def plot_class_balance(
            self,
            y: Any,
            name: str,
            *,
            filename: str = "class_balance.png",
        ) -> Path:
        """Суадносіны Адмова / Купля."""
        counts = pd.Series(np.asarray(y)).value_counts().sort_index()
        labels = ["Адмова (0)", "Купля (1)"]
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sns.barplot(
            x=labels,
            y=counts.values,
            ax=ax,
            hue=labels,
            legend=False,
            palette=[CORAL_SOFT, BLUE],
        )

        for index, value in enumerate(counts.values):
            ax.text(index, value, f"{value:,}", ha="center", va="bottom")

        self.style_ax(
            ax,
            title=f'Баланс класаў па мэтавай зменнай {name}',
            ylabel='Колькасць сесій наведвальнікаў',
        )
        return self._save(fig, filename)

    def plot_cv_pr_auc(
        self,
        table: pd.DataFrame,
        *,
        filename: str = "cv_pr_auc.png",
    ) -> Path:
        """PR-AUC па CV. Выкарыстоўвае супольны фірменны стыль."""
        # 1. Сартуем мадэлі па якасці прагнозу
        data = table.sort_values("pr_auc_mean", ascending=False)
        
        # 2. Ствараем графік з фонам BG і PANEL
        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=BG)
        ax.set_facecolor(PANEL)
        
        # 3. Малюем бары з вашай палітрай (колер BLUE) і вусамі памылак (колер SLATE)
        bars = ax.bar(
            data["model"],
            data["pr_auc_mean"],
            yerr=data["pr_auc_std"],
            capsize=5,
            error_kw={"ecolor": SLATE, "elinewidth": 1.5, "capthick": 1.5},
            color=BLUE,
            edgecolor=GRID,
            linewidth=1,
            zorder=3  # Каб слупкі былі НАД сеткай
        )
        
        # 4. Наладжваем фірменную гарызантальную сетку (колер GRID)
        ax.grid(axis="y", color=GRID, linestyle="-", linewidth=1, zorder=0)
        
        # 5. ВЫКЛІК ВАШАЙ ФУНКЦЫІ (звяртаемся праз self)
        self.style_ax(
            ax,
            title="PR-AUC, StratifiedKFold(5), mean ± std",
            ylabel="PR-AUC"
        )
        
        # 6. Дадатковыя налады лімітаў і колеру шрыфтоў для восяў
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", colors=INK_SOFT, rotation=20, labelsize=10)
        ax.tick_params(axis="y", colors=INK_SOFT, labelsize=10)
        
        # Хаваем непатрэбныя верхнюю і правую рамкі, як патрабуе стыль
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # 7. Тэкставыя лічбавыя пазнакі пасярэдзіне кожнага бара (колер PANEL або INK)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                0.02 if height < 0.05 else height / 2.0,
                f"{height:.3f}",
                ha="center",
                va="center",
                color=PANEL if height >= 0.05 else INK,
                fontweight="bold",
                fontsize=9,
                zorder=4
            )

        return self._save(fig, filename)

    
    # Функцыя дыяграмы рассеяння
    def plot_scatter_clean(
        self,
        df: pd.DataFrame,
        *,
        x_col: str,
        y_col: str,
        hue_col: str | None = None,
        filename: str = "scatter_clean.png",
    ) -> Path:
        """
        Генеруе дыяграму рассеяння па зыходным датасэце.
        """
        # 1. Ствараем фігуру і восі праз subplots з захаваннем кастомнага фону
        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=BG)
        ax.set_facecolor(PANEL)
        
        # 2. Вызначаем палітру колераў для катэгорый мэтавай зменнай (True/False)
        # Калі кропкі разбіваюцца па таргеце Revenue, выкарыстоўваем CORAL_SOFT і BLUE
        custom_palette = {False: BLUE, True: CORAL_SOFT} if hue_col == "Revenue" else None

        # 3. Будуем дыяграму рассеяння па даных, захаваных у self.df
        sns.scatterplot(
            data=df, 
            x=x_col, 
            y=y_col, 
            hue=hue_col,
            ax=ax,
            palette=custom_palette,
            alpha=0.75,     # Дадаем празрыстасць, каб бачыць накладанне кропак у шчыльных месцах
            edgecolor=GRID, # Акуратная абводка вакол кожнай кропкі
            linewidth=0.5,
            zorder=3
        )
        
        # 4. Наладжваем каардынатную сетку
        ax.grid(True, color=GRID, linestyle="-", linewidth=0.5, zorder=0)
        
        # 5. Выклікаем функцыю стылізацыі (звяртаемся праз self)
        self.style_ax(
            ax,
            title=f"Дыяграма рассеяння: {x_col} vs {y_col}",
            xlabel=x_col,
            ylabel=y_col
        )
        
        # 6. Настройка рамак (spines) і шрыфтоў
        ax.tick_params(axis="both", colors=INK_SOFT, labelsize=10)
        
        # Прыбіраем верхнюю і правую рамкі, як патрабуе мінімалістычны стыль
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Калі ёсць легенда (hue), стылізуем яе элементы ў адпаведнасці з палітрай
        legend = ax.get_legend()
        if legend:
            # Правяраем, ці перададзена назва калонкі для колеру
            if hue_col is not None:
                legend.set_title(hue_col)
            legend.get_frame().set_facecolor(PANEL)
            legend.get_frame().set_edgecolor(GRID)
            plt.setp(legend.get_texts(), color=INK)
            plt.setp(legend.get_title(), color=INK_SOFT)

        # 7. Захоўваем фігуру (plt.show() унутры модуляў выкарыстоўваць нельга!)
        return self._save(fig, filename)

    # from sklearn.model_selection import StratifiedKFold
    # from sklearn.base import clone
    # from sklearn.pipeline import Pipeline
    # from sklearn.metrics import precision_recall_curve, average_precision_score

    def plot_pr_curve_cv(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        models_dict: dict,
        preprocessor,
        *,
        filename: str = "pr_curve_cv.png"
    ) -> Path:
        """
        Будуе сумеснае параўнанне PR-крывых для ЎСІХ мадэляў па фолдах.
        Кожная мадэль адлюстроўваецца сваёй сярэдняй лініяй з вашай фірмовай палітры.
        """
        # 1. Ствараем фігуру і вось, фарбуем у вашы фірмовыя колеры
        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=BG)
        ax.set_facecolor(PANEL)
        
        # Налада 5 фолдаў (дакладна як у тваёй крос-валідацыі)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # або выкарыстай свой seed
        
        # Спіс вашых брэндавых колераў для розных мадэляў
        brand_colors = [BLUE, TEAL, AMBER, CORAL, LAVENDER, SLATE]
        
        # Сетка для разліку сярэдняй крывой (іннтэрпаляцыя па паўнаце Recall)
        mean_recall = np.linspace(0, 1, 100)
        y_arr = np.asarray(y, dtype=int)
        
        # 2. Цыкл па мадэлях са слоўніка (як на тваім графіку ROC)
        for (model_name, template), color in zip(models_dict.items(), brand_colors):
            # Пропуск Dummy для сур'ёзнага параўнання (пры жаданні можна пакінуць)
            if model_name == "Dummy":
                continue
                
            precisions = []
            ap_scores = []
            
            # Прабягаем па фолдах для кожнай мадэлі
            for train_idx, val_idx in kf.split(x, y):
                model = clone(template)
                x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
                y_tr, y_va = y_arr[train_idx], y_arr[val_idx]
                
                # Збіраем часовы бяспечны Pipeline
                pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
                pipe.fit(x_tr, y_tr)
                
                # Атрымліваем імавернасці (np.asarray для супакою Pylance)
                proba_matrix = np.asarray(pipe.predict_proba(x_va))
                proba = proba_matrix[:, 1]
                
                # Лічым кропкі PR-крывой
                precision, recall, _ = precision_recall_curve(y_va, proba)
                ap_score = float(average_precision_score(y_va, proba))
                ap_scores.append(ap_score)
                
                # Інтэрпалюем (разварочваем recall, бо ён ідзе па спаданні)
                precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
                
            # Росцім сярэднюю лінію для ГЭТАЙ мадэлі па ўсіх 5 фолдах
            mean_precision = np.mean(precisions, axis=0)
            mean_precision[0] = 1.0  # Карэкціроўка стартавай кропкі Precision
            mean_ap = np.mean(ap_scores)
            std_ap = np.std(ap_scores)
            
            # Малюем ТЛУСТУЮ фірмовую лінію для гэтай мадэлі
            ax.plot(
                mean_recall, mean_precision,
                color=color, linewidth=2, zorder=3,
                label=f"{model_name} (PR-AUC = {mean_ap:.3f} ± {std_ap:.3f})"
            )
            
        # 3. Лінія выпадковага класіфікатара (Baseline = доля пакупнікоў, ~0.15)
        baseline = float(np.sum(y_arr == 1) / len(y_arr))
        ax.axhline(
            y=baseline, 
            color=INK_SOFT, linestyle="--", linewidth=1.2, zorder=2,
            label=f"Baseline (Канверсія = {baseline:.2f})"
        )
        
        # 4. Ужываем ТВОЙ метад для стылізацыі восі ax
        self.style_ax(
            ax=ax,
            title="Крос-валідацыйнае параўнанне PR-крывых (5 фолдаў)",
            xlabel="Recall (Полнота / Доля знойдзеных пакупнікоў)",
            ylabel="Precision (Точность / Дакладнасць прагнозу)"
        )
        
        # Налада кантэксту
        ax.grid(True, linestyle="--", color=GRID, alpha=1.0, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        ax.tick_params(axis="both", colors=INK_SOFT, labelsize=10)
        
        # Легенда ў вашым стылі
        legend = ax.legend(loc="lower left", facecolor=PANEL, edgecolor=GRID, fontsize=9)
        if legend:
            plt.setp(legend.get_texts(), color=INK)
            
        # 5. Захоўваем і вяртаем Path праз твой _save
        return self._save(fig, filename)


    def plot_roc_curve_cv(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        models_dict: dict[str, Any],
        preprocessor: Any,
        *,
        filename: str = "roc_curves_cv.png",
    ) -> Path:
        """
        Будуе сярэднія ROC-крывыя для кожнай мадэлі па выніках 5 фолдаў.
        """

        # 1. Ствараем фігуру і восі з захаваннем кастомнага фону
        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=BG)
        ax.set_facecolor(PANEL)
        
        # 2. Наладжваем аднолькавы падзел на фолды (як у cross_validate)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        mean_fpr = np.linspace(0, 1, 100) # Агульная сетка для інтэрпаляцыі крывых
        
        # Спіс фірменных колераў
        model_colors = [BLUE, TEAL, AMBER, CORAL, LAVENDER, SLATE]
        
        # 3. Цыкл па кожнай мадэлі са слоўніка
        for idx, (model_name, template) in enumerate(models_dict.items()):
            if model_name == "Dummy":
                # Для Baseline малюем ідэальную выпадковую дыяганаль (0.5)
                ax.plot([0, 1], [0, 1], linestyle="--", color=INK_SOFT, linewidth=1.5, 
                        label="Dummy (Baseline, AUC = 0.5000)", zorder=2)
                continue

            tprs = []
            aucs = []
            
            # Праходзім па фолдах, каб сабраць сапраўдныя каардынаты для гэтай мадэлі
            for train_idx, val_idx in kf.split(x, y):
                model = clone(template)
                x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
                y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
                
                # Будуем канвеер
                pipe = Pipeline([('prep', preprocessor), ('model', model)])
                pipe.fit(x_tr, y_tr)
                
                # Атрымліваем імавернасці
                proba = pipe.predict_proba(x_va)[:, 1]
                
                # Вылічваем каардынаты ROC для гэтага фолду
                fpr, tpr, _ = roc_curve(y_va, proba)
                
                # Інтэрпалюем, каб злучыць вынікі розных фолдаў разам
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(auc(fpr, tpr))
            
            # Лічым сярэднія каардынаты пасля ўсіх фолдаў
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            
            # 4. МАЛЮЕМ САПРАЎДНУЮ СЯРЭДНЮЮ КРЫВУЮ (замест кропак з коскай)
            color = model_colors[idx % len(model_colors)]
            ax.plot(
                mean_fpr, 
                mean_tpr,
                color=color,
                linewidth=2.5,
                label=f"{model_name} (Mean AUC = {mean_auc:.4f} ± {std_auc:.4f})",
                zorder=3
            )

        # 5. Наладжваем каардынатную сетку ў тонах GRID
        ax.grid(True, color=GRID, linestyle="-", linewidth=0.5, zorder=0)
        
        # 6. Выклікаем функцыю стылізацыі
        self.style_ax(
            ax,
            title="ROC-AUC на Крос-Валідацыі (Сярэднія крывыя па 5 фолдах)",
            xlabel="False Positive Rate (1 - Спецыфічнасць)",
            ylabel="True Positive Rate (Адчувальнасць / Recall)"
        )
        
        # 7. Налады лімітаў і рамак
        ax.set_xlim((-0.02, 1.02))
        ax.set_ylim((-0.02, 1.02))
        ax.tick_params(axis="both", colors=INK_SOFT, labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Прыгожае афармленне легенды
        legend = ax.legend(loc="lower right", frameon=True)
        if legend:
            legend.get_frame().set_facecolor(PANEL)
            legend.get_frame().set_edgecolor(GRID)
            plt.setp(legend.get_texts(), color=INK, fontsize=9)
            
        return self._save(fig, filename)

    def plot_predictions_comparison(
        self,
        y_true: np.ndarray, 
        preds_dict: dict[str, np.ndarray], 
        num_samples_per_class: int = 10,
        *,
        filename: str = "plot_predictions_comparison.png"
        ) -> Path:    
        """Будуе брэндаваны Heatmap для параўнання прагнозаў мадэляў на збалансаваным наборы."""
        y_arr = np.asarray(y_true, dtype=int)
            
        # 1. Знаходзім індэксы першых пакупнікоў і не-пакупнікоў (дададзены [0] для карэктнасці зрэзу)
        pos_indices = np.where(y_arr == 1)[0][:num_samples_per_class]
        neg_indices = np.where(y_arr == 0)[0][:num_samples_per_class]
            
        # Аб'ядноўваем іх (спачатку ідуць пакупнікі, потым не-пакупнікі)
        selected_indices = np.concatenate([pos_indices, neg_indices])
            
        # 2. Збіраем даныя для цеплавой карты па выбраных індэксах
        heatmap_data = {
            "True Class": y_arr[selected_indices]
        }
            
        for model_name, oof_probas in preds_dict.items():
            heatmap_data[model_name] = oof_probas[selected_indices]
                
        # Ператвараем у DataFrame і транспануем, каб мадэлі сталі радкамі
        df_heat = pd.DataFrame(heatmap_data).T
            
        # Пераназываем слупкі для прыгажосці (напрыклад, P_1, P_2... N_1, N_2...)
        col_names = [f"P_{i+1}" for i in range(len(pos_indices))] + [f"N_{i+1}" for i in range(len(neg_indices))]
        df_heat.columns = col_names
            
            # 3. Малюем графік з фірмовым фонам BG
        fig, ax = plt.subplots(figsize=(10, 2.5 + len(preds_dict) * 0.6), facecolor=BG)
        ax.set_facecolor(PANEL)
            
            # Ствараем кастомны фірмовы градыент: ад белай панэлі да глыбокага BLUE праз SLATE
        from matplotlib.colors import LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list("custom_brand", [PANEL, SLATE, BLUE])
            
        # Будуем цеплавую карту па правілах вашага стылю
        sns.heatmap(
            df_heat,
            annot=True, 
            fmt=".2f", 
            cmap=custom_cmap, 
            ax=ax,
            cbar=True,
            linewidths=1.0,
            linecolor=GRID, # Сетка паміж ячэйкамі strictly ў колер GRID
            annot_kws={"size": 9} # Памер шрыфту лічбаў унутры ячэек
        )
            
        self.style_ax(
            ax=ax,
            title=f"Аналіз імавернасцяў: першыя {num_samples_per_class} пакупнікоў (P) і {num_samples_per_class} не-пакупнікоў (N)",
            xlabel="Выбраныя сесіі карыстальнікаў",
            ylabel="Мадэлі / Сапраўдны клас"
        )
            
        # Дадатковая настройка тэксту палітраў
        ax.tick_params(axis='x', colors=INK, labelsize=10)
        ax.tick_params(axis='y', colors=INK, labelsize=10)
            
        # Настройка каляровай шкалы (cbar), калі яна стварылася
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=9, colors=INK_SOFT)
            cbar.outline.set_edgecolor(GRID) # Межы шкалы ў колер GRID
                
        plt.tight_layout()
            
        # 4. Захоўваем праз ваш унутраны _save
        return self._save(fig, filename)

    def plot_models_boxplot(
        self, 
        fold_scores: dict[str, np.ndarray], 
        metric_name: str = "PR-AUC", 
        *, 
        filename: str = "models_boxplot.png"
    ) -> Path:
        """Будуе Box-plot для параўнання стабільнасці мадэляў."""

        # 1. Ператвараем слоўнік метрык у табліцу DataFrame
        data_list = []
        for model_name, scores in fold_scores.items():
            if "dummy" in model_name.lower():
                continue              
            for fold_idx, score in enumerate(scores):
                data_list.append({
                    "Мадэль": model_name,
                    metric_name: score,
                    "Фолд": f"Fold {fold_idx + 1}"
                })
        # df_scores = pd.DataFrame(data_list)
        
        # # Ствараем палітру для мадэляў
        # custom_colors = [BLUE, TEAL, AMBER, CORAL, LAVENDER, SLATE]
        # active_colors = custom_colors[:len(fold_scores)]
    #-----------------
        # df_scores = pd.DataFrame(data_list)
        
        # # 🚨 ВЫПРАЎЛЕННЕ: Вылічаем колькасць мадэляў strictly пасля адсеўкі dummy-класіфікатара
        # n_unique_models = df_scores["Мадэль"].nunique()
        
        # # Ствараем палітру для мадэляў
        # custom_colors = [BLUE, TEAL, AMBER, CORAL, LAVENDER, SLATE]
        # active_colors = custom_colors[:n_unique_models] # бярэм дакладную колькасць колераў
        
        df_scores = pd.DataFrame(data_list)
        
        # 🚨 НАДЗЕЙНАЕ ВЫПРАЎЛЕННЕ: Ствараем палітру ў выглядзе слоўніка-мапы
        unique_models_list = list(df_scores["Мадэль"].unique())
        custom_colors = [BLUE, TEAL, AMBER, CORAL, LAVENDER, SLATE]
        
        # Спалучаем імя кожнай мадэлі з яе асабістым колерам
        model_palette = {model: color for model, color in zip(unique_models_list, custom_colors)}

    # -----------------
        # 2. Ствараем фігуру і вось, фарбуем фон фігуры ў BG
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
        ax.set_facecolor(PANEL) # Фон самой рабочай вобласці графіка — белая панэль
        
        # Малюем Box-plot у колерах палітры
        # sns.boxplot(
        #     data=df_scores, 
        #     x="Мадэль", 
        #     y=metric_name, 
        #     ax=ax, 
        #     palette=active_colors,
        #     hue = "Мадэль",
        #     legend=False, 
        #     width=0.4,
        #     showmeans=True,
        #     # Налада лініі сярэдняга значэння (ромб колеру INK з белай абводкай)
        #     meanprops={
        #         "marker": "D", 
        #         "markerfacecolor": PANEL, 
        #         "markeredgecolor": INK, 
        #         "markersize": 6
        #     },
        #     # Налада вусоў і ліній самой скрыні пад колер тэксту INK
        #     boxprops={"edgecolor": INK, "linewidth": 1.2},
        #     whiskerprops={"color": INK, "linewidth": 1.2},
        #     capprops={"color": INK, "linewidth": 1.2},
        #     medianprops={"color": INK, "linewidth": 1.5}
        # )
                # Малюем Box-plot у колерах кастомнай мапы
        sns.boxplot(
            data=df_scores, 
            x="Мадэль", 
            y=metric_name, 
            ax=ax, 
            palette=model_palette,  # <--- ПЕРАДАЕМ СЛОЎНІК ЗАМЕСТ СПІСУ!
            hue="Мадэль",
            legend=False, 
            width=0.4,
            showmeans=True,
            meanprops={
                "marker": "D", 
                "markerfacecolor": PANEL, 
                "markeredgecolor": INK, 
                "markersize": 6
            },
            boxprops={"edgecolor": INK, "linewidth": 1.2},
            whiskerprops={"color": INK, "linewidth": 1.2},
            capprops={"color": INK, "linewidth": 1.2},
            medianprops={"color": INK, "linewidth": 1.5}
        )

        # Накладваем кропкі кожнага фолду (мяккі шэра-сіні SLATE з празрыстасцю)
        sns.stripplot(
            data=df_scores, 
            x="Мадэль", 
            y=metric_name, 
            ax=ax, 
            color=SLATE, 
            alpha=0.6, 
            size=5, 
            jitter=0.08
        )
               
        
        # 3. СТВАРАЕМ ПОЎНУЮ РУЧНУЮ ЛЕГЕНДУ
        legend_elements = [
            # 1. Тлумачэнне для каляровых скрыняў (бярэм фірмовы BLUE як прыклад)
            Patch(facecolor=BLUE, edgecolor=INK, linewidth=1.2, label='Размах метрыкі мадэлі'),
            
            # 2. Тлумачэнне для ромба (сярэдняе)
            Line2D([0], [0], marker='D', color='none', markeredgecolor=INK, 
                   markerfacecolor=PANEL, markersize=8, label='Сярэдняе значэнне (Mean)'),
            
            # 3. Тлумачэнне для кропак (фолды) - дадаем унутраны колер, каб кропка не была нябачнай
            Line2D([0], [0], marker='o', color='none', markerfacecolor=SLATE, 
                   markeredgecolor=SLATE, alpha=0.7, markersize=6, label='Метрыка асобнага фолду')
        ]
        
        legend = ax.legend(
            handles=legend_elements, 
            loc='upper left', #'lower left',        # Пераносім у левы ніжні кут, бо там звычайна вольнае месца
            frameon=True,
            facecolor=PANEL,
            edgecolor=GRID,
            fontsize=9,
            framealpha=1.0,          # Шчыльная рамка, каб графік пад ёй не прасвечваў
            shadow=False
        )
        
        #----------------------------
        # # Дадаем легенду на графік
        # legend = ax.legend(
        #     handles=legend_elements, 
        #     loc='upper right', # размяшчэнне ў верхнім правым куце
        #     frameon=True,
        #     facecolor=PANEL,
        #     edgecolor=GRID,
        #     fontsize=9
        # )
        
        # Фарбуем тэкст легенды ў колер INK
        if legend:
            plt.setp(legend.get_texts(), color=INK)
        


        # 3. Стылізацыя восі ax
        self.style_ax(
            ax=ax,
            title=f"Параўнанне стабільнасці мадэляў па метрыцы {metric_name} (5 фолдаў)",
            xlabel="Класіфікатары",
            ylabel=metric_name
        )
        
        # Налада кантэкстнай сеткі strictly ў колер GRID
        ax.grid(True, linestyle="--", color=GRID, alpha=1.0, zorder=0)
        ax.set_axisbelow(True)
        
        # Фарбуем подпісы мадэляў на восі X у колер INK
        ax.tick_params(axis='x', colors=INK, labelsize=10)
        ax.tick_params(axis='y', colors=INK_SOFT, labelsize=10)
        plt.xticks(rotation=15)
        
        return self._save(fig, filename)

    # def plot_predictions(
    #         self, 
    #          y_true: pd.Series, 
    #          y_pred: pd.Series, 
    #          num_points, 
    #          target_name,
    #         *,
    #         filename: str = "plot_predictions.png"):
    #     """Визуализация истинных и предсказанных значений"""
    #     # Ствараем фігуру і восі
    #     fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=BG)
                
        
    #     plt.plot(range(num_points), y_true[:num_points], label='Истинные значения', marker='o')
    #     plt.plot(range(num_points), y_pred[:num_points], label='Предсказанные значения', marker='x')

    #     # Выклікаем функцыю стылізацыі
    #     self.style_ax(
    #         ax,
    #         title="Параўнанне праўдзівых і прадказаных значэнняў {target_name})",
    #         xlabel="Назіранні",
    #         ylabel="Значэнні"
    #     )

    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    #     return self._save(fig, filename)    
        
    def plot_cl_results(self, X, y, feature_names, target_names):
        
        # Создание DataFrame для удобного отображения данных
        df = pd.DataFrame(data=X, columns=feature_names)

        # Ператвараем коды (0 і 1) у тэкст праз слоўнік і робім слупок катэгарыяльным
        df[target_names] = pd.Series(y, dtype=int).map({0: 'No-Buy', 1: 'Buy'}).astype('category').values

        # Построение парных графиков при помощи sns
        sns.pairplot(df, hue=target_names)
        plt.show()    

    def plot_cv_pr_curve(
        self, 
        y_true_folds: list[np.ndarray], 
        y_proba_folds: list[np.ndarray], 
        model_name: str,
        *,
        filename: str = "cv_pr_curve.png"
    ) -> Path:
        """Малюе 5 PR-крывых для кожнага фолду і іх сярэднюю лінію."""
        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=BG)
        ax.set_facecolor(PANEL)
        
        # Спіс для збору значэнняў інтэрпаляцыі (каб палічыць сярэднюю)
        mean_recall = np.linspace(0, 1, 100)
        precisions = []
        
        # 1. Малюем крывую для кожнага фолду асобна (мяккім SLATE з высокай празрыстасцю)
        for i, (y_true, y_proba) in enumerate(zip(y_true_folds, y_proba_folds)):
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            ap_score = average_precision_score(y_true, y_proba)
            
            ax.plot(
                recall, precision, 
                color=SLATE, alpha=0.3, linewidth=1, 
                label=f'Fold {i+1} (PR-AUC = {ap_score:.3f})' if i == 0 else "" # Подпіс толькі для першага, каб не забіваць легенду
            )
            # Інтэрпалюем, каб мець аднолькавую сетку кропак для сярэдняй
            precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
            
        # 2. Разлічваем і малюем СЯРЭДНЮЮ лінію
        mean_precision = np.mean(precisions, axis=0)
        # Першы элемент заўсёды роўны 1.0 па логіцы PR
        mean_precision[0] = 1.0


        mean_np = np.mean([float(average_precision_score(t, p)) for t, p in zip(y_true_folds, y_proba_folds)])

        ax.plot(
            mean_recall, mean_precision, 
            color=BLUE, linewidth=2.5, zorder=4,
            label=f'Mean PR-Curve (AP = {mean_np:.3f})'
        )
        
        # 3. Базавая лінія выпадковага класіфікатара (доля пакупнікоў, у нашым выпадку ~0.15)
        # Бярэм долю з першага фолду як арыенцір
        baseline = sum(y_true_folds[0]) / len(y_true_folds[0])
        ax.axhline(y=baseline, color=CORAL, linestyle='--', linewidth=1.2, label=f'Baseline (Канверсія = {baseline:.2f})')
        
        # Стылізацыя
        self.style_ax(
            ax=ax,
            title=f"Пафолдавая PR-крывая для мадэлі: {model_name}",
            xlabel="Recall (Полнота)",
            ylabel="Precision (Точность)"
        )
        
        ax.grid(True, linestyle="--", color=GRID, alpha=1.0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        # Легенда
        legend = ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID)
        if legend: plt.setp(legend.get_texts(), color=INK)
            
        return self._save(fig, filename)

    def plot_test_pr_curves(
        self, 
        y_test: np.ndarray, 
        test_preds_dict: dict[str, np.ndarray], 
        *,
        filename: str = "test_pr_curves.png"
    ) -> Path:
        """Малюе фінальнае параўнанне PR-крывых розных мадэляў на тэставым сэце."""
        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=BG)
        ax.set_facecolor(PANEL)
        
        # Спіс брэндавых колераў для розных мадэляў
        brand_colors = [BLUE, TEAL, AMBER, CORAL, LAVENDER]
        
        # Малюем лінію для кожнай мадэлі
        for (model_name, y_proba), color in zip(test_preds_dict.items(), brand_colors):
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap_score = average_precision_score(y_test, y_proba)
            
            ax.plot(
                recall, precision, 
                color=color, linewidth=2, 
                label=f'{model_name} (PR-AUC = {ap_score:.3f})'
            )
            
        # Лінія выпадковага выбару (базавая канверсія)
        baseline = sum(y_test) / len(y_test)
        ax.axhline(y=baseline, color=INK_SOFT, linestyle='--', linewidth=1, label=f'Baseline ({baseline:.2f})')
        
        self.style_ax(
            ax=ax,
            title="Фінальнае параўнанне PR-крывых на тэставых даных",
            xlabel="Recall (Доля знойдзеных пакупнікоў)",
            ylabel="Precision (Дакладнасць прагнозу)"
        )
        
        ax.grid(True, linestyle="--", color=GRID, alpha=1.0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        
        legend = ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID)
        if legend: plt.setp(legend.get_texts(), color=INK)
            
        return self._save(fig, filename)
    
    def plot_classification_results(
        self, 
        y_true: np.ndarray, 
        y_probas: np.ndarray, 
        model_name: str,
        threshold: float = 0.20,  # Парог з улікам дысбалансу 15% пакупнікоў
        *,
        filename: str = "confusion_matrix_oof.png"
    ) -> Path:
        """
        Візуалізацыя матрыцы памылак (Confusion Matrix).
        """
        # 1. Пераводзім імавернасці ў бінарныя класы па абранаму парогу
        # Для 15% канверсіі рэкамендуецца парог у раёне 0.15 - 0.25 (па змаўчанні 0.20)
        y_pred = (y_probas >= threshold).astype(int)
        y_arr = np.asarray(y_true, dtype=int)
        
        # Разлічваем саму матрыцу памылак
        cm = confusion_matrix(y_arr, y_pred)
        
        # 2. Ствараем фігуру і вось 
        fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=BG)
        ax.set_facecolor(PANEL)
        
        # Ствараем градыент для матрыцы (ад белага PANEL да BLUE)
        from matplotlib.colors import LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list("cm_brand", [PANEL, BLUE])
        
        # 3. Адлюстроўваем матрыцу з лаканічнымі шопінг-подпісамі
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=['Адмова', 'Купля'] # Нашы лаканічныя подпісы замест 0 і 1
        )
        
        # Малюем матрыцу, прыбіраючы стандартную каляровую шкалу
        disp.plot(cmap=custom_cmap, ax=ax, colorbar=False, values_format='d')
        
        # 4. Ужываем метад стылізацыі восі ax
        self.style_ax(
            ax=ax,
            title=f"Матрыца памылак (OOF): {model_name}\n(Парог прыняцця рашэння = {threshold:.2f})",
            xlabel="Што прадказала мадэль",
            ylabel="Рэальныя паводзіны пакупніка"
        )
        
        # Настройка колераў тэксту ўнутры матрыцы і на восях
        ax.tick_params(axis='both', colors=INK, labelsize=10)
        
        # Тонкая настройка лічбаў унутры квадратаў (каб былі бачныя на цёмным і светлым)
        for text in ax.texts:
            text.set_color(INK)
            text.set_fontsize(11)
            
        plt.tight_layout()
        
        # 5. Захоўваем і вяртаем Path
        return self._save(fig, filename)
        
    def plot_feature_importance(
        self,
        model_pipeline,  # Навучаны фінальны Pipeline мадэлі
        *,
        model_name:str,
        filename: str = "feature_importance.png",
        top_k: int = 15  # Колькасць топ-прыкмет для адлюстравання
        ) -> Path:
        """
        Будуе графік важнасці прыкмет (Feature Importance) для найлепшай мадэлі.
        Аўтаматычна здабывае імёны калонак пасля ColumnTransformer.
        """
        # 1. Дастаем прэпрацэсар і фінальную мадэль з Pipeline
        # preprocessor = model_pipeline.named_steps.get("preprocessor")
        # clf = self._get_estimator(model_pipeline)
        
        # 1. Дастаем прэпрацэсар і фінальную мадэль з Pipeline БЯСПЕЧНА па індэксе
        preprocessor = model_pipeline[0] # заўсёды першы крок
        
        # ВЫПРАЎЛЕННЕ: Бярэм [-1], каб дакладна дастаць голы LightGBM, а не ўвесь Pipeline!
        clf = model_pipeline[-1] 

        # 2. Здабываем імёны прыкмет пасля ўсіх трансфармацый (OHE, Ordinal і г.д.)
        if preprocessor is not None:
            try:
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                # Страхоўка для старых версій scikit-learn
                feature_names = np.array([f"Feature_{i}" for i in range(clf.n_features_in_)])
        else:
            feature_names = np.array([f"Feature_{i}" for i in range(clf.n_features_in_)])

        # 3. Дастаем каэфіцыенты важнасці ў залежнасці ад тыпу мадэлі
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            metric_title = "Важнасць (Gain / Gini)"
        elif hasattr(clf, "coef_"):
            # Для лагістычнай рэгрэсіі бярэм абсалютныя значэнні вагаў (для першага класа)
            importances = np.abs(clf.coef_[0])
            metric_title = "Абсалютная вага каэфіцыента"
        else:
            print(f"[Visualizer] Мадэль {model_name} не падтрымлівае feature_importances_ ці coef_. Графік адменены.")
            return Path()

        # 4. Збіраем DataFrame і адбіраем ТОП-K прыкмет
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(top_k)

        # Пераназываем тэхнічныя OHE-подпісы для прыгожага вываду ў шопінгу
        df_imp["Feature"] = df_imp["Feature"].str.replace("num__", "").str.replace("cat_num_ohe__", "").str.replace("cat_str_ohe__", "")

        # 5. Будуем гарызантальны Bar-plot (іх значна зручней чытаць вачыма)
        fig, ax = plt.subplots(figsize=(10, 2.5 + top_k * 0.35), facecolor=BG)
        ax.set_facecolor(PANEL)

        # Малюем слупкі BLUE
        sns.barplot(
            data=df_imp,
            x="Importance",
            y="Feature",
            ax=ax,
            color=BLUE,
            edgecolor=INK, # Акуратная абводка слупкоў у колер тэксту
            linewidth=0.8
        )

        # 6. Ужываем метад для стылізацыі восі ax
        self.style_ax(
            ax=ax,
            title=f"Топ-{top_k} найважнейшых прыкмет сесіі для прагнозу пакупкі метадам {model_name}",
            xlabel=metric_title,
            ylabel="Прыкметы інтэрнэт-крамы"
        )

        # Тонкая настройка кантэксту і сеткі па вертыкалі ў колер GRID
        ax.grid(True, axis="x", linestyle="--", color=GRID, alpha=1.0, zorder=0)
        ax.set_axisbelow(True)
        
        ax.tick_params(axis='both', colors=INK, labelsize=10)

        plt.tight_layout()
        
        # 7. Захоўваем і вяртаем Path праз ваш унутраны _save
        return self._save(fig, filename)

    def plot_final_feature_importance(
        self,
        model_pipeline,  # Сюды перадаем ваш final_test_pipeline
        *,
        model_name: str,
        filename: str = "final_feature_importance.png",
        top_k: int = 15
    ) -> Path:
        """
        Метад малявання важнасці прыкмет для фінальнага кроку ацэнкі.
        """
        # 1. Дастаем аб'екты па назвах крокаў, якія вы прапісалі ў final_secure_refit
        preprocessor = model_pipeline.named_steps["preprocessor"]
        clf = model_pipeline.named_steps["model"]

        # 2. Здабываем важнасць прыкмет
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            metric_title = "Важнасць прыкметы (Gain / Gini)"
        elif hasattr(clf, "coef_"):
            coef = clf.coef_
            importances = np.abs(coef[0] if coef.ndim > 1 else coef)
            metric_title = "Абсалютная вага каэфіцыента"
        else:
            print(f"[Final Visualizer] Мадэль {model_name} не падтрымлівае важнасць.")
            return Path()

        # 3. Бяспечнае здабыванне назваў прыкмет
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            # Калі get_feature_names_out чамусьці падае на фінальных даных, ствараем індэксы
            feature_names = np.array([f"Feature_{i}" for i in range(len(importances))])

        # Выраўноўваем памеры, калі preprocessor і мадэль разышліся пасля трансфармацыі
        if len(feature_names) != len(importances):
            min_len = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_len]
            importances = importances[:min_len]

        # 4. Падрыхтоўка DataFrame
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(top_k)

        # Ачыстка тэхнічных подпісаў
        df_imp["Feature"] = df_imp["Feature"].astype(str).str.replace(
            r"^(num__|cat_num_ohe__|cat_str_ohe__)", "", regex=True
        )

        # 5. Маляванне (выкарыстоўвае вашы канстанты стыляў)
        fig, ax = plt.subplots(figsize=(10, 2.5 + top_k * 0.35), facecolor=BG)
        ax.set_facecolor(PANEL)

        sns.barplot(
            data=df_imp,
            x="Importance",
            y="Feature",
            ax=ax,
            color=BLUE,
            edgecolor=INK,
            linewidth=0.8
        )
        
        self.style_ax(
            ax=ax,
            title=f"Топ-{top_k} важных прыкмет фінальнай мадэлі {model_name}",
            xlabel=metric_title,
            ylabel="Прыкметы інтэрнэт-крамы"
        )
        
        ax.grid(True, axis="x", linestyle="--", color=GRID, alpha=1.0, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', colors=INK, labelsize=10)

        plt.tight_layout()
        return self._save(fig, filename)



    def _get_estimator(self, model: Any) -> Any:

        steps = getattr(model, "named_steps", None)
        if steps and "clf" in steps:
            return steps["clf"]
        return model

    
    def plot_boosting_early_stopping(
        self,
        evals_result: dict[str, Any],
        *,
        filename: str = "boosting_early_stopping.png",
    ) -> Path:
        """Будуе графік крывых навучання на аснове перададзенага слоўніка evals_result."""
        # структура evals_result 
        # (Крос-валідацыя па фолдах): {'LightGBM_fold_0': {...}}
        # Dataset (Выбарка): {'valid_0': {...}}
        # Metric (Назва метрыкі): {'average_precision': [...], 'binary_logloss': [...]}
        # Values (Спіс лікаў): [0.543, 0.567, 0.591...]
        
        if not evals_result:
            raise ValueError("Перададзены слоўнік evals_result пусты.")

        # Ствараем фігуру і восі
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.set_facecolor(PANEL)
        fig.patch.set_facecolor(BG)

        # Налада сеткі
        ax.grid(True, color=GRID, linestyle="-", linewidth=1, zorder=1)

        # Ствараем мапу для прыгожых подпісаў метрык
        metric_mapping = {
            "binary_logloss": "LogLoss",
            "logloss": "LogLoss",
            "average_precision": "PR-AUC",
            "aucpr": "PR-AUC"
        }

        drawn_labels = set()

        # # ТРОХУЗРОЎНЕВЫ ЦЫКЛ: Разбіраем структуру LightGBM па фолдах
        for fold_name, folds_data in evals_result.items():
            for dataset_name, metrics in folds_data.items():
                for metric_name, values in metrics.items():

                    # Цяпер values — гэта дакладна спіс лікавых значэнняў!
                    clean_values = [float(x) for x in values]

                    if not clean_values:
                        continue

                      # LightGBM: 'valid_1', XGBoost: 'validation_1'
                    is_val = "valid_1" in dataset_name.lower() or "validation_1" in dataset_name.lower()
                    
                    # КАЛІ ГЭТА ТРЭЙН, МЫ ЯГО ІГНАРУЕМ! Нам патрэбна толькі валідацыя
                    if not is_val:
                        continue

                    # Вызначаем назву метрыкі з мапы (калі няма ў мапе, пакідаем як ёсць)
                    pretty_metric_name = metric_mapping.get(metric_name.lower(), metric_name)

                    # Вызначаем тып метрыкі і колер для графіку
                    is_loss_metric = any(
                        x in metric_name.lower()
                        for x in ["loss", "error", "brier", "deviance"]
                    )
                    
                    # Каб графік не быў кашай, падзелім колеры: страты (loss) — CORAL_SOFT, якасць — BLUE
                    line_color = CORAL_SOFT if is_loss_metric else BLUE

                    # Кантралюем унікальнасць подпісу ў легендзе
                    if pretty_metric_name not in drawn_labels:
                        current_label = pretty_metric_name
                        drawn_labels.add(pretty_metric_name)
                    else:
                        current_label = ""

                    # Малюем лінію навучання для бягучага фолду
                    ax.plot(
                        clean_values,
                        color=line_color,
                        linewidth=1.2,
                        alpha=0.5,  # Празрыстасць, каб 5 фолдаў прыгожа накладваліся
                        zorder=3,
                        label= current_label # f"{metric_name}" if "fold_0" in fold_name else "" # Пазбягаем дублявання ў легендзе
                    )

                    # Шукаем пункт ранняга прыпынку (мінімум для страт, максімум для якасці)
                    if is_loss_metric:
                        best_val = min(clean_values)
                        best_iteration = clean_values.index(best_val)
                    else:
                        best_val = max(clean_values)
                        best_iteration = clean_values.index(best_val)

                    # Вертыкальная лінія ранняга прыпынку для кожнага фолду
                    ax.axvline(
                        x=best_iteration,
                        color=CORAL,
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.3,
                        zorder=2
                    )

                    # Ставім маркер-кропку на найлепшае значэнне фолду
                    ax.scatter(
                        best_iteration,
                        best_val,
                        color=CORAL,
                        s=30,
                        edgecolors=INK,
                        linewidths=0.5,
                        zorder=4
                    )
        keys_list = list(evals_result.keys())
        first_fold = keys_list[0] if keys_list else ""
        # Цяпер first_fold — гэта дакладна радок, і .split() адпрацуе ідэальна
        method_name = first_fold.split('_fold_')[0] if first_fold else "Бустынг"


        self.style_ax(
            ax,
            title=f"Крывыя навучання бустынгу {method_name} (Early Stopping)",
            xlabel="Колькасць дрэў (Ітэрацыі)",
            ylabel="Значэнне метрыкі якасці",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Ствараем адзін супольны маркер для легенды, каб патлумачыць чырвоныя кропкі і лініі
        ax.plot([], [], color=CORAL, linestyle="--", marker="o", 
                markersize=6, markeredgecolor=INK, label="Early Stopping (Best Iteration)")


        # Стылізацыя легенды і шкалы
        legend = ax.legend(
            loc="center right", frameon=True, facecolor=PANEL, edgecolor=GRID
        )
        for text in legend.get_texts():
            text.set_color(INK)
        ax.tick_params(colors=INK_SOFT, which="both")

        plt.tight_layout()

        return self._save(fig, filename)

    def plot_boosting_train_vs_val(
        self,
        evals_result: dict[str, Any],
        *,
        filename: str = "boosting_train_vs_val.png",
    ) -> Path:
        """
        Будуе графікі навучання (Train vs Validation па ітэрацыях) з пазнакай Early Stopping.
        Падтрымлівае адначасова і сіметрычна натыўныя структуры LightGBM і XGBoost.
        """
        # import matplotlib.pyplot as plt
        # from pathlib import Path

        INK = '#22303F'
        INK_SOFT = '#7A8AA0'
        BLUE = "#4682B4"
        CORAL = '#C46B5E'
        CORAL_SOFT = "#E9967A" 
        GRID = '#E3E8EE'

        if not evals_result:
            raise ValueError("Перададзены слоўнік evals_result пусты.")

        metric_mapping = {
            "binary_logloss": "LogLoss",
            "logloss": "LogLoss",
            "average_precision": "PR-AUC",
            "aucpr": "PR-AUC"
        }

        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=100)
        ax.set_facecolor("#FFFFFF")  # PANEL
        fig.patch.set_facecolor("#F3F5F8")  # BG
        ax.grid(True, color=GRID, linestyle="-", linewidth=1, zorder=1)

        drawn_labels = set()

        # ТРОХУЗРОЎНЕВЫ ЦЫКЛ
        for fold_name, folds_data in evals_result.items():
            for dataset_name, metrics in folds_data.items():
                for metric_name, values in metrics.items():
                    
                    clean_values = [float(x) for x in values]
                    if not clean_values:
                        continue

                    # Вызначаем колер (памылка ці якасць)
                    is_loss_metric = any(
                        x in metric_name.lower()
                        for x in ["loss", "error", "brier", "deviance"]
                    )
                    line_color = CORAL_SOFT if is_loss_metric else BLUE

                    # ВЫПРАЎЛЕНА: Строгае і дакладнае вызначэнне Валідацыі для абедзвюх бібліятэк
                    # Калі гэта 'valid_1' (LightGBM) альбо 'validation_1' (XGBoost), то гэта 100% ВАЛІДАЦЫЯ
                    is_val = "valid_1" in dataset_name.lower() or "validation_1" in dataset_name.lower()
                    is_train = not is_val  # Усё астатняе (training / validation_0) — гэта ТРЭЙН

                    line_style = "--" if is_train else "-"
                    subset_title = "Train" if is_train else "Val"
                    
                    pretty_metric = metric_mapping.get(metric_name.lower(), metric_name)
                    full_label = f"{pretty_metric} ({subset_title})"

                    if full_label not in drawn_labels:
                        current_label = full_label
                        drawn_labels.add(full_label)
                    else:
                        current_label = ""

                    # Малюем лінію (пункцір для Train, суцэльная для Val)
                    ax.plot(
                        clean_values,
                        color=line_color,
                        linestyle=line_style,
                        linewidth=1.4,
                        alpha=0.5,
                        zorder=3,
                        label=current_label
                    )

                    # 4. Кропку Early Stopping лічым і малюем ТОЛЬКІ па валідацыйных лініях
                    if is_val:
                        if is_loss_metric:
                            best_val = min(clean_values)
                            best_iteration = clean_values.index(best_val)
                        else:
                            best_val = max(clean_values)
                            best_iteration = clean_values.index(best_val)

                        ax.axvline(
                            x=best_iteration,
                            color=CORAL,
                            linestyle=":",
                            linewidth=0.8,
                            alpha=0.3,
                            zorder=2
                        )

                        ax.scatter(
                            best_iteration,
                            best_val,
                            color=CORAL,
                            s=25,
                            edgecolors=INK,
                            linewidths=0.5,
                            zorder=4
                        )

        # Супольны маркер для легенды
        ax.plot([], [], color=CORAL, linestyle=":", marker="o", 
                markersize=5, markeredgecolor=INK, label="Early Stopping Peak")

        # Назва метаду для загалоўка
        first_fold_str = next(iter(evals_result.keys()), "")
        method_name = first_fold_str.split('_fold_')[0] if first_fold_str else "Бустынг"

        self.style_ax(
            ax,
            title=f"{method_name}: Дынаміка навучання Train vs Validation",
            xlabel="Колькасць дрэў (Ітэрацыі)",
            ylabel="Значэнне метрыкі",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend = ax.legend(loc="center right", frameon=True, facecolor="#FFFFFF", edgecolor=GRID)
        for text in legend.get_texts():
            text.set_color(INK)
        ax.tick_params(colors=INK_SOFT, which="both")

        plt.tight_layout()
        return self._save(fig, filename)


    def plot_learning_curves(
        self,
        curves_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        filename: str = "models_learning_curves.png",
    ) -> Path:
        """
        Будуе супольны графік Learning Curves (залежнасць якасці ад памеру выбаркі).
        Большы разрыў (gap) паміж пункцірам (Train) і суцэльнай лініяй (Val) = перавучэнне.
        """
        # import matplotlib.pyplot as plt
        # import numpy as np
        # from pathlib import Path

        # # ВЫПРАЎЛЕНА: Фірмовая палітра колераў раскаментавана для працы метаду
        # BG        = '#F3F5F8'
        # PANEL     = '#FFFFFF'
        # INK       = '#22303F'
        # INK_SOFT  = '#7A8AA0'
        # BLUE      = "#4682B4"      # Колер для Валідацыі (Val)
        # SLATE     = '#4A5C73'     # Колер для Трэніроўкі (Train)
        # GRID      = '#E3E8EE'

        n_models = len(curves_data)
        if n_models == 0:
            raise ValueError("Слоўнік curves_data пусты. Няма чаго маляваць.")

        # Ствараем дынамічную сетку subplots (усе мадэлі ў адзін радок)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True, dpi=100)
        fig.patch.set_facecolor(BG)  # BG колер вонкавага фону
        
        # Калі мадэль усяго адна, ператвараем у спіс для зручнасці ітэрацыі
        if n_models == 1:
            axes = [axes]

        # Распакоўка ідзе дакладна па структуры tuple: (sizes, train_scores, val_scores)
        for i, (model_name, (sizes, train_scores, val_scores)) in enumerate(curves_data.items()):
            ax = axes[i]
            ax.set_facecolor(PANEL)
            ax.grid(True, color=GRID, linestyle="-", linewidth=1, zorder=1)

            # Вылічваем сярэдняе і std па 5 фолдах (вось 1)
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            # 1. Малюем ТРЭЙН (Пункцірная лінія колеру SLATE з заліўкай std карыдора)
            ax.plot(sizes, train_mean, linestyle="--", color=SLATE, linewidth=2, zorder=3, label="Train")
            ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, color=SLATE, alpha=0.1, zorder=2)

            # 2. Малюем ВАЛІДАЦЫЮ (Суцэльная лінія колеру BLUE з заліўкай std карыдора)
            ax.plot(sizes, val_mean, linestyle="-", color=BLUE, linewidth=2, zorder=3, label="Validation")
            ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, color=BLUE, alpha=0.1, zorder=2)

            # Стылізацыя кожнай асобнай панэлі праз ваш метад style_ax
            self.style_ax(
                ax,
                title=f"Крывая навучання {model_name}",
                xlabel="Памер выбаркі (радкоў)",
                ylabel="PR-AUC" if i == 0 else None  # Подпіс вертыкалі толькі на самай першай панэлі
            )

            # Прыбіраем непатрэбныя верхнюю і правую рамкі (як у папярэдніх метадах)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(colors=INK_SOFT, which="both")

            # Дадаем легенду толькі на першую панэль, каб не дубляваць прастору
            if i == 0:
                legend = ax.legend(loc="center right", frameon=True, facecolor=PANEL, edgecolor=GRID)
                for text in legend.get_texts():
                    text.set_color(INK)

        plt.tight_layout()
        
        # Захоўваем малюнак праз ваш унутраны метад і вяртаем Path
        return self._save(fig, filename)

    def plot_learning_curves_ES(self, curves_data: dict[str, Any], *, filename: str = "models_learning_curves_ES.png") -> Path:
        """Будуе панэльны графік з параўнаннем двух гэпаў і зоркай Early Stopping."""
        # import matplotlib.pyplot as plt
        # from pathlib import Path
        
        BG, PANEL, INK, INK_SOFT, BLUE, SLATE, GRID = '#F3F5F8', '#FFFFFF', '#22303F', '#7A8AA0', "#4682B4", '#4A5C73', '#E3E8EE'

        curves_data_clean = {k: v for k, v in curves_data.items() if "dummy" not in k.lower()}
        
        n_models = len(curves_data_clean)

        # n_models = len(curves_data)
        if n_models == 0:
            raise ValueError("Слоўнік curves_data пусты.")

        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5.5), sharey=True, dpi=100)
        fig.patch.set_facecolor(BG)
        if n_models == 1: axes = [axes]

        for i, (model_name, data) in enumerate(curves_data_clean.items()):
                
            ax = axes[i]

            ax.set_facecolor(PANEL)
            ax.grid(True, color=GRID, linestyle="-", linewidth=1, zorder=1)
            
            sizes = data["train_sizes"]
            
            # Малюем карыдоры стабільнасці і лініі (як у вашым кодэ)
            ax.fill_between(sizes, data["train_mean"] - data["train_std"], data["train_mean"] + data["train_std"], alpha=0.1, color=SLATE, zorder=2)
            ax.fill_between(sizes, data["val_mean"] - data["val_std"], data["val_mean"] + data["val_std"], alpha=0.1, color=BLUE, zorder=2)
            
            ax.plot(sizes, data["train_mean"], 'o--', color=SLATE, linewidth=2, zorder=3, label='Train')
            ax.plot(sizes, data["val_mean"], 'o-', color=BLUE, linewidth=2, zorder=3, label='Validation')
            
            # === ЧЫСТАЕ МАЛЯВАННЕ ЗОРКІ ПА НАЯЎНАСЦІ ДАДЗЕНЫХ ===
            best_es_iter = data.get("best_es_iter")
            
            if best_es_iter is not None:
                x_pos = sizes[-1]                  # Вось 100% памеру выбаркі
                y_pos = data.get("val_score_es")   # Вышыня якасці на піку ES
                
                ax.scatter(
                    x_pos, y_pos, color='#C99A3E', marker='*', s=280,
                    edgecolors=INK, linewidths=1.2, zorder=5,
                    label=f"ES Peak (Tree №{best_es_iter})"
                )

            # Двухпавярховая працэнтная разметка восі X пад 4 стратыфікаваныя крокі (25% - 100%)
            percent_steps = ["25%", "50%", "75%", "100%"]
            combined_labels = [f"{pct}\n({int(s)})" for pct, s in zip(percent_steps, sizes)]
            ax.set_xticks(sizes)
            ax.set_xticklabels(combined_labels, fontsize=8, color=INK_SOFT) #, rotation=10)
            
            for s in sizes:
                ax.axvline(x=s, color=GRID, linestyle=":", linewidth=1.0, zorder=1)

            # === СІНХРАНІЗАЦЫЯ: Падзагалоўкі з падпраўленымі назвамі як у табліцы ===
            gap_100 = data["gap_100"]
            gap_es = data["gap_es"]
            
            if gap_es is not None:
                title_text = f"{model_name}\nGap 100%: {gap_100:.4f}\nGap ES: {gap_es:.4f}"
            else:
                title_text = f"{model_name}\nGap 100%: {gap_100:.4f}"
                
            ax.set_title(title_text, fontsize=9.5, color=INK, pad=10)
            ax.set_xlabel('Памер выбаркі')
            if i == 0: ax.set_ylabel('PR-AUC')
            
            legend = ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor=GRID)
            for text in legend.get_texts(): text.set_color(INK)
            
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(colors=INK_SOFT, which="both")

        plt.tight_layout()
        return self._save(fig, filename)

    def plot_pirson_correlation_matrix(
        self, 
        x_train: pd.DataFrame, 
        filename: str = "correlation_matrix.png"
    ) -> Path:
        """
        Будуе матрыцу карэляцыі для лічбавых прыкмет на навучальнай выбарцы.
        Дапамагае знайсці дублюючыя прыкметы.
        """
        # 1. Адбіраем толькі сапраўдныя лічбавыя калонкі (без катэгорый і OHE)
        numeric_cols = [
            "Administrative", "Administrative_Duration", 
            "Informational", "Informational_Duration", 
            "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay"
        ]
        
        # Калі ў x_train калонкі маюць тэхнічныя прэфіксы пасля Pipeline, бярэм іх, інакш звычайныя
        available_cols = [c for c in numeric_cols if c in x_train.columns]
        if not available_cols:
            # Калі калонкі яшчэ не падзяліліся або маюць прэфікс num__
            available_cols = [c for c in x_train.columns if any(n in c for n in numeric_cols)]

        # 2. Лічым карэляцыю Пірсана
        corr_matrix = x_train[available_cols].corr()

        # 3. Маляванне цеплавой карты
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
        ax.set_facecolor(PANEL)

        # Малюем маску для верхняга трыкутніка, каб графік лепей чытаўся
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,          # Выводзім лічбы каэфіцыентаў
            fmt=".2f",           # Два знакі пасля коскі
            cmap="coolwarm",     # Колеравая схема (сіні — мінус, чырвоны — плюс)
            vmax=1, vmin=-1,     # Мяжы карэляцыі
            center=0,
            ax=ax,
            cbar_kws={"shrink": 0.8}
        )

        self.style_ax(
            ax=ax,
            title="Матрыца карэляцыі лічбавых прыкмет",
            xlabel="",
            ylabel=""
        )
        
        # Паварочваем подпісы для прыгажосці
        plt.xticks(rotation=45, ha='right', color=INK)
        plt.yticks(color=INK)

        plt.tight_layout()
        return self._save(fig, filename)
    
    def plot_validation_curve(
        self,
        curves_results: dict,
        model_name: str = "LightGBM",
        *,
        filename: str = "lgbm_val_curve.png"
    ) -> Path:
        """
        Будуе крывую валідацыі на аснове стандартызаванага слоўніка вынікаў.
        """
        # 1. Распакоўка даных па ключы мадэлі (Гэта галоўная змена!)
        data = curves_results[model_name]
        param_name = data["param_name"]
        param_range = data["param_range"]
        
        # 2. Ствараем фігуру і вось у вашых брэндавых колерах
        fig, ax = plt.subplots(figsize=(6.5, 5), facecolor=BG)
        ax.set_facecolor(PANEL)
        
        # 3. Лінія навучання (Тэаль) — дадаем вердыкт прама ў легенду
        ax.plot(
            param_range, data["train_mean"],
            color=TEAL, linewidth=2, marker="o", zorder=3,
            label=f"Training Score ({data['verdict']})"
        )
        ax.fill_between(
            param_range, 
            data["train_mean"] - data["train_std"], 
            data["train_mean"] + data["train_std"],
            alpha=0.12, color=TEAL, zorder=2
        )
        
        # 4. Лінія крос-валідацыі (Блю)
        ax.plot(
            param_range, data["val_mean"],
            color=BLUE, linewidth=2, marker="s", zorder=3,
            label="Cross-Validation Score (Валідацыя)"
        )
        ax.fill_between(
            param_range, 
            data["val_mean"] - data["val_std"], 
            data["val_mean"] + data["val_std"],
            alpha=0.12, color=BLUE, zorder=2
        )
        
        # 5. стылізацыя восі ax

        self.style_ax(
            ax=ax,
            title=f"Крывая валідацыі для кантролю перанавучання",
            xlabel=f"Значэнне параметра {param_name} (Колькасць лісця)",
            ylabel="Якасць мадэлі (Метрыка F1-Score)"
        )
        
        # НАЛАДА ВОСІ X: Жорстка ставім падпісы ТОЛЬКІ там, дзе ёсць рэальныя значэнні
        # np.asarray гарантуе, што перададзены спіс прачытаецца карэктна
        ticks = np.asarray(param_range)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=9) # Пераводзім у тэкст для дакладнасці
        
        # Калі кропак шмат і яны блізка, можна трохі нахіліць подпісы, каб не перакрываліся:
        # ax.set_xticklabels([str(t) for t in ticks], fontsize=9, rotation=45)

        ax.grid(True, linestyle="--", color=GRID, alpha=1.0, zorder=0)
        ax.set_axisbelow(True)
        
        # Водступы па баках робім мінімальнымі, каб вобласць графіка выкарыстоўвалася эфектыўна
        ax.set_xlim(min(param_range) - 1, max(param_range) + 5)
        ax.set_ylim(-0.02, 1.02)
        
        ax.tick_params(axis="both", colors=INK_SOFT, labelsize=10)

        
        # 6. Легенда па цэнтры справа (за межамі рамкі)
        legend = ax.legend(
            loc="upper left", 
            # bbox_to_anchor=(1.02, 0.5), 
            facecolor=PANEL, 
            edgecolor=GRID, 
            fontsize=9
        )
        if legend:
            plt.setp(legend.get_texts(), color=INK)
            
        # 7. Захоўваем і вяртаем Path праз твой унутраны метад _save
        return self._save(fig, filename)

    def plot_correlation_matrix(
        self, 
        x_train: pd.DataFrame, 
        filename: str = "correlation_matrix.png"
    ) -> Path:
        """
        Будуе сумесную матрыцу карэляцый (Лікі + Катэгорыі) праз бібліятэку Dython.
        Выкарыстоўвае Pearson's r, Cramér's V і Correlation Ratio аўтаматычна.
        """
        from dython.nominal import associations
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        
        # Фірмовая каляровая палітра
        BG, PANEL, INK, INK_SOFT, GRID = '#F3F5F8', '#FFFFFF', '#22303F', '#7A8AA0', '#E3E8EE'

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
