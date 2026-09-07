"""
Візуалізацыі, спецыфічныя для задачы кластарызацыі Spotify Tracks Dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# from modules.visualization import (
#     DataVisualizer, BG, PANEL, INK, INK_SOFT, SLATE, BLUE, TEAL, AMBER,
#     CORAL, CORAL_SOFT, LAVENDER, GRID,
# )

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


# class ClusteringVisualizer(DataVisualizer):
class ClusteringVisualizer():    
    """Графікі для EDA і кластарызацыі. Усе метады вяртаюць Path да захаванага PNG."""

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
    
    # ================================================================== #
    # EDA
    # ================================================================== #

    def plot_feature_distributions(
        self, df: pd.DataFrame, cols: list[str], *, filename: str = "feature_distributions.png", n_cols: int = 4,
    ) -> Path:
        """Гісторограмы размеркавання лічбавых аўдыё-прыкмет (для пошуку скошанасці/выкідаў)."""
        n_rows = int(np.ceil(len(cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), facecolor=BG)
        axes = np.asarray(axes).reshape(-1)

        for i, col in enumerate(cols):
            ax = axes[i]
            ax.set_facecolor(PANEL)
            sns.histplot(x=df[col], bins=40, color=BLUE, edgecolor=GRID, ax=ax, kde=True, line_kws={"color": CORAL})
            self.style_ax(ax, title=col, xlabel=None, ylabel=None)
            ax.tick_params(colors=INK_SOFT, labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for j in range(len(cols), len(axes)):
            axes[j].axis("off")

        fig.suptitle("Размеркаванне аўдыё-прыкмет (да маштабавання)", color=INK, fontsize=14, y=1.01)
        return self._save(fig, filename)

    def plot_outlier_boxplots(
        self, df: pd.DataFrame, cols: list[str], *, filename: str = "outlier_boxplots.png",
    ) -> Path:
        """Boxplot для хуткай візуальнай ацэнкі выкідаў (IQR "вусы") па некалькіх прыкметах."""
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(cols)), 5.5), facecolor=BG)
        ax.set_facecolor(PANEL)

        # Z-нармалізуем толькі для сумеснага візуальнага параўнання маштабаў на адным графіку
        z = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0)
        sns.boxplot(
            data=z, ax=ax, color=BLUE, width=0.5, fliersize=2.5,
            flierprops={"markerfacecolor": CORAL, "markeredgecolor": CORAL, "alpha": 0.5},
            boxprops={"edgecolor": INK, "linewidth": 1.1},
            whiskerprops={"color": INK, "linewidth": 1.1},
            capprops={"color": INK, "linewidth": 1.1},
            medianprops={"color": CORAL, "linewidth": 1.6},
        )
        self.style_ax(
            ax, title="Выкіды па IQR (z-нармалізавана для сумеснага маштабу)",
            ylabel="Z-значэнне",
        )
        ax.tick_params(axis="x", colors=INK, rotation=30, labelsize=9)
        ax.tick_params(axis="y", colors=INK_SOFT)
        ax.grid(axis="y", color=GRID, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return self._save(fig, filename)

    def plot_feature_correlation_matrix(
        self, x_raw: pd.DataFrame, cols: list[str], *, filename: str = "feature_correlation_matrix.png",
    ) -> Path:
        """Матрыца карэляцыі Пірсана паміж лічбавымі аўдыё-прыкметамі (пошук мультыкалінеарнасці)."""
        corr = x_raw[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(9, 7.5), facecolor=BG)
        ax.set_facecolor(PANEL)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, ax=ax, cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor=GRID,
        )
        self.style_ax(ax, title="Матрыца карэляцыі аўдыё-прыкмет (Пірсан)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color=INK)
        plt.setp(ax.get_yticklabels(), color=INK)
        plt.tight_layout()
        return self._save(fig, filename)

    # ================================================================== #
    # Падбор гіперпараметраў (elbow / silhouette / BIC / HDBSCAN-scan)
    # ================================================================== #

    def plot_elbow_and_silhouette(
        self, scan_table: pd.DataFrame, *, chosen_k: int | None = None, filename: str = "kmeans_elbow_silhouette.png",
    ) -> Path:
        """Сумесны графік elbow (inertia) і Silhouette Score для выбару k у KMeans."""
        fig, ax1 = plt.subplots(figsize=(8, 5.5), facecolor=BG)
        ax1.set_facecolor(PANEL)

        ax1.plot(scan_table["k"], scan_table["inertia"], "o-", color=SLATE, linewidth=2, label="Inertia (elbow)")
        self.style_ax(ax1, title="Падбор колькасці кластараў k (KMeans)", xlabel="Колькасць кластараў k",
                       ylabel="Inertia (сума квадратаў адлегласцяў)")
        ax1.tick_params(axis="y", colors=SLATE)
        ax1.yaxis.label.set_color(SLATE)
        ax1.grid(True, color=GRID, linestyle="--", zorder=0)

        ax2 = ax1.twinx()
        ax2.plot(scan_table["k"], scan_table["silhouette"], "s-", color=CORAL, linewidth=2, label="Silhouette Score")
        ax2.set_ylabel("Silhouette Score", color=CORAL)
        ax2.tick_params(axis="y", colors=CORAL)
        ax2.spines["top"].set_visible(False)

        if chosen_k is not None:
            ax1.axvline(chosen_k, color=INK_SOFT, linestyle=":", linewidth=1.5)
            ax1.text(chosen_k, ax1.get_ylim()[1], f" k={chosen_k}", color=INK, va="top", fontsize=9)

        ax1.set_xticks(list(scan_table["k"]))
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor=PANEL, edgecolor=GRID)
        plt.setp(legend.get_texts(), color=INK)

        plt.tight_layout()
        return self._save(fig, filename)

    def plot_gmm_bic_aic(self, scan_table: pd.DataFrame, *, chosen_k: int | None = None,
                          filename: str = "gmm_bic_aic.png") -> Path:
        """BIC/AIC для падбору n_components GaussianMixture (мінімум = аптымальны выбар)."""
        fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=BG)
        ax.set_facecolor(PANEL)

        ax.plot(scan_table["k"], scan_table["bic"], "o-", color=BLUE, linewidth=2, label="BIC")
        ax.plot(scan_table["k"], scan_table["aic"], "s--", color=AMBER, linewidth=2, label="AIC")

        if chosen_k is not None:
            ax.axvline(chosen_k, color=INK_SOFT, linestyle=":", linewidth=1.5)

        self.style_ax(ax, title="Падбор колькасці кампанентаў (Gaussian Mixture)",
                       xlabel="Колькасць кампанентаў", ylabel="Крытэрый інфармацыі")
        ax.set_xticks(list(scan_table["k"]))
        ax.grid(True, color=GRID, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend = ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID)
        plt.setp(legend.get_texts(), color=INK)
        return self._save(fig, filename)

    def plot_hdbscan_scan(self, scan_table: pd.DataFrame, *, filename: str = "hdbscan_scan.png") -> Path:
        """Чутлівасць HDBSCAN да min_cluster_size: колькасць кластараў і доля шуму."""
        fig, ax1 = plt.subplots(figsize=(7.5, 5), facecolor=BG)
        ax1.set_facecolor(PANEL)

        ax1.plot(scan_table["min_cluster_size"], scan_table["n_clusters"], "o-", color=TEAL, linewidth=2,
                 label="Колькасць кластараў")
        self.style_ax(ax1, title="Чутлівасць HDBSCAN да min_cluster_size", xlabel="min_cluster_size",
                       ylabel="Колькасць кластараў")
        ax1.tick_params(axis="y", colors=TEAL)
        ax1.yaxis.label.set_color(TEAL)
        ax1.grid(True, color=GRID, linestyle="--", zorder=0)

        ax2 = ax1.twinx()
        ax2.plot(scan_table["min_cluster_size"], scan_table["noise_share_%"], "s-", color=CORAL, linewidth=2,
                  label="Доля шуму, %")
        ax2.set_ylabel("Доля шуму, %", color=CORAL)
        ax2.tick_params(axis="y", colors=CORAL)
        ax2.spines["top"].set_visible(False)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor=PANEL, edgecolor=GRID)
        plt.setp(legend.get_texts(), color=INK)

        plt.tight_layout()
        return self._save(fig, filename)

    # ================================================================== #
    # Візуалізацыя вынікаў кластарызацыі (PCA / t-SNE / UMAP)
    # ================================================================== #

    def plot_cluster_scatter_2d(
        self, embedding: np.ndarray, labels: np.ndarray, *, method_name: str, model_name: str,
        filename: str = "cluster_scatter.png",
    ) -> Path:
        """Дыяграма рассеяння 2D-праекцыі (PCA/t-SNE/UMAP), пункты пафарбаваны па кластары."""
        fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=BG)
        ax.set_facecolor(PANEL)

        labels = np.asarray(labels)
        unique_labels = sorted(set(labels))
        palette = sns.color_palette("husl", n_colors=max(3, len(unique_labels)))

        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            color = INK_SOFT if lab == -1 else palette[i % len(palette)]
            label_text = "Шум (HDBSCAN)" if lab == -1 else f"Кластар {lab}"
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1], s=6, alpha=0.5 if lab == -1 else 0.65,
                color=color, edgecolor="none", label=label_text, zorder=3 if lab != -1 else 2,
            )

        self.style_ax(ax, title=f"{method_name}-праекцыя, пафарбавана па кластарах ({model_name})",
                       xlabel=f"{method_name}-1", ylabel=f"{method_name}-2")
        ax.grid(True, color=GRID, linestyle="--", alpha=0.7, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=INK_SOFT, labelsize=8)

        if len(unique_labels) <= 20:
            legend = ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID, fontsize=7, markerscale=2, ncol=1)
            plt.setp(legend.get_texts(), color=INK)

        return self._save(fig, filename)

    def plot_cluster_sizes(self, labels: np.ndarray, *, model_name: str, filename: str = "cluster_sizes.png") -> Path:
        """Слупковая дыяграма памераў кластараў (шукаем "кластары-аскепкі" ці занадта вялікія групы)."""
        labels = np.asarray(labels)
        counts = pd.Series(labels).value_counts().sort_index()
        names = ["Шум" if i == -1 else f"К{i}" for i in counts.index]

        fig, ax = plt.subplots(figsize=(max(6.5, 0.55 * len(counts)), 5), facecolor=BG)
        ax.set_facecolor(PANEL)
        colors = [INK_SOFT if i == -1 else BLUE for i in counts.index]
        # bars = ax.bar(names, counts.values, color=colors, edgecolor=GRID, linewidth=1, zorder=3)
        bars = ax.bar(names, counts.to_numpy(), color=colors, edgecolor=GRID, linewidth=1, zorder=3)

        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK)

        self.style_ax(ax, title=f"Памеры кластараў ({model_name})", ylabel="Колькасць трэкаў")
        ax.grid(axis="y", color=GRID, linestyle="--", zorder=0)
        ax.tick_params(axis="x", colors=INK, labelsize=9)
        ax.tick_params(axis="y", colors=INK_SOFT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return self._save(fig, filename)

    def plot_cluster_feature_heatmap(
        self, x_raw: pd.DataFrame, labels: np.ndarray, cols: list[str], *, model_name: str,
        filename: str = "cluster_feature_heatmap.png",
    ) -> Path:
        """
        Цеплавая карта сярэдніх Z-значэнняў прыкмет па кластарах — асноўны
        інструмент інтэрпрэтацыі "музычнага партрэта" кожнага кластара.
        """
        df = x_raw[cols].copy()
        z = (df - df.mean()) / df.std(ddof=0)
        z["cluster"] = np.asarray(labels)
        z = z[z["cluster"] != -1]  # шум HDBSCAN не мае сэнсу профіляваць
        profile = z.groupby("cluster")[cols].mean()

        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(cols)), max(4, 0.6 * len(profile))), facecolor=BG)
        ax.set_facecolor(PANEL)
        custom_cmap = LinearSegmentedColormap.from_list("brand_diverge", [BLUE, PANEL, CORAL])

        sns.heatmap(
            profile, annot=True, fmt=".2f", cmap=custom_cmap, center=0, ax=ax,
            linewidths=1.0, linecolor=GRID, cbar_kws={"label": "Z-адхіленне ад сярэдняга"},
        )
        self.style_ax(ax, title=f"Музычны профіль кластараў ({model_name}, Z-адхіленні)",
                       xlabel="Прыкметы", ylabel="Кластар")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", color=INK)
        plt.setp(ax.get_yticklabels(), color=INK, rotation=0)
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_metrics_comparison(self, comparison_table: pd.DataFrame, *, filename: str = "metrics_comparison.png") -> Path:
        """Бар-чарты Silhouette / Davies-Bouldin / Calinski-Harabasz для ўсіх мадэляў побач."""
        metrics = [("silhouette", "Silhouette (вышэй = лепш)", True),
                   ("davies_bouldin", "Davies-Bouldin (ніжэй = лепш)", False),
                   ("calinski_harabasz", "Calinski-Harabasz (вышэй = лепш)", True)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=BG)
        colors_map = {"KMeans": BLUE, "GMM": TEAL, "HDBSCAN": CORAL}

        for ax, (col, title, higher_better) in zip(axes, metrics):
            ax.set_facecolor(PANEL)
            data = comparison_table.dropna(subset=[col])
            colors = [colors_map.get(m, SLATE) for m in data["model"]]
            bars = ax.bar(data["model"], data[col], color=colors, edgecolor=GRID, linewidth=1, zorder=3)
            for bar, v in zip(bars, data[col]):
                ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}",
                        ha="center", va="bottom", fontsize=9, color=INK)
            self.style_ax(ax, title=title)
            ax.grid(axis="y", color=GRID, linestyle="--", zorder=0)
            ax.tick_params(axis="x", colors=INK, labelsize=9, rotation=10)
            ax.tick_params(axis="y", colors=INK_SOFT)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("Параўнанне метадаў кластарызацыі", color=INK, fontsize=13, y=1.03)
        plt.tight_layout()
        return self._save(fig, filename)

    def plot_genre_cluster_crosstab(
        self, metadata: pd.DataFrame, labels: np.ndarray, *, model_name: str, top_genres: int = 20,
        filename: str = "genre_cluster_crosstab.png",
    ) -> Path:
        """
        Крос-табуляцыя (жанр x кластар) для найбольш частых жанроў — паказвае,
        ці збіраюцца музычна блізкія жанры (напр. death-metal / black-metal) у
        адзін і той жа кластар, нягледзячы на тое, што жанр НЕ выкарыстоўваўся
        як прыкмета пры кластарызацыі.
        """
        df = pd.DataFrame({"track_genre": metadata["track_genre"].to_numpy(), "cluster": np.asarray(labels)})
        df = df[df["cluster"] != -1]
        top = df["track_genre"].value_counts().head(top_genres).index
        df = df[df["track_genre"].isin(top)]

        crosstab = pd.crosstab(df["track_genre"], df["cluster"], normalize="index") * 100
        crosstab = crosstab.loc[top]

        fig, ax = plt.subplots(figsize=(max(7, 0.8 * crosstab.shape[1]), max(6, 0.4 * crosstab.shape[0])), facecolor=BG)
        ax.set_facecolor(PANEL)
        sns.heatmap(crosstab, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar_kws={"label": "% трэкаў жанру"},
                    linewidths=0.5, linecolor=GRID)
        self.style_ax(ax, title=f"Размеркаванне жанраў па кластарах, % ({model_name})",
                       xlabel="Кластар", ylabel="Жанр")
        plt.setp(ax.get_yticklabels(), color=INK, rotation=0)
        plt.setp(ax.get_xticklabels(), color=INK)
        plt.tight_layout()
        return self._save(fig, filename)
