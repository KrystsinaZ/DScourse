"""Параўнанне і інтэрпрэтацыя вынікаў кластарызацыі."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)


class ClusteringEvaluator:
    """Метрыкі якасці кластарызацыі і профіль кластараў."""

    def __init__(self, silhouette_sample: int = 8000, random_state: int = 42) -> None:
        self.silhouette_sample = silhouette_sample
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    # 1. Табліца параўнання метадаў
    # ------------------------------------------------------------------ #
    def compile_comparison_table(
        self, x: np.ndarray, labels_dict: dict[str, np.ndarray], fit_times: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Зводная табліца па ўсіх метадах:
          - Silhouette (вышэй=лепш, [-1;1]) — на падвыбарцы дзеля хуткасці;
          - Davies-Bouldin (ніжэй=лепш, [0;+inf));
          - Calinski-Harabasz (вышэй=лепш, [0;+inf));
          - n_clusters і, калі ёсць, доля пазначаных як "шум" (HDBSCAN, label=-1).
        Для HDBSCAN шумавыя кропкі (-1) выключаюцца з разліку silhouette/DB/CH,
        бо гэтыя метрыкі не разлічаны на паняцце "адсутнасць кластара".
        """
        rows = []
        for name, labels in labels_dict.items():
            labels = np.asarray(labels)
            mask = labels != -1  # прыбіраем шум HDBSCAN пры разліку метрык
            n_noise = int((~mask).sum())
            x_eval, lab_eval = x[mask], labels[mask]
            n_clusters = len(set(lab_eval))

            if n_clusters < 2:
                rows.append({"model": name, "n_clusters": n_clusters, "noise_share_%": 100 * n_noise / len(labels),
                             "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
                             "fit_seconds": (fit_times or {}).get(name, np.nan)})
                continue

            sil = silhouette_score(x_eval, lab_eval, sample_size=min(self.silhouette_sample, len(lab_eval)),
                                    random_state=self.random_state)
            db = davies_bouldin_score(x_eval, lab_eval)
            ch = calinski_harabasz_score(x_eval, lab_eval)

            rows.append({
                "model": name, "n_clusters": n_clusters, "noise_share_%": 100 * n_noise / len(labels),
                "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
                "fit_seconds": (fit_times or {}).get(name, np.nan),
            })

        table = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
        return table

    # ------------------------------------------------------------------ #
    # 2. Параўнанне з жанрам (знешні "арыенцір", не мэта аптымізацыі)
    # ------------------------------------------------------------------ #
    def compare_with_genre(self, genre: pd.Series, labels_dict: dict[str, np.ndarray]) -> pd.DataFrame:
        """
        ARI/NMI паміж знойдзенымі кластарамі і жанрам track_genre.
        УВАГА: гэта НЕ мэтавая метрыка выбару мадэлі (кластарызацыя —
        unsupervised, а 114 жанраў фармуюцца па маркетынгавых, а не чыста
        акустычных прынцыпах, і адна песня можа насіць некалькі жанраў).
        Але супадзенне дае карысны арыенцір: ці лавяць кластары буйныя
        музычныя сямействы (напр. класіка/акустыка супраць метал/электронікі).
        """
        rows = []
        genre_codes = genre.astype("category").cat.codes.to_numpy()
        for name, labels in labels_dict.items():
            labels = np.asarray(labels)
            mask = labels != -1
            ari = adjusted_rand_score(genre_codes[mask], labels[mask])
            nmi = normalized_mutual_info_score(genre_codes[mask], labels[mask])
            rows.append({"model": name, "ARI_vs_genre": ari, "NMI_vs_genre": nmi})
        return pd.DataFrame(rows).sort_values("NMI_vs_genre", ascending=False)

    # ------------------------------------------------------------------ #
    # 3. Профіль кластараў: сярэднія значэнні прыкмет + дамінантныя жанры
    # ------------------------------------------------------------------ #
    def profile_clusters(
        self, x_raw: pd.DataFrame, labels: np.ndarray, metadata: pd.DataFrame, *, top_genres: int = 3,
    ) -> pd.DataFrame:
        """Для кожнага кластара: памер, сярэднія аўдыё-прыкметы, найчасцейшыя жанры."""
        df = x_raw.copy()
        df["cluster"] = labels
        df["track_genre"] = metadata["track_genre"].to_numpy()

        numeric_cols = [c for c in x_raw.columns if x_raw[c].dtype != object]
        agg = df.groupby("cluster")[numeric_cols].mean()
        agg["n_tracks"] = df.groupby("cluster").size()
        agg["share_%"] = 100 * agg["n_tracks"] / len(df)

        top_genre_strs = []
        for cluster_id, group in df.groupby("cluster"):
            counts = group["track_genre"].value_counts().head(top_genres)
            top_genre_strs.append(
                ", ".join(f"{g} ({c})" for g, c in counts.items())
            )
        agg["top_genres"] = top_genre_strs

        return agg.reset_index().sort_values("cluster")

    def pick_best(self, comparison_table: pd.DataFrame) -> str:
        """Абірае мадэль з найвышэйшым Silhouette Score сярод сапраўдных (не-дэгенератыўных) кластарызацый."""
        valid = comparison_table.dropna(subset=["silhouette"])
        best_row = valid.sort_values("silhouette", ascending=False).iloc[0]
        return str(best_row["model"])
