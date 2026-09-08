"""
Кластарызацыя Spotify Tracks Dataset (114 000 трэкаў, 114 жанраў).
Модулі clusterer.py, clustering_evaluation.py, dataset_clustering.py і
visualization_clustering.py
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from modules.dataset_clustering import PrepareClusteringDataset
from modules.clusterer import ClustererBench
from modules.clustering_evaluation import ClusteringEvaluator
from modules.visualization_clustering import ClusteringVisualizer

ROOT = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DATA = ROOT / "data"
OUT = ROOT / "output"

RANDOM_STATE = 42
EMBEDDING_SAMPLE_SIZE = 12000   # для t-SNE/UMAP: разлік на ўсіх 114k радках непрактычна павольны


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    dataset = PrepareClusteringDataset(DATA)
    bench = ClustererBench(random_state=RANDOM_STATE)
    evaluator = ClusteringEvaluator(silhouette_sample=8000, random_state=RANDOM_STATE)
    viz = ClusteringVisualizer(OUT)
    plots_list: list[Path] = []

    # ================================================================== #
    section("1. Загрузка датасета Spotify Tracks")
    frame = dataset.load_csv(file_name="spotify-tracks-dataset.csv")

    # ================================================================== #
    section("2. EDA: прапушчаныя значэнні, дублікаты, размеркаванні, карэляцыі, выкіды")
    dataset.report_missing_and_duplicates(frame)

    plots_list.append(viz.plot_feature_distributions(
        frame, dataset.continuous_cols, filename="01_feature_distributions.png",
    ))
    plots_list.append(viz.plot_outlier_boxplots(
        frame, dataset.continuous_cols, filename="02_outlier_boxplots.png",
    ))
    plots_list.append(viz.plot_feature_correlation_matrix(
        frame, dataset.continuous_cols, filename="03_feature_correlation_matrix.png",
    ))
    print("[Visualizer] Захаваны графікі размеркаванняў, выкідаў і карэляцыі.")

    # ================================================================== #
    section("3. Падрыхтоўка прыкмет: ачыстка, OneHotEncoder (НЕ LabelEncoder!), StandardScaler")
    result = dataset.prepare_for_clustering(frame)
    x, x_raw, metadata, feat_names = result.x_scaled, result.x_raw, result.metadata, result.feature_names_out

    with open(OUT / "prepared_data.pkl", "wb") as f:
        pickle.dump(result, f)

    # ================================================================== #
    section("4. Базавы метад: KMeans + падбор k (elbow + Silhouette Score)")
    kmeans_scan = bench.kmeans_k_scan(x, k_range=range(2, 16), silhouette_sample=5000)
    kmeans_scan.to_csv(OUT / "kmeans_k_scan.csv", index=False)

    CHOSEN_K = 9  # лакальны максімум Silhouette сярод суседніх k
    plots_list.append(viz.plot_elbow_and_silhouette(
         kmeans_scan, chosen_k=CHOSEN_K, filename="04_kmeans_elbow_silhouette.png",
    ))

    labels_kmeans = bench.fit_kmeans(x, n_clusters=CHOSEN_K)
    print(f"[KMeans] Абрана k={CHOSEN_K}. Памеры кластараў:\n{pd.Series(labels_kmeans).value_counts().sort_index()}")

    # ================================================================== #
    section("5. Больш прыдатныя метады: GaussianMixture і HDBSCAN")

    print("\n5.1. GaussianMixture — падбор колькасці кампанентаў праз BIC/AIC")
    gmm_scan = bench.gmm_component_scan(x, k_range=range(2, 16), covariance_type="diag")
    gmm_scan.to_csv(OUT / "gmm_component_scan.csv", index=False)
    # print(
    #     "[GMM] АГУЛЬНАЯ ЗАЎВАГА: BIC/AIC манатонна змяншаюцца без унутранага мінімуму ў "
    #     "дыяпазоне k=2..15. Гэта чакана: 17 з 30 прыкмет — гэта OneHot-дамі (0/1), якія НЕ "
    #     "маюць нармальнага размеркавання, і Gaussian-кампанента можа неабмежавана "
    #     "'падганяць' да іх дысперсію, штучна паляпшаючы правападобнасць. Каб карэктна "
    #     "параўнаць GMM з KMeans (a не проста абраць найбольшы k), бяром аднолькавы "
    #     "n_components = 9, як і ў KMeans."
    # )
    plots_list.append(viz.plot_gmm_bic_aic(gmm_scan, chosen_k=CHOSEN_K, filename="05_gmm_bic_aic.png"))

    labels_gmm = bench.fit_gmm(x, n_components=CHOSEN_K, covariance_type="diag")
    print(f"[GMM] Памеры кластараў:\n{pd.Series(labels_gmm).value_counts().sort_index()}")

    print("\n5.2. HDBSCAN — шчыльнасны метад, сам вызначае колькасць кластараў і адзначае шум")
    # Дыягностыку min_cluster_size робім на падвыбарцы 30k (хутка), фінальны фіт - на ўсіх даных
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx_hdb = rng.choice(len(x), size=30000, replace=False)
    hdbscan_scan = bench.hdbscan_min_cluster_size_scan(
        x[sample_idx_hdb], pca_components=10, sizes=[150, 250, 400, 600, 900],
    )
    hdbscan_scan.to_csv(OUT / "hdbscan_min_cluster_size_scan.csv", index=False)
    plots_list.append(viz.plot_hdbscan_scan(hdbscan_scan, filename="06_hdbscan_scan.png"))

    labels_hdbscan = bench.fit_hdbscan(x, pca_components=10, min_cluster_size=400, min_samples=25)

    # ================================================================== #
    section("6. Параўнанне метадаў кластарызацыі")
    all_labels = {"KMeans": labels_kmeans, "GMM": labels_gmm, "HDBSCAN": labels_hdbscan}
    # all_labels = {"KMeans": labels_kmeans, "GMM": labels_gmm}

    comparison_table = evaluator.compile_comparison_table(x, all_labels, fit_times=bench.fit_times_)
    print("\n[Comparison] Табліца метрык якасці кластарызацыі:")
    print(comparison_table.to_string(index=False))
    comparison_table.to_csv(OUT / "clustering_comparison_table.csv", index=False)

    genre_comparison = evaluator.compare_with_genre(metadata["track_genre"], all_labels)
    print("\n[Comparison] ARI/NMI супраць track_genre (даведачны, не мэтавы, арыенцір):")
    print(genre_comparison.to_string(index=False))
    genre_comparison.to_csv(OUT / "clustering_vs_genre.csv", index=False)

    plots_list.append(viz.plot_metrics_comparison(comparison_table, filename="07_metrics_comparison.png"))

    best_name = evaluator.pick_best(comparison_table)
    print(f"\n[Comparison] Найлепшая мадэль па Silhouette Score: {best_name}")

    # ================================================================== #
    section("7. Візуалізацыя вынікаў: PCA (усе даныя) + t-SNE/UMAP (падвыбарка)")

    print("7.1. PCA(2) на ўсіх 114k радках")
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    embedding_pca = pca2.fit_transform(x)
    print(f"     Тлумачаная дысперсія PC1+PC2 = {sum(pca2.explained_variance_ratio_):.3f}")

    for model_name, labels in all_labels.items():
        plots_list.append(viz.plot_cluster_scatter_2d(
            embedding_pca, labels, method_name="PCA", model_name=model_name,
            filename=f"08_pca_scatter_{model_name}.png",
        ))

    print(f"\n7.2. t-SNE і UMAP на выпадковай падвыбарцы {EMBEDDING_SAMPLE_SIZE} радкоў "
          f"(поўны разлік на 114k радках вылічальна непрактычны для гэтых метадаў)")
    sample_idx = rng.choice(len(x), size=EMBEDDING_SAMPLE_SIZE, replace=False)
    x_sample = x[sample_idx]

    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca", perplexity=30, n_jobs=-1)
    embedding_tsne = tsne.fit_transform(x_sample)
    embedding_tsne = np.asarray(embedding_tsne, dtype=np.float64)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)

    embedding_umap = reducer.fit_transform(x_sample)
    embedding_umap = np.asarray(embedding_umap, dtype=np.float64)
    
    for model_name, labels in all_labels.items():
        labels_sample = np.asarray(labels)[sample_idx]
        plots_list.append(viz.plot_cluster_scatter_2d(
            embedding_tsne, labels_sample, method_name="t-SNE", model_name=model_name,
            filename=f"09_tsne_scatter_{model_name}.png",
        ))
        plots_list.append(viz.plot_cluster_scatter_2d(
            embedding_umap, labels_sample, method_name="UMAP", model_name=model_name,
            filename=f"10_umap_scatter_{model_name}.png",
        ))
    print("[Visualizer] Захаваны PCA/t-SNE/UMAP-праекцыі для ўсіх трох метадаў.")

    # ================================================================== #
    section(f"8. Інтэрпрэтацыя кластараў найлепшай мадэлі ({best_name})")
    best_labels = all_labels[best_name]

    plots_list.append(viz.plot_cluster_sizes(best_labels, model_name=best_name, filename="11_cluster_sizes.png"))
    plots_list.append(viz.plot_cluster_feature_heatmap(
        x_raw, best_labels, dataset.continuous_cols, model_name=best_name, filename="12_cluster_feature_heatmap.png",
    ))
    plots_list.append(viz.plot_genre_cluster_crosstab(
        metadata, best_labels, model_name=best_name, top_genres=20, filename="13_genre_cluster_crosstab.png",
    ))

    profile_table = evaluator.profile_clusters(x_raw, best_labels, metadata, top_genres=4)
    print(f"\n[Profile] Музычны профіль кластараў ({best_name}):")
    print(profile_table.to_string(index=False))
    profile_table.to_csv(OUT / f"cluster_profile_{best_name}.csv", index=False)

    # Профіль і для базавага KMeans, каб можна было наглядна параўнаць з абраным лепшым метадам
    if best_name != "KMeans":
        profile_kmeans = evaluator.profile_clusters(x_raw, labels_kmeans, metadata, top_genres=4)
        profile_kmeans.to_csv(OUT / "cluster_profile_KMeans.csv", index=False)

    # ================================================================== #
    section("Падрахуем")
    print(f"Крыніца даных: {DATA / 'spotify-tracks-dataset.csv'}")
    print(f"Абраны найлепшы метад па Silhouette Score: {best_name}")
    print("Графікі і табліцы захаваны ў:", OUT)
    for p in plots_list:
        print(f"  • {p.name}")


if __name__ == "__main__":
    main()
