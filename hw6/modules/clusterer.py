from __future__ import annotations
import pandas as pd
import numpy as np



from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

import time
from typing import Literal

class ClustererBench:

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        # self.silhouette_sample = 5000
        self.fit_times_: dict[str, float] = {}
    

    def kmeans_k_scan(self, x: np.ndarray, k_range: range, silhouette_sample: int = 5000) -> pd.DataFrame:
        """
        Скануе колькасць кластараў k для KMeans.
        Вылічвае інерцыю (для Elbow) і Silhouette Score.
        """
        rows = []
        print(f"[KMeans Scan] Пачынаецца падбор k ў дыяпазоне {list(k_range)}...")
        
        for k in k_range:
            # Ініцыялізуем KMeans (выкарыстоўваем n_init="auto" для сучасных версій)
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            labels = kmeans.fit_predict(x)
            
            # 1. Метрыка для метаду локця (Inertia / WCSS)
            inertia = float(kmeans.inertia_)
            
            # 2. Метрыка Silhouette (абмяжоўваем sample_size дзеля хуткасці)
            sil = float(silhouette_score(
                x, labels, 
                sample_size=min(silhouette_sample, len(x)), 
                random_state=self.random_state
            ))
            
            rows.append({
                "k": k,
                "inertia": inertia,
                "silhouette": sil
            })
            print(f"  k={k:<2} | Inertia: {inertia:.2f} | Silhouette: {sil:.4f}")
            
        return pd.DataFrame(rows)

        
    def fit_kmeans(self, x: np.ndarray, n_clusters: int) -> np.ndarray:
        """Навучае фінальную мадэль KMeans і вяртае масіў ідэнтыфікатараў кластараў."""

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")

        # return kmeans.fit_predict(x)
        start_time = time.perf_counter() # Старт
        labels = kmeans.fit_predict(x)
        end_time = time.perf_counter()   # Фініш
        
        # Запісваем у слоўнік
        self.fit_times_["KMeans"] = float(end_time - start_time)
        return labels
     
    def gmm_component_scan(self, x: np.ndarray, k_range: range, covariance_type: Literal["full", "tied", "diag", "spherical"] = "diag") -> pd.DataFrame:
        """
        Скануе колькасць кампанентаў k для GaussianMixture.
        Вылічвае інфармацыйныя крытэрыі BIC і AIC.
        """
        rows = []
        print(f"[GMM Scan] Пачынаецца падбор кампанентаў у дыяпазоне {list(k_range)}...")
        
        for k in k_range:
            # Ініцыялізацыя мадэлі сумесяў Гаўсіяна
            gmm = GaussianMixture(
                n_components=k, 
                covariance_type=covariance_type, 
                random_state=self.random_state
            )
            gmm.fit(x)
            
            # Спампоўваем убудаваныя значэнні крытэрыяў
            bic = float(gmm.bic(x))
            aic = float(gmm.aic(x))
            
            rows.append({
                "k": k,
                "bic": bic,
                "aic": aic
            })
            print(f"  k={k:<2} | BIC: {bic:.2f} | AIC: {aic:.2f}")
            
        return pd.DataFrame(rows)

    def fit_gmm(self, x: np.ndarray, n_components: int, covariance_type: Literal["full", "tied", "diag", "spherical"] = "diag") -> np.ndarray:
        """Навучае фінальную мадэль GaussianMixture і зварачае масіў пазнак кластараў."""
        gmm = GaussianMixture(
            n_components=n_components, 
            covariance_type=covariance_type, 
            random_state=self.random_state
        )
        # gmm.fit(x)
        # return gmm.predict(x)

        start_time = time.perf_counter()
        gmm.fit(x)
        labels = gmm.predict(x)
        end_time = time.perf_counter()
        
        self.fit_times_["GMM"] = float(end_time - start_time)
        return labels
       
    # Шум (-1): HDBSCAN — гэта шчыльнасны метад, які пазначыць частку вашых трэкаў Spotify як "шум" (label = -1), 
    # калі яны ляжаць далёка ад асноўных скапленняў. Ваша функцыя compile_comparison_table, якую мы разбіралі вышэй, 
    # ужо ідэальна гатова апрацаваць гэтую маску mask = labels != -1.
    # Параметры: Галоўныя параметры для падбору ў ім — гэта min_cluster_size (мінімальны памер кластара) і min_samples.


    def hdbscan_min_cluster_size_scan(
        self, x_sample: np.ndarray, pca_components: int = 10, sizes: list[int] = [150, 250, 400, 600, 900]
    ) -> pd.DataFrame:
        """
        Скануе ўплыў min_cluster_size на вынікі HDBSCAN (на падвыбарцы дадзеных).
        Папярэдне зніжае размернасць праз PCA.
        """
        print(f"[HDBSCAN Scan] Зніжэнне размернасці да {pca_components} кампанент праз PCA...")
        # 1. Сціскаем прастору прыкмет, каб шчыльнасны метад працаваў эфектыўна
        pca = PCA(n_components=pca_components, random_state=self.random_state)
        x_pca = pca.fit_transform(x_sample)
        
        rows = []
        print(f"[HDBSCAN Scan] Пачынаецца сканаванне min_cluster_size: {sizes}...")
        
        for size in sizes:
            # Ініцыялізацыя HDBSCAN. Параметр min_samples можна зафіксаваць альбо звязаць з size
            hdb = HDBSCAN(min_cluster_size=size, min_samples=25, algorithm="brute", copy=True, n_jobs=-1)
            labels = hdb.fit_predict(x_pca)
            
            # Разлічваем колькасць кластараў (выключаючы шум -1)
            unique_labels = set(labels)
            if -1 in unique_labels:
                n_clusters = len(unique_labels) - 1
            else:
                n_clusters = len(unique_labels)
                
            # Падлічваем долю шуму ў адсотках
            n_noise = int((labels == -1).sum())
            noise_share = float((n_noise / len(labels)) * 100)
            
            rows.append({
                "min_cluster_size": size,
                "n_clusters": n_clusters,
                "noise_share_%": noise_share
            })
            print(f"  Size: {size:<3} | Вызначана кластараў: {n_clusters:<2} | Доля шуму: {noise_share:.2f}%")
            
        return pd.DataFrame(rows)
 
    
    def fit_hdbscan(self, x: np.ndarray, pca_components: int = 10, min_cluster_size: int = 400, min_samples: int = 25) -> np.ndarray:
        """
        Выконвае фінальнае навучанне HDBSCAN на поўным наборы дадзеных з папярэднім PCA.
        Вяртае масіў пазнак кластараў (дзе шумавыя трэкі пазначаны як -1).
        """
        print(f"[HDBSCAN] Фінальны фіт на ўсіх дадзеных... PCA={pca_components}, min_cluster_size={min_cluster_size}")
        
        # 1. Прымяняем PCA да ўсяго датасэту
        pca = PCA(n_components=pca_components, random_state=self.random_state)
        x_pca = pca.fit_transform(x)
        
        # 2. Навучаем фінальны HDBSCAN algorithm="brute"
        hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, algorithm="brute", copy=True, n_jobs=-1)
        labels = hdb.fit_predict(x_pca)
        
        # Справаздача аб выніках у кансоль
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - 1 if -1 in unique_labels else len(unique_labels)
        n_noise = int((labels == -1).sum())
        
        print(f"[HDBSCAN Done] Знойдзена кластараў: {n_clusters}. Радкоў у шуме: {n_noise} ({ (n_noise/len(labels))*100 :.2f}%)")
        return labels
   