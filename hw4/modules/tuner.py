from dataclasses import dataclass, field
import json
import numpy as np
import optuna
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from typing import Any


# Адключаем лішнія лагі Optuna, каб не засмечваць тэрмінал
optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class TuneResult:
    """Выніковыя параметры оптуны."""

    source: str
    params: dict[str, Any]
    cv_pr_auc: float
    history: list[float] = field(default_factory=list)

class BayesianTuner:
    """
    Модуль аптымізацыі гіперпараметраў на базе Optuna (Bayesian Optimization).
    Падтрымлівае адаптыўны пошук, бесперапынныя прасторы і прунінг.
    """
    def __init__(self, bench_instance: Any, random_state: int = 42):
        self.bench = bench_instance
        self.random_state = random_state
        self.best_params_report: dict[str, dict[str, Any]] = {}

    def _objective(self, trial: optuna.Trial, name: str, x: pd.DataFrame, y: pd.Series) -> float:
        """Мэтавая функцыя для Optuna па максімізацыі 5-фолдавай PR-AUC на ЧЫСТЫМ прэпрацэсары."""
        params: dict[str, Any] = {}

        # Вызначаем scale_pos_weight бяспечна з мінулага пайплайна
        base_pipeline = self.bench.fitted[name]
        scale_pos = base_pipeline.named_steps["clf"].get_params().get("scale_pos_weight", 1.0)

        # max_depth = trial.suggest_int("max_depth", 3, 6)
        # # num_leaves выбіраецца з улікам абранай глыбіні, але не больш за 17
        # max_leaves = min(17, 2**max_depth) 
        # num_leaves = trial.suggest_int("num_leaves", 2, max_leaves)
        
        # 1. Вызначаем адаптыўную прастору (выпраўлена праверка імёнаў праз .lower())
        if "light" in name.lower():
            params = {
                "n_estimators": 500,  
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 17),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "class_weight": "balanced",
                "objective": "binary",
                "metric": "auc",  
                "n_jobs": -1,  # ВАЖНА: 1 замест -1, каб Tkinter не падаў
                "random_state": self.random_state,
                "verbosity": -1
            }

        elif "xgb" in name.lower():
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
                "scale_pos_weight": scale_pos,                                          
                "eval_metric": "aucpr",                                                 
                "n_jobs": -1  # Ізаляцыя патокаў для бяспекі GUI
            }
        elif name == "DecisionTree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "class_weight": "balanced"
            }            
        else:
            # Калі мадэль не падтрымліваецца ў гэтым Objective
            return 0.0

        # === 2. КРОС-ВАЛІДАЦЫЯ З РЭАЛЬНЫМ EARLY STOPPING УНУТРЫ OPTUNA ===
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        fold_scores = []
        y_np = y.to_numpy().astype(int)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x, y_np)):
            
            # ВЫПРАЎЛЕННЕ ВЫЦЕКУ ДАДЗЕНЫХ:
            # Ствараем АБСАЛЮТНА ЧЫСТЫ, свежы прэпрацэсар менавіта для гэтага фолду
            preprocessor = self.bench._get_adapted_preprocessor(name)
            
            # Кланіруем толькі ГОЛЫ класіфікатар з мінулага пайплайна і задаем яму параметры Optuna
            clf_model = clone(base_pipeline.named_steps["clf"])
            clf_model.set_params(**params)
            
            x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
            y_tr, y_va = y_np[train_idx], y_np[val_idx]

            # Прэпрацэсінг адбываецца ў строгай ізаляцыі ўнутры фолду
            x_tr_trans = preprocessor.fit_transform(x_tr, y_tr)
            x_va_trans = preprocessor.transform(x_va)

            # Запускаем навучанне з улікам ранняга прыпынення (30 раўндаў)
            if "light" in name.lower():
                clf_model.set_params(early_stopping_rounds=30, verbose=-1)
                clf_model.fit(x_tr_trans, y_tr, eval_set=[(x_va_trans, y_va)])
            elif "xgb" in name.lower():
                clf_model.set_params(early_stopping_rounds=30, verbose=0)
                clf_model.fit(x_tr_trans, y_tr, eval_set=[(x_va_trans, y_va)], verbose=False)

            # Вылічэнне PR-AUC на валідацыйным масіве
            proba = clf_model.predict_proba(x_va_trans)[:, 1]
            score = float(average_precision_score(y_va, proba))
            fold_scores.append(score)

            # === УБУДАНЫ ПРУНІНГ OPTUNA ===
            current_mean = np.mean(fold_scores)
            trial.report(float(current_mean), fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    def _objective2(self, trial: optuna.Trial, name: str, x: pd.DataFrame, y: pd.Series) -> float:
        """Мэтавая функцыя для Optuna па максімізацыі 5-фолдавай PR-AUC."""
        # 1. Вызначаем адаптыўную прастору гіперпараметраў у залежнасці ад мадэлі
        params: dict[str, Any] = {}

        # Дастаем структуру Pipeline для гэтай мадэлі, якую мы захавалі ў мінулых кроках
        base_pipeline = self.bench.fitted[name]
        scale_pos = base_pipeline.named_steps["clf"].get_params().get("scale_pos_weight", 1.0)
        if name == "LightGBM":
            params = {

            "n_estimators": 500,  # Задаем з запасам, бо Early Stopping спыніць навучанне своечасова
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            
            # Строга абмяжоўваем колькасць лісцікаў (дэфолт 31 — занадта шмат для ўмеранага дысбалансу)
            "num_leaves": trial.suggest_int("num_leaves", 15, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            
            # Жорсткі мінімум аб'ектаў у лісце, каб пазбегнуць ізаляцыі рэдкіх пакупнікоў
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            
            # Дадаем рэгулярызацыю (L1 і L2 штрафы) для стабілізацыі ваг дрэў
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            
            # Кантроль выбаркі прыкмет і радкоў (барацьба з гэпам)
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            
            "class_weight": "balanced",
            # ВАЖНА: забараняем унутраныя вагі, пакідаем баланс натуральным
            # "scale_pos_weight": 1.0,
            
            "objective": "binary",
            "metric": "auc",  # альбо пакідаем пустым, бо выкарыстоўваем eval_metric вонку
            "n_jobs": -1,
            "random_state": self.random_state,
            "verbose": -1
            }

        # if name == "LightGBM":
        #     params = {
        #         # Абмяжоўваем колькасць дрэў і зніжаем хуткасць навучання для плаўнасці
        #         "n_estimators": trial.suggest_int("n_estimators", 100, 450, step=50),
        #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            
        #         # ЖОРСТКАЕ АБМЕЖАВАННЕ СКЛАДАНАСЦІ (Збіваем перанавучанне!)
        #         "num_leaves": trial.suggest_int("num_leaves", 20, 96),          # шчыльна вакол 15
        #         "max_depth": -1, # trial.suggest_int("max_depth", 4, 10),             # Не даем расці глыбей 5 паверхаў
                
        #         # Павялічваем мінімальны памер ліста (абарона ад ізаляваных сесій)
        #         "min_child_samples": trial.suggest_int("min_child_samples", 40, 100), # старт ад 40!
                
        #         # Шчыльны стахастычны адбор
        #         "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        #         "subsample_freq": 1, 
        #         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
                
        #         # Задаем рэальную стартавую рэгулярызацыю (прыбіраем мікра-значэнні 1e-8)
        #         "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0, log=True),
        #         "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 15.0, log=True), # Моцны L2 штраф ваг дрэў
                
        #         "class_weight": "balanced",
        #         "verbose": -1,
        #         "n_jobs": 1
        #     }

        # if name == "LightGBM":
        #     params = {
        #         "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        #         "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        #         "max_depth": trial.suggest_int("max_depth", 3, 12),
        #         "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        #         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        #         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        #         "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        #         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        #         "verbose": -1,
        #         "n_jobs": 1 #"n_jobs": -1 # 1 выключыла унутраную паралелізацыю, каб патокі не канфліктавалі у бустынгаў
        #     }
        # if name == "LightGBM":
        #     params = {
        #         "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        #         "num_leaves": trial.suggest_int("num_leaves", 7, 31),          # Шчыльны пошук вакол 15
        #         "max_depth": trial.suggest_int("max_depth", 3, 6),             # Абмяжоўваем глыбіню дрэва
        #         "min_child_samples": trial.suggest_int("min_child_samples", 30, 150), # Павялічаны інтэрвал
        #         "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        #         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        #         "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
        #         "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True), # Моцны L2 штраф
        #         "class_weight": "balanced",
        #         "verbose": -1,
        #         "n_jobs": 1 
        #     }            
        elif name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
                "scale_pos_weight": scale_pos,                                          # Фіксуем вылічаны на дадзеных каэфіцыент балансу
                "eval_metric": "aucpr",                                                 # Мэтавая метрыка
                "n_jobs": 1 #"n_jobs": -1                                                             # Ізаляцыя патокаў для бяспекі n_jobs=-1
                
            }
        # elif name == "XGBoost":
        #     params = {
        #         "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        #         "max_depth": trial.suggest_int("max_depth", 3, 6),                     # Шчыльны кантроль глыбіні дрэва
        #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True), # Больш плыўны крок навучання
        #         "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),       # Абарона ад ізаляцыі рэдкіх сесій
        #         "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        #         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
        #         "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),     # Адаптыўны падбор L1-штрафу
        #         "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),   # Адаптыўны падбор моцнага L2-штрафу
        #         "scale_pos_weight": scale_pos,                                          # Фіксуем ваш каэфіцыент балансу
        #         "eval_metric": "aucpr",                                                 # Мэтавая метрыка
        #         "n_jobs": 1                                                             # Ізаляцыя патокаў для бяспекі n_jobs=-1
        #     }

        elif name == "DecisionTree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "class_weight": "balanced"
            }
        else:
            # Калі нейкая лінейная або іншая мадэль
            return 0.0

        # === 2. КРОС-ВАЛІДАЦЫЯ З РЭАЛЬНЫМ EARLY STOPPING УНУТРЫ OPTUNA ===
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        fold_scores = []

        y_np = y.to_numpy().astype(int)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x, y_np)):
            # Стварэнне чыстага клона пайплайна на кожны фолд
            pipe = clone(base_pipeline)
            pipe.named_steps["clf"].set_params(**params)
            
            x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
            y_tr, y_va = y_np[train_idx], y_np[val_idx]

            # Вылучаем прэпрацэсар, каб падрыхтаваць лікавы eval_set для Early Stopping
            preprocessor = pipe.named_steps["preprocessor"]
            x_tr_trans = preprocessor.fit_transform(x_tr, y_tr)
            x_va_trans = preprocessor.transform(x_va)

            # Вылучаем сам класіфікатар
            clf_model = pipe.named_steps["clf"]

            # Запускаем навучанне з улікам ранняга прыпынення (30 раўндаў)
            if "light" in name.lower():
                clf_model.set_params(early_stopping_rounds=30, verbose=-1)
                clf_model.fit(x_tr_trans, y_tr, eval_set=[(x_va_trans, y_va)])
            elif "xgb" in name.lower():
                clf_model.set_params(early_stopping_rounds=30, verbose=0)
                clf_model.fit(x_tr_trans, y_tr, eval_set=[(x_va_trans, y_va)], verbose=False)
            else:
                clf_model.fit(x_tr_trans, y_tr)

            # Вылічэнне PR-AUC на валідацыйным масіве
            proba = clf_model.predict_proba(x_va_trans)[:, 1]
            score = float(average_precision_score(y_va, proba))
            fold_scores.append(score)

            # === УБУДАНЫ ПРУНІНГ OPTUNA (ЗАСТАЕЦЦА) ===
            current_mean = np.mean(fold_scores)
            trial.report(float(current_mean), fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    def tune_model(self, name: str, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Запускае баесаўскую аптымізацыю для канкрэтнай мадэлі."""
        # Вызначаем колькасць ітэрацый паводле патрабавання
        if "XGB" in name or "LightGBM" in name:
            n_trials = 70  # 50–100 для бустынгаў
        else:
            n_trials = 20  # 20–30 для лінейных / класічных дрэў

        print(f"\n[Optuna] Старт аптымізацыі для {name} ({n_trials} trials, pruning уключаны)...")
        
        # Выкарыстоўваем MedianPruner для адсячэння безнадзейных камбінацый
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )         
        
        study.optimize(lambda trial: self._objective(trial, name, x, y), n_trials=n_trials, show_progress_bar=True)
        
        print(f"[Optuna] Найлепшы PR-AUC для {name}: {study.best_value:.4f}")
        self.best_params_report[name] = study.best_params

        
        return study.best_params

    def tune_or_load(self, best_name: str, x: pd.DataFrame, y: pd.Series, cache_filename: str = "output/optuna_best_params.json") -> dict[str, Any]:
        """
        Архітэктурны менеджэр: шукае параметры ў JSON-кэшы. 
        Калі файла няма — запускае Optuna, а вынік запісвае ў кэш.
        """

        cache_path = Path(cache_filename)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Спрабуем прачытаць існуючы кэш
        if cache_path.exists():
            print(f"\n[Tuner Cache] Знойдзены захаваны файл з параметрамі: {cache_path}")
            print("[Tuner Cache] Імгненная загрузка лепшых параметраў без паўторных вылічэнняў...")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 2. Калі кэшу няма, праводзім гнуткі пошук ключа ў models
        matched_key = None

        for pipe_key in self.bench.models.keys():
            if best_name.lower() in pipe_key.lower():
                matched_key = pipe_key
                break

        if matched_key is None:
            print(f"[Tuner Warning] Мадэль '{best_name}' не знойдзена ў models. Падбор немагчымы.")
            return {}

        # 3. Запускаем доўгі баесаўскі падбор
        best_hyperparams = self.tune_model(matched_key, x, y)
        
        # Фармуем структуру для захавання
        optimized_results = {matched_key: best_hyperparams}

        # 4. Запісваем вынік у JSON-файл
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(optimized_results, f, indent=4)
        print(f"[Tuner Cache] Параметры паспяхова захаваны ў кэш: {cache_path}")

        return optimized_results
