"""
BayesianRegressionTuner — аналаг BayesianTuner (tuner.py) для рэгрэсіі PageValues.

Адрозненні ад класіфікацыйнага tuner.py:
  - Мэтавая функцыя мінімізуе RMSE (не максімізуе PR-AUC), таму
    study = optuna.create_study(direction="minimize")
  - Прасторы пошуку пабудаваны для LightGBM/XGBoost/CatBoost (бустынгі
    з Early Stopping), а таксама DecisionTree/ExtraTrees/Ridge (лёгкія
    мадэлі без Early Stopping)
  - Клон базавай мадэлі бярэцца НАПРАМУЮ з bench.models[name], а не з
    bench.fitted[name] — у рэгрэсіі няма scale_pos_weight, дзеля якога
    арыгінальны tuner.py патрабаваў ужо навучаны pipeline
  - Фолды будуюцца праз тыя ж стратыфікаваныя біны нуль/квантылі
    PageValues, што і ў RegressorBench, каб пошук гіперпараметраў не
    трапляў выпадкова на "лёгкі" ці "цяжкі" фолд
"""
from __future__ import annotations

import json
import numpy as np
import optuna
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

from typing import Any

from modules.regressor import make_zero_inflated_bins

optuna.logging.set_verbosity(optuna.logging.WARNING)

EARLY_STOPPING_ROUNDS = 30
BOOSTING_TRIALS = 70
LIGHT_TRIALS = 25


class BayesianRegressionTuner:
    """Оптуна-аптымізацыя гіперпараметраў для RegressorBench (мінімізацыя RMSE)."""

    def __init__(self, bench_instance: Any, random_state: int = 42):
        self.bench = bench_instance
        self.random_state = random_state
        self.best_params_report: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Прастора пошуку па назве мадэлі
    # ------------------------------------------------------------------ #
    def _suggest_params(self, trial: optuna.Trial, name: str) -> dict[str, Any]:
        seed = self.random_state

        if "light" in name.lower():
            return {
                "n_estimators": 500,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 31),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "objective": "regression",
                "metric": "rmse",
                "n_jobs": -1,
                "random_state": seed,
                "verbosity": -1,
            }

        if "xgb" in name.lower():
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "n_jobs": -1,
                "random_state": seed,
            }

        if "cat" in name.lower():
            return {
                "iterations": 500,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": seed,
                "verbose": False,
                "allow_writing_files": False,
            }

        if name == "DecisionTree":
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse"]),
                "random_state": seed,
            }

        if name == "ExtraTrees":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
                "max_features": trial.suggest_float("max_features", 0.4, 1.0),
                "n_jobs": -1,
                "random_state": seed,
            }

        if name == "Ridge":
            return {
                "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
                "random_state": seed,
            }

        return {}

    # ------------------------------------------------------------------ #
    # Мэтавая функцыя: сярэдні RMSE па 5 фолдах (менш = лепш)
    # ------------------------------------------------------------------ #
    def _objective(self, trial: optuna.Trial, name: str, x: pd.DataFrame, y: pd.Series) -> float:
        params = self._suggest_params(trial, name)
        if not params:
            return float("inf")

        bins = make_zero_inflated_bins(y)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        fold_scores: list[float] = []

        y_np = y.to_numpy()

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x, bins)):
            preprocessor = self.bench._get_adapted_preprocessor(name)

            reg_model = clone(self.bench.models[name])
            reg_model.set_params(**params)

            x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
            y_tr, y_va = y_np[train_idx], y_np[val_idx]

            x_tr_trans = preprocessor.fit_transform(x_tr, y_tr)
            x_va_trans = preprocessor.transform(x_va)

            if "light" in name.lower():
                from lightgbm import early_stopping, log_evaluation
                reg_model.fit(
                    x_tr_trans, y_tr,
                    eval_set=[(x_va_trans, y_va)],
                    eval_metric="rmse",
                    callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
                )
            elif "xgb" in name.lower():
                reg_model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
                reg_model.fit(x_tr_trans, y_tr, eval_set=[(x_va_trans, y_va)], verbose=False)
            elif "cat" in name.lower():
                reg_model.fit(
                    x_tr_trans, y_tr,
                    eval_set=(x_va_trans, y_va),
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=False,
                )
            else:
                reg_model.fit(x_tr_trans, y_tr)

            preds = np.asarray(reg_model.predict(x_va_trans)).ravel()
            rmse = float(np.sqrt(mean_squared_error(y_va, preds)))
            fold_scores.append(rmse)

            current_mean = float(np.mean(fold_scores))
            trial.report(current_mean, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    # ------------------------------------------------------------------ #
    # Запуск пошуку і кэшаванне
    # ------------------------------------------------------------------ #
    def tune_model(self, name: str, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        n_trials = BOOSTING_TRIALS if any(k in name.lower() for k in ("light", "xgb", "cat")) else LIGHT_TRIALS

        print(f"\n[Optuna] Старт аптымізацыі для {name} ({n_trials} trials, мінімізацыя RMSE)...")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )

        study.optimize(lambda trial: self._objective(trial, name, x, y), n_trials=n_trials, show_progress_bar=True)

        print(f"[Optuna] Найлепшы RMSE для {name}: {study.best_value:.4f}")
        self.best_params_report[name] = study.best_params

        return study.best_params

    def tune_or_load(
        self,
        best_name: str,
        x: pd.DataFrame,
        y: pd.Series,
        cache_filename: str = "output/optuna_best_params_regression.json",
    ) -> dict[str, Any]:
        """Кэш-менеджэр: чытае JSON, калі ён ужо ёсць, іначай запускае Optuna і захоўвае вынік."""
        cache_path = Path(cache_filename)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"\n[Tuner Cache] Знойдзены захаваны файл: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        matched_key = None
        for pipe_key in self.bench.models.keys():
            if best_name.lower() in pipe_key.lower():
                matched_key = pipe_key
                break

        if matched_key is None:
            print(f"[Tuner Warning] Мадэль '{best_name}' не знойдзена ў bench.models.")
            return {}

        best_hyperparams = self.tune_model(matched_key, x, y)
        optimized_results = {matched_key: best_hyperparams}

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(optimized_results, f, indent=4)
        print(f"[Tuner Cache] Параметры захаваны ў: {cache_path}")

        return optimized_results
