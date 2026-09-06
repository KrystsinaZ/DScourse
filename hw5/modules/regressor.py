"""
RegressorBench — аналаг ClassifierBench для рэгрэсійнай задачы (таргет PageValues).

Захоўвае тую ж архітэктуру, што і classifier.py:
  - слоўнік шаблонаў мадэляў (self.models)
  - адаптыўны прэпрацэсар (скалер уключаецца толькі для лінейных мадэляў)
  - крос-валідацыя KFold(5) з OOF-прагнозамі
  - Early Stopping для бустынгаў (LightGBM, XGBoost, CatBoost) з захаваннем
    evals_result для тых жа графікаў, што і DataVisualizer ужо ўмее маляваць

ВАЖНА: catboost не ўваходзіць у requirements.txt праекта.
    Устанавіце яго асобна:  pip install catboost
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Any, Dict, cast

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor, early_stopping, log_evaluation, record_evaluation
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50


@dataclass(frozen=True)
class ModelSpec:
    """
    Метаданыя мадэлі: адзіная крыніца праўды замест некалькіх паралельных
    спісаў імёнаў (NO_SCALE_REQUIRED, BOOSTING, COMPLEXITY), якія раней
    даводзілася ўручную сінхранізаваць пры кожным новым метадзе.
    """
    needs_scaling: bool
    supports_early_stopping: bool
    complexity: int  # для tie-break у RegressionEvaluator.pick_best (менш = прасцей)


# Рэестр па змаўчанні. Складаныя/састаўныя мадэлі (напр. TwoStageRegressor)
# рэгіструюцца асобна праз RegressorBench.add_two_stage_model(), а не
# заганяюцца сюды праз падстроку ў імені.
_DEFAULT_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Dummy":        ModelSpec(needs_scaling=False, supports_early_stopping=False, complexity=0),
    "Ridge":        ModelSpec(needs_scaling=True,  supports_early_stopping=False, complexity=1),
    "DecisionTree": ModelSpec(needs_scaling=False, supports_early_stopping=False, complexity=2),
    "ExtraTrees":   ModelSpec(needs_scaling=False, supports_early_stopping=False, complexity=3),
    "LightGBM":     ModelSpec(needs_scaling=False, supports_early_stopping=True,  complexity=4),
    "XGBoost":      ModelSpec(needs_scaling=False, supports_early_stopping=True,  complexity=4),
    "CatBoost":     ModelSpec(needs_scaling=False, supports_early_stopping=True,  complexity=4),
}


def make_zero_inflated_bins(y: pd.Series, n_quantile_bins: int = 4) -> np.ndarray:
    """
    PageValues мае ~78% нулявых значэнняў. Звычайны KFold можа выпадкова
    накідаць нераўнамерную долю нулявых/ненулявых сесій па фолдах.
    Гэтая функцыя стварае "псеўда-клас" для StratifiedKFold:
      бін 0     — усе сесіі з PageValues == 0
      бін 1..n  — квантыльныя бліны сярод ненулявых значэнняў
    Выкарыстоўваецца толькі для падзелу на фолды, не трапляе ў X ці y мадэляў.
    """
    y_arr = np.asarray(y)
    bins = np.zeros(len(y_arr), dtype=int)
    nonzero_mask = y_arr > 0
    if nonzero_mask.sum() > 0:
        nonzero_vals = y_arr[nonzero_mask]
        quantile_bins = pd.qcut(nonzero_vals, q=n_quantile_bins, labels=False, duplicates="drop")
        bins[nonzero_mask] = quantile_bins.astype(int) + 1
    return bins


class RegressorBench:
    """Аналаг ClassifierBench: збор мадэляў, крос-валідацыя, OOF-прагнозы."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.models: dict[str, Any] = {}
        self.fitted: dict[str, Any] = {}

        self.fold_rmse: dict[str, np.ndarray] = {}
        self.fold_mae: dict[str, np.ndarray] = {}
        self.fold_r2: dict[str, np.ndarray] = {}

        self.evals_result: dict[str, Any] | None = None
        self.cv_evals_results: dict[str, dict[str, Any]] = {}

        self.preprocessor = ColumnTransformer(transformers=[], remainder="passthrough")

        # Уласная копія рэестра для гэтага экзэмпляра (не глабальны стан),
        # каб add_two_stage_model() не муляла іншыя бэнчмаркі ў той жа праграме.
        self.model_registry: dict[str, ModelSpec] = dict(_DEFAULT_MODEL_REGISTRY)

    # ------------------------------------------------------------------ #
    # Шаблоны мадэляў
    # ------------------------------------------------------------------ #
    def _make_models(self) -> dict[str, Any]:
        seed = self.random_state

        models: dict[str, Any] = {
            "Dummy": DummyRegressor(strategy="median"),
            # median, а не mean: пры 78% нулёў і хвасце да 362,
            # медыяна (=0) — гэта сумленны, неабражальны baseline.

            "Ridge": Ridge(
                alpha=1.0,
                random_state=seed,
            ),
            # L2-рэгулярызацыя канкрэтна пад вашу мультыкалінеарнасць:
            # BounceRates/ExitRates corr=0.91, ProductRelated*/_Duration corr=0.86

            "DecisionTree": DecisionTreeRegressor(
                max_depth=6,
                min_samples_leaf=20,   # абарона ад зазубрывання нулявых лісцяў
                random_state=seed,
            ),

            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=seed,
            ),

            "LightGBM": LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=4,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=5.0,
                objective="regression",
                metric="rmse",
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            ),

            "XGBoost": XGBRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                min_child_weight=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=6.0,
                objective="reg:squarederror",
                eval_metric="rmse",
                n_jobs=-1,
                random_state=seed,
            ),
        }

        if _CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostRegressor(
                iterations=500,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=5.0,
                loss_function="RMSE",
                eval_metric="RMSE",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
        else:
            print("[RegressorBench] catboost не ўстаноўлены — прапускаю CatBoost. "
                  "Выканайце: pip install catboost")

        return models

    def build(self) -> dict[str, Any]:
        self.models = self._make_models()
        print("[RegressorBench] Метады:")
        print("  Dummy Regressor      — baseline (медыяна, улічвае 78% нулёў)")
        print("  Ridge                — лінейная, L2 супраць мультыкалінеарнасці")
        print("  DecisionTree         — простае нелінейнае дрэва")
        print("  ExtraTrees           — бэггінг")
        print("  LightGBM, XGBoost" + (", CatBoost" if _CATBOOST_AVAILABLE else "") + " — бустынг")
        return self.models

    # ------------------------------------------------------------------ #
    # 8-ы шлях: TwoStageRegressor (hurdle-мадэль)
    # ------------------------------------------------------------------ #
    def add_two_stage_model(
        self,
        name: str = "TwoStage",
        classifier: Any | None = None,
        regressor: Any | None = None,
        combine: str = "gated",
        threshold: float = 0.5,
        complexity: int = 5,
    ) -> None:
        """
        Рэгіструе TwoStageRegressor (класіфікатар-варотнік P(PageValues>0) +
        рэгрэсар на ненулявой частцы) як асобны, восьмы метад у бэнчмарку.

        Мадэль атрымлівае ЎЛАСНЫ запіс у self.model_registry, каб адаптыўны прэпрацэсар і праверка
        Early Stopping працавалі
        Дэфолтны combine="gated" (не "probability") абраны наўмысна: дыягностыка
        паказала, што "probability"-рэжым дае false-positive rate 99.8% на
        сапраўды нулявых сесіях і рэзка пагаршае MAE (гл. справаздачу, раздзел 5).

        Калі classifier/regressor не пераданы, выкарыстоўваюцца LightGBM-кампаненты
        з тымі ж гіперпараметрамі, што паказалі лепшы вынік у дыягностыцы.
        Класіфікатар свядома без class_weight="balanced" — балансаванне класаў
        завышае P(nonzero) і руйнуе якасць варотніка (тая ж дыягностыка).
        """
        from lightgbm import LGBMClassifier
        from modules.two_stage_regressor import TwoStageRegressor

        seed = self.random_state

        if classifier is None:
            classifier = LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=5,
                # min_child_samples=50, subsample=0.8, subsample_freq=1,
                # colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=5.0,
                random_state=seed, n_jobs=-1, verbosity=-1,                 
            )
        if regressor is None:
            regressor = LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=15, max_depth=5,
                random_state=seed, n_jobs=-1, verbosity=-1,
            )

        self.models[name] = TwoStageRegressor(
            classifier=classifier, regressor=regressor, combine=combine, threshold=threshold,
        )
        # Уласны рэестр-запіс: TwoStage не патрабуе маштабавання (унутры дрэвы)
        # і не праходзіць праз _fit_boosting (сваё ранняе спыненне робіць, калі
        # наогул робіць, унутры ўласнага fit(), а не праз eval_set у Pipeline).
        self.model_registry[name] = ModelSpec(
            needs_scaling=False, supports_early_stopping=False, complexity=complexity,
        )
        print(f"[RegressorBench] Дададзены 8-ы метад: {name} "
              f"(hurdle-мадэль, combine='{combine}', threshold={threshold})")

    def get_complexity_map(self) -> dict[str, int]:
        """Мапа складанасці ўсіх зарэгістраваных мадэляў — для RegressionEvaluator.pick_best()."""
        return {name: spec.complexity for name, spec in self.model_registry.items()}

    # ------------------------------------------------------------------ #
    # Адаптыўны прэпрацэсар (той жа прынцып, што ў ClassifierBench)
    # ------------------------------------------------------------------ #
    def _get_adapted_preprocessor(self, model_name: str) -> Any:
        if self.preprocessor is None:
            raise AttributeError(
                "[RegressorBench] self.preprocessor не ініцыялізаваны. "
                "Выклічце PrepareRegressionDataset.create_preprocessor_regression() "
                "і прысвойце вынік bench.preprocessor перад крос-валідацыяй."
            )

        preprocessor_clone = clone(self.preprocessor)

        spec = self.model_registry.get(model_name)
        if spec is None:
            # Невядомая (незарэгістраваная) мадэль: бяспечны дэфолт — маштабаваць,
            # бо для дрэў/бустынгаў лішні StandardScaler бяскодны, а для лінейнай
            # мадэлі без яго было б сапраўднай памылкай.
            print(
                f"[RegressorBench] Увага: '{model_name}' адсутнічае ў model_registry. "
                f"Дэфолтна ўключаю маштабаванне. Зарэгіструйце ModelSpec для гэтай мадэлі, "
                f"каб пазбегнуць гэтага папярэджання."
            )
            needs_scaling = True
        else:
            needs_scaling = spec.needs_scaling

        preprocessor_clone.set_params(num=StandardScaler() if needs_scaling else "passthrough")
        return preprocessor_clone

    def _supports_early_stopping(self, model_name: str) -> bool:
        spec = self.model_registry.get(model_name)
        return bool(spec and spec.supports_early_stopping)

    # ------------------------------------------------------------------ #
    # Early Stopping для бустынгаў
    # ------------------------------------------------------------------ #
    def _fit_boosting(
        self,
        name: str,
        pipe: Pipeline,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        *,
        record: bool = True,
    ) -> Pipeline:
        prep = pipe.named_steps["prep"]
        x_tr = prep.fit_transform(x_train, y_train)
        x_va = prep.transform(x_val)

        y_tr = y_train.to_numpy().ravel() if hasattr(y_train, "to_numpy") else np.asarray(y_train).ravel()
        y_va = y_val.to_numpy().ravel() if hasattr(y_val, "to_numpy") else np.asarray(y_val).ravel()

        if name == "LightGBM":
            evals: dict[str, Any] = {}
            callbacks = [
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                log_evaluation(0),
                record_evaluation(evals),
            ]
            # Замяняем reg__eval_set на reg__eval_X і reg__eval_y для валідацыйнага сэта.
            # Калі трэба перадаць менавіта ТЭСТАВЫ/ВАЛІДАЦЫЙНЫ сэт (x_va, y_va):
            pipe.fit(
                x_train, y_train,
                reg__eval_X=x_va,
                reg__eval_y=y_va,
                reg__eval_metric="rmse",
                reg__callbacks=callbacks,
            )
            if record:
                self.evals_result = evals
            return pipe

        # if name == "LightGBM":
        #     evals: dict[str, Any] = {}
        #     callbacks = [
        #         early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        #         log_evaluation(0),
        #         record_evaluation(evals),
        #     ]
        #     pipe.fit(
        #         x_train, y_train,
        #         reg__eval_set=[(x_tr, y_tr), (x_va, y_va)],
        #         reg__eval_metric="rmse",
        #         reg__callbacks=callbacks,
        #     )
        #     if record:
        #         self.evals_result = evals
        #     return pipe

        if name == "XGBoost":
            reg = pipe.named_steps["reg"]
            reg.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS, eval_metric="rmse")
            pipe.fit(
                x_train, y_train,
                reg__eval_set=[(x_tr, y_tr), (x_va, y_va)],
                reg__verbose=False,
            )
            if record and hasattr(reg, "evals_result"):
                raw = reg.evals_result() if callable(reg.evals_result) else reg.evals_result
                # self.evals_result = raw
                self.evals_result = cast(dict[str, Any], raw)
            return pipe

        if name == "CatBoost":
            reg = pipe.named_steps["reg"]
            pipe.fit(
                x_train, y_train,
                reg__eval_set=(x_va, y_va),
                reg__early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                reg__verbose=False,
            )
            if record and hasattr(reg, "get_evals_result"):
                self.evals_result = reg.get_evals_result()
            return pipe

        pipe.fit(x_train, y_train)
        return pipe

    # ------------------------------------------------------------------ #
    # Крос-валідацыя
    # ------------------------------------------------------------------ #
    def cross_validate_newV(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        models_conf: dict[str, Any] | None = None,
        *,
        use_zero_inflated_bins: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        KFold(5) (альбо StratifiedKFold па бінах нуль/квантылі, калі
        use_zero_inflated_bins=True) з OOF-прагнозамі.
        Вяртае: (fold_rmse, fold_mae, fold_r2, oof_predictions)
        """
        if models_conf is None:
            models_conf = self.models

        if use_zero_inflated_bins:
            bins = make_zero_inflated_bins(y)
            kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=self.random_state)
            split_iter_factory = lambda: kf.split(x, bins)
        else:
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=self.random_state)
            split_iter_factory = lambda: kf.split(x)

        oof_predictions: dict[str, np.ndarray] = {}

        for name, template in models_conf.items():
            rmse_scores: list[float] = []
            mae_scores: list[float] = []
            r2_scores: list[float] = []
            print(f"[Cross-Validation] {name} ...")

            oof_preds = np.zeros(len(y))

            for train_idx, val_idx in split_iter_factory():
                model = clone(template)
                x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
                y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

                preprocessor = self._get_adapted_preprocessor(name)
                pipe = Pipeline([("prep", preprocessor), ("reg", model)])

                if self._supports_early_stopping(name):
                    pipe = self._fit_boosting(name, pipe, x_tr, y_tr, x_va, y_va, record=True)
                    if self.evals_result:
                        self.cv_evals_results.setdefault(name, {})
                        fold_name = f"{name}_fold_{len(rmse_scores)}"
                        self.cv_evals_results[name][fold_name] = self.evals_result
                else:
                    pipe.fit(x_tr, y_tr)

                preds = np.asarray(pipe.predict(x_va)).ravel()
                oof_preds[val_idx] = preds

                y_va_np = y_va.to_numpy()
                rmse_scores.append(float(root_mean_squared_error(y_va_np, preds)))
                mae_scores.append(float(mean_absolute_error(y_va_np, preds)))
                r2_scores.append(float(r2_score(y_va_np, preds)))

            self.fold_rmse[name] = np.asarray(rmse_scores)
            self.fold_mae[name] = np.asarray(mae_scores)
            self.fold_r2[name] = np.asarray(r2_scores)
            self.fitted[name] = pipe
            oof_predictions[name] = oof_preds

            print(
                f"     RMSE={self.fold_rmse[name].mean():.4f} ± {self.fold_rmse[name].std():.4f}   "
                f"MAE={self.fold_mae[name].mean():.4f} ± {self.fold_mae[name].std():.4f}   "
                f"R2={self.fold_r2[name].mean():.4f} ± {self.fold_r2[name].std():.4f}"
            )

        return self.fold_rmse, self.fold_mae, self.fold_r2, oof_predictions

    def get_metrics_table(self) -> pd.DataFrame:
        """Табліца метрык, адсартаваная па росце RMSE (лепшыя першымі)."""
        if not self.fold_rmse:
            return pd.DataFrame()

        rows = []
        for name in self.fold_rmse:
            if name == "Dummy":
                continue
            rmse_mean = float(self.fold_rmse[name].mean())
            rows.append({
                "model": name,
                "RMSE (Mean ± STD)": f"{rmse_mean:.4f} ± {float(self.fold_rmse[name].std()):.4f}",
                "MAE (Mean ± STD)": f"{float(self.fold_mae[name].mean()):.4f} ± {float(self.fold_mae[name].std()):.4f}",
                "R2 (Mean ± STD)": f"{float(self.fold_r2[name].mean()):.4f} ± {float(self.fold_r2[name].std()):.4f}",
                "_rmse_sort": rmse_mean,
            })

        table = pd.DataFrame(rows).sort_values(by="_rmse_sort", ascending=True)
        return table.drop(columns=["_rmse_sort"])

    # ------------------------------------------------------------------ #
    # Фінальныя канфігурацыі
    # ------------------------------------------------------------------ #
    def prepare_final_models(
        self,
        best_name: str | None = None,
        optimized_results: Dict[str, Dict[str, Any]] | None = None,
        models_to_keep: list[str] | None = None,
    ) -> dict[str, Any]:
        if models_to_keep is None:
            models_to_keep = [best_name] if best_name else list(self.models.keys())

        final_configs: dict[str, Any] = {}
        opt_res = optimized_results or {}

        for name in models_to_keep:
            if name not in self.models:
                print(f"[Warning] Мадэль '{name}' не знойдзена ў self.models.")
                continue

            model_copy = clone(self.models[name])

            if best_name and name == best_name:
                best_params = opt_res.get(name, {})
                if best_params:
                    model_copy.set_params(**best_params)

            final_configs[name] = model_copy

        return final_configs

    def final_secure_refit(
        self,
        final_configs: dict[str, Any],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Any:
        """Поўнае навучанне лепшай мадэлі на x_train, ES-кантроль на x_val."""
        model_name = list(final_configs.keys())[0]
        base_model = final_configs[model_name]
        reg_model = clone(base_model)

        preprocessor = self._get_adapted_preprocessor(model_name)
        pipe = Pipeline([("prep", preprocessor), ("reg", reg_model)])

        if self._supports_early_stopping(model_name):
            pipe = self._fit_boosting(model_name, pipe, x_train, y_train, x_val, y_val, record=True)
        else:
            pipe.fit(x_train, y_train)

        self.fitted[model_name] = pipe
        return pipe
