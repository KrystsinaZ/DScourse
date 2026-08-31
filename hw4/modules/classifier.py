"""Модели, CV, Optuna/Hyperopt."""
from __future__ import annotations

import warnings
from lightgbm.basic import LightGBMError

# Спецыяльны фільтр, які цалкам ігнаруе LGBMDeprecationWarning
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import numpy as np
import pandas as pd
from lightgbm import (
    LGBMClassifier,
    early_stopping,
    log_evaluation,
    record_evaluation,
)

from typing import (List, Dict, cast, Any, Tuple, Optional)

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ( StratifiedKFold, learning_curve, validation_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import time

from xgboost import XGBClassifier

from modules.tuner import BayesianTuner, TuneResult


N_SPLITS = 5
BOOSTING = {"LightGBM", "XGBoost"}
EARLY_STOPPING_ROUNDS = 50

def scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Для XGBoost у анлайн-шопінгу: колькі сышло без пакупкі на адну куплю, 
    аптымальная вага для канверсіі на умераным дысбалансе (15%)."""
    # Пераводзім у int, каб нават True/False або тэкставыя 1/0 сталі лічбамі
    y_arr = np.asarray(y, dtype=int)
    
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    
    # Калі мэтавага класа няма, альбо яго назіранняў раптам больш, чым нулёў — вагу не змяняем
    if n_pos == 0 or n_pos >= n_neg:
        return 1.0
        
    return n_neg / n_pos


class ClassifierBench:
    
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.models: dict[str, Any] = {}
        self.fitted: dict[str, Any] = {}
        self.fold_pr: dict[str, np.ndarray] = {}
        self.fold_roc: dict[str, np.ndarray] = {}
        self.best_params: dict[str, Any] | None = None
        self.tune_source: str | None = None
        self.evals_result: dict[str, Any] | None = None
        self.optuna_result: TuneResult | None = None

        self.cv_evals_results: dict[str, dict[str, Any]] = {}
        self.best_cv_iterations: dict[str, int] = {}

        self.last_verify_train = None
        self.last_verify_val = None
        self.last_verify_gap = None

        self.preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')              


    def _make_models(self, y_train: pd.Series) -> dict[str, Any]:
            """Мадэлі з наладамі баланса класаў."""
            
            scale_pos = scale_pos_weight(y_train)
            seed = self.random_state
            
            return {
                "Dummy": DummyClassifier(strategy="stratified", random_state=seed),
              
                "LogisticRegression": LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=seed,
                ),
                "SVM": CalibratedClassifierCV(
                    estimator=SVC(
                        class_weight="balanced",# Аўтаматычна балансуе вагі класаў
                        random_state=seed, 
                        max_iter=1000),     # Абмяжоўваем ітэрацыі, бо SVM доўга вучыцца на вялікіх даных
                    ensemble=False          # Cцяжок гарантуе больш хуткую працу, бо іначай быў бы ансамбль (некалькі копій) мадэляў SVM, 
                                            # навучаў бы кожную на сваім кавалку даных, а потым усярэднедваў іх прагнозы. 
                                            # Пры дысбалансе гэта вельмі доўга лічыцца
                ),
                "KNN": KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",  # Дапамагае пры дысбалансе, улічвае адлегласць
                    n_jobs=-1
                ),
                "DecisionTree": DecisionTreeClassifier(
                    class_weight="balanced",
                    max_depth=4, #6,              # Абмяжоўваем глыбіню
                    random_state=seed,
                ),
                "ExtraTrees": ExtraTreesClassifier(
                    n_estimators=150,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=seed,
                ),
                "LightGBM": LGBMClassifier(
                    # 1. Складанасць і хуткасць (Тандэм плаўнасці)
                    n_estimators=300,            # Запас дрэў для стабільнага Early Stopping
                    learning_rate=0.03,          # Зніжаны крок (0.03 замест 0.1) прымушае мадэль вучыцца павольна
                    
                    # 2. Жорсткае абмежаванне структуры (Галоўны ўдар па гэпу)
                    num_leaves=15,               # Палова ад дэфолтных 31. Мадэль фізічна не зможа зазубрыць трэйн
                    max_depth=4,                 # Жорсткая столь глыбіні дрэва (не вышэй за 4 паверхі)
                    
                    # 3. Абарона ад ізаляцыі рэдкіх пакупнікоў
                    min_child_samples=50,        # Ставім 50 радкоў (замест 20), каб мадэль не выдумляла лакальныя правілы
                    
                    # 4. Выпадковасць (Стахастычная рэгулярызацыя)
                    subsample=0.8,               # Кожнае дрэва бачыць толькі 80% выпадковых сесій сайта
                    subsample_freq=1,            # Перавыбіраць радкі на кожнай ітэрацыі
                    colsample_bytree=0.8,        # Кожнае дрэва бачыць толькі 80% калонак (прыкмет)
                    
                    # 5. Матэматычныя штрафы ваг
                    reg_alpha=0.5,               # Уключаем L1-рэгулярызацыю для адсячэння шумавых фіч
                    reg_lambda=5.0,              # Уключаем моцны L2-штраф, каб прыціснуць каэфіцыенты ў лісці
                    
                    # 6. Спецыфіка вашай бізнес-задачы
                    class_weight="balanced",     # Абсалютна крытычна для дысбалансу класаў! Падымае Recall і PR-AUC
                    random_state=seed,           # Фіксаваная пераменная сіда (напрыклад, 42)
                    n_jobs=-1,                   # Першая CV павінна выкарыстоўваць усе ядры на максімальнай хуткасці
                    verbosity=-1,                # Прыбіраем тэхнічныя папярэджанні з кансолі
                ),               
                
                # "XGBoost": XGBClassifier(
                #     n_estimators=400,
                #     max_depth=4,
                #     learning_rate=0.05,
                #     min_child_weight=1,
                #     subsample=0.8,
                #     colsample_bytree=0.8,
                #     reg_alpha=0.0,
                #     reg_lambda=1.0,
                #     scale_pos_weight=scale_pos,
                #     eval_metric="aucpr",
                #     n_jobs=-1,
                #     random_state=seed,
                #     ),
                # 
                "XGBoost": XGBClassifier(
                    n_estimators=400,
                    max_depth=4,                 # Абмежаванне глыбіні для таблічных дадзеных
                    learning_rate=0.03,          # Больш плыўны крок для барацьбы з гэпам
                    min_child_weight=10,         # Жорсткі фільтр супраць ізаляцыі рэдкіх пакупнікоў
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0, #1.0,         # Уключаны штраф L1 для адсячэння шуму
                    reg_lambda=6.0, #10.0,       # Моцны L2 штраф для стабілізацыі ваг дрэў
                    scale_pos_weight=scale_pos,  # Разлічаны каэфіцыент балансіроўкі класаў
                    eval_metric="aucpr",         # Мэтавая метрыка для PR-AUC
                    n_jobs=-1,
                    random_state=seed,
                ),

            }
    
    
    def build(self, y_train: pd.Series) -> dict[str, Any]:
        """Печатаю, какие модели беру."""
        self.models = self._make_models(y_train)
        scale_pos = scale_pos_weight(y_train)

        print("[Classifier] Метады:")
        print("  Dummy Classifier — baseline")
        print("  LogisticRegression — лінейная")
        print("  SVM — лінейная")
        print("  KNN — простае дрэва")
        print("  DecisionTree — простае дрэва")
        print("  ExtraTrees   — бэггінг")
        print("  LightGBM, XGBoost — бустынг")
        print(f"[Classifier] Маштабаванне вагі станоўчага класа: scale_pos_weight={scale_pos:.1f}. Перадаю параметрам для XGBoost")
        return self.models

    
    def cross_validate_newV(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        models_conf: dict[str, Any] | None = None, 
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        5 фолдаў крос-валідацыі з адаптыўным маштабаваннем пад кожны метад.
        Вяртае: (fold_pr, fold_roc, oof_predictions)
        """
                    
        kf = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state,
        )
        
        # self.fold_pr = {}
        # self.fold_roc = {}

        # Ствараем слоўнік для захавання Out-of-Fold прагнозаў
        oof_predictions: dict[str, np.ndarray] = {}

        if models_conf is None:
            models_conf = self.models

        for name, template in models_conf.items():
        # for name, template in models_conf:
            pr_scores: list[float] = []
            roc_scores: list[float] = []
            print(f"[Cross-Validation] {name} ...")
            
            # Ствараем пусты масіў памерам з увесь таргет для OOF-прагнозаў бягучай мадэлі
            oof_probas = np.zeros(len(y))
            
            for train_idx, val_idx in kf.split(x, y):
                model = clone(template)
                x_tr = x.iloc[train_idx]
                y_tr = y.iloc[train_idx]
                x_va = x.iloc[val_idx]
                y_va = y.iloc[val_idx]
         
                preprocessor = self._get_adapted_preprocessor(model_name=name)

                # КРОК 3: Будуем Pipeline, каб пазбегнуць Dataleake
                if name == "Dummy":
                    pipe = model
                else:
                    pipe = Pipeline([('prep', preprocessor), ('clf', model)])

                # КРОК 4: Навучанне і прагноз з улікам спецыфікі бустынгаў
                
                if name in BOOSTING:
                    pipe = self._fit_boosting(name, pipe, x_tr, y_tr, x_va, y_va, record=True)
                    
                    if hasattr(self, "evals_result") and self.evals_result:
                        # Ствараем прастору для мадэлі, калі яе яшчэ няма
                        if name not in self.cv_evals_results:
                            self.cv_evals_results[name] = {}
                        
                        fold_name = f"{name}_fold_{len(pr_scores)}"
                        self.cv_evals_results[name][fold_name] = self.evals_result
                else:
                    pipe.fit(x_tr, y_tr)                
                
                
                # Прагноз імавернасцей (загортваем у np.asarray для супакою Pylance)
                proba_matrix = np.asarray(pipe.predict_proba(x_va))
                proba = proba_matrix[:, 1]
                
                # ВАЖНА: Запісваем прагнозы бягучага фолду на свае пазіцыі па валідацыйных індэксах
                oof_probas[val_idx] = proba

                y_va_np = y_va.to_numpy()
                
                pr_scores.append(float(average_precision_score(y_va_np, proba)))
                roc_scores.append(float(roc_auc_score(y_va_np, proba)))
                
            # Захоўваем пафолдавыя метрыкі
            self.fold_pr[name] = np.asarray(pr_scores)
            self.fold_roc[name] = np.asarray(roc_scores)
            # Захоўваем цалкам гатовы і навучаны Pipeline вонку цыкла па фолдах
            self.fitted[name] = pipe
    
            # Захоўваем цалкам сабраны вектар прагнозаў для гэтай мадэлі
            oof_predictions[name] = oof_probas
            
            print(f"     PR-AUC={self.fold_pr[name].mean():.4f} ± {self.fold_pr[name].std():.4f}")
            
        
        return self.fold_pr, self.fold_roc, oof_predictions

   
    def predict_proba(
        self,
        name: str,
        x_test: pd.DataFrame,
    ) -> np.ndarray:
        """Вероятность класса fraud=1."""
        return self.fitted[name].predict_proba(x_test)[:, 1]

    def _fit_boosting(
        self,
        name: str,
        model: Any,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        *,
        record: bool = True,
    ) -> Any:
        """Early stopping на val. С автоматическим препроцессингом для Pipeline."""
        
        # 1. Проверяем, завернут ли бустинг в scikit-learn Pipeline
        pipe = hasattr(model, "named_steps") and "clf" in getattr(
            model, "named_steps", {}
        )
        
        # 2. Если это Pipeline, трансформируем валидационные наборы вручную
        if pipe:
            prep = model.named_steps["prep"]
            x_train_proc = prep.fit_transform(x_train, y_train)
            x_val_proc = prep.transform(x_val)
            eval_sets = [(x_train_proc, y_train), (x_val_proc, y_val)]
        else:
            x_train_proc, x_val_proc = x_train, x_val
            eval_sets = [(x_train, y_train), (x_val, y_val)]

        # --- БЛОК LIGHTGBM ---
        if name == "LightGBM":
            evals: dict[str, Any] = {}
            callbacks = [
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                log_evaluation(0),
                record_evaluation(evals),
            ]
            
            x_train_proc, y_train_data = eval_sets[0]
            x_val_proc, y_val_data = eval_sets[1]

            # Выпростваем і ператвараем таргеты ў патрэбныя аднамерныя масівы
            y_train_processed = y_train_data.to_numpy().ravel() if hasattr(y_train_data, "to_numpy") else np.asarray(y_train_data).ravel()
            y_val_processed = y_val_data.to_numpy().ravel() if hasattr(y_val_data, "to_numpy") else np.asarray(y_val_data).ravel()

            lgb_metrics = ["binary_logloss", "average_precision"]
                                                # PR-AUC
            fit_kw: dict[str, Any]
            if pipe:
                fit_kw = {
                    "clf__eval_set": [
                        (x_train_proc, y_train_processed), 
                        (x_val_proc, y_val_processed)
                    ],
                    "clf__eval_metric": lgb_metrics,#"average_precision",
                    "clf__callbacks": callbacks,
                }
                model.fit(x_train, y_train, **fit_kw)
            else:
                fit_kw = {
                    "eval_set": [
                        (x_train_proc, y_train_processed),
                        (x_val_proc, y_val_processed)
                    ],
                    "eval_metric": lgb_metrics, #"average_precision",
                    "callbacks": callbacks,
                }
                model.fit(x_train, y_train, **fit_kw)
                
            if record:
                self.evals_result = evals
            return model
        
        if "XGB" in name or name == "XGBoost":
            clf = model.named_steps["clf"] if pipe else model
            
            
            x_train_proc, y_train_data = eval_sets[0]
            x_val_proc, y_val_data = eval_sets[1]
            
            y_train_processed = y_train_data.to_numpy().astype(int).ravel() if hasattr(y_train_data, "to_numpy") else np.asarray(y_train_data).astype(int).ravel()
            y_val_processed = y_val_data.to_numpy().astype(int).ravel() if hasattr(y_val_data, "to_numpy") else np.asarray(y_val_data).astype(int).ravel()

            
            xgb_metrics = ["logloss", "aucpr"]

            # Жорстка і празрыста задаем параметры без дубляў
            clf.set_params(
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                eval_metric=xgb_metrics
            )
            
            if pipe:
                model.fit(
                    x_train,
                    y_train,
                    # clf__eval_set=[(x_val_data, y_val_processed)],
                    clf__eval_set=[(x_train_proc, y_train_processed), (x_val_proc, y_val_processed)],
                    clf__verbose=False
                )
            else:
                model.fit(
                    x_train,
                    y_train,
                    # eval_set=[(x_val_data, y_val_processed)],
                    eval_set=[(x_train_proc, y_train_processed), (x_val_proc, y_val_processed)],
                    verbose=False
                )
                
            if record and hasattr(clf, "evals_result"):
                raw_evals = clf.evals_result() if callable(clf.evals_result) else clf.evals_result
                self.evals_result = cast(dict[str, Any], raw_evals)
                
            return model
   
        model.fit(x_train, y_train)
        return model

    def run_dummy_baseline(
        self, 
        *, 
        split: Any, 
        evaluator: Any, 
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Trains a DummyClassifier baseline on raw features, evaluates it,
        and prints the initial performance metrics table.
        """
        dummy_model = DummyClassifier(strategy="stratified", random_state=random_state)

        # Засякаем час ПЕРАД пачаткам працы
        start_time = time.perf_counter()

        # 2. Train the baseline model (Dummy only learns target class ratios)
        # Note: We use .X_train and .X_val based on your previous split objects
        dummy_model.fit(split.x_train, split.y_train)

        dummy_preds = dummy_model.predict(split.x_val)
        
        proba_matrix = np.asarray(dummy_model.predict_proba(split.x_val))
        dummy_probs = proba_matrix[:, 1]

        # Засякаем час ПА завяршэнні і лічым розніцу
        elapsed_time = time.perf_counter() - start_time

        dummy_row = evaluator.evaluate_model_performance(
            model_name="DummyClassifier (Baseline)",
            y_true=split.y_val,
            y_pred=dummy_preds,
            y_probs=dummy_probs
        )
        
        metrics_table = pd.concat([dummy_row], ignore_index=True)

        metrics_table["Час (сек)"] = [elapsed_time]

        
        return metrics_table

    def run_logistic_regression(
        self,
        *,
        split: Any,
        preprocessor: Any,
        evaluator: Any,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Стварае Pipeline з апрацоўкай прыкмет і Лагістычнай рэгрэсіяй,
        навучае яго, вылічвае метрыкі на валідацыі і вяртае радок для табліцы.
        """

        lr_preprocessor = self._get_adapted_preprocessor(model_name="LogisticRegression")


        # 1. Збіраем Pipeline (апрацоўка даных + мадэль)
        # Параметр class_weight='balanced' аўтаматычна штрафуе за памылкі ў класе True
        lr_pipeline = Pipeline([
            ('preprocessor', lr_preprocessor),
            ('classifier', LogisticRegression(
                class_weight='balanced', 
                random_state=random_state,
                max_iter=1000  # Павялічваем колькасць ітэрацый для збежнасці
            ))
        ])

        # Засякаем час ПЕРАД пачаткам працы
        start_time = time.perf_counter()

        
        # 2. Навучаем увесь канвеер на трэніровачных даных
        # Дзякуючы Pipeline, маштабаванне адбудзецца ТОЛЬКІ на X_train (без выцеку даных)
        lr_pipeline.fit(split.x_train, split.y_train)

        # 3. Робім прагноз для валідацыйнай выбаркі
        lr_preds = lr_pipeline.predict(split.x_val)
        lr_probs = lr_pipeline.predict_proba(split.x_val)[:, 1]

        # Засякаем час ПАСЛЯ завяршэння і лічым розніцу
        elapsed_time = time.perf_counter() - start_time

        # 4. Ствараем радок з метрыкамі праз ваш evaluator
        lr_row = evaluator.evaluate_model_performance(
            model_name="Logistic Regression (Balanced)",
            y_true=split.y_val,
            y_pred=lr_preds,
            y_probs=lr_probs
        )

        lr_row["Час (сек)"] = [elapsed_time]

        return lr_row

    def run_knn(
        self,
        *,
        split: Any,
        preprocessor: Any,
        evaluator: Any,
    ) -> pd.DataFrame:
        """
        Стварае Pipeline з апрацоўкай прыкмет і KNN класіфікатарам,
        навучае яго, вылічвае метрыкі на валідацыі і вяртае радок для табліцы.
        Аптымізавана пад выкіды і дысбаланс класаў.
        """
        # from sklearn.neighbors import KNeighborsClassifier

        knn_preprocessor = self._get_adapted_preprocessor(model_name="KNN")

        # 2. Збіраем Pipeline (апрацоўка даных + мадэль)
        # Параметр weights="distance" аддае больш вагі бліжэйшым суседзям, вырараючы праблему дысбалансу.
        knn_pipeline = Pipeline([
            ('preprocessor', knn_preprocessor),
            ('classifier', KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=1
            ))
        ])

        # Засякаем час ПЕРАД пачаткам працы
        start_time = time.perf_counter()

        # 3. Навучаем увесь канвеер на трэніровачных даных (без выцеку даных)
        knn_pipeline.fit(split.x_train, split.y_train)

        # 4. Робім прагноз для валідацыйнай выбаркі
        knn_preds = knn_pipeline.predict(split.x_val)
        knn_probs = knn_pipeline.predict_proba(split.x_val)[:, 1]

        # Засякаем час ПАСЛЯ завяршэння і лічым розніцу
        elapsed_time = time.perf_counter() - start_time

        # 5. Ствараем радок з метрыкамі праз ваш evaluator
        knn_row = evaluator.evaluate_model_performance(
            model_name="KNN (Robust + Distance Weights)",
            y_true=split.y_val,
            y_pred=knn_preds,
            y_probs=knn_probs
        )

        knn_row["Час (сек)"] = [elapsed_time]

        self.fitted["KNN"] = Pipeline([
            ('preprocessor', knn_preprocessor),
            ('model', knn_pipeline)
        ])

        return knn_row

    def run_svm(
        self,
        *,
        split: Any,
        preprocessor: Any,
        evaluator: Any,
    ) -> pd.DataFrame:
        """
        Запускае навучанне SVM на спліце 60/20/20 з аўтаматычнай унутранай каліброўкай.
        Павялічаны max_iter ратуе ад ConvergenceWarning.
        """

        # 1. Ствараем АБСАЛЮТНА СВЕЖЫ прэпрацэсар (аўтаматычна атрымае RobustScaler)
        svm_preprocessor = self._get_adapted_preprocessor(model_name="SVM")

        # 2. Ствараем ГОЛУЮ базавую мадэль з павялічаным max_iter
        # Павелічэнне ітэрацый да 5000 цалкам ліквідуе памылку ConvergenceWarning!
        base_svm = LinearSVC(max_iter=5000, random_state=self.random_state)

        # 3. Заварочваем у каліброўку па змаўчанні (яна сама зробіць стабільны ўнутраны спліц)
        # Ніякіх ручных абнуленняў і .cv = "prefit". Стандартны надзейны рэжым!
        calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=3) 

        # 4. ЗБІРАЕМ ЧЫСТЫ ПАЙПЛАЙН ПА ЗМАЎЧАННЮ
        svm_pipeline = Pipeline([
            ('preprocessor', svm_preprocessor),
            ('classifier', calibrated_svm)
        ])

        start_time = time.perf_counter()

        # 5. Навучаем увесь канвеер адным выклікам (Чыста і бяспечна)
        svm_pipeline.fit(split.x_train, split.y_train)

        # 6. Робім прагноз для валідацыйнай выбаркі
        svm_preds = svm_pipeline.predict(split.x_val)
        svm_probs = svm_pipeline.predict_proba(split.x_val)[:, 1]

        elapsed_time = time.perf_counter() - start_time

        # 7. Ствараем радок з метрыкамі
        svm_row = evaluator.evaluate_model_performance(
            model_name="SVM (Robust + Calibrated Config)",
            y_true=split.y_val,
            y_pred=svm_preds,
            y_probs=svm_probs
        )
        svm_row["Час (сек)"] = [elapsed_time]

        # Захоўваем для гісторыі праекта
        self.fitted["SVM"] = svm_pipeline

        return svm_row


    def run_decision_tree(
        self,
        *,
        split: Any,
        preprocessor: Any,
        evaluator: Any,
    ) -> pd.DataFrame:
        """
        Запускае адзінкавае навучанне DecisionTree, выкарыстоўваючы налады са слоўніка self.models.
        Адключае маштабаванне лікаў для захавання зыходнай прыроды дрэў рашэнняў.
        """

        # 1. Дастаем гатовую мадэль са слоўніка self.models
        if "DecisionTree" not in self.models:
            raise KeyError("[ClassifierBench] Мадэль 'DecisionTree' не знойдзена ў self.models. Праверце _make_models().")
        
        # Клануем шаблон, каб не сапсаваць арыгінал
        dt_model = clone(self.models["DecisionTree"])

        dt_preprocessor = self._get_adapted_preprocessor(model_name="DecisionTree")

        # 3. Збіраем і навучаем Pipeline (катэгарыяльнае кадыраванне + ваша гатовая мадэль)
        dt_pipeline = Pipeline([
            ('preprocessor', dt_preprocessor),
            ('model', dt_model)
        ])

        # Засякаем час ПЕРАД пачаткам працы
        start_time = time.perf_counter()
        # Навучаем увесь канвеер
        dt_pipeline.fit(split.x_train, split.y_train)

        # 4. Робім прагноз на валідацыйнай выбарцы
        dt_preds = dt_pipeline.predict(split.x_val)
        dt_probs = dt_pipeline.predict_proba(split.x_val)[:, 1]

        # Засякаем час ПАСЛЯ завяршэння і лічым розніцу
        elapsed_time = time.perf_counter() - start_time

        # 5. Генеруем радок метрык праз ваш evaluator
        dt_row = evaluator.evaluate_model_performance(
            model_name="DecisionTree (Balanced + Raw Numeric)",
            y_true=split.y_val,
            y_pred=dt_preds,
            y_probs=dt_probs
        )

        dt_row["Час (сек)"] = [elapsed_time]
        
        return dt_row

    def run_extra_trees(
        self,
        *,
        split: Any,
        preprocessor: Any,
        evaluator: Any,
    ) -> pd.DataFrame:
        """
        Запускае адзінкавае навучанне ансамбля ExtraTrees з падлікам часу выканання.
        Адключае маштабаванне лікаў для захавання зыходнай прыроды дрэў.
        """
    
        # 1. Дастаем гатовую мадэль са слоўніка self.models
        if "ExtraTrees" not in self.models:
            raise KeyError("[ClassifierBench] Мадэль 'ExtraTrees' не знойдзена ў self.models. Праверце _make_models().")
        
        et_model = clone(self.models["ExtraTrees"])

        et_preprocessor = self._get_adapted_preprocessor(model_name="ExtraTrees")

        # 3. Збіраем Pipeline (катэгарыяльнае кадыраванне + ансамбль)
        et_pipeline = Pipeline([
            ('preprocessor', et_preprocessor),
            ('model', et_model)
        ])
        
        # 4. Засякаем час і выконваем навучанне/прагноз
        start_time = time.perf_counter()

        et_pipeline.fit(split.x_train, split.y_train)
        et_preds = et_pipeline.predict(split.x_val)
        et_probs = et_pipeline.predict_proba(split.x_val)[:, 1]

        elapsed_time = time.perf_counter() - start_time

        # 5. Генеруем радок метрык і дадаем час выканання
        et_row = evaluator.evaluate_model_performance(
            model_name="ExtraTrees (Ensemble)",
            y_true=split.y_val,
            y_pred=et_preds,
            y_probs=et_probs
        )
        et_row["Час (сек)"] = [elapsed_time]

        return et_row
        
    def get_metrics_table(self) -> pd.DataFrame:
        """
        Збірае выніковую табліцу DataFrame з сярэднімі метрыкамі і STD вонку.
        Аўтаматычна сартуе мадэлі па спаданні якасці PR-AUC.
        """
        # Бяспечная праверка: калі крос-валідацыя яшчэ не запускалася
        if not hasattr(self, 'fold_pr') or not self.fold_pr:
            return pd.DataFrame()

        rows = []
        for name, pr_scores in self.fold_pr.items():

            if "dummy" in name.lower():
                continue

            roc_scores = self.fold_roc[name]
            
            # Разлічваем сярэднія значэнні (як у вашай cv_table)
            pr_mean = float(pr_scores.mean())
            roc_mean = float(roc_scores.mean())
            
            # Ствараем прыгожыя радкі з захаваннем лічбавага значэння для сартавання
            rows.append({
                "model": name,
                "PR-AUC (Mean ± STD)": f"{pr_mean:.4f} ± {float(pr_scores.std()):.4f}",
                "ROC-AUC (Mean ± STD)": f"{roc_mean:.4f} ± {float(roc_scores.std()):.4f}",
                "_pr_sort": pr_mean  # Тэхнічнае схаванае поле для сартавання
            })
            
        # Ператвараем у DataFrame і сартуем па спаданні PR-AUC (моцны бок cv_table)
        table = pd.DataFrame(rows).sort_values(
            by="_pr_sort",
            ascending=False
        )
        
        # Выдаляем часовае поле для сартавання, каб справаздача была чыстай
        table = table.drop(columns=["_pr_sort"])
        
       
        return table

    def collect_learning_curves(self, x: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, Any]]:
        """
        Вылічвае 5-фолдавыя крывыя навучання, аўтаматычна выкарыстоўваючы 
        структуру прэпрацэсараў з паспяхова выкананай cross_validate_newV.
        """

        # Правераем, ці быў ужо запушчаны асноўны працэс CV
        if not hasattr(self, "fitted_pipelines") or not self.fitted:
            raise ValueError(
                "Спачатку трэба запусціць cross_validate_newV, каб сабраць структуру пайплайнаў!"
            )

        curves_results = {}
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for name, fitted_pipe in self.fitted.items():
            if "dummy" in name.lower():
                continue

            print(f"[Learning Curve] Вылічэнне для {name} на аснове існуючага Pipeline...")
            
            pipe_template = clone(fitted_pipe)

            # 2. Калі гэта бустынг, цалкам адключаем early_stopping_rounds для гэтай працэдуры
            if hasattr(pipe_template, "named_steps") and "clf" in pipe_template.named_steps:
                clf_model = pipe_template.named_steps["clf"]
                
                # Скідваем параметры ранняга прыпынку, каб яны не патрабавалі eval_set
                if hasattr(clf_model, "set_params"):
                    # Для XGBoost і LightGBM Scikit-Learn API
                    current_params = clf_model.get_params()
                    
                    if "early_stopping_rounds" in current_params and current_params["early_stopping_rounds"] is not None:
                        clf_model.set_params(early_stopping_rounds=None)
                    
                    # Прыбіраем калбэкі LightGBM, калі яны зашыліся ў параметры
                    if "callbacks" in current_params and current_params["callbacks"] is not None:
                        clf_model.set_params(callbacks=None)

            train_sizes, train_scores, val_scores, *_= learning_curve(
                estimator=pipe_template,  # Перадаем ачышчаную ад ранніх стопаў копію структуры вашага Pipeline
                X=x,
                y=y.to_numpy().astype(int),
                cv=kf,
                scoring="average_precision", 
                train_sizes=np.linspace(0.1, 1.0, 5), 
                n_jobs=-1, 
                random_state=self.random_state
            )

            curves_results[name] = (
                train_sizes,
                train_scores,
                val_scores
            )

        return curves_results


    def get_single_learning_curve_split(
        self,
        pipe_or_model: Any,
        split: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вылічвае кропкі крывой навучання для адной канкрэтнай мадэлі/пайплайна
        на аснове фіксаванага падзелу 60/20 без унутранай крос-валідацыі.
        """
        # from sklearn.metrics import average_precision_score
        # from sklearn.base import clone
        # import numpy as np

        # Прапорцыі (25%, 50%, 75%, 100%)
        fractions = np.linspace(0.25, 1.0, 4)
        
        x_train = split.x_train
        y_train = split.y_train
        x_val = split.x_cv
        y_val = split.y_cv

        n_samples = len(y_train)
        sizes = []
        train_scores = []
        val_scores = []

        # Часова абмяжоўваем колькасць дрэў, калі гэта ансамбль
        # Каб разлік памераў выбаркі праходзіў хутка
        pipe_structure = hasattr(pipe_or_model, "named_steps") and "clf" in getattr(pipe_or_model, "named_steps", {})
        clf = pipe_or_model.named_steps["clf"] if pipe_structure else pipe_or_model
        
        if hasattr(clf, "n_estimators"):
            clf.set_params(n_estimators=min(int(getattr(clf, "n_estimators", 80)), 80))
            if hasattr(clf, "early_stopping_rounds"):
                clf.set_params(early_stopping_rounds=None)

        # Цыкл па памерах выбаркі (60% трэйну наразаем уручную)
        for frac in fractions:
            current_size = int(n_samples * frac)
            sizes.append(current_size)

            # Клануем мадэль на кожны крок памеру, каб навучанне было сумленным і аўтаномным
            current_pipe = clone(pipe_or_model)

            # Выразаем кавалак дадзеных
            x_tr_chunk = x_train.iloc[:current_size]
            y_tr_chunk = y_train.iloc[:current_size]

            # Навучаем на кавалку
            current_pipe.fit(x_tr_chunk, y_tr_chunk)

            # Лічым PR-AUC на трэніровачным кавалку
            tr_proba = current_pipe.predict_proba(x_tr_chunk)[:, 1]
            train_score = average_precision_score(y_tr_chunk.to_numpy(), tr_proba)
            train_scores.append([train_score]) # Загортваем у спіс для сумяшчальнасці з функцыяй plot

            # Лічым PR-AUC на фіксаванай валідацыі (яна заўсёды 20%)
            val_proba = current_pipe.predict_proba(x_val)[:, 1]
            val_score = average_precision_score(y_val.to_numpy(), val_proba)
            val_scores.append([val_score])

        return np.array(sizes), np.array(train_scores), np.array(val_scores)

    def analyze_learning_gaps(self, curves_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """
        Аналізуе разрыў (gap) паміж Train і Val на максімальным памеры выбаркі.
        Вяртае прыгожую аналітычную табліцу DataFrame.
        """
        # import pandas as pd
        # import numpy as np

        analysis_list = []

        for model_name, (sizes, train_scores, val_scores) in curves_data.items():

            if "dummy" in model_name.lower():
                continue    
            # Нам патрэбны апошні індэкс (-1), які адпавядае 100% памеру выбаркі
            train_mean_final = float(np.mean(train_scores[-1, :]))
            val_mean_final = float(np.mean(val_scores[-1, :]))
            
            # Вылічваем разрыў
            gap = train_mean_final - val_mean_final
            
            # Вашы правільныя развагі і вердыкты:
            if gap < 0.05:
                status = "✓ Стабільная (gap < 0.05)"
            elif gap < 0.10:
                status = "⚠️ Невялікае перанавучанне"
            else:
                status = "Моцнае перанавучанне"

            analysis_list.append({
                "Мадэль": model_name,
                "Train PR-AUC (100%)": round(train_mean_final, 4),
                "Val PR-AUC (100%)": round(val_mean_final, 4),
                "Разрыў (Gap)": round(gap, 4),
                "Вердыкт": status
            })

        # Загортваем у DataFrame і сартуем па павелічэнні разрыву (самыя стабільныя зверху)
        df_analysis = pd.DataFrame(analysis_list)
        return df_analysis.sort_values(by="Разрыў (Gap)")

    #     """Для 5 фолдаў (дзе навучальная выбарка — гэта максімум 80% ад агульных дадзеных) 
    #     падбіраем крокі з развагаў:
    #     Крок 1 (25% ад навучальнай): роўна 1 блок дадзеных (20% ад усяго датасэта).
    #     Крок 2 (50% ад навучальнай): роўна 2 блокі дадзеных (40% ад усяго датасэта).
    #     Крок 3 (75% ад навучальнай): роўна 3 блокі дадзеных (60% ад усяго датасэта).
    #     Крок 4 (100% ад навучальнай): усе 4 блокі дадзеных (80% ад усяго датасэта)
    #     """ 


    def collect_learning_curves_ES(
            self, 
            x: pd.DataFrame, 
            y: pd.Series, 
            selected_models: List[str] | None = None
        ) -> dict[str, dict[str, Any]]:
            """
            Вылічвае стратыфікаваныя крывыя навучання (крок 25%) толькі для абраных мадэлей.
            Дынамічна збірае ізаляваны чысты Pipeline і разлічвае каардынаты Early Stopping.
            """
            curves_results = {}
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

            # 1. Калі спіс лепшых мадэлей не перададзены, бярэм усе з даступных канфігурацый
            if selected_models is None:
                selected_models = list(self.models.keys())

            for name in selected_models:
                if "dummy" in name.lower():
                    continue
                
                # Перастрахоўка: правяраем, ці ёсць такая мадэль у нашых шаблонах
                if name not in self.models:
                    continue
                    
                # 2. ВЫПРАЎЛЕННЕ ВЫЦЕКУ: Атрымліваем АБСАЛЮТНА ЧЫСТЫЯ ізаляваныя кампаненты
                clean_preprocessor = self._get_adapted_preprocessor(name)

                # 1. РАСПАКОЎКА ПА НАЗВЕ: Дастаем базавы SVC толькі для SVM
                raw_template = self.models[name]
                
                if "svm" in name.lower() or "svc" in name.lower():
                    # Калі ўнутры SVM ляжыць каліброўка, забіраем яе базавы эстыматар
                    if hasattr(raw_template, "estimator"):
                        clean_clf = clone(raw_template.estimator)
                    else:
                        clean_clf = clone(raw_template)
                else:
                    # Усе астатнія мадэлі (уключаючы ExtraTrees) клануем цалкам як ёсць
                    clean_clf = clone(raw_template)

                # 2. АПТЫМІЗАЦЫЯ І ІЗАЛЯЦЫЯ ПАТОКАЎ
                name_lower = name.lower()

                if "lightgbm" in name_lower or "lgbm" in name_lower:
                    clean_clf.set_params(early_stopping_rounds=None, n_jobs=1)
                    if hasattr(clean_clf, "callbacks"):
                        clean_clf.set_params(callbacks=None)

                elif "xgb" in name_lower:
                    clean_clf.set_params(n_estimators=100, n_jobs=1)

                elif "extratrees" in name_lower:
                    # Цяпер ExtraTrees застаўся цэлым ансамблем, і n_jobs=1 спрацуе ідэальна!
                    clean_clf.set_params(n_jobs=1, random_state=self.random_state)

                elif "logisticregression" in name_lower:
                    clean_clf.set_params(max_iter=1500, random_state=self.random_state)

                elif "decisiontree" in name_lower:
                    clean_clf.set_params(random_state=self.random_state)
            
                

                # Збіраем свежы Pipeline з нуля (скалер будзе лічыцца строга ўнутры кожнага фолду)
                pipe_template = Pipeline(steps=[
                    ('preprocessor', clean_preprocessor),
                    ('model', clean_clf)
                ])

                stratified_steps = np.array([0.25, 0.50, 0.75, 1.0])

                # 3. Вылічэнне кропак для ліній графіку (25%, 50%, 75%, 100%)
                # n_jobs=1 ратуе ад завісання Tkinter
                train_sizes, train_scores, val_scores, *_ = learning_curve(
                    estimator=pipe_template, 
                    X=x, 
                    y=y.to_numpy().astype(int),
                    cv=kf, 
                    scoring="average_precision", 
                    train_sizes=stratified_steps, 
                    n_jobs=1,
                    error_score=0.0, 
                    random_state=self.random_state
                )

                train_mean = train_scores.mean(axis=1)
                val_mean = val_scores.mean(axis=1)
                gap_100 = float(train_mean[-1] - val_mean[-1])

                # Ініцыялізацыя каардынат зоркі
                best_es_iter = None
                gap_es = None
                val_score_es = float(val_mean[-1]) 

                # 4. АДНАПРАХОДНЫ РАЗЛІК ЭФЕКТЫЎНАСЦІ (Ваш зыходны код без зменаў)
                if hasattr(self, "cv_evals_results") and name in self.cv_evals_results:
                    best_iters = []
                    all_folds_metric_histories = [] 
                    
                    folds_dict = self.cv_evals_results[name]
                    for fold_data in folds_dict.values():
                        for dataset in fold_data.values():
                            for metric_name, values in dataset.items():
                                if any(m in metric_name.lower() for m in ["auc", "precision"]):
                                    clean_vals = [float(v) for v in values]
                                    if clean_vals:
                                        best_iters.append(clean_vals.index(max(clean_vals)))
                                        all_folds_metric_histories.append(clean_vals)

                    if best_iters and ("light" in name.lower() or "xgb" in name.lower()):
                        best_es_iter = int(np.mean(best_iters))
                        
                        val_scores_at_es = []
                        for history in all_folds_metric_histories:
                            idx = min(best_es_iter, len(history) - 1)
                            val_scores_at_es.append(history[idx])
                            
                        if val_scores_at_es:
                            val_score_es = float(np.mean(val_scores_at_es))
                            gap_es = float(train_mean[-1] - val_score_es)

                # Вызначэнне вердыкту
                final_check_gap = gap_es if gap_es is not None else gap_100
                if final_check_gap < 0.05:
                    verdict = "✓ Стабільная мадэль (gap < 0.05)"
                elif final_check_gap < 0.10:
                    verdict = "⚠️ Невялікае перанавучанне"
                else:
                    verdict = "Моцнае перанавучанне"

                # Захоўваем усё ў выніковы слоўнік
                curves_results[name] = {
                    "train_sizes": train_sizes,
                    "train_mean": train_mean,
                    "train_std": train_scores.std(axis=1),
                    "val_mean": val_mean,
                    "val_std": val_scores.std(axis=1),
                    "gap_100": gap_100,
                    "gap_es": gap_es,
                    "val_score_es": val_score_es,
                    "best_es_iter": best_es_iter, 
                    "verdict": verdict
                }
                
            return curves_results


    def analyze_learning_gaps_table(self, curves_data: dict[str, dict[str, Any]]) -> pd.DataFrame:
        """
        Аналізуе і параўноўвае два тыпы гэпаў з улікам двухпавярховай структуры cv_evals_results.
        Чыстая версія без лішніх праверак радкоў і рэгістраў.
        """
        analysis_list = []

        for model_name, data in curves_data.items():
            gap_100 = data["gap_100"]
            train_mean_final = data["train_mean"][-1]
            val_mean_final = data["val_mean"][-1]
            
            best_iters = []
            val_scores_at_es = []
            
            # Наўпрост звяртаемся да патрэбнай мадэлі ў слоўніку без лішніх цыклаў
            if hasattr(self, "cv_evals_results") and model_name in self.cv_evals_results:
                folds_dict = self.cv_evals_results[model_name]
                
                # Збіраем індэксы найлепшых ітэрацый па фолдах
                for fold_data in folds_dict.values():
                    for dataset in fold_data.values():
                        for metric_name, values in dataset.items():
                            if any(m in metric_name.lower() for m in ["auc", "precision"]):
                                clean_vals = [float(v) for v in values]
                                if clean_vals:
                                    best_iters.append(clean_vals.index(max(clean_vals)))

            # Калі знайшлі кропкі прыпынку (гэта бустынг) — вылічваем рэальны рабочы гэп
            if best_iters:
                mean_es_tree = int(np.mean(best_iters))
                
                # Збіраем якасць менавіта на гэтым аптымальным дрэве
                folds_dict = self.cv_evals_results[model_name]
                for fold_data in folds_dict.values():
                    for dataset in fold_data.values():
                        for metric_name, values in dataset.items():
                            if any(m in metric_name.lower() for m in ["auc", "precision"]):
                                idx = min(mean_es_tree, len(values) - 1)
                                val_scores_at_es.append(float(values[idx]))
                
                val_score_es = float(np.mean(val_scores_at_es)) if val_scores_at_es else val_mean_final
                gap_es = float(train_mean_final - val_score_es)
                
                best_es_str = f"Дрэва №{mean_es_tree}"
                gap_es_str = f"{gap_es:.4f}"
                sort_val = gap_es  # Лік для правільнага сартавання
            else:
                # Класічныя мадэлі без Early Stopping (DecisionTree)
                best_es_str = "Не выкар."
                gap_es_str = "-"
                sort_val = 9.99  # Скідаем класічныя мадэлі ў канец табліцы
 

            # Выніковы вердыкт (строга па рабочым гэпе, калі ён ёсць, іначай)
            final_gap = gap_es if best_iters else gap_100
            if final_gap < 0.05:
                verdict = "✓ Стабільная мадэль (gap < 0.05)"
            elif final_gap < 0.10:
                verdict = "⚠️ Невялікае перанавучанне"
            else:
                verdict = "Моцнае перанавучанне"

            analysis_list.append({
                "Мадэль": model_name,
                "Gap 100% (Без EarlStop)": f"{gap_100:.4f}",
                "Gap EarlStop (Працоўны)": gap_es_str,
                "Кропка прыпынку": best_es_str,
                "Высновы": verdict,
                "sort_key": sort_val  # Нябачны лікавы ключ для бяспечнага сартавання
            })

        df_report = pd.DataFrame(analysis_list)

        # Сартуем па тэхнічным лікавым ключы, каб пазбегнуць крашу з-за сімвала "-"
        df_report = df_report.sort_values(by="sort_key")

        return df_report.drop(columns=["sort_key"])

    def verify_optimized_models(self, optimized_results: dict[str, dict[str, Any]], x: pd.DataFrame, y: pd.Series) -> None:
        """
        Архітэктурны менеджэр верыфікацыі: часова падмяняе слоўнік мадэлей,
        запускае cross_validate_newV толькі для аптымізаваных мадэлей і друкуе справаздачу.
        """

        if not optimized_results:
            print("[Sanity Check Error] Спіс аптымізаваных мадэлей пусты. Няма чаго верыфікаваць.")
            return

        
        # 1. Захоўваем арыгінальны зыходны слоўнік мадэлей, каб не зламаць клас на будучыню
        original_models_dict = self.models.copy()
        tuned_models_dict = {}

        # 2. Вычышчаем і напаўняем часовы слоўнік толькі пракачанымі шаблонамі
        for m_name, best_params in optimized_results.items():
            if m_name in original_models_dict:
                print(f"[Sanity Check] Наладжваем параметры шаблона для {m_name}...")
                
                # Чысты клон структуры Pipeline (абнуленне памяці)
                clean_template = clone(original_models_dict[m_name])
                
                # Копіруем параметры, каб не зацерці арыгінал, і ставім максімальную хуткасць
                copied_params = best_params.copy()
                copied_params["n_jobs"] = -1
                clean_template.set_params(**copied_params)
                
                tuned_models_dict[m_name] = clean_template

        # 3. Галоўная архітэктурная падмена: ізалюем бэнчмарк
        self.models = tuned_models_dict

        print(f"[Sanity Check] Запуск крос-валідацыі для кастомнага спісу: {list(tuned_models_dict.keys())}...")
        # 4. Выклікаем унутраны метад крос-валідацыі (ён адпрацуе імгненна толькі па патрэбных мадэлях)
        opt_fold_pr, opt_fold_roc, opt_oof_preds = self.cross_validate_newV(x, y)

        # 5. Вывад фінальнай інфарматыўнай справаздачы эфектыўнасці
        print("\n" + "="*75)
        print("ФІНАЛЬНАЯ СПРАВАЗДАЧА ЭФЕКТЫЎНАСЦІ ПАСЛЯ OPTUNA (GAPS REPORT):")
        print("="*75)
        
        for m_name in tuned_models_dict.keys():
            mean_val = float(np.mean(opt_fold_pr[m_name]))
            
            # Дастаем трэйн з унутранай гісторыі cv_evals_results
            train_scores_from_history = []
            if hasattr(self, "cv_evals_results") and m_name in self.cv_evals_results:
                folds_dict = self.cv_evals_results[m_name]
                for fold_data in folds_dict.values():
                    for dataset_name, metrics_dict in fold_data.items():
                        if "train" in dataset_name.lower():
                            for metric_name, values in metrics_dict.items():
                                if any(m in metric_name.lower() for m in ["auc", "precision"]):
                                    if values:
                                        train_scores_from_history.append(float(values[-1]))
            
            mean_train = float(np.mean(train_scores_from_history)) if train_scores_from_history else 1.0
            new_gap = mean_train - mean_val

            self.last_verify_train = mean_train
            self.last_verify_val = mean_val
            self.last_verify_gap = new_gap
       

            print(f"Мадэль: {m_name}")
            print(f"  -> Новы Сярэдні Train PR-AUC: {mean_train:.4f}")
            print(f"  -> Новы Сярэдні Val PR-AUC:   {mean_val:.4f}")
            print(f"  => Новы выніковы Gap:          {new_gap:.4f}")
            
            if new_gap < 0.05:
                print("  => Вердыкт: ✓ Перанавучанне цалкам ПЕРАМАГЛІ (gap < 0.05)")
            elif new_gap < 0.10:
                print("  => Вердыкт: ⚠️ Невялікае бяспечнае перанавучанне (gap < 0.10)")
            else:
                print("  => Вердыкт: ✗ Мадэль усё яшчэ схільная да перанавучання")
            print("-" * 55)
            
        print("="*75 + "\n")
        
        # 6. Абавязкова вяртаем класу яго зыходны поўны слоўнік мадэлей (рэстаўрацыя стану)
        self.models = original_models_dict

    def compare_before_after_optuna(self, base_curves_data: dict[str, dict[str, Any]], optimized_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
        """
        Універсальны метад: проста параўноўвае статыстыку мадэлі ДА і ПАСЛЯ Optuna.
        Чыстая версія: забірае гатовыя вылічэнні з атрыбутаў класа.
        """
        # import pandas as pd

        comparison_list = []

        for model_name in optimized_results.keys():
            # 1. Дадзеныя ДА аптымізацыі (з першапачатковых крывых)
            if model_name in base_curves_data:
                b_data = base_curves_data[model_name]
                val_before = float(b_data["val_mean"][-1])
                gap_before = float(b_data["gap_es"] if b_data.get("gap_es") is not None else b_data["gap_100"])
                tree_before = f"Дрэва №{b_data['best_es_iter']}" if b_data.get("best_es_iter") is not None else "100% выбаркі"
            else:
                val_before, gap_before, tree_before = 0.0, 0.0, "Няма дадзеных"

            # 2. ВЫПРАЎЛЕНА: Ніякіх паўторных цыклаў! Проста забіраем лічбы з атрыбутаў
            val_after = getattr(self, "last_verify_val", val_before)
            gap_after = getattr(self, "last_verify_gap", 0.0)
            
            total_trees = optimized_results[model_name].get("n_estimators", 400)
            tree_after = f"{total_trees} дрэў"

            # 3. Лічым чыстую дэльту якасці
            auc_delta = val_after - val_before

            comparison_list.append({
                "Параметр аналізу": [
                    "Валідацыйны PR-AUC (Якасць)", 
                    "Памер ансамбля (Колькасць дрэў)", 
                    "Перанавучанне (Gap)"
                ],
                "ДА аптымізацыі (Базавы + ES)": [
                    f"{val_before:.4f}", 
                    str(tree_before), 
                    f"{gap_before:.4f}"
                ],
                "ПАСЛЯ Optuna (Стрэс-тэст)": [
                    f"{val_after:.4f}", 
                    str(tree_after), 
                    f"{gap_after:.4f}"
                ],
                "Чысты эфект аптымізацыі": [
                    f"{auc_delta:+.4f} дэльта якасці",
                    f"Павелічэнне до {tree_after}",
                    "Штучны аверфіт (без ES)"
                ]
            })

        dfs = [pd.DataFrame(res) for res in comparison_list]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def final_secure_refit(
        self, 
        final_configs: dict[str, Any],  
        x_train: pd.DataFrame, 
        y_train: pd.Series, 
        x_val: pd.DataFrame, 
        y_val: pd.Series, 
        evaluator: Any,  
        random_state: int = 42
    ) -> tuple[Any, float]:  
        """
        КРОК 4.1: Поўнае фінальнае навучанне на X_cv (без удзелу X_test).
        Навучае мадэль з Early Stopping і падбірае лепшы парог для F2.
        """
        # 1. Забіраем першае імя мадэлі са слоўніка як радок (напрыклад, 'LightGBM')
        model_name = list(final_configs.keys())[0]
        
        if model_name not in self.models:
            raise KeyError(f"Мадэль '{model_name}' адсутнічае ў self.models.")

        print(f"\n[Secure Refit] Старт навучання фінальнага пайплайна для {model_name}...")
        
        # 2. Дастаем ГОЛУЮ мадэль з канфігурацыі і клануем яе
        base_model = final_configs[model_name]
        clf_model = clone(base_model)
        
        # 3. Налада параметраў ранняга прыпынку прама ў мадэлі (без named_steps!)
        if hasattr(clf_model, "set_params"):
            clf_model.set_params(early_stopping_rounds=30, n_jobs=1, random_state=random_state)

        # 4. Збіраем СВЕЖЫ прэпрацэсар праз ваш адаптатар
        preprocessor = self._get_adapted_preprocessor(model_name)
        
        # Трансфармацыя дадзеных strictly ўнутры метаду
        x_refit_train_trans = preprocessor.fit_transform(x_train, y_train)
        x_refit_eval_trans = preprocessor.transform(x_val)

        # Скідаем індэксацыю масіваў
        y_tr_np = y_train.to_numpy().astype(int)
        y_va_np = y_val.to_numpy().astype(int)

        print(f"[Secure Refit] Навучанне мадэлі з Early Stopping...")
        if "light" in model_name.lower():
            clf_model.set_params(verbose=-1)
            # clf_model.fit(x_refit_train_trans, y_tr_np, eval_set=[(x_refit_eval_trans, y_va_np)])
            clf_model.fit(x_refit_train_trans, y_tr_np, eval_X=x_refit_eval_trans, eval_y=y_va_np)
        elif "xgb" in model_name.lower():
            # clf_model.fit(x_refit_train_trans, y_tr_np, eval_set=[(x_refit_eval_trans, y_va_np)], verbose=False)
            clf_model.fit(x_refit_train_trans, y_tr_np, eval_X=x_refit_eval_trans, eval_y=y_va_np, verbose=False)
        else:
            clf_model.fit(x_refit_train_trans, y_tr_np)
        
        # 5. ЗБІРАЕМ СУЦЭЛЬНЫ ПАЙПЛАЙН ДЛЯ ТЭСТУ МЕНАВІТА ТУТ
        final_pipe = Pipeline([('preprocessor', preprocessor), ('model', clf_model)])
        self.fitted[model_name] = final_pipe
        
        # 6. Аўтаматычны падбор бізнес-парога па валідацыйных прагнозах
        val_probas = clf_model.predict_proba(x_refit_eval_trans)[:, 1]
        
        best_threshold, _ = evaluator.optimize_threshold_for_fbeta(            
            y_true=y_val, 
            oof_probas=val_probas, 
            model_name=model_name,
            beta=2.0
        )
        
        return final_pipe, best_threshold


    def predict_with_final_threshold(self, trained_pipeline, x_test: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Прагноз на ТЭСТАВАЙ выбарцы з выкарыстаннем зафіксаванага парога.
        Вяртае: (test_proba, test_pred_binary)
        """
        print(f"[Final Predict] Адзін чысты прагноз на тэсце з парогам {threshold:.2f}...")
        
        test_proba = trained_pipeline.predict_proba(x_test)[:, 1]
        
        test_pred_binary = (test_proba >= threshold).astype(int)
        
        return test_proba, test_pred_binary

    def evaluate_baseline_models(self, split: Any, evaluator: Any, random_state: int = 42) -> pd.DataFrame:
        """
        МЕНЭДЖЭР ЭТАПУ 1: Запускае пачатковую ацэнку ўсіх мадэлей на спліце 60/20/20.
        Чыстая версія: выкарыстоўвае адзіны адаптатар без ручных індэксаў.
        """
        metrics_table = pd.DataFrame()

        for name, template in self.models.items():

            if name == "Dummy":
                print("--- Запуск Dummy у якасці baseline ---")
                print("Маштабаванне прыкмет Dummy не патрабуе")
                dummy_row = self.run_dummy_baseline(split=split, evaluator=evaluator, random_state=random_state)
                metrics_table = pd.concat([metrics_table, dummy_row], ignore_index=True)
                continue # Імгненна пераходзім да наступнай мадэлі

            if name == "LogisticRegression":
                print("--- Запуск Logistic Regression ---")
                print("Маштабаванне прыкмет абавязковае")
                preprocessor = self._get_adapted_preprocessor(model_name=name)
                lr_row = self.run_logistic_regression(split=split, preprocessor=preprocessor, evaluator=evaluator)
                metrics_table = pd.concat([metrics_table, lr_row], ignore_index=True)

            elif name == "KNN":   
                print("--- Запуск KNN ---")
                print("Маштабаванне прыкмет абавязковае (Robust + Calibrated Config)")
                preprocessor = self._get_adapted_preprocessor(model_name=name)
                knn_row = self.run_knn(split=split, preprocessor=preprocessor, evaluator=evaluator)
                metrics_table = pd.concat([metrics_table, knn_row], ignore_index=True)

            elif name == "SVM":
                print("--- Запуск SVM ---")
                print("Маштабаванне прыкмет абавязковае (Robust + Calibrated Config)")
                preprocessor = self._get_adapted_preprocessor(model_name=name)
                svm_row = self.run_svm(split=split, preprocessor=preprocessor, evaluator=evaluator)
                metrics_table = pd.concat([metrics_table, svm_row], ignore_index=True)

            elif name == "DecisionTree":
                print("--- Запуск Дрэва рашэнняў ---")
                print("Маштабаванне катэгарыяльных прыкмет абавязковае (Balanced + Raw Numeric)")
                preprocessor = self._get_adapted_preprocessor(model_name=name)
                dt_row = self.run_decision_tree(split=split, preprocessor=preprocessor, evaluator=evaluator)
                metrics_table = pd.concat([metrics_table, dt_row], ignore_index=True)

            elif name == "ExtraTrees":
                print("--- Запускаем ансамбль ExtraTrees ---")
                print("Маштабаванне катэгарыяльных прыкмет абавязковае (Balanced + Raw Numeric)")
                preprocessor = self._get_adapted_preprocessor(model_name=name)
                et_row = self.run_extra_trees(split=split, preprocessor=preprocessor, evaluator=evaluator)
                metrics_table = pd.concat([metrics_table, et_row], ignore_index=True)

        return metrics_table

    def _get_adapted_preprocessor(self, model_name: str) -> Any:
        """
        УНУТРАНЫ АДАПТАТАР: прымае імя мадэлі і вяртае індывідуальна 
        настроены, чысты і абнулены клон self.preprocessor.
        """

        if self.preprocessor is None:
            raise AttributeError("[Benchmark Error] self.preprocessor не быў ініцыялізаваны з main.py.")

        # 1. Ствараем абсалютна свежы, сляпы клон эталоннага прэпрацэсара
        preprocessor_clone = clone(self.preprocessor)
        
        # 2. Вызначаем патрэбны матэматычны скалер паводле прыроды сямействаў мадэлей
        ROBUST_REQUIRED = ["KNN", "SVM"]
        NO_SCALE_REQUIRED = ["DecisionTree", "ExtraTrees", "LightGBM", "XGBoost", "CatBoost"]

        if any(m in model_name for m in NO_SCALE_REQUIRED):
            scaler = 'passthrough'
        elif any(m in model_name for m in ROBUST_REQUIRED):
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        raw_transformers = self.preprocessor.get_params().get("transformers", [])
        
        if not raw_transformers:
            raise ValueError("[Benchmark Error] Структура transformers пустая. Праверце create_preprocessor.")

        # 3. Дынамічна выцягваем спісы калонак і трансфарматары з нашага афіцыйнага эталона
        # preprocessor.transformers змяшчае спіс картэжаў.
        # [0][2] — гэта спіс true_numeric_cols, [1] — cat_num_ohe, [2] — cat_str_ohe, [3] — cat_ord
        
        preprocessor_clone.transformers = [
            ('num', scaler, raw_transformers[0][2]),
            ('cat_num_ohe', raw_transformers[1][1], raw_transformers[1][2]),
            ('cat_str_ohe', raw_transformers[2][1], raw_transformers[2][2]),
            ('cat_ord', raw_transformers[3][1], raw_transformers[3][2])
        ]
                
        return preprocessor_clone

    def prepare_final_models(
        self, 
        best_name: str | None = None, 
        optimized_results: Dict[str, Dict[str, Any]] | None = None, 
        models_to_keep: list[str] | None = None
    ) -> dict[str, Any]:    
        """
        Стварае слоўнік фінальных канфігурацый мадэлей.
        
        Калі best_name ці optimized_results пустыя — вяртае чыстыя клоны мадэляў.
        Калі models_to_keep не зададзены, а best_name пусты — бярэ ўсе мадэлі з self.models.
        """
        # 1. Вызначаем спіс мадэлей для апрацоўкі
        if models_to_keep is None:
            if best_name:
                models_to_keep = [best_name]
            else:
                # Калі найлепшай мадэлі няма — бяром наогул усе мадэлі, што ёсць у бэнчмарку
                models_to_keep = list(self.models.keys())
        
        final_configs: dict[str, Any] = {}
        # Бяспечна прыводзім optimized_results да слоўніка, калі прыйшоў None
        opt_res = optimized_results or {}

        for name in models_to_keep:
            if name not in self.models:
                print(f"[Warning] Мадэль '{name}' не знойдзена ў агульным спісе.")
                continue
                
            # Клануем мадэль
            model_copy = clone(self.models[name])
            
            # 2. Накідваем параметры, толькі калі зададзена best_name і гэта менавіта яна
            if best_name and name == best_name:
                model_entry = opt_res.get(name, {})
                
                if isinstance(model_entry, dict) and "params" in model_entry:
                    best_params = model_entry["params"]
                else:
                    best_params = model_entry  # Страхоўка, калі там ляжаць проста параметры
                
                if best_params:  # Правераем, ці не пусты сам слоўнік параметраў
                    model_copy.set_params(**best_params)
                
            final_configs[name] = model_copy
            
        return final_configs

    def collect_lgbm_validation_curves(
        self, 
        x: pd.DataFrame, 
        y: pd.Series, 
        param_name: str = "num_leaves",
        param_range: list | np.ndarray | None = None,
        base_params: dict | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Вылічвае крос-валідацыйныя крывыя валідацыі для LightGBM па зададзеным параметру.
        Дынамічна збірае ізаляваны чысты Pipeline для прадухілення ўцечкі даных (Data Leakage),
        аўтаматычна імпартуючы ўсе параметры з існуючага шаблону мадэлі.
        """
        if param_range is None:
            param_range = [7, 15, 31, 63, 127]
            
        if base_params is None:
            base_params = {}

        curves_results = {}
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        model_key = "LightGBM"
        
        # 1. ВЫПРАЎЛЕННЕ ВЫЦЕКУ: Атрымліваем АБСАЛЮТНА ЧЫСТЫЯ ізаляваныя кампаненты
        clean_preprocessor = self._get_adapted_preprocessor(model_key)
        
        # 2. АЎТАМАЦЫЧНЫ ЗБОР ПАРАМЕТРАЎ З ШАБЛОНУ (self.models)
        # Выкарыстоўваем каментарый альбо лакальную анатацыю, каб растлумачыць, што гэта можа быць Pipeline або Any мадэль
        raw_template = self.models[model_key]
        
        # Бяспечна дастаем параметры як звычайны слоўнік
        if isinstance(raw_template, Pipeline):
            model_params = dict(raw_template.named_steps['model'].get_params())
        else:
            # Выклікаем get_params() і відавочна прыводзім да dict, каб ачысціць сувязь з SVC
            model_params = dict(raw_template.get_params())
            
        # ВЫКЛЮЧАЕМ параметр, які ідзе пад замену на графіку, каб пазбегнуць канфліктаў
        model_params.pop(param_name, None)
        
        # Збіраем фінальныя kwargs: шаблонныя параметры + дадатковыя карыстальніцкія
        lgbm_kwargs = {
            **model_params,
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": 1,
            **base_params
        }
        
        clean_clf = LGBMClassifier(**lgbm_kwargs)
        
        # Збіраем свежы Pipeline з нуля. 
        # Відавочна анатаруеце тып змяняемага пайплайна як Pipeline з sklearn.pipeline
        pipe_template = Pipeline(steps=[
            ('preprocessor', clean_preprocessor),
            ('model', clean_clf)
        ])
        
        print(f"[{self.__class__.__name__}] Крос-валідацыя LightGBM па параметру '{param_name}'...")

        # 3. Вылічэнне кропак для ліній графіку праз убудаваны validation_curve
        # Звяртаемся да параметру мадэлі ўнутры Pipeline праз 'model__'
        train_scores, val_scores = validation_curve(
            estimator=pipe_template, # type: ignore[reportArgumentType]
            X=x,
            y=y.to_numpy().astype(int),
            param_name=f"model__{param_name}",
            param_range=param_range,
            cv=kf,
            scoring="f1",
            n_jobs=1
        )
        
        # Разлік сярэдніх па фолдах
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)
        gap_100 = float(train_mean[-1] - val_mean[-1])
        
        # 4. Вызначэнне вердыкту па велічыні перанавучання (Gap)
        if gap_100 < 0.05:
            verdict = "✓ Стабільная мадэль (gap < 0.05)"
        elif gap_100 < 0.10:
            verdict = "⚠️ Невялікае перанавучанне"
        else:
            verdict = "Моцнае перанавучанне"
            
        # 5. Захоўваем усё ў лаканічны слоўнік
        curves_results[model_key] = {
            "param_name": param_name,
            "param_range": np.asarray(param_range),
            "train_mean": train_mean,
            "train_std": train_scores.std(axis=1),
            "val_mean": val_mean,
            "val_std": val_scores.std(axis=1),
            "gap_100": gap_100,
            "verdict": verdict
        }
        
        return curves_results