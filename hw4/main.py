import warnings
# Глушым стандартны клас папярэджанняў Python
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Глушым любыя папярэджанні, якія ўтрымліваюць тэкст пра eval_set
warnings.filterwarnings("ignore", message=".*eval_set.*is deprecated.*")

import gc

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sys
import numpy as np
from pathlib import Path

from modules import (
    BayesianTuner,
    ClassifierBench,
    DataVisualizer,
    ModelEvaluator,
    PrepareDataset,    
)
    

ROOT = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd() # рашэнне для .ipny  і .py
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DATA = ROOT / "data"
OUT = ROOT / "output"

def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evaluator = ModelEvaluator() #(n_splits=5, random_state=split.random_state)
    bench = ClassifierBench()
    viz = DataVisualizer(OUT)
    dataset = PrepareDataset(DATA)


    section("1. Датасет Online Shoppers Purchasing Intention")

    file_name="online_shoppers_intention.csv"
    print(f"Крыніца даных: {DATA/file_name}")

    frame = dataset.load_csv(file_name=file_name)
    print("Слупкі frame:", frame.columns.tolist()) 

    # # Будуем матрыцу карэляцыі для ўсіх прыкмет, выключыўшы таргет "Revenue"
    plots_list = []
    plot_path = viz.plot_correlation_matrix(x_train=frame.iloc[:, :-1], filename="CM_engin_feat.png")
    plots_list.append(plot_path)
    # # Feature Engineering і Оптымізацыя прыкмет      
    # frame = dataset.engineer_features(frame)

    target_name = "Revenue"
    
    plot_path = viz.plot_class_balance(frame[target_name], name=target_name)
    plots_list.append(plot_path)

    # 1. Спіс ТОЛЬКІ тых калонак-прыкмет, для якіх OneHotEncoder патрабуе катэгорыі
    ohe_feature_list = ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]

    # 2. Збіраем метаданыя АЎТАМАЦЫЧНА да падзелу (выключаючы таргет "Revenue")
    known_categories = {
        col: sorted(list(frame[col].dropna().unique())) for col in ohe_feature_list
    }
            
    section("2. Падзел дадзеных спосабам 'static_3way' і падрыхтоўка фармата да 'cv_holdout' ")

    split = dataset.prepare(frame, target_name = target_name, test_size = 0.2, val_size = 0.2, random_state=42)   

    bench.build(split.y_train)
    
    # bench.preprocessor = dataset.create_preprocessor2(df=frame)
    # А функцыя create_preprocessor проста бярэ гэты гатовы слоўнік
    bench.preprocessor = dataset.create_preprocessor(categories_dict=known_categories)
    
    section("3. Classifier на падзеле (60/20/20) ")
    
    metrics_table = bench.evaluate_baseline_models(
        split=split,        
        evaluator=evaluator,
        random_state=42
    )
        
    print("\n=== Параўноўчая табліца метрык для падзелу 60/20/20 ===")
    print(metrics_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


    section("4. Classifier на Крос-Валідацыі праз StratifiedKFold(5) ")

    CV_methods_names = ["LogisticRegression", "DecisionTree", "ExtraTrees", "LightGBM", "XGBoost"]
    CV_configs = bench.prepare_final_models(models_to_keep=CV_methods_names)
    
    fold_pr2, fold_roc2, oof_predictions = bench.cross_validate_newV(split.x_cv, split.y_cv, models_conf=CV_configs)

    # табліца метрык PR i ROC
    cv_metrics_table = bench.get_metrics_table()
    print("\n[Cross-Validation] Табліца метрык PR i ROC на StratifiedKFold(5), адсартавана па метрыцы выбару PR-AUC:")
    print(cv_metrics_table.to_string(index=False))

    # df_base_report = evaluator.compile_performance_report(
    #     y_true=split.y_cv, 
    #     probas_dict=oof_predictions, 
    #     threshold=0.20
    # )
    # print(df_base_report.to_string(index=False))

    best_threshold, best_score = evaluator.optimize_threshold_for_fbeta(            
            y_true=split.y_cv, 
            oof_probas=oof_predictions["LightGBM"], 
            model_name="LightGBM",
            beta=2.0
        )
    
    df_base_report = evaluator.compile_performance_report(
            y_true=split.y_cv, 
            probas_dict=oof_predictions, 
            threshold=best_threshold
        )
    print(df_base_report.to_string(index=False))
    
    print("\n[Cross-Validation] Малюем і захоўваем PR-крывую для ўсіх мадэляў") 
    plot_path = viz.plot_pr_curve_cv(x=split.x_train, y=split.y_train, models_dict=CV_configs, preprocessor=bench.preprocessor)
    plots_list.append(plot_path)
    
    print("\n[Cross-Validation] boxplot ") 
    plot_path = viz.plot_models_boxplot(fold_pr2, metric_name="PR-AUC", filename="models_stability_pr_auc.png")
    plots_list.append(plot_path)

    cv_table = evaluator.cv_table(fold_pr2, fold_roc2)
    cv_pr_auc = viz.plot_cv_pr_auc(cv_table)
    plots_list.append(cv_pr_auc)

    print("\n[Cross-Validation] Малюем і захоўваем ROC-крывую для ўсіх мадэляў")
    roc_plot_path = viz.plot_roc_curve_cv(x=split.x_train, y=split.y_train, models_dict=CV_configs, preprocessor=bench.preprocessor)    
    plots_list.append(roc_plot_path)    

    print("\n[Cross-Validation] Прагноз метадаў для 40 назіранняў: па 20 на кожны клас ")
    path_heat = viz.plot_predictions_comparison(y_true=np.asarray(split.y_train), preds_dict=oof_predictions, num_samples_per_class=20)
    plots_list.append(path_heat)
    
    # Для бустынгаў выклік функцыі ранняга прыпынку па фолдах
    if bench.cv_evals_results:
        print("\n[Cross-Validation] Выклік функцыі ранняга прыпынку па фолдах для бустынгаў")
        for model_name, evals_data in bench.cv_evals_results.items():
            path_early_stopping = viz.plot_boosting_early_stopping(
                evals_result=evals_data, 
                filename=f"cv_es_{model_name.lower()}.png"
            )
            plots_list.append(path_early_stopping)
    
    if bench.cv_evals_results:
        print("\n[Cross-Validation] Выклік функцыі графікаў Train vs Validation для бустынгаў")
        for model_name, evals_data in bench.cv_evals_results.items():
            path_train_vs_val = viz.plot_boosting_train_vs_val(
                evals_result=evals_data, 
                filename=f"train_vs_val_{model_name.lower()}.png"
            )
            plots_list.append(path_train_vs_val)


    plt.close('all')  # Закрываем усё, што магло выпадкова застацца адкрытым
    gc.collect()      # Прымусова чысцім памяць прама ЗАРАЗ у галоўным патоку

    section("5. Выбар мадэлі па парным t-тэсце Сцюдэнта")
    # Выбар паміж двума лепшымі па PR
    best_name = evaluator.pick_best(fold_pr2)   
    
    # Матрыца памылак для выбранай топ-мадэлі па яе імені карыстаецца гатовым вектарам імавернасцяў OOF
    cm_path = viz.plot_classification_results(
        #y_true=split.y_cv, 
        y_true=split.y_cv.to_numpy(),
        y_probas=oof_predictions[best_name], 
        model_name=best_name,
        threshold=best_threshold,#0.20, # наладжваем парог пад нашы 15% канверсіі
        filename=f"confusion_matrix_{best_name}.png"
    )
    plots_list.append(cm_path)

    # Важнасць прыкмет карыстаецца гатовым навучаным пайплайнам
    importance_path = viz.plot_feature_importance(model_pipeline=bench.fitted[best_name], model_name=best_name, filename=f"feature_importance_{best_name}.png")
    plots_list.append(importance_path)        

    plt.close('all')  # Закрываем усё, што магло выпадкова застацца адкрытым
    gc.collect()      # Прымусова чысцім памяць прама ЗАРАЗ у галоўным патоку
    
    section("6. Learning curves. Эфектыўнасць выкарыстання дадзеных ")
    # print("Праверка на перанавучанне метадаў. Залежнасць train/val score ад памеру выбаркі ")
    
    print("\n[Visualizer] Вылічэнне крывых навучання...")

    LC_methods_names = ["LogisticRegression", "DecisionTree", "ExtraTrees", "LightGBM", "XGBoost"]
    # LC_configs = bench.prepare_final_models(models_to_keep=LC_methods_names)
        
    learning_curves_data_ES = bench.collect_learning_curves_ES(split.x_cv, split.y_cv, LC_methods_names)
    
    path_es = viz.plot_learning_curves_ES(learning_curves_data_ES, filename="learning_curves_ES.png")
    plots_list.append(path_es)
        
    print("Параўнальны аналіз перанавучання (БЕЗ Early Stopping vs З УЛІКАМ Early Stopping).")
    print("Высновы (ацэнка gap) па Early Stopping, калі ён мае месца быць, іначай напрыканцы кросвалідацыі на 100% (7891) даных, вылучаных пад яе")
                
    df_comparison = bench.analyze_learning_gaps_table(learning_curves_data_ES)
    print(df_comparison.to_string(index=False))  
        
    plt.close('all')  # Закрываем усё, што магло выпадкова застацца адкрытым
    gc.collect()      # Прымусова чысцім памяць прама ЗАРАЗ у галоўным патоку

    section("6.2 Валідацыйныя крывыя. Аптымізацыя LightGBM і аналіз перанавучання")
    
    
    leaves_range = [2, 4, 8, 10, 15, 20, 31]
    # Задаем толькі той дыяпазон лісця, які хочам даследаваць на графіку
    
    # Вылічэнне ў модулі класіфікатара
    lgbm_results = bench.collect_lgbm_validation_curves(
        x=split.x_cv,  # Перадаем прыкметы для CV
        y=split.y_cv,  # Перадаем таргет для CV
        param_name="num_leaves",
        param_range=leaves_range,
        base_params={"max_depth": -1}  # Дазваляем лісцю расці #     base_params={}  # - перазапісаць дэфолтны значэнне параметру мадэлі
    )
    
    val_curve_path = viz.plot_validation_curve(curves_results=lgbm_results, model_name="LightGBM", filename="lgbm_leaves_curve.png")
    plots_list.append(val_curve_path)


    section("7. БАЕСАЎСКАЯ АПТЫМІЗАЦЫЯ ПАРАМЕТРАЎ (OPTUNA)")

    tuner = BayesianTuner(bench_instance=bench, random_state=42)
    
    # Выклікаем адзіны інкапсуляваны метад-менеджэр
    optimized_results = tuner.tune_or_load(best_name, split.x_cv, split.y_cv)
    
    print("СПРАВАЗДАЧА ПАДБОРУ ГІПЕРПАРАМЕТРАЎ:")
    for m_name, params in optimized_results.items():
        print(f"Мадэль: {m_name}")
        for p_key, p_val in params.items():
            print(f"  -> {p_key}: {p_val:.5f}" if isinstance(p_val, float) else f"  -> {p_key}: {p_val}")

    # section ("Стрэс-тэст. Верыфікацыя параметраў Optuna на крос-валідацыі.")
    # print("Адлучаны механізм Early Stoping .")
    # # Выклік ізаляванай верыфікацыі
    # bench.verify_optimized_models(optimized_results, split.x_cv, split.y_cv)

    # print("Стрэс-тэст. МАТЭМАТЫЧНАЕ ПАРАЎНАННЕ ЭФЕКТЫЎНАСЦІ: ДА vs ПАСЛЯ OPTUNA")
    # # Перадаем першапачатковыя крывыя (вынікі Этапу 1) і вынікі Оптуны
    # df_delta = bench.compare_before_after_optuna(learning_curves_data_ES, optimized_results)
    # print(df_delta.to_string(index=False))
    
    plt.close('all')  # Закрываем усё, што магло выпадкова застацца адкрытым
    gc.collect()      # Прымусова чысцім памяць прама ЗАРАЗ у галоўным патоку

    section("Фінальная валідацыя ")

    final_valid_name = [best_name]
    final_configs = bench.prepare_final_models(best_name, optimized_results, final_valid_name)

    opt_pr, opt_roc, opt_oof = bench.cross_validate_newV(split.x_cv, split.y_cv, models_conf=final_configs)

    best_threshold, best_score = evaluator.optimize_threshold_for_fbeta(            
            y_true=split.y_cv, 
            oof_probas=opt_oof[best_name], 
            model_name=best_name,
            beta=2.0
        )
        
    df_opt_oof_report = evaluator.compile_performance_report(
        y_true=split.y_cv, 
        probas_dict=opt_oof, 
        threshold=best_threshold       
    )
    print(f"\n[Report] МЕТРЫКІ МАДЭЛІ ПАСЛЯ OPTUNA НА ВАЛІДАЦЫІ (5-Fold OOF, аптымальны парог = {best_threshold}):")
    print(df_opt_oof_report.to_string(index=False))

    # 4. Малюем матрыцу памылак валідацыі
    cm_path = viz.plot_classification_results(
        y_true=split.y_cv.to_numpy(), 
        y_probas=opt_oof[best_name], 
        model_name=best_name,
        threshold=best_threshold,  
        filename=f"confusion_matrix_CV_{best_name}.png"
    )
    plots_list.append(cm_path)

    plt.close('all')  # Закрываем усё, што магло выпадкова застацца адкрытым
    gc.collect()      # Прымусова чысцім памяць прама ЗАРАЗ у галоўным патоку
        
    section("SECURE REFIT")

    final_methods = [best_name]
    final_configs = bench.prepare_final_models(
        best_name=best_name, 
        optimized_results=optimized_results, 
        models_to_keep=final_methods
    )

    final_test_pipeline, best_threshold = bench.final_secure_refit(
        final_configs=final_configs,
        x_train=split.x_train,
        y_train=split.y_train,
        x_val=split.x_val,
        y_val=split.y_val,
        evaluator=evaluator
    )

    section("Test")

    test_proba, test_pred = bench.predict_with_final_threshold(
            trained_pipeline=final_test_pipeline,
            x_test=split.x_test,
            threshold=best_threshold
        )    

    section("Табліца метрык на тэставай выбарцы")
  
    test_probas_dict = {f"{best_name} (Фінальны тэст)": test_proba}
    df_final_test_report = evaluator.compile_performance_report(
        y_true=split.y_test,
        probas_dict=test_probas_dict,
        threshold=best_threshold
    )
    print(df_final_test_report.to_string(index=False))

    final_cm_path = viz.plot_classification_results(
        y_true=split.y_test.to_numpy(),
        y_probas=test_proba, model_name=best_name,
        threshold=best_threshold, filename=f"confusion_matrix_FT_{best_name}.png"
    )
    plots_list.append(final_cm_path)
        
    importance_path = viz.plot_final_feature_importance(
            model_pipeline=final_test_pipeline, 
            model_name=best_name, 
            filename=f"feature_importance_FT_{best_name}.png"
        )
    plots_list.append(importance_path)
    
    plt.close('all')  # Закрываем усё, што магло выпадкова застацца адкрытым
    gc.collect()      # Прымусова чысцім памяць прама ЗАРАЗ у галоўным патоку
        
    # section("Падрахуем:")

    # print(f"Крыніца даных: {DATA/file_name}")

    # print("Графікі:")
    # for p in plots_list:
    #     print(f"  • {p.name}")

    


if __name__ == "__main__":
    main()