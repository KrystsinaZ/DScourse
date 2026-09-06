import warnings
warnings.filterwarnings("ignore", message=".*The argument 'eval_set' is deprecated.*")


import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from modules.dataset_regression import PrepareRegressionDataset
from modules.regressor import RegressorBench
from modules.regression_evaluation import RegressionEvaluator
from modules.regression_tuner import BayesianRegressionTuner
from modules.visualization import DataVisualizer

ROOT = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
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
    dataset = PrepareRegressionDataset(DATA)
    bench = RegressorBench()
    evaluator = RegressionEvaluator()
    viz = DataVisualizer(OUT)
    plots_list: list[Path] = []

    section("1. Датасет: рэгрэсія PageValues")
    frame = dataset.load_csv(file_name="online_shoppers_intention.csv")

    ohe_feature_list = ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
    known_categories = {
        col: sorted(list(frame[col].dropna().unique())) for col in ohe_feature_list
    }

    section("2. Падзел 60/20/20 (Revenue выдалены з прыкмет, стратыфікацыя па бінах PageValues)")
    split = dataset.prepare_regression(
        frame,
        target_name="PageValues",
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    )

    bench.build()
    bench.preprocessor = dataset.create_preprocessor_regression(categories_dict=known_categories)

    section("2.1. Дадаем 8-ы метад: TwoStageRegressor (hurdle-мадэль)")
    # Асобны шлях: класіфікатар-варотнік P(PageValues>0) + LightGBM-рэгрэсія
    # толькі на ненулявой частцы. combine="gated" (дэфолт) — канфігурацыя,
    # якая паказала найлепшы MAE ў дыягностыцы (гл. справаздачу, раздзел 5).
    bench.add_two_stage_model(name="TwoStage")

    section("3. Крос-валідацыя (5 фолдаў, стратыфікаваных па нуль/квантылі PageValues)")
    CV_methods = ["Dummy", "Ridge", "DecisionTree", "ExtraTrees", "LightGBM", "XGBoost", "CatBoost", "TwoStage"]
    CV_methods = [m for m in CV_methods if m in bench.models]  # прапусціць мадэлі, якіх няма (напр. CatBoost без catboost)
    CV_configs = bench.prepare_final_models(models_to_keep=CV_methods)

    fold_rmse, fold_mae, fold_r2, oof_predictions = bench.cross_validate_newV(
        split.x_cv, split.y_cv, models_conf=CV_configs
    )

    metrics_table = bench.get_metrics_table()
    print("\n[Cross-Validation] Табліца метрык, адсартавана па RMSE:")
    print(metrics_table.to_string(index=False))

    df_report = evaluator.compile_performance_report(y_true=split.y_cv, preds_dict=oof_predictions)
    print("\n[Report] Поўная справаздача OOF:")
    print(df_report.to_string(index=False))

    section("3.1. Візуалізацыя параўнання метадаў")
    plots_list.append(viz.plot_regression_metric_comparison(
        fold_rmse, metric_name="RMSE", lower_is_better=True, filename="regression_rmse_comparison.png",
    ))
    plots_list.append(viz.plot_regression_metric_comparison(
        fold_mae, metric_name="MAE", lower_is_better=True, filename="regression_mae_comparison.png",
    ))
    plots_list.append(viz.plot_regression_metric_comparison(
        fold_r2, metric_name="R2", lower_is_better=False, filename="regression_r2_comparison.png",
    ))
    print("\n[Visualizer] Захаваны графікі параўнання RMSE/MAE/R2.")

    section("4. Выбар лепшай мадэлі (парны t-тэст па RMSE)")
    best_name = evaluator.pick_best(fold_rmse, complexity=bench.get_complexity_map())
    print(f"\nАбраная мадэль: {best_name}")

    section("4.1. Дыягностыка нулявых прагнозаў (zero-inflation)")
    diag_models = [best_name] + (["TwoStage"] if "TwoStage" in oof_predictions and best_name != "TwoStage" else [])
    plots_list.append(viz.plot_zero_prediction_diagnostic(
        y_true=split.y_cv.to_numpy(), oof_predictions=oof_predictions, model_names=diag_models,
        filename="regression_zero_prediction_diagnostic.png",
    ))
    print(f"[Visualizer] Захаваны графік дыягностыкі нулявых прагнозаў ({', '.join(diag_models)}).")

    section("5. Баесаўская аптымізацыя гіперпараметраў (Optuna)")
    tuner = BayesianRegressionTuner(bench_instance=bench, random_state=42)
    optimized_results = tuner.tune_or_load(best_name, split.x_cv, split.y_cv)

    print("СПРАВАЗДАЧА ПАДБОРУ ГІПЕРПАРАМЕТРАЎ:")
    for m_name, params in optimized_results.items():
        print(f"Мадэль: {m_name}")
        for p_key, p_val in params.items():
            print(f"  -> {p_key}: {p_val:.5f}" if isinstance(p_val, float) else f"  -> {p_key}: {p_val}")

    section("6. Фінальны рэфіт і тэст")
    final_configs = bench.prepare_final_models(
        best_name=best_name,
        optimized_results=optimized_results,
        models_to_keep=[best_name],
    )
    final_pipeline = bench.final_secure_refit(
        final_configs=final_configs,
        x_train=split.x_train, y_train=split.y_train,
        x_val=split.x_val, y_val=split.y_val,
    )

    test_preds = final_pipeline.predict(split.x_test)
    test_report = evaluator.compile_performance_report(
        y_true=split.y_test,
        preds_dict={f"{best_name} (Фінальны тэст)": test_preds},
    )
    print("\n[Report] Метрыкі на тэставай выбарцы:")
    print(test_report.to_string(index=False))

    section("6.1. Візуалізацыя фінальнай мадэлі")
    plots_list.append(viz.plot_actual_vs_predicted(
        y_true=split.y_test.to_numpy(), y_pred=test_preds, model_name=best_name,
        filename=f"regression_actual_vs_predicted_{best_name}.png",
    ))
    plots_list.append(viz.plot_regression_feature_importance(
        model_pipeline=final_pipeline, model_name=best_name,
        filename=f"regression_feature_importance_{best_name}.png",
    ))
    print(f"[Visualizer] Захаваны графікі actual-vs-predicted і feature importance для {best_name}.")

    section("Падрахуем")
    print(f"Крыніца даных: {DATA / 'online_shoppers_intention.csv'}")
    print("Графікі захаваны ў:", OUT)
    for p in plots_list:
        print(f"  • {p.name}")


if __name__ == "__main__":
    main()
