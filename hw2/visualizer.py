import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from dataclasses import dataclass

# Клас-кантэйнер для дадзеных і подпісаў
@dataclass
class ChartConfig:
    df: pd.DataFrame
    x_col: str
    y_col: str
    hue_col: str
    title: str
    x_label: str
    y_label: str

# Функцыя дыяграмы рассеяння
def plot_scatter_clean(config: ChartConfig):
    plt.figure(figsize=(10, 6))
    
    # Усе дадзеныя бяруцца з аб'екта config праз кропку
    sns.scatterplot(
        data=config.df, 
        x=config.x_col, 
        y=config.y_col, 
        hue=config.hue_col
    )
    plt.title(config.title)
    plt.xlabel(config.x_label)
    plt.ylabel(config.y_label)
    plt.show()

# Визуализация истинных и предсказанных значений
def plot_predictions(y_true, y_pred, num_points, target_name, type_vis_method):

    if type_vis_method == 1:
        plt.figure(figsize=(10, 6))
        plt.scatter(range(num_points), y_true[:num_points], color='blue', label='Истинные значения')
        plt.scatter(range(num_points), y_pred[:num_points], color='red', label='Предсказанные значения')
        plt.xlabel('Индекс')
        plt.ylabel('Значение {target_name}')
        plt.title(f'Истинные и предсказанные значения {target_name}(первые {num_points} точек)')
        plt.legend()
        plt.show()

    elif  type_vis_method == 2:
        # Визуализация истинных и предсказанных значений
                # Визуализация истинных и предсказанных значений
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_points), y_true[:num_points], label='Истинные значения', marker='o')
        plt.plot(range(num_points), y_pred[:num_points], label='Предсказанные значения', marker='x')
        plt.xlabel('Наблюдения')
        plt.ylabel('Значения')
        plt.title('Сравнение истинных и предсказанных значений {target_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
def plot_cl_results(X, y, feature_names, target_names):
    
    # Создание DataFrame для удобного отображения данных
    df = pd.DataFrame(data=X, columns=feature_names)

    #  передаем список вместо строки target_name
    df[target_names] = pd.Categorical.from_codes(y, categories=['Died', 'Survived'])

    # Построение парных графиков при помощи sns
    sns.pairplot(df, hue=target_names)
    plt.show()    


def plot_classification_results(model, X_test, y_test, feature_names, type_vis_method):
    #Візуалізацыя вынікаў класіфікацыі трыма рознымі спосабамі.
    #fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if type_vis_method == 1:
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        # Спосаб 1: Матрыца памылак (Confusion Matrix)

        axes[0].set(xlabel="Што прадказала мадэль", ylabel="Што прадказала мадэль", title="Матрыца памылак")

        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, ax=axes[0], cmap="Blues"
        )

        # Спосаб 2: ROC-крывая (ROC Curve)
        #elif type_vis_method == 2:
        axes[1].set_title("ROC-крывая (Якасць мадэлі)")

        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1])

        axes[1].plot([0, 1], [0, 1], "r--", label="Выпадковы выбар")
        axes[1].legend()
 

        plt.tight_layout()
        plt.show()
     
    elif  type_vis_method == 2:
        # Спосаб 3: Важнасць прыкмет
        fig, axes = plt.subplots(1, 1, figsize=(18, 5))

        plt.title("Вага каэфіцыентаў мадэлі")
        # Для бінарнай класіфікацыі coef_ мае памер (1, n_features), робім яго аднамерным
        coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        indices = np.argsort(np.abs(coefficients))


        plt.barh(
            range(len(indices)), coefficients[indices], color="seagreen", align="center"
        )
        axes.set_yticks(range(len(indices)))
        axes.set_yticklabels([feature_names[i] for i in indices])
        axes.set_xlabel("Значэнне каэфіцыента") 

        plt.tight_layout()

        plt.show()



def plot_visual_histogram(analysis_df):
    """Будуе прыгожую графічную гістаграму пропускаў."""
    # Фільтруем слупкі з пропускамі
    only_missing = analysis_df[analysis_df["Total"] > 0]
    
    if only_missing.empty:
        print("Пропускаў няма, няма чаго маляваць.")
        return

    # Ствараем графік (назвы слупкоў па восі X, колькасць па восі Y)
    plt.figure(figsize=(8, 5))
    plt.bar(only_missing.index, only_missing["Total"], color="skyblue", edgecolor="black")
    
    # Налады дызайну
    plt.title("Колькасць пропушчаных значэнняў па слупках", fontsize=14)
    plt.xlabel("Назвы слупкоў", fontsize=12)
    plt.ylabel("Колькасць пропускаў (Total)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Дадаем сетку для зручнасці
    
    # Паказваем графік на экране
    plt.show()




