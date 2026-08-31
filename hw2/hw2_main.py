# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Імпарт модуляў
import custom_imputer as ci
import data_loader as dl
import visualizer as vis
import pandas as pd

from visualizer import ChartConfig, plot_scatter_clean

# Крок 1: Загрузка дадзеных (напрыклад, Titanic з OpenML)

print("Загрузка даДзеных з OpenML...")
#X, y = dl.load_from_openml(name_or_id="titanic", target_column="survived")
dfmodel = dl.load_from_openml(name_or_id="titanic")

target_column="survived"
X = dfmodel.drop(columns=[target_column])
y = dfmodel[target_column]
# Спрашчаем мэтавы слупок для класіфікацыі (выдаляем пропускі, прыводзім да int)
# y = y.fillna(0).astype(int) # у VSc не сварыўся чамусьці
y = y.astype(object).fillna(0).astype(int) # з'явіўся па просьбе Gc

# Крок 2: Першасны аналіз дадзеных  
# 2.1 аналіз пропускаў, вывад статыстыкі
# 2.2 знаёмства з дадзенымі для выбара прыкмет для мадэлі класіфікацыі
#-------

# 2.1 аналіз пропускаў, вывад статыстыкі
ci.analyze_missing_data(X) # прасцяцкая справаздача

# analisis = ci.analyze_missing_data(X)
# ci.print_missing_report(X, analisis) #Генеруе падрабязную справаздачу аб пропушчаных значэннях.

# Структуруем усе подпісы і дадзеныя ў адным месцы
titanic_chart = ChartConfig( # на адпаведнасць тэме занятку, прапрацоўваю панятак клас: 
    df=dfmodel,
    x_col='age',
    y_col='fare',
    hue_col='survived',
    title='Дыяграма рассейвання для пасажыраў Тытаніка',
    x_label='Узрост (Age)',
    y_label='Кошт квітка (Fare)'
)

plot_scatter_clean(titanic_chart)

# 2.2 Адбор прыкмет па якіх будзем праводзіць класіфікацыю
features_cat_freq = ['embarked']    # Катэгорыі, дзе мала пропускаў (заменім модай)
features_cat_CONST = ['sex']             # Катэгорыі, дзе шмат пропускаў (заменім на 'Unknown')
features_num = ['age', 'fare', 'pclass'] # Лічбавыя слупкі (заменім медыянай)

#all_features = features_num + features_cat_freq + features_cat_CONST
#X = X[all_features]

# Крок 3: Папярэдняя апрацоўка дадзеных

print("Запаўненне пропускаў і падрыхтоўка дадзеных...")

# Выклікаем функцыю preprocess_pipeline і распакоўваем картэж

custom_reg_col="age"
reg_predictors=["pclass", "sex", "fare"]

X_numpy, my_preprocessor = ci.preprocess_pipeline(X, numeric_features=features_num, categorical_const_features=features_cat_freq, categorical_freq_features=features_cat_CONST)

my_preprocessor

import pandas as pd

# Ператвараем масіў NumPy назад у табліцу Pandas DataFrame
X_cleaned = pd.DataFrame(
    X_numpy,
    columns=my_preprocessor.get_feature_names_out(),
    index=X.index  # захоўваем арыгінальныя індэксы радкоў
)

# правяраем як яно там
X_cleaned.head(3)

# Крок 4: Падзел на навучальную і тэставую выбаркі
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.2, random_state=42, stratify=y
)

# Крок 5: Стварэнне і навучанне мадэлі лагістычнай рэгрэсіі
print("Навучанне мадэлі Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Крок 6: Тэкставая выснова метрык якасці
y_pred = model.predict(X_test)
print("\nФiнальная статыстыка класіфікацыі:")
print(classification_report(y_test, y_pred))

# Крок 7: Выклік візуалізацыі
print("Генерацыя графікаў з модуля 'visualizer'...")
# матрыца памылак і ROC
vis.plot_classification_results(model, X_test, y_test, X_train.columns,type_vis_method = 1)
# гістаграма
vis.plot_classification_results(model, X_test, y_test, X_train.columns, type_vis_method = 2)

# прыемна глядзець на кропачкі: зыходныя дадзеныя таргет і прагноз
vis.plot_predictions(y_test, y_pred, num_points=20, type_vis_method=1)

# vis.plot_predictions(y_test, y_pred, num_points=10, type_vis_method=2)

# Парныя графікі з дапамогай sns
# vis.plot_cl_results(X_test, y_test, X_train.columns, target_column)