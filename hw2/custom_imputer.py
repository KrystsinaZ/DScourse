import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # Абавязкова для IterativeImputer
        


import pandas as pd

def analyze_missing_data(df: pd.DataFrame):
    """Аналіз колькасці пропускаў у кожным слупку."""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    analysis = pd.concat([missing, missing_pct], axis=1, keys=["Total", "%"])
    # просценькі вывад без лішняга
    print("--- Аналіз пропускаў у даных ---")
    print(analysis[analysis["Total"] > 0])
    print("-" * 32)

    return analysis 

def print_missing_report(df: pd.DataFrame, analysis_df: pd.DataFrame):
    """Выводзіць на экран прыгожую статыстыку і справаздачу аб пропусках."""
    print("--- АНАЛІЗ ПРОПУСКАЎ У СЛУПКАХ ---")
    # Фільтруем і паказваем толькі праблемныя слупкі
    only_missing = analysis_df[analysis_df["Total"] > 0]
    
    if only_missing.empty:
        print("Усе слупкі чыстыя, пропускаў няма!")
    else:
        print(only_missing)
    print("-" * 34)

    # Лічым агульную статыстыку па ўсёй табліцы
    total_cells = df.size
    total_missing = analysis_df["Total"].sum()
    total_missing_pct = 100 * total_missing / total_cells if total_cells > 0 else 0

    print(f"Агульная колькасць ячэек: {total_cells}")
    print(f"Агульная колькасць пропускаў: {int(total_missing)} ({total_missing_pct:.2f}%)")
    print("-" * 34)


def preprocess_pipeline(
    X: pd.DataFrame, numeric_features, categorical_freq_features, categorical_const_features, custom_reg_col: str = None, reg_predictors: list = None
):
     
    #Поўны цыкл падрыхтоўкі і маштабавання дадзеных.
    X = X.copy()

    #numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    #categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not (custom_reg_col and reg_predictors):
          
        # 1. Пайплайн для лічбавых даных: спачатку запаўненне медыянай, потым маштабаванне
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())  # Дадалі маштабаванне
        ])

        # 2. Пайплайн для катэгорый: Частыя катэгорыі запаўняем модай, потым кадыруем у OneHot (0 і 1)
        # handle_unknown='ignore' абароніць ад памылак, калі ў тэсце з'явіцца новы порт
        categorical_freq_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))  # Дадалі кадаванне
        ])
  
        # 3. Рэдкія катэгорыі (кшталту каюта) запаўняем тэкстам 'Unknown', потым OneHot
        categorical_const_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))  # Дадалі кадаванне
        ])
            
        # 4. Злучаем усе правілы ў адзін суцэльны трансформер
    
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat_f', categorical_freq_transformer, categorical_freq_features),
                ('cat_с', categorical_const_transformer, categorical_const_features)
                ],
                remainder='drop'  # 'drop' па змоўчванні, непазначаныя слупкі аўтаматычна выдаляюцца
                                #remainder='passthrough' 
        )

        # 4. Прымяненне да дадзеных (запускае ўвесь ланцужок)
        X_processed = preprocessor.fit_transform(X)

        print("Пропускі запоўненыя, лічбы адмаштабаваныя, катэгорыі закадаваныя!")

    else:
        # месца пад лінейную рэгрэсію для 'age'
        X_processed = X
        print("такі рэжым не наладжаны. Пропускі не запоўненыя, засталіся без масштабавання!")
        

    return X_processed, preprocessor #, y



"""
    def regression_impute(
        df: pd.DataFrame, target_col: str, predictor_cols: list
    ):
        #Запаўненне пропускаў праз LinearRegression з папярэдняй ачысткай прэдыктараў.

        df_copy = df.copy()

        # Калі ў мэтавым слупку няма пропускаў, нічога не робім
        if df_copy[target_col].isnull().sum() == 0:
            return df_copy

        # 1. Дадатковая праверка на адсутнасць пропускаў у прэдыктары (выкарыстоўваем базавы imputer)
        
        predictors_df = df_copy # часовы кастылек

        # Дадаем мэтавы слупок для падзелу выбаркі
        working_df = predictors_df.copy()
        working_df[target_col] = df_copy[target_col]

        # 3. Падзел на навучальную (дзе ёсць target) і прагнозную (дзе target пустая) выбаркі
        train_data = working_df[working_df[target_col].notnull()]
        predict_data = working_df[working_df[target_col].isnull()]

        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        X_predict = predict_data.drop(columns=[target_col])

        # 4. Навучанне лінейнай рэгрэсіі і запаўненне
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        predicted_values = lr.predict(X_predict)

        # Вяртаем вылічаныя значэнні ў зыходны датафрэйм
        df_copy.loc[df_copy[target_col].isnull(), target_col] = predicted_values

        print("Пропускі паспяхова запоўнены праз LinearRegression!")
        return df_copy

"""

"""def reg (X):
    # =====================================================================
    # ЭТАП 1: Падрыхтоўка базавых прыкмет (без 'age')
    # =====================================================================

    # Спісы слупкоў
    numeric_reg_features = ['fare']            # 'pclass' звычайна катэгорыя ці парадкавы лік
    categorical_reg_features = ['pclass', 'sex']

    #categorical_reg_features

    # Ствараем звычайныя трансформеры для базавых прыкмет
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    # Збіраем базавы прэпрацэсар (ён пакуль НЕ чапае 'age', астатнія слупкі прапускае)
    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('cat', cat_transformer, categorical_features)
        ],
        remainder='passthrough' # Пакідае 'age' у канцы матрыцы без зменаў
    )

    # =====================================================================
    # ЭТАП 2: Разумнае запаўненне пропускаў у 'age' праз рэгрэсію
    # =====================================================================

    # Ствараем Імп'ютар на базе Лінейнай рэгрэсіі
    # Увага: пасля base_preprocessor слупок 'age' будзе апошнім
    reg_imputer = IterativeImputer(
        estimator=LinearRegression(), 
        missing_values=float('nan'), 
        random_state=42
    )

    # Маштабаванне для 'age' пасля таго, як яго запоўніць рэгрэсія
    final_scaler = ColumnTransformer(
        transformers=[
            ('age_scale', StandardScaler(), [-1]) # [-1] азначае апошні слупок (гэта наш 'age')
        ],
        remainder='passthrough' # Не чапаем ужо апрацаваныя fare, pclass, sex
    )

    # =====================================================================
    # ФІНАЛЬНЫ ПАЙПЛАЙН АПРАЦОЎКІ
    # =====================================================================

    full_preprocessor = Pipeline(steps=[
        ('base_preprocessing', base_preprocessor), # 1. Рыхтуем fare, pclass, sex
        ('regression_impute', reg_imputer),       # 2. Прадказваем age праз рэгрэсію
        ('scale_age', final_scaler)               # 3. Маштабуем знойдзены age
    ])

    # Запуск усяго ланцужка на даных
    X_processed = full_preprocessor.fit_transform(X)

    print("Пропускі ў 'age' паспяхова прадказаныя лінейнай рэгрэсіяй!")

    
return X_processed"""