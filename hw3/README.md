
# Дамашняя работа №3

'''
1. Выберите любой открытый датасет и скачайте открытый датасет, соответствующий вашим интересам или области обучения.
2. Создайте новую базу данных в системе управления базами данных (например, SQLite, PostgreSQL).
3. Создайте таблицу (или несколько таблиц) в базе данных с различными типами данных (INTEGER, TEXT, DATE), которые требуются для вашего датасета. 
	Импортируйте данные из датасета в созданные таблицы.
4. Напишите несколько SQL-запросов для извлечения данных из таблиц базы данных. 
	Используйте условия фильтрации (например, WHERE) для получения нужных данных.
5. Напишите SQL-запросы, использующие агрегатные функции (SUM, AVG, COUNT) для выполнения расчетов по данным таблицы.
6. Визуализируйте данные. Используйте библиотеки Python, такие как Matplotlib или Seaborn, для визуализации данных, извлеченных из базы данных. 
	Постройте графики или диаграммы, которые помогут проанализировать и понять данные.
'''
## Заданне

| # | Рэалізацыя |
|---|------------|
| 1 | `data/screen_time_mental_health.csv` і `data/bdi_screen_items.csv` - датасет screen time vs mental health на [Kaggle](/kaggle/input/datasets/kylefengkfeng209/screen-time-vs-mental-health-ml-ready/screen_time_mental_health.csv) |
| 2 | SQLite БД: `db/screen_time.db` |
| 3 | Табліцы `screen_time` + `dbi_screen_items` (INTEGER, TEXT, REAL) + імпарт CSV - `modules/database.py`|
| 4 | SQL з `WHERE` - функцыі `tables_exist` і `table_info` з `modules/database.py`|
|   | `WHERE` і `COUNT` - у функцыі `screen_by_sex_over_5_hours`, `screen_by_sex_over_5_depresed` з `modules/queries.py` |
|   | `WHERE` і `AVG` - у функцыі `get_screen_time_by_gender` з `modules/queries.py` |
|   | `SUM` / `COUNT` - у функцыі `get_screen_and_depression_stats` з `modules/queries.py` |
| 5 | Matplotlib + Seaborn - `modules/visualization.py` -> `output/` |

## Датасет

`screen_time_mental_health.csv` - [4810x10] вынікі тэставання дэпрэсіі ў падлеткаў.

Структура табліцы:  
```
subject_id, 
sex, 
screen_time_index, 
est_leisure_screen_hours - (Estimated Daily Leisure Screen Time) Штодзённы вольны час перад экранам, у гадзінах
sleep_quality_index, 
avg_sleep_hours, 
midsleep_weekend_hours, 
social_jetlag_hors, 
bdi_total, 
depressed
```

`bdi_screen_item.csv` - [4810x29] змяшчае вынікі апытанкі па BDI-II (Beck Depression Inventory-II)  — гэта другая версія шкалы дэпрэсіі Бэка. Псіхалагічны тэст з 21 пытання, які дапамагае вызначыць узровень і цяжкасць сымптомаў дэпрэсіі ў дарослых і падлеткаў з 13 год.
Ранжыраванне па балах:  
  0–13 балаў: мінімальны ці нормальны ўзровень.  
  14–19 балаў: лёгкая дэпрэсія.  
  20–28 балаў: сярэдняя (ўмераная) дэпрэсія.  
  29–63 балы: цяжкая дэпрэсія

  Структура табліцы:  
  ```
  subject_id, 
  screen_normal_day_1to6, screen_weekday_1to6, screen_weekend_1to6, 
  sqi_fall_asleep_1to6, sqi_repeated_awake_1to6, sqi_disturbed_1to6, sqi_early_awake_1to6, 
  bdi_item_01, bdi_item_02, bdi_item_03, bdi_item_04,
  bdi_item_05, bdi_item_06, bdi_item_07, bdi_item_08, bdi_item_09,
  bdi_item_10, bdi_item_11, bdi_item_12, bdi_item_13, bdi_item_14, bdi_item_15,
  bdi_item_16, bdi_item_17, bdi_item_18, bdi_item_19, bdi_item_20, bdi_item_21
```
## Структура

```
main.py
hw3.ipynb
requirements.txt
data/
db/
  .db          # стварэнне пры запуску
modules/
  database.py
  queries.py
  visualization.py
output/
  *.png       # стварэнне і напаўненне па выкананню
```
## Тыпы дадзеных у табліцах

**:** ` INTEGER`, ` TEXT`, ` REAL`

**:** ` INTEGER`

## Поўторны запуск

- Табліцы: `CREATE IF NOT EXISTS` — калі ёсць, іначай ствараюцца наноў.
- Поўны скід: `db.create_tables(force=True)`.
- Імпарт: upsert по `subject_id` (`ON CONFLICT DO UPDATE`) — абнаўленне радкоў, без паўтораў.


## Запуск

Патрэбен Python 3.11+.

### 1. Кланаваць рэпазіторы

```bash
git clone https://github.com/KrystsinaZ/DScourse/hw3.git
cd hw3
```

### 2. Стварыць віртуальны асяродак

```bash
python3 -m venv .venv
```

На Windows, калі `python3` не знайшлі:

```bash
python -m venv .venv
```

### 3. Актываваць асяродак

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```bat
.\.venv\Scripts\activate
```

**Linux / macOS (Apple):**

```bash
source .venv/bin/activate
```

### 4. Усталяваць залежнасці і запусціць

```bash
pip install -r requirements.txt
python main.py
```

Графікі размешчаны ў тэчцы `output/`.  
Ноутбук: `hw3.ipynb` (па актывацыі асяродка).

## Узор вываду

```

============================================================
5. Візуалізацыя (Matplotlib + Seaborn)
============================================================
[Visualizer] Захавана: output/est_leisure_screen_hours_by_gender.png
...

```

