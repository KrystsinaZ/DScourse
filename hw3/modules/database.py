
from __future__ import annotations

import sqlite3
import pandas as pd

from pathlib import Path
from typing import Self


class DatabaseManager:
    """
    Кіраванне БД: стварэнне табліц и загрузка CSV.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """
        Злучэнне з БД
        """
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row             # вяртае радкі як слоўнікі
        self.conn.execute("PRAGMA foreign_keys = ON")   # сочыць за сувязямі паміж табліцамі праз падтрымку знешніх ключоў (Foreign Keys) 
        return self.conn

    def close(self) -> None:
        """
        Бяспечнае закрыццё БД
        """
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> Self:    #"DatabaseManager":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: #def __exit__(self, *args) -> None:
        if self.conn is not None:
            if exc_type is None:
                # Калі памылак не было — захоўваем змены
                self.conn.commit()
            else:
                # Калі здарылася памылка — адмяняем змены, каб не сапсаваць БД
                self.conn.rollback()
        self.close()

    def tables_exist(self, expected_tables: list[str] | tuple[str, ...] = ("screen_time", "bdi_screen_items")) -> bool:
        """
        Правярае наяўнасць зададзеных табліц у БД.
        
        Па змаўчанні шукае табліцы праекта: 'screen_time' і 'bdi_screen_items'.
        Вяртае True, калі знойдзены ЎСЕ табліцы, што патрабуюцца.
        """
        assert self.conn is not None, "База дадзеных не падключана!"
        
        # Пераводзім чаканыя табліцы ў мноства для хуткага параўнання
        expected_set = set(expected_tables)
        
        # Дынамічна атрымліваем з БД спіс ТОЛЬКІ тых табліц, якія рэальна існуюць
        # Выкарыстоўваем аптымізаваны запыт праз оператар IN
        placeholders = ", ".join("?" for _ in expected_set)
        rows = self.conn.execute(
            f"""
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name IN ({placeholders})
            """,
            list(expected_set)
        ).fetchall()
        
        # Збіраем імёны знойдзеных табліц
        existing_set = {r["name"] for r in rows}
        
        # Правяраем, ці з'яўляецца мноства чаканых табліц падмноствам існуючых
        return expected_set.issubset(existing_set)

    
    def create_tables(self, *, force: bool = False) -> None:
        """
        Стварае табліцы, калі іх яшчэ няма.

        force=True — выдаліць і створыць наноў.
        """

        if self.conn is None:
            raise RuntimeError("Злучэнне з базай дадзеных не ўсталявана.")

        if force:
            self.conn.executescript(
                """
                DROP TABLE IF EXISTS bdi_screen_items;
                DROP TABLE IF EXISTS screen_time;                
                """
            )
            self.conn.commit()
            print("[Database] Старыя табліцы выдалены (force=True)")

        if self.tables_exist() and not force: # tables_exist=True - калі знойдзены абедзве табліцы у БД
            print("[Database] Табліцы у БД існуюць і каманды стварыць наноў не было — прапускаю стварэнне (tables_exist=True і force=False)")
            return
        

        self.conn.executescript("""

            CREATE TABLE IF NOT EXISTS screen_time (
                subject_id                  INTEGER PRIMARY KEY,
                sex                         TEXT NOT NULL,
                screen_time_index           REAL,
                est_leisure_screen_hours    REAL,
                sleep_quality_index         REAL,
                avg_sleep_hours             REAL,
                midsleep_weekend_hours      REAL,
                social_jetlag_hours         REAL,
                bdi_total                   INTEGER,  
                depressed                   INTEGER 
                );

            CREATE TABLE IF NOT EXISTS bdi_screen_items (
                subject_id               INTEGER PRIMARY KEY,
                screen_normal_day_1to6   INTEGER,
                screen_weekday_1to6      INTEGER,
                screen_weekend_1to6      INTEGER,
                sqi_fall_asleep_1to6     INTEGER,
                sqi_repeated_awake_1to6  INTEGER,
                sqi_disturbed_1to6       INTEGER,
                sqi_early_awake_1to6     INTEGER,
                bdi_item_01 INTEGER, bdi_item_02 INTEGER, bdi_item_03 INTEGER, bdi_item_04 INTEGER,
                bdi_item_05 INTEGER, bdi_item_06 INTEGER, bdi_item_07 INTEGER, bdi_item_08 INTEGER,
                bdi_item_09 INTEGER, bdi_item_10 INTEGER, bdi_item_11 INTEGER, bdi_item_12 INTEGER,
                bdi_item_13 INTEGER, bdi_item_14 INTEGER, bdi_item_15 INTEGER, bdi_item_16 INTEGER,
                bdi_item_17 INTEGER, bdi_item_18 INTEGER, bdi_item_19 INTEGER, bdi_item_20 INTEGER,
                bdi_item_21 INTEGER,

                FOREIGN KEY (subject_id) REFERENCES screen_time(subject_id)
                ); 
            """
            )
        
        self.conn.commit()
        print(f"[Database] Табліцы створаныя: screen_time, bdi_screen_items -> {self.db_path}")

    def table_info(self) -> None:
        """
        Выводзіць інфармацыю аб усіх існуючых у БД табліцах.
        """
        assert self.conn is not None
        
        # Аўтаматычна атрымліваем спіс ТОЛЬКІ тых табліц, якія існуюць у БД
        tables = [r["name"] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        
        if not tables:
            print("База дадзеных пакуль пустая (няма табліц).")
            return

        # Выводзім кампактную інфармацыю па кожнай знойдзенай табліцы
        for table in tables:
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cols = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            types = ", ".join(f"{c['name']} {c['type']}" for c in cols)
            print(f"  {table}: {count} радкоў | {types}")
        
    def import_csv(self, csv_path: str | Path, csv_path2: str | Path) -> dict[str, int]:
        """
        Запаўненне табліц дадзенымі з CSV: дубляў няма, поўторны запуск абнаўляе значэнні.

        -############ developers: INSERT OR IGNORE по UNIQUE(name) # спраба дадаць новы радок у табліцу, але калі радок з такім самым name ужо існуе, база дадзеных проста ігнаруе гэты запіс і ідзе далей, не выклікаючы памылкі.
        - screen_time: INSERT ... ON CONFLICT(subject_id) DO UPDATE (абнаўленне значэнняў UPSERT) - устаў новы радок, але калі такі subject_id ужо існуе, не выклікай памылку, а проста абнаві дадзеныя для яго
        
        """
        assert self.conn is not None
        
        csv_path = Path(csv_path)
        df_screen = pd.read_csv(csv_path)
        print(f"\nscreen_time_mental_health.csv: {df_screen.shape[0]:,} радкоў x {df_screen.shape[1]} слупкоў")

        csv_path2 = Path(csv_path2)
        df_bdi = pd.read_csv(csv_path2)
        print(f"\nbdi_and_screen_items.csv: {df_bdi.shape[0]:,} радкоў x {df_bdi.shape[1]} слупкоў")
        
        print(f"\n Слупок subject_id прысутны у абодвух наборах дадзеных: {set(df_bdi.subject_id) == set(df_screen.subject_id)}")
        
        # ачыстка датасэта ад дублікатаў па слупку subject_id
        # keep="last" Калі знойдуцца некалькі радкоў з аднолькавым subject_id, 
        # то захоўваецца самы апошні радок (які быў ніжэй за ўсё ў файле),
        # а ўсе папярэднія радкі з гэтым жа subject_id — выдаляюцца. 
        # (Гэта робіцца для таго, каб пакінуць самую актуальную альбо познюю інфармацыю пра subject_id).
        before = len(df_screen)

        if not (set(df_bdi.subject_id) == set(df_screen.subject_id)):
            df_screen = df_screen.drop_duplicates(subset=["subject_id"], keep="last")

        csv_dupes = before - len(df_screen)
        if csv_dupes > 0:
            print(f"  Выдалена дублікатаў у CSV: {csv_dupes}")
        
        before_count = self.conn.execute("SELECT COUNT(*) FROM screen_time").fetchone()[0]
            
        records = []        
        for _, row in df_screen.iterrows():
            records.append(
                (
                int(row["subject_id"]),
                str(row["sex"]),
                None 
                if pd.isna(row.get("screen_time_index")) else float(row["screen_time_index"]),
                None 
                if pd.isna(row.get("est_leisure_screen_hours")) else float(row["est_leisure_screen_hours"]),
                None
                if pd.isna(row.get("sleep_quality_index")) else float(row["sleep_quality_index"]),
                None 
                if pd.isna(row.get("avg_sleep_hours")) else float(row["avg_sleep_hours"]),
                None
                if pd.isna(row.get("midsleep_weekend_hours")) else float(row["midsleep_weekend_hours"]),
                None
                if pd.isna(row.get("social_jetlag_hours")) else float(row["social_jetlag_hours"]),
                None 
                if pd.isna(row.get("bdi_total")) else int(row["bdi_total"]),
                None 
                if pd.isna(row.get("depressed")) else int(row["depressed"]),
                )
            )

            self.conn.executemany(
                """
                INSERT INTO screen_time (
                    subject_id, sex, screen_time_index, est_leisure_screen_hours, sleep_quality_index, 
                    avg_sleep_hours, midsleep_weekend_hours, social_jetlag_hours, bdi_total, depressed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject_id) DO UPDATE SET
                    sex                 = excluded.sex,
                    screen_time_index   = excluded.screen_time_index,
                    est_leisure_screen_hours    = excluded.est_leisure_screen_hours,
                    sleep_quality_index = excluded.sleep_quality_index,
                    avg_sleep_hours     = excluded.avg_sleep_hours,
                    midsleep_weekend_hours      = excluded.midsleep_weekend_hours,
                    social_jetlag_hours = excluded.social_jetlag_hours,
                    bdi_total           = excluded.bdi_total,
                    depressed           = excluded.depressed                
                """,
                records,
            )
            self.conn.commit()
            
        after_count = self.conn.execute("SELECT COUNT(*) FROM screen_time").fetchone()[0]
        inserted = after_count - before_count
        updated = len(records) - inserted

        
        before_count_bdi = self.conn.execute("SELECT COUNT(*) FROM bdi_screen_items").fetchone()[0]
        records_bdi = []        
        for _, row in df_bdi.iterrows():
            records_bdi.append(
                (
                    int(row["subject_id"]),
                    None 
                    if pd.isna(row.get("screen_normal_day_1to6")) else int(row["screen_normal_day_1to6"]),
                    None 
                    if pd.isna(row.get("screen_weekday_1to6")) else int(row["screen_weekday_1to6"]),
                    None 
                    if pd.isna(row.get("screen_weekend_1to6")) else int(row["screen_weekend_1to6"]),                     
                    None 
                    if pd.isna(row.get("sqi_fall_asleep_1to6")) else int(row["sqi_fall_asleep_1to6"]),
                    None 
                    if pd.isna(row.get("sqi_repeated_awake_1to")) else int(row["sqi_repeated_awake_1to"]),
                    None 
                    if pd.isna(row.get("sqi_disturbed_1to6")) else int(row["sqi_disturbed_1to6"]),
                    None 
                    if pd.isna(row.get("sqi_early_awake_1to6")) else int(row["sqi_early_awake_1to6"]),
                    None 
                    if pd.isna(row.get("bdi_item_1")) else int(row["bdi_item_1"]),
                    None 
                    if pd.isna(row.get("bdi_item_2")) else int(row["bdi_item_2"]),
                    None 
                    if pd.isna(row.get("bdi_item_3")) else int(row["bdi_item_3"]),
                    None 
                    if pd.isna(row.get("bdi_item_4")) else int(row["bdi_item_4"]),
                    None
                    if pd.isna(row.get("bdi_item_5")) else int(row["bdi_item_5"]),
                    None 
                    if pd.isna(row.get("bdi_item_6")) else int(row["bdi_item_6"]),
                    None 
                    if pd.isna(row.get("bdi_item_7")) else int(row["bdi_item_7"]),
                    None 
                    if pd.isna(row.get("bdi_item_8")) else int(row["bdi_item_8"]),
                    None 
                    if pd.isna(row.get("bdi_item_9")) else int(row["bdi_item_9"]),
                    None 
                    if pd.isna(row.get("bdi_item_10")) else int(row["bdi_item_10"]),
                    None 
                    if pd.isna(row.get("bdi_item_11")) else int(row["bdi_item_11"]),                   
                    None 
                    if pd.isna(row.get("bdi_item_12")) else int(row["bdi_item_12"]),                   
                    None 
                    if pd.isna(row.get("bdi_item_13")) else int(row["bdi_item_13"]),                    
                    None 
                    if pd.isna(row.get("bdi_item_14")) else int(row["bdi_item_14"]),                   
                    None
                    if pd.isna(row.get("bdi_item_15")) else int(row["bdi_item_15"]),                    
                    None 
                    if pd.isna(row.get("bdi_item_16")) else int(row["bdi_item_16"]),                   
                    None 
                    if pd.isna(row.get("bdi_item_17")) else int(row["bdi_item_17"]),                   
                    None 
                    if pd.isna(row.get("bdi_item_18")) else int(row["bdi_item_18"]),
                    None 
                    if pd.isna(row.get("bdi_item_19")) else int(row["bdi_item_19"]),
                    None 
                    if pd.isna(row.get("bdi_item_20")) else int(row["bdi_item_20"]),
                    None 
                    if pd.isna(row.get("bdi_item_21")) else int(row["bdi_item_21"])
                )
            )

        self.conn.executemany(
            """
            INSERT INTO bdi_screen_items (
                subject_id, 
                screen_normal_day_1to6, screen_weekday_1to6, screen_weekend_1to6, 
                sqi_fall_asleep_1to6, sqi_repeated_awake_1to6, sqi_disturbed_1to6, sqi_early_awake_1to6, 
                bdi_item_01, bdi_item_02, bdi_item_03, bdi_item_04,
                bdi_item_05, bdi_item_06, bdi_item_07, bdi_item_08, bdi_item_09,
                bdi_item_10, bdi_item_11, bdi_item_12, bdi_item_13, bdi_item_14, bdi_item_15,
                bdi_item_16, bdi_item_17, bdi_item_18, bdi_item_19, bdi_item_20, bdi_item_21     
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject_id) DO UPDATE SET
                screen_normal_day_1to6   = excluded.screen_normal_day_1to6,
                screen_weekday_1to6      = excluded.screen_weekday_1to6,
                screen_weekend_1to6      = excluded.screen_weekend_1to6,
                sqi_fall_asleep_1to6     = excluded.sqi_fall_asleep_1to6,
                sqi_repeated_awake_1to6  = excluded.sqi_repeated_awake_1to6,
                sqi_disturbed_1to6       = excluded.sqi_disturbed_1to6,
                sqi_early_awake_1to6     = excluded.sqi_early_awake_1to6,
                bdi_item_01 = excluded.bdi_item_01, bdi_item_02 = excluded.bdi_item_02,
                bdi_item_03 = excluded.bdi_item_03, bdi_item_04 = excluded.bdi_item_04,
                bdi_item_05 = excluded.bdi_item_05, bdi_item_06 = excluded.bdi_item_06,
                bdi_item_07 = excluded.bdi_item_07, bdi_item_08 = excluded.bdi_item_08,
                bdi_item_09 = excluded.bdi_item_09, bdi_item_10 = excluded.bdi_item_10,
                bdi_item_11 = excluded.bdi_item_11, bdi_item_12 = excluded.bdi_item_12,
                bdi_item_13 = excluded.bdi_item_13, bdi_item_14 = excluded.bdi_item_14,
                bdi_item_15 = excluded.bdi_item_15, bdi_item_16 = excluded.bdi_item_16,
                bdi_item_17 = excluded.bdi_item_17, bdi_item_18 = excluded.bdi_item_18,
                bdi_item_19 = excluded.bdi_item_19, bdi_item_20 = excluded.bdi_item_20,
                bdi_item_21 = excluded.bdi_item_21
            """,
            records_bdi,
        )
        self.conn.commit()

        after_count_bdi = self.conn.execute("SELECT COUNT(*) FROM bdi_screen_items").fetchone()[0]
        inserted_bdi = after_count_bdi - before_count_bdi
        updated_bdi = len(records_bdi) - inserted_bdi
        

        print(
            f"[Database] Загрузка з {csv_path.name}: "
            f"\nрадкоў CSV={before}, унікальных subject_id={len(records)}, "
            f"дубляў у CSV={csv_dupes}, "
            f"\nУ screen_time дададзена = {inserted}, адноўлена={updated}, агулам ў БД={after_count}"
            f"\n[Database] Загрузка з {csv_path2.name}: "
            f"\nрадкоў CSV={before_count_bdi}, унікальных subject_id={len(records_bdi)}, "
            f"\nУ bdi_screen_items дададзена = {inserted_bdi}, адноўлена={updated_bdi}, агулам ў БД={after_count_bdi}"
        )
        return {
            "csv_rows": before,
            "unique": len(records),
            "csv_dupes": csv_dupes,
            "inserted": inserted,
            "updated": updated,
            "total": after_count,
        }
