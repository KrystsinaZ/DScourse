import sys
from pathlib import Path

from modules import DatabaseManager, DataVisualizer, SqlQueries

ROOT = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd() # рашэнне для .ipny  і .py
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DATA = ROOT / "data" / "screen_time_mental_health.csv"  # '/kaggle/input/datasets/kylefengkfeng209/screen-time-vs-mental-health-ml-ready/screen_time_mental_health.csv'
DATA2 = ROOT / "data" / "bdi_and_screen_items.csv"      # '/kaggle/input/datasets/kylefengkfeng209/screen-time-vs-mental-health-ml-ready/bdi_and_screen_items.csv'

DB = ROOT / "db" / "screen_time.db"
OUT = ROOT / "output"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    section("1-3. Датасет -> SQLite (стварэнне табліц + загрузка дадзеных)")

    print(f"Крыніца дадзеных: {DATA} {DATA2}")

    with DatabaseManager(DB) as db:
        db.create_tables(force=True) #(force=False) #(force=True) #
        db.import_csv(DATA, DATA2)
        print("Схема:")
        db.table_info()
        n = db.conn.execute("SELECT COUNT(*) FROM screen_time").fetchone()[0]
        n_bdi = db.conn.execute("SELECT COUNT(*) FROM bdi_screen_items").fetchone()[0]

        section("4. SQL  Агрэгаты `SUM` / `AVG` / `COUNT` у тым ліку з фільтрацыяй (WHERE)")

        q = SqlQueries(db.conn)

        sleep = q.sleep()
        print(f"\n Гадзіны сну ўпарадкаваныя: {len(sleep)}")
        print(sleep.head(2).to_string(index=False))

        screen_over_5 = q.screen_by_sex_over_5_hours()
        print(f"\n Колькасць хлапцоў і дзяўчат, чы экранны час (WHERE > 5): {len(screen_over_5)}")
        print(screen_over_5.head(3).to_string(index=False))
                     
        screen = q.get_screen_time_by_gender()
        print(f"\n Гадзіны экраннага часу: {len(screen)}")

        df_merged = q.get_screen_and_depression_stats()
        print(f"\n Дэпрэсія і экранны час у падлеткаў: {len(df_merged)}")
        print(df_merged.head(4).to_string(index=False))

        df_bdi = q.get_depression_stats()
        print(f"\n Катэгорыі дэпрэсіі падлеткаў: {len(df_bdi)}")
        print(df_bdi.head(4).to_string(index=False))

        df_bdi_cat = q.get_bdi_with_categories()
        print(f"\n Адсартэраваны спіс падлеткаў па катэгорыях дэпрэсіі: {len(df_bdi_cat)}")
               

        section("5. Візуалізацыя (Matplotlib + Seaborn)")

        viz = DataVisualizer(OUT)
        paths = [
            viz.plot_leisure_scr(screen),
            viz.plot_screen_by_sex_donut(screen_over_5),
            viz.plot_screen_by_sex(df_merged),                                       
            viz.plot_avg_sleep_hours(sleep),
            #viz.plot_bdi_histogram_tagged(df_bdi_cat),
            viz.plot_bdi_histogram_tagged4(df_bdi_cat),
        ]

    section("Падрахуем:")
    print(f"БД: {DB}")
    print(f"у БД 2 табліцы: {n} і {n_bdi}")
    print("Графікі:")
    for p in paths:

        print(f"  • {p.name}")



if __name__ == "__main__":
    main()
