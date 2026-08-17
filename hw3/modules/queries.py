"""
SQL-запыты: фільтрацыя (WHERE) і агрэгаты (SUM, AVG, COUNT).
"""

from __future__ import annotations

import sqlite3

import pandas as pd


class SqlQueries:
    """
    Запыт да БД screen time.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def execute(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def screen(self) -> pd.DataFrame:
        """Вяртае экранны час, адсартаваны ад большага да меншага."""

        return self.execute(
            """
            SELECT est_leisure_screen_hours 
            FROM screen_time
            ORDER BY est_leisure_screen_hours DESC;
            """
        )

    def get_screen_time_by_gender(self) -> pd.DataFrame:
        """Вяртае пол, экранны час і сярэдняе значэнне па полу, вылічанае праз SQL."""
        return self.execute(
            """
            SELECT 
                sex,
                est_leisure_screen_hours,
                AVG(est_leisure_screen_hours) OVER(PARTITION BY sex) AS gender_avg
            FROM screen_time
            WHERE est_leisure_screen_hours IS NOT NULL
            ORDER BY est_leisure_screen_hours DESC;
            """
        )

    def screen_by_sex_over_5_hours(self) -> pd.DataFrame:
        """Вяртае колькасць хлопчыкаў і дзяўчынак з экранным часам > 5 гадзін."""
        return self.execute(
            """
            SELECT sex, COUNT(*) as count
            FROM screen_time
            WHERE est_leisure_screen_hours > ?
            GROUP BY sex;
            """,
            params=(5,)
        )

    def screen_by_sex_over_5_depresed(self) -> pd.DataFrame:
        """Вяртае колькасць хлопчыкаў і дзяўчынак у дэпрэсіі з экранным часам > 5 гадзін."""
        return self.execute(
            """
            SELECT sex, COUNT(*) as count
            FROM screen_time
            WHERE est_leisure_screen_hours > ?  AND depressed >= ?
            GROUP BY sex;
            """,
            params=(5, 1)
        )

    
    def get_screen_and_depression_stats(self) -> pd.DataFrame:
            """Вяртае зводную табліцу: 
            агульная колькасць хлопчыкаў і дзяўчынак, 
            колькасць падлеткаў ў дэпрэсіі, 
            колькасць падлеткаў, што спажывае болей за 5 гадзін экраннага часу ў дзень і колькі з іх ў дэпрэсіі."""
            return self.execute(
                """
                SELECT 
                    sex AS sex,
                    COUNT(*) AS "total",
                    SUM(CASE WHEN est_leisure_screen_hours > 5 THEN 1 ELSE 0 END) AS "screen_over_5",
                    SUM(CASE WHEN depressed >= 1 THEN 1 ELSE 0 END) AS "depressed",
                    SUM(CASE WHEN est_leisure_screen_hours > 5 AND depressed >= 1 THEN 1 ELSE 0 END) AS "depressed_screen_over_5"
                FROM screen_time
                GROUP BY sex;
                """
            )
        
    def sleep(self) -> pd.DataFrame:
        """Вяртае адсартэраваныя гадзіны сну."""

        return self.execute(
            """
            SELECT avg_sleep_hours 
            FROM screen_time
            ORDER BY avg_sleep_hours DESC;
            """
        )


    def get_bdi_with_categories(self) -> pd.DataFrame:
        """Вылічвае балы Бэка для кожнага падлетка і прысвойвае катэгорыю дэпрэсіі праз SQL."""
        return self.execute(
            """
            WITH bdi_sums AS (
                SELECT 
                    subject_id,
                    (
                        COALESCE(bdi_item_01, 0) + COALESCE(bdi_item_02, 0) + COALESCE(bdi_item_03, 0) + 
                        COALESCE(bdi_item_04, 0) + COALESCE(bdi_item_05, 0) + COALESCE(bdi_item_06, 0) + 
                        COALESCE(bdi_item_07, 0) + COALESCE(bdi_item_08, 0) + COALESCE(bdi_item_09, 0) + 
                        COALESCE(bdi_item_10, 0) + COALESCE(bdi_item_11, 0) + COALESCE(bdi_item_12, 0) + 
                        COALESCE(bdi_item_13, 0) + COALESCE(bdi_item_14, 0) + COALESCE(bdi_item_15, 0) + 
                        COALESCE(bdi_item_16, 0) + COALESCE(bdi_item_17, 0) + COALESCE(bdi_item_18, 0) + 
                        COALESCE(bdi_item_19, 0) + COALESCE(bdi_item_20, 0) + COALESCE(bdi_item_21, 0)
                    ) AS total_score
                FROM bdi_screen_items
            )
            SELECT 
                subject_id,
                total_score,
                CASE 
                    WHEN total_score BETWEEN 0 AND 13 THEN 'Мінімальная / Норма'
                    WHEN total_score BETWEEN 14 AND 19 THEN 'Лёгкая дэпрэсія'
                    WHEN total_score BETWEEN 20 AND 28 THEN 'Умераная дэпрэсія'
                    WHEN total_score BETWEEN 29 AND 63 THEN 'Цяжкая дэпрэсія'
                    ELSE 'Невядома'
                END AS depression_category
            FROM bdi_sums
            ORDER BY total_score DESC;
            """
        )

    def get_depression_stats(self) -> pd.DataFrame:
        """Вяртае колькасць падлеткаў у кожнай катэгорыі дэпрэсіі."""
        return self.execute(
            """
            WITH bdi_sums AS (
                SELECT 
                    (COALESCE(bdi_item_01, 0) + COALESCE(bdi_item_02, 0) + COALESCE(bdi_item_03, 0) + 
                    COALESCE(bdi_item_04, 0) + COALESCE(bdi_item_05, 0) + COALESCE(bdi_item_06, 0) + 
                    COALESCE(bdi_item_07, 0) + COALESCE(bdi_item_08, 0) + COALESCE(bdi_item_09, 0) + 
                    COALESCE(bdi_item_10, 0) + COALESCE(bdi_item_11, 0) + COALESCE(bdi_item_12, 0) + 
                    COALESCE(bdi_item_13, 0) + COALESCE(bdi_item_14, 0) + COALESCE(bdi_item_15, 0) + 
                    COALESCE(bdi_item_16, 0) + COALESCE(bdi_item_17, 0) + COALESCE(bdi_item_18, 0) + 
                    COALESCE(bdi_item_19, 0) + COALESCE(bdi_item_20, 0) + COALESCE(bdi_item_21, 0)) AS total_score
                FROM bdi_screen_items
            )
            SELECT 
                CASE 
                    WHEN total_score BETWEEN 0 AND 13 THEN 'Мінімальная / Норма'
                    WHEN total_score BETWEEN 14 AND 19 THEN 'Лёгкая дэпрэсія'
                    WHEN total_score BETWEEN 20 AND 28 THEN 'Умераная дэпрэсія'
                    WHEN total_score BETWEEN 29 AND 63 THEN 'Цяжкая дэпрэсія'
                    ELSE 'Невядома'
                END AS depression_category,
                COUNT(*) AS count_teenagers
            FROM bdi_sums
            GROUP BY depression_category
            ORDER BY count_teenagers DESC;
            """
        )
