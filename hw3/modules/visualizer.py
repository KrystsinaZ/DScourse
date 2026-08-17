"""Візуалізацыя дадзеных, што захоўваліся ў SQLite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- Palette ----
BG        = '#F3F5F8'
PANEL     = '#FFFFFF'
INK       = '#22303F'
INK_SOFT  = '#7A8AA0'
SLATE     = '#4A5C73'
BLUE      = "#4682B4"
TEAL      = '#4C8C8C'
AMBER     = '#C99A3E'
CORAL     = '#C46B5E'
CORAL_SOFT= "#E9967A" 
LAVENDER  = '#8C7FB0'
GRID      = '#E3E8EE'

def style_ax(ax, title=None, xlabel=None, ylabel=None):
        if title: ax.set_title(title, color=INK, pad=12, fontsize=13)
        if xlabel: ax.set_xlabel(xlabel, color=INK_SOFT)
        if ylabel: ax.set_ylabel(ylabel, color=INK_SOFT)
        ax.spines['left'].set_color(GRID)
        ax.spines['bottom'].set_color(GRID)
        return ax


class DataVisualizer:
    """
    Графікі Matplotlib / Seaborn па выніках SQL запыту.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[Visualizer] Захавана: {path}")
        return path


    def plot_leisure_scr(self, df: pd.DataFrame) -> Path:
        """Гістаграмма: ацэнка вольнага экраннага часу ў гадзінах з падзелам па полу."""

        # Падрыхтоўка дадзеных (выкарыстоўваем .replace для хуткасці)
        df['sex'] = (
            df['sex']
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({'girl': 'Дзяўчаты', 'boy': 'Хлопцы'})
        )
        
        fig, ax = plt.subplots(figsize=(17, 5.2))
        
        # 1. Сярэднія значэнні па групах 
        '''# (разлічваем унутры Pandas)
        mean_male = df[df['sex'] == 'Хлопцы']['est_leisure_screen_hours'].mean()
        mean_female = df[df['sex'] == 'Дзяўчаты']['est_leisure_screen_hours'].mean()
        '''
        # (разлічваем у СКБД)
        mean_female = df[df['sex'] == 'Дзяўчаты']['gender_avg'].iloc[0] if not df[df['sex'] == 'Дзяўчаты'].empty else None
        mean_male = df[df['sex'] == 'Хлопцы']['gender_avg'].iloc[0] if not df[df['sex'] == 'Хлопцы'].empty else None

 
        # 2. Малюем дзве гістаграмы адначасова з празрыстасцю (alpha), каб яны не перакрывалі адна адну
        # Калі ў вас выкарыстоўваюцца кастомныя колеры, замяніце іх тут (напрыклад, MALE_COLOR, FEMALE_COLOR)
        ax.hist(
            df[df['sex'] == 'Дзяўчаты']['est_leisure_screen_hours'], 
            bins=30, alpha=0.7, color=CORAL_SOFT, edgecolor=BG, linewidth=0.6, label="Дзяўчаты"
        )
        ax.hist(
            df[df['sex'] == 'Хлопцы']['est_leisure_screen_hours'], 
            bins=30, alpha=0.6, color=BLUE, edgecolor=BG, linewidth=0.6, label="Хлопцы"
        )
        
        # 3. Пункцірная лінія сярэдняга значэння для дзяўчат
        if not pd.isna(mean_female):
            ax.axvline(
                mean_female, 
                color=CORAL_SOFT, linestyle='--', linewidth=2, 
                label=f"Сярэдняе (дзяўчаты) = {mean_female:.1f} г."
            )
            
        # 4. Пункцірная лінія сярэдняга значэння для хлопцаў
        if not pd.isna(mean_male):
            ax.axvline(
                mean_male, 
                color=BLUE, linestyle=':', linewidth=2, 
                label=f"Сярэдняе (хлопцы) = {mean_male:.1f} г."
            )
        
        style_ax(
            ax, 
            'Прыблізны штодзённы вольны час перад экранам у падлеткаў', 
            'Гадзіны', 
            'Колькасць падлеткаў'
        )
        
        # Павялічым памер шрыфту легенды, бо параметраў стала больш
        ax.legend(frameon=False, fontsize=10, loc="upper right")

        return self._save(fig, "est_leisure_screen_hours_by_gender.png")

    
    def plot_screen_by_sex(self, df_merged: pd.DataFrame) -> Path:
        """Barplot: Падлеткі з экранным часам > 5 гадзін на тле агульнай колькасці асоб."""

        # 1. Падрыхтоўка дадзеных (выкарыстоўваем .replace для хуткасці)
        df_merged["sex"] = (
            df_merged["sex"]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace({"girl": "Дзяўчаты", "boy": "Хлопцы"})
        )

        # 2. Ініцыялізацыя палатна
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
        ax.set_facecolor(BG)

        # Агульныя параметры для ўсіх barplot (аптымізацыя)
        base_kwargs = {"data": df_merged, "y": "sex", "legend": False, "ax": ax}

        # 3. Адмалёўка пластоў (Усяго і Дэпрэсія - шырокія слупкі)
        sns.barplot(x="total", color=GRID, width=0.6, **base_kwargs)
        
        sns.barplot(
            x="depressed",
            facecolor=(0, 0, 0, 0),
            edgecolor=INK,
            linewidth=1.8,
            linestyle="--",
            width=0.6,
            **base_kwargs
        )
        
        # 4. Адмалёўка пластоў (>5г і Дэпрэсія >5г - вузкія слупкі)
        sns.barplot(
            x="screen_over_5",
            hue="sex",
            palette={"Дзяўчаты": CORAL_SOFT, "Хлопцы": BLUE},
            width=0.4,
            **base_kwargs
        )
        
        sns.barplot(
            x="depressed_screen_over_5",
            facecolor=(0, 0, 0, 0),
            edgecolor=INK,
            linewidth=1.8,
            width=0.4,
            **base_kwargs
        )

        # 5. Тэкставыя меткі-падказкі
        for i, row in df_merged.iterrows():
            # Метка для экранаванага часу (>5г)
            ax.text(
                x=row["screen_over_5"], y=i, s=f">5г: {row['screen_over_5']}",
                va="center", ha="left", fontsize=10, fontweight="bold", color=INK
            )
            # Метка для агульнай колькасці
            ax.text(
                x=row["total"] - 2, y=i, s=f"усяго: {row['total']}",
                va="center", ha="right", fontsize=9, color=INK, fontstyle="italic"
            )
            # Метка для дэпрэсіі (па цэнтру)
            ax.text(
                x=row["depressed"] / 2, y=i - 0.22, s=f"{row['depressed']}",
                va="bottom", ha="center", fontsize=9, color=INK, fontweight="bold", fontstyle="italic"
            )
            # Метка для дэпрэсіі звыш 5 гадзін 
            ax.text(
                x=row["depressed_screen_over_5"] / 2, y=i, s=f"{row['depressed_screen_over_5']}",
                va="bottom", ha="center", fontsize=9, color=INK, fontweight="bold"
            )            

        # 6. Стылізацыя восяў і межаў праз функцыю style_ax
        title_text = "Падлеткі з экранным часам > 5 гадзін на тле агульнай колькасці удзельнікаў тэставання дэпрэсіі"
        
        # Перадаем ylabel="", каб схаваць надпіс "sex"
        style_ax(ax, title=title_text, xlabel="", ylabel="")
        
        ax.get_xaxis().set_visible(False)  # хаваем вось X для чысціні
        sns.despine(left=True, bottom=True) # прыбіраем межы, як і планавалася

        ax.set_ylabel(None)
        
        # 7. Легенда
        legend_elements = [
            mpatches.Patch(
                facecolor="none", edgecolor=INK, linestyle="--", linewidth=1,
                label="падлеткі з дэпрэсіяй з усёй сукупнасці асоб дадзенага полу",
            ),
            mpatches.Patch(
                facecolor="none", edgecolor=INK, linestyle="-", linewidth=1,
                label="падлеткі з дэпрэсіяй, чый экранны час перавысіў 5 гадзін",
            ),
        ]
        
        # loc="bottom left" вызначае, што кропка прывязкі легенды — яе левы ніжні кут.
        # bbox_to_anchor=(0.0, 1.02) адносна восяў (ад 0 да 1): 
        # 0.0 — левы край графіка, 1.02 — крыху вышэй за верхнюю мяжу слупкоў.
        ax.legend(
            handles=legend_elements, 
            loc="lower left", 
            bbox_to_anchor=(0.0, 0.9), 
            frameon=False,
            fontsize=9
        )
            
        plt.tight_layout()
        return self._save(fig, "screen_by_sex_seaborn.png")
    
    def plot_screen_by_sex_donut(self, df_sex: pd.DataFrame) -> Path:
        """Донатная дыяграма: доля хлопцаў і дзяўчат сярод тых, хто > 5 г. за экранам."""
        
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        
        color_map = {'Girl': CORAL_SOFT, 'Boy': BLUE}
        label_map = {'Girl': 'Дзяўчаты', 'Boy': 'Хлопцы'}
        
        labels = [label_map.get(x, x) for x in df_sex['sex']]
        colors = [color_map.get(x, BLUE) for x in df_sex['sex']]

        # Малюем кругавую дыяграму
        wedges, texts, autotexts = ax.pie(
            df_sex['count'], 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%', # паказваем адсоткі
            startangle=90,
            pctdistance=0.75, # змяшчае адсоткі бліжэй да вонкавага краю 
            textprops=dict(color=INK, fontsize=11),
            wedgeprops=dict(width=0.4, edgecolor=BG, linewidth=2) # width=0.4 робіць дзірку ў цэнтры
        )

        # Стылізуем адсоткі ўнутры круга
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # агульная колькасць падлеткаў
        total_children = df_sex['count'].sum()

        # тэкст у цэнтры (два радкі: лічба і слова "чал.")
        ax.text(
            0, 0, f"{total_children}\nчал.", 
            ha='center', va='center', 
            fontsize=14, fontweight='bold', 
            color=INK
        )

        style_ax(ax, title='Суадносіны палоў (экранны час > 5 гадзін)')
      
        plt.tight_layout()
        return self._save(fig, "screen_by_sex_donut.png")


    def plot_avg_sleep_hours(self, df: pd.DataFrame) -> Path:
        """Гістаграмма: сярэдняя працягласць сну ў гадзінах."""

        # Бяром слупок 'avg_sleep_hours', каб пазбегнуць памылак
        sleep_hours = df['avg_sleep_hours']
        
        fig, ax = plt.subplots(figsize=(17, 5.2))

        # Малюем гістаграму
        ax.hist(sleep_hours, bins=30, color=TEAL, edgecolor=BG, linewidth=0.6)
        
        ax.axvline(
            9, 
            color=INK, 
            linestyle='--', 
            linewidth=1.5, 
            label='рэкамендацыя для падлеткаў (~9 г.)'
        )
        
        style_ax(
            ax, 
            'Сярэдняя працягласць сну падлеткаў', 
            'Гадзіны', 
            'Колькасць падлеткаў'
        )
        
        ax.legend(frameon=False, fontsize=9)

        return self._save(fig, "avg_sleep_hours.png")

    def plot_bdi_histogram_tagged(self, df: pd.DataFrame) -> Path:
        """Гістаграма: размеркаванне балаў Бэка з падсветкай катэгорый."""
        # Праверка: калі датафрэйм пусты, адразу выходзім, каб не ламаць код
        if df.empty or 'depression_category' not in df.columns:
            print("Памылка: Датафрэйм пусты альбо адсутнічае калонка 'depression_category'")
            return Path()

        fig, ax = plt.subplots(figsize=(15, 5.5))
        
        category_order = [
                    'Мінімальная / Норма', 
                    'Лёгкая дэпрэсія', 
                    'Умераная дэпрэсія', 
                    'Цяжкая дэпрэсія'
                ]
        # Вызначаем толькі тыя катэгорыі, якія ёсць у нашых даных
        existing_categories = [cat for cat in category_order if cat in df['depression_category'].unique()]
        
        sns.histplot(
            data=df,
            x="total_score",
            hue="depression_category",
            hue_order=existing_categories, #category_order,
            multiple="stack", # Слупкі растуць адзін на адным
            bins=30,
            palette="coolwarm", # пераход ад сіняга (норма) да чырвонага (цяжкая)
            edgecolor=BG,
            linewidth=0.6,
            ax=ax
        )
        
        # Стылізуем легенду, якая ствараецца аўтаматычна бібліятэкай Seaborn
        sns.move_legend(ax, "upper right", frameon=False, title="Катэгорыі")
        
        style_ax(
            ax, 
            'Шчыльнасць размеркавання балаў Бэка сярод даследаваных падлеткаў', 
            'Набраныя балы', 
            'Колькасць падлеткаў'
        )
        
        return self._save(fig, "bdi_stacked_histogram.png")

    def plot_bdi_histogram_tagged4(self, df: pd.DataFrame) -> Path:
        """Малюе асобныя гістаграмы для кожнай катэгорыі дэпрэсіі, каб бачыць меншыя групы."""
        
        if df.empty or 'depression_category' not in df.columns:
            print("Памылка: Датафрэйм пусты альбо адсутнічае калонка 'depression_category'")
            return Path()

        category_order = ['Мінімальная / Норма', 'Лёгкая дэпрэсія', 'Умераная дэпрэсія', 'Цяжкая дэпрэсія']
        existing_categories = [cat for cat in category_order if cat in df['depression_category'].unique()]
        
        # Ствараем сетку з 4 графікаў у адзін радок (1 радок, калонак столькі, колькі груп)
        n_cats = len(existing_categories)
        fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 4.5), sharex=True)
        
        # Калі катэгорыя адна, axes будзе не спісам, а адным аб'ектам. Робім спісам:
        if n_cats == 1:
            axes = [axes]
            
        # Спіс колераў для кожнай катэгорыі (каб адпавядала палітры coolwarm)
        colors = {"Мінімальная / Норма": BLUE, "Лёгкая дэпрэсія": TEAL, 
                "Умераная дэпрэсія": AMBER, "Цяжкая дэпрэсія": CORAL}

        for ax, cat in zip(axes, existing_categories):
            sub_df = df[df['depression_category'] == cat]
            
            # Малюем гістаграму толькі для гэтай групы
            sns.histplot(
                data=sub_df,
                x="total_score",
                bins=15,
                color=colors.get(cat, SLATE),
                edgecolor=BG,
                linewidth=0.6,
                ax=ax
            )
            
            # Стылізуем кожны асобны мікра-графік
            ax.set_title(f"{cat}\n(N = {len(sub_df)})", fontsize=10, fontweight='bold', color=SLATE)
            ax.set_xlabel('Балы BDI', fontsize=9)
            ax.set_ylabel('Колькасць падлеткаў', fontsize=9)
            
            # Прыбіраем лішнія межы рэдактара
            # Альбо проста пакідаем чысты выгляд
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.suptitle('Размеркаванне балаў Бэка ўнутры кожнай катэгорыі дэпрэсіі', fontsize=12, fontweight='bold', color=SLATE, y=1.02)
        plt.tight_layout()
        
        return self._save(fig, "bdi_facets_histogram.png")
