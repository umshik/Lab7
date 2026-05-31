import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np

# _________________ Глобальные переменные _______________________________
VARIANT_NUMBER = 14
CURRENT_ID = 1
CURRENT_SCALE = "день"
CURRENT_AGG = "sum"
HEAT_VALUE = "v_oil"
BAR_GROUP = "season"
SCATTER_X = "press"
SCATTER_Y = "v_oil"
SCATTER_COLOR = "risk"

# _________________ Вспомогательные функции _____________________________
# Распределение по временам года
def get_season(month):
    if month in [12, 1, 2]:
        return 'зима'
    elif month in [3, 4, 5]:
        return 'весна'
    elif month in [6, 7, 8]:
        return 'лето'
    else:
        return 'осень'

# _____________ метрики IQR _________________________________
def replace_outliers(group):
    q1 = group.quantile(0.25)
    q3 = group.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    median = group.median()

    mask = (group < lower) | (group > upper)
    group[mask] = median
    return group

# ________________ обработчики кнопок _______________________
# изменение текущей скважины
def set_well_id():
    global CURRENT_ID
    value = well_entry.get()

    if not value.isdigit():
        messagebox.showerror("Ошибка", "Введите число")
        return

    well_num = int(value)

    if well_num not in df_work['well_id'].values:
        messagebox.showerror("Ошибка", f"Скважины {well_num} не существует")
        return

    CURRENT_ID = well_num
    print(f"Текущая скважина: {CURRENT_ID}")
    refresh_data()

# агрегация в зависимости от масштаба (день / месяц / год)
def aggregate_by_scale(df, scale, agg_method):
    df = df.copy()
    df = df.set_index('date')

    if agg_method == "sum":
        df = df.resample(scale).sum()
    elif agg_method == "mean":
        df = df.resample(scale).mean()
    elif agg_method == "median":
        df = df.resample(scale).median()

    return df.reset_index()

# масштаб (день / месяц / год)
def set_scale():
    global CURRENT_SCALE
    CURRENT_SCALE = scale_combo.get()
    refresh_data()

# изменение типа агрегации (сумма / среднее / медиана)
def set_aggregation():
    global CURRENT_AGG
    CURRENT_AGG = agg_var.get()
    refresh_data()

# Обработчик кнопки для тепловой карты
def set_heat_param():
    global HEAT_VALUE
    HEAT_VALUE = heat_value_combo.get()
    if current_chart == "heat_map":
        plot_heat_map()

# Обработчик кнопки для столбчатого графика
def set_bar_group():
    global BAR_GROUP
    BAR_GROUP = bar_group_combo.get()
    if current_chart == "bar":
        plot_bar()

# Обработчик кнопки для точечной диаграммы
def set_scatter_params():
    global SCATTER_X, SCATTER_Y, SCATTER_COLOR
    SCATTER_X = scatter_x_combo.get()
    SCATTER_Y = scatter_y_combo.get()
    SCATTER_COLOR = scatter_color_combo.get()
    if current_chart == "scatter":
        plot_scatter()

#_______ ПРЕДОБРАБОТКА ДАННЫХ (фильтрация, обрезка, вычисление производных признаков, оптимизация) _________
def preprocess_data():
    df_work = df_raw.copy()
    # ________________1. Фильтрация по условию варианта________________
    float_fields = ['press', 'temp', 'vib', 'fluid']

    mask_ni = np.zeros(df_work.shape[0], dtype=bool)
    for field in float_fields:
        mask_ni = mask_ni | np.isnan(df_work[field]) | np.isinf(df_work[field])

    df_work = df_work[~mask_ni]
    df_work = df_work[df_work['press'] >= 0]
    df_work['fluid'] = np.clip(df_work['fluid'], None, 100)
    vib_95 = np.percentile(df_work['vib'], 95)
    df_work['vib'] = np.clip(df_work['vib'], None, vib_95)

    # _______________ 2. Обрезка выбросов (IQR по группам) ________________________
    df_work['vib'] = df_work.groupby('well_id')['vib'].transform(replace_outliers)

    # _________________ 3. Безопасное вычисление производного признака_____
    # Перевод из секунд в дату
    df_work['date'] = pd.to_datetime(df_work['ts'], unit='s')

    # Время года
    df_work['season'] = df_work['date'].dt.month.apply(get_season)

    # Условный объем нефти (для различного типа агрегации (сумма / среднее / медиана)
    df_work['v_oil'] = (df_work['press'] * (df_work['fluid'] / 100) * (1 - df_work['vib'] / 100)).clip(lower=0)

    # Требует ли скважина внимания (уровень вибрации <33 процентиля low, >66 high, иначе medium)
    p_33 = np.percentile(df_work['vib'], 33)
    p_66 = np.percentile(df_work['vib'], 66)
    df_work['risk'] = 'medium'
    df_work.loc[df_work['vib'] < p_33, 'risk'] = 'low'
    df_work.loc[df_work['vib'] > p_66, 'risk'] = 'high'

    # ____________________ 4. Оптимизация категориальных полей_______________
    # Перевод категориальных полей из типа objet в category для оптимизации
    df_work['season'] = df_work['season'].astype('category')
    df_work['risk'] = df_work['risk'].astype('category')

    return df_work

#______________ ФУНКЦИИ ОТРИСОВКИ ГРАФИКОВ _______________________
#Линейный график
def plot_line():
    global current_chart
    current_chart = "line"
    well_frame.pack(side=tk.LEFT, padx=10)
    agg_frame.pack(side=tk.LEFT, padx=10)
    scale_frame.pack(side=tk.LEFT, padx=10)
    bar_frame.pack_forget()
    scatter_frame.pack_forget()
    heat_frame.pack_forget()

    fig.clear()
    ax = fig.add_subplot(111)

    df_one_well = df_work.loc[df_work['well_id'] == CURRENT_ID, ['date', 'v_oil']].copy()
    df_one_well = df_one_well.sort_values('date')

    if CURRENT_SCALE == "день":
        rule = 'D'
    elif CURRENT_SCALE == "месяц":
        rule = 'ME'
    else:                   # год
        rule = 'YE'

    df_agg = aggregate_by_scale(df_one_well, rule, CURRENT_AGG)

    ax.plot(df_agg['date'], df_agg['v_oil'], linewidth=1.5, color='saddlebrown', marker='o', markersize=3)

    agg_names = {"sum": "Сумма", "mean": "Среднее", "median": "Медиана"}
    ax.set_title(
        f'Динамика добычи нефти (скважина {CURRENT_ID}) - масштаб: {CURRENT_SCALE}, агрегация: {agg_names[CURRENT_AGG]}')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Объём нефти (усл. ед.)')
    ax.grid(True, alpha=0.3)

    if CURRENT_SCALE == "день":
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=120))
    elif CURRENT_SCALE == "месяц":
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    elif CURRENT_SCALE == "год":
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    canvas.draw_idle()

#Столбчатая диаграмма
def plot_bar():
    global current_chart
    current_chart = "bar"
    well_frame.pack(side=tk.LEFT, padx=10)
    agg_frame.pack(side=tk.LEFT, padx=10)
    bar_frame.pack(side=tk.LEFT, padx=10)
    scale_frame.pack_forget()
    scatter_frame.pack_forget()
    heat_frame.pack_forget()

    fig.clear()
    ax = fig.add_subplot(111)

    if BAR_GROUP == "season":
        grouped = df_work[df_work['well_id'] == CURRENT_ID].groupby('season')['v_oil'].agg(CURRENT_AGG)
        x_labels = grouped.index.tolist()
        values = grouped.values
        x_label = "Сезон"
        title_group = "сезонам"
    else:  # risk
        grouped = df_work[df_work['well_id'] == CURRENT_ID].groupby('risk')['v_oil'].agg(CURRENT_AGG)
        order = ['low', 'medium', 'high']
        grouped = grouped.reindex(order)
        x_labels = grouped.index.tolist()
        values = grouped.values
        x_label = "Уровень риска"
        title_group = "риску"

    agg_names = {"sum": "Сумма", "mean": "Среднее", "median": "Медиана"}

    ax.bar(x_labels, values, color=['mediumseagreen', 'darkorange', 'indianred'] if BAR_GROUP == "risk" else ['mediumpurple', 'lightskyblue', 'lightgreen', 'coral'])

    ax.set_title(f'Добыча нефти по {title_group} (скважина {CURRENT_ID}) - {agg_names[CURRENT_AGG]}')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Объём нефти (условный)')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    canvas.draw_idle()

#Точечная диаграмма
def plot_scatter():
    global current_chart
    current_chart = "scatter"
    well_frame.pack(side=tk.LEFT, padx=10)
    agg_frame.pack_forget()
    scatter_frame.pack(side=tk.LEFT, padx=10)
    scale_frame.pack_forget()
    bar_frame.pack_forget()
    heat_frame.pack_forget()

    fig.clear()
    ax = fig.add_subplot(111)

    df_one_well = df_work[df_work['well_id'] == CURRENT_ID]

    if SCATTER_COLOR == "risk":
        colors = {'low': 'lightgreen', 'medium': 'darkorange', 'high': 'indianred'}
        point_colors = df_one_well['risk'].map(colors)
        legend_title = "Риск"
        legend_elements = [
            Patch(facecolor='lightgreen', label='low'),
            Patch(facecolor='darkorange', label='medium'),
            Patch(facecolor='indianred', label='high')
        ]
    else:  # season
        season_colors = {'зима': 'lightskyblue', 'весна': 'mediumpurple', 'лето': 'lightgreen', 'осень': 'coral'}
        point_colors = df_one_well['season'].map(season_colors)
        legend_title = "Сезон"
        legend_elements = [
            Patch(facecolor='lightskyblue', label='зима'),
            Patch(facecolor='mediumpurple', label='весна'),
            Patch(facecolor='lightgreen', label='лето'),
            Patch(facecolor='coral', label='осень')
        ]

    ax.scatter(
        df_one_well[SCATTER_X],
        df_one_well[SCATTER_Y],
        c=point_colors,
        alpha=0.6,
        s=30
    )

    names = {"press": "Давление (атм)", "temp": "Температура (°C)",
             "vib": "Вибрация (мм/с)", "v_oil": "Объём нефти (условный)"}

    ax.set_xlabel(names.get(SCATTER_X, SCATTER_X))
    ax.set_ylabel(names.get(SCATTER_Y, SCATTER_Y))
    ax.set_title(f'Зависимость {names[SCATTER_Y]} от {names[SCATTER_X]} (скважина {CURRENT_ID})')
    ax.grid(True, alpha=0.3)

    ax.legend(handles=legend_elements, title=legend_title)

    fig.tight_layout()
    canvas.draw_idle()

#Тепловая карта
def plot_heat_map():
    global current_chart
    current_chart = "heat_map"
    agg_frame.pack(side=tk.LEFT, padx=10)
    heat_frame.pack(side=tk.LEFT, padx=10)
    scale_frame.pack_forget()
    bar_frame.pack_forget()
    scatter_frame.pack_forget()
    well_frame.pack_forget()

    fig.clear()
    ax = fig.add_subplot(111)

    pivot_data = df_work.pivot_table(
        values=HEAT_VALUE,
        index='well_id',
        columns='season',
        aggfunc=CURRENT_AGG
    )

    sns.heatmap(pivot_data, annot=False, cmap='viridis', ax=ax)

    agg_names = {"sum": "Сумма", "mean": "Среднее", "median": "Медиана"}
    ax.set_title(f'Тепловая карта: {HEAT_VALUE} ({agg_names[CURRENT_AGG]})')
    ax.set_xlabel('Сезон')
    ax.set_ylabel('Скважина')

    fig.tight_layout()
    canvas.draw_idle()

#Обновление данных
def refresh_data():
    global df_work
    df_work = preprocess_data()
    if current_chart == "line":
        plot_line()
    elif current_chart == "bar":
        plot_bar()
    elif current_chart == "scatter":
        plot_scatter()
    elif current_chart == "heat_map":
        plot_heat_map()

#Экспорт в .png и .pdf
def export_plot():
    filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches='tight')

# ___________________ ОСНОВНАЯ ОБЛАСТЬ (ИНТЕРФЕЙС) ______________________________________
df_raw = None  # Исходные данные
df_work = None  # Рабочая копия
fig = plt.Figure(figsize=(9, 5.5), dpi=100)
canvas = None
current_chart = "line"
# ________________________Настройка шрифтов для кириллицы__________________
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
# ── Загрузка и диагностика ──
df_raw = pd.read_csv('data.csv')

root = tk.Tk()
root.title(f"Дашборд: Вариант {VARIANT_NUMBER}")
root.geometry("1000x700")
root.configure(bg="#f0f2f5")
# ___________________ Контейнер для кнопок ___________________________
ctrl_frame = tk.Frame(root, bg="#f0f2f5")
ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
# ___________________ Контейнер для параметров _______________________
par_frame = tk.Frame(root, bg="#f0f2f5")
par_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
# ___________________ Контейнер для графика __________________________
plot_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, bd=1)
plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
# ____________________Адаптер matplotlib __________________
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
# _________________ Панель инструментов _____
toolbar = NavigationToolbar2Tk(canvas, plot_frame)
toolbar.update()
toolbar.pack(side=tk.TOP, fill=tk.X)

# _____________________Панель кнопок_____________________
tk.Button(ctrl_frame, text="Линейный", command=plot_line, width=16).pack(side=tk.LEFT, padx=4)
tk.Button(ctrl_frame, text="Столбчатый", command=plot_bar, width=16).pack(side=tk.LEFT, padx=4)
tk.Button(ctrl_frame, text="Точечная диаграмма", command=plot_scatter, width=16).pack(side=tk.LEFT, padx=4)
tk.Button(ctrl_frame, text="Тепловая карта", command=plot_heat_map, width=16).pack(side=tk.LEFT, padx=4)

tk.Button(ctrl_frame, text=" Обновить", command=refresh_data, width=12).pack(side=tk.RIGHT, padx=4)
tk.Button(ctrl_frame, text=" Экспорт", command=export_plot, width=12).pack(side=tk.RIGHT, padx=4)

for widget in par_frame.winfo_children():
    widget.destroy()

#_____________ Выбор скважины ________________
well_frame = tk.Frame(par_frame, bg="#f0f2f5")
tk.Label(well_frame, text="Скважина:", bg="#f0f2f5").pack(side=tk.LEFT, padx=4)
well_entry = tk.Entry(well_frame, width=8)
well_entry.pack(side=tk.LEFT, padx=4)
well_entry.insert(0, str(CURRENT_ID))
tk.Button(well_frame, text="Выбрать", command=set_well_id).pack(side=tk.LEFT, padx=4)
well_frame.pack(side=tk.LEFT, padx=10)

#_____________ Агрегация _________________________
agg_frame = tk.Frame(par_frame, bg="#f0f2f5")
tk.Label(agg_frame, text="Агрегация:", bg="#f0f2f5").pack(side=tk.LEFT, padx=4)
agg_var = tk.StringVar(value="sum")
tk.Radiobutton(agg_frame, text="Сумма", variable=agg_var, value="sum", command=set_aggregation, bg="#f0f2f5").pack(
    side=tk.LEFT, padx=2)
tk.Radiobutton(agg_frame, text="Среднее", variable=agg_var, value="mean", command=set_aggregation, bg="#f0f2f5").pack(
    side=tk.LEFT, padx=2)
tk.Radiobutton(agg_frame, text="Медиана", variable=agg_var, value="median", command=set_aggregation, bg="#f0f2f5").pack(
    side=tk.LEFT, padx=2)
agg_frame.pack(side=tk.LEFT, padx=10)

#________________ Масштаб (день / месяц / год) только для линейного графика _____________
scale_frame = tk.Frame(par_frame)
tk.Label(scale_frame, text="Масштаб:").pack(side=tk.LEFT)
scale_combo = ttk.Combobox(scale_frame, values=["день", "месяц", "год"], width=6)
scale_combo.set("день")
scale_combo.pack(side=tk.LEFT)
tk.Button(scale_frame, text="Применить", command=set_scale).pack(side=tk.LEFT, padx=4)

#______________ Группировка только для столбчатой диаграммы ___________________________
bar_frame = tk.Frame(par_frame)
tk.Label(bar_frame, text="Группировка:").pack(side=tk.LEFT)
bar_group_combo = ttk.Combobox(bar_frame, values=["season", "risk"], width=6)
bar_group_combo.set("season")
bar_group_combo.pack(side=tk.LEFT)
tk.Button(bar_frame, text="Применить", command=set_bar_group).pack(side=tk.LEFT, padx=4)

#______________ Выбор осей (только для точечной диаграммы) ___________________________
scatter_frame = tk.Frame(par_frame)
tk.Label(scatter_frame, text="X:").pack(side=tk.LEFT)
scatter_x_combo = ttk.Combobox(scatter_frame, values=["press", "temp", "vib", "v_oil"], width=5)
scatter_x_combo.set("press")
scatter_x_combo.pack(side=tk.LEFT)
tk.Label(scatter_frame, text="Y:").pack(side=tk.LEFT)
scatter_y_combo = ttk.Combobox(scatter_frame, values=["v_oil", "press", "temp", "vib"], width=5)
scatter_y_combo.set("v_oil")
scatter_y_combo.pack(side=tk.LEFT)
tk.Label(scatter_frame, text="Цвет:").pack(side=tk.LEFT)
scatter_color_combo = ttk.Combobox(scatter_frame, values=["risk", "season"], width=6)
scatter_color_combo.set("risk")
scatter_color_combo.pack(side=tk.LEFT)
tk.Button(scatter_frame, text="Применить", command=set_scatter_params).pack(side=tk.LEFT, padx=4)

#_________ Параметры для тепловой карты ______________________________________________
heat_frame = tk.Frame(par_frame)
tk.Label(heat_frame, text="Параметр:").pack(side=tk.LEFT)
heat_value_combo = ttk.Combobox(heat_frame, values=["v_oil", "press", "temp", "vib"], width=6)
heat_value_combo.set("v_oil")
heat_value_combo.pack(side=tk.LEFT)
tk.Button(heat_frame, text="Применить", command=set_heat_param).pack(side=tk.LEFT, padx=4)

scale_frame.pack_forget()
bar_frame.pack_forget()
scatter_frame.pack_forget()
heat_frame.pack_forget()

df_work = preprocess_data()
plot_line()
root.mainloop()