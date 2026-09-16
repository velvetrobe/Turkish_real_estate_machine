import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Zingat", page_icon="🏠", layout="wide")

DATA = "data"

# ---------- Загрузка метрик и данных ----------
@st.cache_data
def load_metrics():
    try:
        val  = pd.read_csv(f"{DATA}/model_comparison_val.csv", index_col=0)
        test = pd.read_csv(f"{DATA}/model_comparison_test.csv", index_col=0)
        return val, test
    except Exception:
        return None, None

@st.cache_data
def load_test():
    try:
        X = pd.read_csv(f"{DATA}/X_test_raw.csv")
        y = pd.read_csv(f"{DATA}/y_test.csv").squeeze()
        return X, y
    except Exception:
        return None, None

val_df, test_df = load_metrics()
X_test, y_test = load_test()

# ---------- Меню ----------
st.sidebar.title("🏠 Zingat")
page = st.sidebar.radio("Разделы", ["Обзор", "Данные", "Модели", "Справка"])

# ---------- 1. Обзор ----------
if page == "Обзор":
    st.title("Прогнозирование стоимости недвижимости")
    st.write("Дашборд по данным платформы Zingat.")

    if test_df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Строк в датасете", "402 772")
        c2.metric("Признаков", "3 480")
        c3.metric("Лучшая модель", test_df["RMSE"].idxmin())
    else:
        st.warning("Метрики моделей не найдены. Запустите ноутбук.")

    if y_test is not None:
        st.subheader("Распределение цены")
        price = y_test.dropna()
        price = price[price > 0]
        st.plotly_chart(px.histogram(price, nbins=50, log_y=True),
                        use_container_width=True)

    st.subheader("Пропуски в данных")
    miss = pd.DataFrame({
        "Столбец": ["furnished", "size", "end_date", "floor_no",
                    "total_floor_count", "heating_type", "building_age"],
        "Пропуски, %": [100.0, 36.2, 34.0, 8.8, 6.9, 6.9, 6.8],
    })
    st.dataframe(miss, use_container_width=True, hide_index=True)

# ---------- 2. Данные ----------
elif page == "Данные":
    st.title("Исследование данных")

    if X_test is None or y_test is None:
        st.error("Файлы данных не найдены в папке data/.")
    else:
        st.subheader("Цена")
        price = y_test.dropna()
        price = price[price > 0]
        st.plotly_chart(px.histogram(price, nbins=50, log_y=True),
                        use_container_width=True)

        st.subheader("Площадь vs Цена")
        tmp = pd.DataFrame({
            "Площадь, м²": X_test["size"],
            "Цена": y_test.values,
        }).dropna()
        # Убираем выбросы по обоим осям, чтобы точки не сжимались в угол
        tmp = tmp[(tmp["Цена"] > 0) & (tmp["Цена"] < tmp["Цена"].quantile(0.99))]
        tmp = tmp[(tmp["Площадь, м²"] > 0) & (tmp["Площадь, м²"] < tmp["Площадь, м²"].quantile(0.99))]
        st.plotly_chart(
            px.scatter(tmp, x="Площадь, м²", y="Цена", opacity=0.3, log_y=True),
            use_container_width=True,
        )

        st.subheader("Цена по типу объявления")
        tmp = pd.DataFrame({
            "Тип": X_test["listing_type"].astype(str),
            "Цена": y_test.values,
        }).dropna()
        tmp = tmp[tmp["Цена"] > 0]
        st.plotly_chart(px.box(tmp, x="Тип", y="Цена", log_y=True),
                        use_container_width=True)

# ---------- 3. Модели ----------
elif page == "Модели":
    st.title("Сравнение моделей")

    if val_df is None or test_df is None:
        st.error("Файлы метрик не найдены в папке data/.")
    else:
        st.subheader("Валидация")
        st.dataframe(val_df.style.format("{:,.2f}"), use_container_width=True)

        st.subheader("Тест")
        st.dataframe(test_df.style.format("{:,.2f}"), use_container_width=True)

        best = test_df["RMSE"].idxmin()
        st.success(f"Лучшая модель: {best}  |  RMSE = {test_df.loc[best, 'RMSE']:,.0f}")

        st.subheader("Сравнение по метрикам")
        metric = st.selectbox("Метрика", ["MAE", "RMSE", "MAPE (%)", "R²"])
        fig = px.bar(test_df, x=test_df.index, y=metric, color=test_df.index)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------- 4. Справка ----------
elif page == "Справка":
    st.title("Справка")

    st.header("О приложении")
    st.write(
        "Дашборд по данным платформы Zingat (Турция). "
        "Показывает результаты разведочного анализа и сравнение "
        "трёх моделей регрессии: Ridge, Random Forest и KNN."
    )

    st.header("Разделы")
    st.markdown(
        """
        - **Обзор** — общие метрики, распределение цены, пропуски
        - **Данные** — интерактивные графики по цене и площади
        - **Модели** — таблицы метрик и сравнение моделей
        """
    )

    st.header("Признаки")
    st.markdown(
        """
        - **size** — площадь, м²
        - **total_rooms** — всего комнат
        - **living_rooms** — гостиных
        - **building_age_num** — возраст здания, лет
        - **tom** — время на рынке, дней
        - **listing_days** — дней активности объявления
        - **sub_type** — подтип (Daire, Villa, …)
        - **listing_type** — тип (1, 2 = продажа; 3 = аренда)
        - **heating_type** — отопление
        - **address** — адрес
        """
    )

    st.header("Ограничения")
    st.markdown(
        """
        - Целевая переменная смешивает валюты (TRY, EUR, USD, GBP)
        - Продажа и аренда моделируются вместе
        - Распределение цены сильно скошено
        """
    )

    st.header("Автор")
    st.write("Ваше Имя, 2026")