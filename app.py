import streamlit as st
import pandas as pd
import plotly.graph_objects as go
# Импортируем функции из utils.py
from utils import generate_realistic_temperature_data, seasonal_temperatures
from utils import analyze_city, get_current_temp_sync, check_anomaly_from_history

st.title("Анализ температуры городов")
st.info("Приложение всегда имеет исторические данные либо реальные из CSV, либо сгенерированные.")

# Пример структуры CSV для пользователя
st.subheader("Пример структуры CSV")
st.markdown("""
CSV должен содержать следующие столбцы:
- `city` — название города  
- `timestamp` — дата в формате `YYYY-MM-DD`  
- `temperature` — среднесуточная температура (°C)  
- `season` — сезон года (`winter`, `spring`, `summer`, `autumn`)  
""")
example_data = pd.DataFrame({
    "city": ["Moscow", "Moscow", "Moscow", "Berlin", "Berlin"],
    "timestamp": ["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-01", "2010-01-02"],
    "temperature": [-7.0, -5.2, -6.5, 0.5, 1.2],
    "season": ["winter", "winter", "winter", "winter", "winter"]})
st.dataframe(example_data)

# Кэшированные функции
@st.cache_data
def load_or_generate_data(uploaded_file, cities):
    """Загрузка CSV или генерация данных"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = generate_realistic_temperature_data(cities)

    # Приведение timestamp к datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Приведение температуры к float
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    # Убираем строки с некорректными данными
    df = df.dropna(subset=["timestamp", "temperature"])
    return df

@st.cache_data
def analyze_city_cached(city_df):
    """Анализ исторических данных для города"""
    return analyze_city(city_df)

# Загрузка исторических данных
uploaded_file = st.file_uploader("Загрузите CSV с историческими данными", type=["csv"])
df = load_or_generate_data(uploaded_file, list(seasonal_temperatures.keys()))

# Ввод API-ключа OpenWeatherMap
api_key = st.text_input("Введите API-ключ OpenWeatherMap", type="password")
show_current_weather = api_key != ""  # показывать текущую погоду только если ключ есть

# Ошибка API-ключа
if show_current_weather:
    current_temp_check = get_current_temp_sync(df["city"].iloc[0], api_key)
    if "error" in current_temp_check:
        st.error(f"Ошибка API-ключа: {current_temp_check['error']}")
        show_current_weather = False  # не показываем текущую температуру дальше

# Выбор города для анализа
cities = df["city"].unique()
selected_city = st.selectbox("Выберите город для анализа", cities)
city_df = df[df["city"] == selected_city]

# Кнопка запуска анализа и визуализаций
if st.button("Показать статистику и графики"):

    # Анализ исторических данных
    analyzed_city_df = analyze_city_cached(city_df)
    # Описательная статистика
    st.subheader("Описательная статистика")
    st.dataframe(analyzed_city_df[["temperature", "rolling_mean_30d", "rolling_std_30d"]].describe())
    # Временной ряд с аномалиями
    st.subheader("Временной ряд температуры с аномалиями")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=analyzed_city_df["timestamp"],
        y=analyzed_city_df["temperature"],
        mode='lines',
        name='Температура'))
    anomalies = analyzed_city_df[analyzed_city_df["is_anomaly"]]
    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"],
        y=anomalies["temperature"],
        mode='markers',
        name='Аномалии',
        marker=dict(color='red', size=6)))
    fig.update_layout(xaxis_title='Дата', yaxis_title='Температура °C')
    st.plotly_chart(fig, use_container_width=True)
    # Сезонный профиль
    st.subheader("Сезонный профиль температуры")
    seasonal_stats = analyzed_city_df.groupby("season")["temperature"].agg(["mean", "std"]).reset_index()
    st.dataframe(seasonal_stats)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=seasonal_stats["season"],
        y=seasonal_stats["mean"],
        mode='lines+markers',
        name='Средняя температура'))
    fig2.add_trace(go.Scatter(
        x=seasonal_stats["season"],
        y=seasonal_stats["mean"] + seasonal_stats["std"],
        mode='lines',
        name='mean+std',
        line=dict(dash='dash', color='gray')))
    fig2.add_trace(go.Scatter(
        x=seasonal_stats["season"],
        y=seasonal_stats["mean"] - seasonal_stats["std"],
        mode='lines',
        name='mean-std',
        line=dict(dash='dash', color='gray')))
    fig2.update_layout(xaxis_title='Сезон', yaxis_title='Температура °C')
    st.plotly_chart(fig2, use_container_width=True)

    # Текущая температура через API
    if show_current_weather:
        current_temp_info = get_current_temp_sync(selected_city, api_key)
        if "error" in current_temp_info:
            st.error(f"Ошибка получения текущей температуры: {current_temp_info['error']}")
        else:
            temp = current_temp_info["temp"]
            desc = current_temp_info["description"]
            is_anomaly, season = check_anomaly_from_history(selected_city, temp, analyzed_city_df)
            st.subheader("Текущая температура")
            st.write(f"{temp}°C, {desc},  сезон: {season},  статус: {'аномальная' if is_anomaly else 'в пределах нормы'}")
