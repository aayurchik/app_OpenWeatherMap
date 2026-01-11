import pandas as pd
import numpy as np
from datetime import datetime
import requests

# Реальные средние температуры для городов по сезонам
seasonal_temperatures = {
    "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
    "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
    "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
    "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
    "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
    "Sydney": {"winter": 12, "spring": 18, "summer": 25, "autumn": 20},
    "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
    "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
    "Rio de Janeiro": {"winter": 20, "spring": 25, "summer": 30, "autumn": 25},
    "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
    "Los Angeles": {"winter": 15, "spring": 18, "summer": 25, "autumn": 20},
    "Singapore": {"winter": 27, "spring": 28, "summer": 28, "autumn": 27},
    "Mumbai": {"winter": 25, "spring": 30, "summer": 35, "autumn": 30},
    "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
    "Mexico City": {"winter": 12, "spring": 18, "summer": 20, "autumn": 15},}

# Сопоставление месяцев с сезонами
month_to_season = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}

#  Функции 
def generate_realistic_temperature_data(cities, num_years=10):
    """Генерация исторических данных о температуре"""
    dates = pd.date_range(start="2010-01-01", periods=365 * num_years, freq="D")
    data = []

    for city in cities:
        for date in dates:
            season = month_to_season[date.month]
            mean_temp = seasonal_temperatures[city][season]
            temperature = np.random.normal(loc=mean_temp, scale=5)
            data.append({"city": city, "timestamp": date, "temperature": temperature})

    df = pd.DataFrame(data)
    df['season'] = df['timestamp'].dt.month.map(lambda x: month_to_season[x])
    return df

def analyze_city(city_df):
    """Анализ исторических данных для одного города"""
    city_df = city_df.sort_values("timestamp")
    city_df["rolling_mean_30d"] = city_df["temperature"].rolling(30, min_periods=1).mean()
    city_df["rolling_std_30d"] = city_df["temperature"].rolling(30, min_periods=1).std()
    stats = city_df.groupby("season")["temperature"].agg(["mean", "std"]).reset_index()
    city_df = city_df.merge(stats, on="season", how="left")
    city_df["is_anomaly"] = (city_df["temperature"] < city_df["mean"] - 2*city_df["std"]) | \
                             (city_df["temperature"] > city_df["mean"] + 2*city_df["std"])
    return city_df

def get_current_temp_sync(city_name, api_key):
    """Синхронный запрос текущей температуры через OpenWeatherMap API"""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": response.json().get("message", "Unknown error")}
    data = response.json()
    return {"temp": data["main"]["temp"], "description": data["weather"][0]["description"]}

def check_anomaly_from_history(city, temp, historical_df):
    """Проверка текущей температуры на аномальность относительно исторических данных"""
    now = datetime.now()
    season = month_to_season[now.month]
    stats = historical_df[(historical_df["city"] == city) & (historical_df["season"] == season)].iloc[0]
    mean = stats["mean"]
    std = stats["std"]
    is_anomaly = (temp < mean - 2*std) or (temp > mean + 2*std)
    return is_anomaly, season
