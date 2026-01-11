{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMLE/4pdwiVAaxwMPurblRr",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/aayurchik/app_OpenWeatherMap/blob/main/analysis_tasks.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import time\n",
        "from datetime import datetime\n",
        "from multiprocessing import Pool, cpu_count\n",
        "import requests\n",
        "import aiohttp\n",
        "import asyncio\n",
        "import nest_asyncio\n",
        "nest_asyncio.apply()  # чтобы asyncio работал в Jupyter/Colab"
      ],
      "metadata": {
        "id": "FMHADKfaV0IH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# *Загрузка csv*"
      ],
      "metadata": {
        "id": "7eRJdNrbX43q"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Od_RUg4dRjN1"
      },
      "outputs": [],
      "source": [
        "# Загрузка csv\n",
        "# Реальные средние температуры (примерные данные) для городов по сезонам\n",
        "seasonal_temperatures = {\n",
        "    \"New York\": {\"winter\": 0, \"spring\": 10, \"summer\": 25, \"autumn\": 15},\n",
        "    \"London\": {\"winter\": 5, \"spring\": 11, \"summer\": 18, \"autumn\": 12},\n",
        "    \"Paris\": {\"winter\": 4, \"spring\": 12, \"summer\": 20, \"autumn\": 13},\n",
        "    \"Tokyo\": {\"winter\": 6, \"spring\": 15, \"summer\": 27, \"autumn\": 18},\n",
        "    \"Moscow\": {\"winter\": -10, \"spring\": 5, \"summer\": 18, \"autumn\": 8},\n",
        "    \"Sydney\": {\"winter\": 12, \"spring\": 18, \"summer\": 25, \"autumn\": 20},\n",
        "    \"Berlin\": {\"winter\": 0, \"spring\": 10, \"summer\": 20, \"autumn\": 11},\n",
        "    \"Beijing\": {\"winter\": -2, \"spring\": 13, \"summer\": 27, \"autumn\": 16},\n",
        "    \"Rio de Janeiro\": {\"winter\": 20, \"spring\": 25, \"summer\": 30, \"autumn\": 25},\n",
        "    \"Dubai\": {\"winter\": 20, \"spring\": 30, \"summer\": 40, \"autumn\": 30},\n",
        "    \"Los Angeles\": {\"winter\": 15, \"spring\": 18, \"summer\": 25, \"autumn\": 20},\n",
        "    \"Singapore\": {\"winter\": 27, \"spring\": 28, \"summer\": 28, \"autumn\": 27},\n",
        "    \"Mumbai\": {\"winter\": 25, \"spring\": 30, \"summer\": 35, \"autumn\": 30},\n",
        "    \"Cairo\": {\"winter\": 15, \"spring\": 25, \"summer\": 35, \"autumn\": 25},\n",
        "    \"Mexico City\": {\"winter\": 12, \"spring\": 18, \"summer\": 20, \"autumn\": 15},}\n",
        "\n",
        "# Сопоставление месяцев с сезонами\n",
        "month_to_season = {12: \"winter\", 1: \"winter\", 2: \"winter\",\n",
        "                   3: \"spring\", 4: \"spring\", 5: \"spring\",\n",
        "                   6: \"summer\", 7: \"summer\", 8: \"summer\",\n",
        "                   9: \"autumn\", 10: \"autumn\", 11: \"autumn\"}\n",
        "\n",
        "# Генерация данных о температуре\n",
        "def generate_realistic_temperature_data(cities, num_years=10):\n",
        "    dates = pd.date_range(start=\"2010-01-01\", periods=365 * num_years, freq=\"D\")\n",
        "    data = []\n",
        "\n",
        "    for city in cities:\n",
        "        for date in dates:\n",
        "            season = month_to_season[date.month]\n",
        "            mean_temp = seasonal_temperatures[city][season]\n",
        "            # Добавляем случайное отклонение\n",
        "            temperature = np.random.normal(loc=mean_temp, scale=5)\n",
        "            data.append({\"city\": city, \"timestamp\": date, \"temperature\": temperature})\n",
        "\n",
        "    df = pd.DataFrame(data)\n",
        "    df['season'] = df['timestamp'].dt.month.map(lambda x: month_to_season[x])\n",
        "    return df\n",
        "\n",
        "# Генерация данных\n",
        "data = generate_realistic_temperature_data(list(seasonal_temperatures.keys()))\n",
        "data.to_csv('temperature_data.csv', index=False)\n",
        "\n",
        "\n",
        "# загрузка CSV как входных данных (как будет в Streamlit)\n",
        "df = pd.read_csv(\"temperature_data.csv\")\n",
        "\n",
        "# Приводим timestamp к datetime\n",
        "df[\"timestamp\"] = pd.to_datetime(df[\"timestamp\"])"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 1. **Анализ исторических данных**:\n",
        "   - Вычислить **скользящее среднее** температуры с окном в 30 дней для сглаживания краткосрочных колебаний.\n",
        "   - Рассчитать среднюю температуру и стандартное отклонение для каждого сезона в каждом городе.\n",
        "   - Выявить аномалии, где температура выходит за пределы $ \\text{среднее} \\pm 2\\sigma $.\n",
        "   - Попробуйте распараллелить проведение этого анализа. Сравните скорость выполнения анализа с распараллеливанием и без него."
      ],
      "metadata": {
        "id": "LfoSgVwMRnEJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Функция анализа одного города\n",
        "def analyze_city(city_df):\n",
        "    city_df = city_df.sort_values(\"timestamp\")\n",
        "    # Скользящее среднее и стандартное отклонение за 30 дней\n",
        "    city_df[\"rolling_mean_30d\"] = city_df[\"temperature\"].rolling(30, min_periods=1).mean()\n",
        "    city_df[\"rolling_std_30d\"] = city_df[\"temperature\"].rolling(30, min_periods=1).std()\n",
        "    # Рассчитать среднюю температуру и стандартное отклонение для каждого сезона в каждом городе.\n",
        "    stats = city_df.groupby(\"season\")[\"temperature\"].agg([\"mean\", \"std\"]).reset_index()\n",
        "    city_df = city_df.merge(stats, on=\"season\", how=\"left\")\n",
        "    # Выявить аномалии, где температура выходит за пределы  среднее±2σ\n",
        "    city_df[\"is_anomaly\"] = (city_df[\"temperature\"] < city_df[\"mean\"] - 2*city_df[\"std\"]) | (city_df[\"temperature\"] > city_df[\"mean\"] + 2*city_df[\"std\"])\n",
        "    return city_df\n",
        "\n",
        "# Последовательный вариант\n",
        "start_seq = time.time()\n",
        "sequential_result = pd.concat([analyze_city(group) for _, group in df.groupby(\"city\")])\n",
        "seq_time = time.time() - start_seq\n",
        "\n",
        "# Параллельный вариант\n",
        "start_par = time.time()\n",
        "with Pool(cpu_count()) as pool:\n",
        "    parallel_result = pd.concat(pool.map(analyze_city, [g for _, g in df.groupby(\"city\")]))\n",
        "par_time = time.time() - start_par\n",
        "\n",
        "# Вывод\n",
        "print(\"Скользящее среднее и std по 30 дням (первые 5 строк):\")\n",
        "print(parallel_result[[\"city\",\"timestamp\",\"temperature\",\"rolling_mean_30d\",\"rolling_std_30d\"]].head(), end=\"\\n\\n\")\n",
        "seasonal_stats = parallel_result.groupby([\"city\",\"season\"]).agg(season_mean_temp=(\"mean\",\"first\"),season_std_temp=(\"std\",\"first\")).reset_index()\n",
        "print(\"Сезонная статистика по городам (первые 5 строк):\")\n",
        "print(seasonal_stats.head(), end=\"\\n\\n\")\n",
        "print(\"Аномалии (первые 5 строк):\")\n",
        "print(parallel_result[[\"city\",\"timestamp\",\"temperature\",\"is_anomaly\"]].head(), end=\"\\n\\n\")\n",
        "print(f\"Время выполнения последовательного анализа: {seq_time:.2f} сек\")\n",
        "print(f\"Время выполнения параллельного анализа: {par_time:.2f} сек\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iHnVA8TLaC-i",
        "outputId": "222abe07-a795-45ea-e478-7ea8d89100c7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Скользящее среднее и std по 30 дням (первые 5 строк):\n",
            "      city  timestamp  temperature  rolling_mean_30d  rolling_std_30d\n",
            "0  Beijing 2010-01-01    -7.003093         -7.003093              NaN\n",
            "1  Beijing 2010-01-02     5.427891         -0.787601         8.790033\n",
            "2  Beijing 2010-01-03    -4.783678         -2.119627         6.629873\n",
            "3  Beijing 2010-01-04    -8.363924         -3.680701         6.249103\n",
            "4  Beijing 2010-01-05     2.695251         -2.405511         6.117109\n",
            "\n",
            "Сезонная статистика по городам (первые 5 строк):\n",
            "      city  season  season_mean_temp  season_std_temp\n",
            "0  Beijing  autumn         16.099292         5.203604\n",
            "1  Beijing  spring         13.082266         5.092471\n",
            "2  Beijing  summer         27.145010         5.168124\n",
            "3  Beijing  winter         -1.940413         4.910997\n",
            "4   Berlin  autumn         11.116922         5.197808\n",
            "\n",
            "Аномалии (первые 5 строк):\n",
            "      city  timestamp  temperature  is_anomaly\n",
            "0  Beijing 2010-01-01    -7.003093       False\n",
            "1  Beijing 2010-01-02     5.427891       False\n",
            "2  Beijing 2010-01-03    -4.783678       False\n",
            "3  Beijing 2010-01-04    -8.363924       False\n",
            "4  Beijing 2010-01-05     2.695251       False\n",
            "\n",
            "Время выполнения последовательного анализа: 0.13 сек\n",
            "Время выполнения параллельного анализа: 0.19 сек\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Сначала прогоняем analyze_city для всех городов\n",
        "full_result = pd.concat([analyze_city(group) for _, group in df.groupby(\"city\")])\n",
        "# Смотрим, есть ли хотя бы одна аномалия\n",
        "num_anomalies = full_result[\"is_anomaly\"].sum()\n",
        "print(f\"Всего аномалий в датасете: {num_anomalies}\")\n",
        "# вывести первые пару аномалий\n",
        "anomalies = full_result[full_result[\"is_anomaly\"]]\n",
        "print(\"Первые 5 аномалий:\")\n",
        "print(anomalies.head())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hZi5lyNufRxV",
        "outputId": "b2c278e8-16d6-46a6-b478-dfb3065e0fef"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Всего аномалий в датасете: 2446\n",
            "Первые 5 аномалий:\n",
            "        city  timestamp  temperature  season  rolling_mean_30d  \\\n",
            "45   Beijing 2010-02-15    10.296674  winter         -0.302975   \n",
            "55   Beijing 2010-02-25     8.492467  winter          0.316137   \n",
            "103  Beijing 2010-04-14     1.580806  spring         11.316172   \n",
            "111  Beijing 2010-04-22     1.180995  spring         11.903849   \n",
            "160  Beijing 2010-06-10    42.250482  summer         16.939747   \n",
            "\n",
            "     rolling_std_30d       mean       std  is_anomaly  \n",
            "45          5.033413  -1.940413  4.910997        True  \n",
            "55          5.164791  -1.940413  4.910997        True  \n",
            "103         4.493819  13.082266  5.092471        True  \n",
            "111         5.585082  13.082266  5.092471        True  \n",
            "160         9.533882  27.145010  5.168124        True  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 2. **Мониторинг текущей температуры**:\n",
        "   - Подключить OpenWeatherMap API для получения текущей температуры города. Для получения API Key (бесплатно) надо зарегистрироваться на сайте. Обратите внимание, что API Key может активироваться только через 2-3 часа, это нормально. Посему получите ключ заранее.\n",
        "   - Получить текущую температуру для выбранного города через OpenWeatherMap API.\n",
        "   - Определить, является ли текущая температура нормальной, исходя из исторических данных для текущего сезона.\n",
        "   - Данные на самом деле не совсем реальные (сюрпрайз). Поэтому на момент эксперимента погода в Берлине, Каире и Дубае была в рамках нормы, а в Пекине и Москве аномальная. Протестируйте свое решение для разных городов.\n",
        "   - Попробуйте для получения текущей температуры использовать синхронные и асинхронные методы. Что здесь лучше использовать?"
      ],
      "metadata": {
        "id": "g56Kg6yygHMM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#  Функция синхронного запроса текущей температуры для одного города\n",
        "def get_current_temp_sync(city_name, api_key):\n",
        "    # формируем URL запроса к OpenWeatherMap API\n",
        "    url = f\"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric\"\n",
        "    response = requests.get(url)\n",
        "    # обработка ошибок\n",
        "    if response.status_code != 200:\n",
        "        return {\"error\": response.json().get(\"message\", \"Unknown error\")}\n",
        "    data = response.json()\n",
        "    # возвращаем текущую температуру и описание погоды\n",
        "    return {\"temp\": data[\"main\"][\"temp\"], \"description\": data[\"weather\"][0][\"description\"]}\n",
        "\n",
        "# Асинхронная функция запроса текущей температуры для одного города\n",
        "async def fetch_temp_async(session, city, api_key):\n",
        "    url = f\"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric\"\n",
        "    async with session.get(url) as resp:\n",
        "        # проверка ошибок ответа\n",
        "        if resp.status != 200:\n",
        "            return city, {\"error\": await resp.json()}\n",
        "        data = await resp.json()\n",
        "        # возвращаем текущую температуру и описание погоды\n",
        "        return city, {\"temp\": data[\"main\"][\"temp\"], \"description\": data[\"weather\"][0][\"description\"]}\n",
        "\n",
        "# Асинхронная функция для получения температуры сразу для нескольких городов\n",
        "async def get_multiple_temps_async(cities, api_key):\n",
        "    async with aiohttp.ClientSession() as session:\n",
        "        tasks = [fetch_temp_async(session, city, api_key) for city in cities]\n",
        "        results = await asyncio.gather(*tasks)\n",
        "        return dict(results)\n",
        "\n",
        "# Функция проверки аномалии на основе исторических данных\n",
        "def check_anomaly_from_history(city, temp, historical_df):\n",
        "    # определяем текущий месяц и сезон\n",
        "    now = datetime.now()\n",
        "    season = month_to_season[now.month]\n",
        "    # извлекаем исторические статистики для данного города и сезона\n",
        "    stats = historical_df[(historical_df[\"city\"] == city) & (historical_df[\"season\"] == season)].iloc[0]\n",
        "    mean = stats[\"mean\"]\n",
        "    std = stats[\"std\"]\n",
        "    # температура считается аномальной, если выходит за пределы mean ± 2*std\n",
        "    is_anomaly = (temp < mean - 2*std) or (temp > mean + 2*std)\n",
        "    return is_anomaly, season\n",
        "\n",
        "# Настройка API и список городов\n",
        "api_key = \"932cf927ad1e580b34ca7784b02376ff\"\n",
        "cities = [\"Moscow\", \"Berlin\", \"Cairo\", \"Dubai\", \"Beijing\"]\n",
        "\n",
        "# Синхронный запрос всех городов и измерение времени\n",
        "start_sync = time.time()  # начало таймера\n",
        "sync_results = {}\n",
        "for city in cities:\n",
        "    sync_results[city] = get_current_temp_sync(city, api_key)\n",
        "sync_time = time.time() - start_sync  # время выполнения синхронного метода\n",
        "\n",
        "# Вывод результатов синхронного метода с проверкой аномалий\n",
        "print(\"Синхронный метод:\")\n",
        "for c, info in sync_results.items():\n",
        "    if \"error\" in info:\n",
        "        print(c, \"Ошибка:\", info[\"error\"])  # обработка ошибок запроса\n",
        "    else:\n",
        "        temp = info[\"temp\"]\n",
        "        desc = info[\"description\"]\n",
        "        # проверка аномальности через исторические данные\n",
        "        is_anomaly, season = check_anomaly_from_history(c, temp, parallel_result)\n",
        "        status = \"аномальная\" if is_anomaly else \"в пределах нормы\"\n",
        "        # вывод текущей температуры, сезона и статуса\n",
        "        print(f\"{c}: {temp}°C, {desc}, сезон {season}, {status}\")\n",
        "print(f\"Время выполнения синхронного метода: {sync_time:.2f} сек\\n\")  # сравнение методов\n",
        "\n",
        "# Асинхронный запрос всех городов и измерение времени\n",
        "start_async = time.time()  # начало таймера\n",
        "async_results = asyncio.run(get_multiple_temps_async(cities, api_key))\n",
        "async_time = time.time() - start_async  # время выполнения асинхронного метода\n",
        "\n",
        "# Вывод результатов асинхронного метода с проверкой аномалий\n",
        "print(\"Асинхронный метод:\")\n",
        "for c, info in async_results.items():\n",
        "    if \"error\" in info:\n",
        "        print(c, \"Ошибка:\", info[\"error\"])  # обработка ошибок запроса\n",
        "    else:\n",
        "        temp = info[\"temp\"]\n",
        "        desc = info[\"description\"]\n",
        "        # проверка аномальности через исторические данные\n",
        "        is_anomaly, season = check_anomaly_from_history(c, temp, parallel_result)\n",
        "        status = \"аномальная\" if is_anomaly else \"в пределах нормы\"\n",
        "        # вывод текущей температуры, сезона и статуса\n",
        "        print(f\"{c}: {temp}°C, {desc}, сезон {season}, {status}\")\n",
        "print(f\"Время выполнения асинхронного метода: {async_time:.2f} сек\\n\")  # сравнение методов\n",
        "\n",
        "# Вывод, какой метод лучше по скорости для этого числа городов\n",
        "if sync_time < async_time:\n",
        "    print(\"синхронный метод быстрее\")\n",
        "else:\n",
        "    print(\"асинхронный метод быстрее\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hwy6vmPJuIuT",
        "outputId": "9d9434bc-0296-493b-9333-46865c81ea02"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Синхронный метод:\n",
            "Moscow: -4.76°C, overcast clouds, сезон winter, в пределах нормы\n",
            "Berlin: -3.55°C, clear sky, сезон winter, в пределах нормы\n",
            "Cairo: 20.42°C, scattered clouds, сезон winter, в пределах нормы\n",
            "Dubai: 22.96°C, clear sky, сезон winter, в пределах нормы\n",
            "Beijing: -9.06°C, clear sky, сезон winter, в пределах нормы\n",
            "Время выполнения синхронного метода: 0.70 сек\n",
            "\n",
            "Асинхронный метод:\n",
            "Moscow: -4.76°C, overcast clouds, сезон winter, в пределах нормы\n",
            "Berlin: -3.55°C, clear sky, сезон winter, в пределах нормы\n",
            "Cairo: 20.42°C, scattered clouds, сезон winter, в пределах нормы\n",
            "Dubai: 22.96°C, clear sky, сезон winter, в пределах нормы\n",
            "Beijing: -9.06°C, clear sky, сезон winter, в пределах нормы\n",
            "Время выполнения асинхронного метода: 0.24 сек\n",
            "\n",
            "асинхронный метод быстрее\n"
          ]
        }
      ]
    }
  ]
}
