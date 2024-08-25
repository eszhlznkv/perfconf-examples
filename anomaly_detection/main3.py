import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from adtk.detector import ThresholdAD, PersistAD
from adtk.visualization import plot
from adtk.transformer import DoubleRollingAggregate
from sklearn.preprocessing import StandardScaler

# Шаг 1: Чтение данных из CSV файлов
data = pd.read_csv("/Users/evgenii/Documents/GitHub/perfconf/perfcong-jmeter-examples/data_api/ar_api.jtl")
data_new_release = pd.read_csv("/Users/evgenii/Documents/GitHub/perfconf/perfcong-jmeter-examples/data_api/ar_api_power.jtl")

# Создание временного ряда для данных первого и второго тестов
data['timeStamp'] = pd.to_datetime(data['timeStamp'], unit='ms')
data = data.set_index('timeStamp')
data_new_release['timeStamp'] = pd.to_datetime(data_new_release['timeStamp'], unit='ms')
data_new_release = data_new_release.set_index('timeStamp')

# Получение уникальных меток (labels) из первого набора данных
unique_labels = data['label'].unique()

for label in unique_labels:
    # Отфильтруем данные по текущей метке для обоих тестов
    filtered_data = data[data['label'] == label]['elapsed']
    filtered_data_new_release = data_new_release[data_new_release['label'] == label]['elapsed']

    # Применение стандартного масштабирования
    scaler = StandardScaler()
    filtered_data_scaled = pd.Series(scaler.fit_transform(filtered_data.values.reshape(-1, 1)).flatten(), index=filtered_data.index)
    filtered_data_new_release_scaled = pd.Series(scaler.fit_transform(filtered_data_new_release.values.reshape(-1, 1)).flatten(), index=filtered_data_new_release.index)

    # Шаг 2: Применение детектора аномалий
    threshold_ad = ThresholdAD(high=2.5, low=-2.5)  # Определяем высокие и низкие пороги
    anomalies = threshold_ad.detect(filtered_data_scaled)
    anomalies_new_release = threshold_ad.detect(filtered_data_new_release_scaled)

    # Применение PersistAD для поиска точек разладки (сначала обучаем модель)
    persist_ad = PersistAD(window=10, c=3.0)  # окно и параметр постоянства
    persist_ad.fit(filtered_data_scaled)  # Обучение на данных первого теста
    breakpoints = persist_ad.detect(filtered_data_scaled)

    persist_ad.fit(filtered_data_new_release_scaled)  # Обучение на данных второго теста
    breakpoints_new_release = persist_ad.detect(filtered_data_new_release_scaled)

    # Шаг 3: Визуализация данных
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # График для первого теста
    plot(filtered_data_scaled, anomaly=anomalies, anomaly_color="red", ax=ax[0], title=f"Аномалии и точки разладки для метки: {label} (Тест 1)")
    ax[0].scatter(filtered_data_scaled.index, breakpoints * filtered_data_scaled, color='orange', s=20, label='Точка разладки')
    ax[0].legend()

    # График для второго теста
    plot(filtered_data_new_release_scaled, anomaly=anomalies_new_release, anomaly_color="red", ax=ax[1], title=f"Аномалии и точки разладки для метки: {label} (Тест 2)")
    ax[1].scatter(filtered_data_new_release_scaled.index, breakpoints_new_release * filtered_data_new_release_scaled, color='orange', s=20, label='Точка разладки')
    ax[1].legend()

    plt.tight_layout()
    plt.show()
