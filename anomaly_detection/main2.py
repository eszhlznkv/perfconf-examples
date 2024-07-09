# Установка библиотек

# Импорт библиотек
import pandas as pd
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD, QuantileAD, GeneralizedESDTestAD, PersistAD, SeasonalAD, VolatilityShiftAD
import matplotlib.pyplot as plt

# Установка стиля для графиков
plt.style.use('seaborn-white')  # Или 'ggplot', или 'bmh'

# Считывание данных из CSV-файла
df = pd.read_csv('raw_data.csv')

# Преобразование временных меток в формат datetime
df['startTime'] = pd.to_datetime(df['startTime'], unit='ms')
df.set_index('startTime', inplace=True)

# Оставляем только нужные столбцы для анализа
df = df[['responseTime']]

# Валидация временного ряда
s = validate_series(df)

# Использование различных детекторов

# Пороговый детектор
threshold_ad = ThresholdAD(high=100, low=50)
anomalies_threshold = threshold_ad.detect(s)
count_threshold = int(anomalies_threshold.sum())

# Визуализация результатов
#plot(s, anomaly=anomalies_threshold, anomaly_color='red')

# Вывод количества аномалий и сообщения об ошибке
anomaly_threshold = 10  # Заданный порог количества аномалий

print("Количество аномалий (ThresholdAD):", count_threshold)
if count_threshold > anomaly_threshold:
    print("Error: ThresholdAD выявил слишком много аномалий")
