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
count_threshold = anomalies_threshold.sum()

# Детектор квантилей
quantile_ad = QuantileAD(high=0.99, low=0.01)
quantile_ad.fit(s)
anomalies_quantile = quantile_ad.detect(s)
count_quantile = anomalies_quantile.sum()

# Generalized ESD Test детектор
esd_ad = GeneralizedESDTestAD()
esd_ad.fit(s)
anomalies_esd = esd_ad.detect(s)
count_esd = anomalies_esd.sum()

# Persist детектор
persist_ad = PersistAD(c=7.0, side='both')
persist_ad.fit(s)
anomalies_persist = persist_ad.detect(s)
count_persist = anomalies_persist.sum()

# Seasonal детектор
# seasonal_ad = SeasonalAD()
# seasonal_ad.fit(s)
# anomalies_seasonal = seasonal_ad.detect(s)
# count_seasonal = anomalies_seasonal.sum()

# Volatility Shift детектор
volatility_shift_ad = VolatilityShiftAD(c=6.0, side='both', window=10)
volatility_shift_ad.fit(s)
anomalies_volatility_shift = volatility_shift_ad.detect(s)
count_volatility_shift = anomalies_volatility_shift.sum()

# Визуализация результатов
# plot(s, anomaly=anomalies_threshold, anomaly_color='red')
# plot(s, anomaly=anomalies_quantile, anomaly_color='blue')
# plot(s, anomaly=anomalies_esd, anomaly_color='green')
# plot(s, anomaly=anomalies_persist, anomaly_color='purple')
# # plot(s, anomaly=anomalies_seasonal, anomaly_color='orange')
# plot(s, anomaly=anomalies_volatility_shift, anomaly_color='brown')

# Вывод количества аномалий и сообщения об ошибке
anomaly_threshold = 10  # Заданный порог количества аномалий

print("Количество аномалий (ThresholdAD):", count_threshold)
if count_threshold > anomaly_threshold:
    print("Error: ThresholdAD выявил слишком много аномалий")

print("Количество аномалий (QuantileAD):", count_quantile)
if count_quantile > anomaly_threshold:
    print("Error: QuantileAD выявил слишком много аномалий")

print("Количество аномалий (GeneralizedESDTestAD):", count_esd)
if count_esd > anomaly_threshold:
    print("Error: GeneralizedESDTestAD выявил слишком много аномалий")

print("Количество аномалий (PersistAD):", count_persist)
if count_persist > anomaly_threshold:
    print("Error: PersistAD выявил слишком много аномалий")
#
# print("Количество аномалий (SeasonalAD):", count_seasonal)
# if count_seasonal > anomaly_threshold:
#     print("Error: SeasonalAD выявил слишком много аномалий")

print("Количество аномалий (VolatilityShiftAD):", count_volatility_shift)
if count_volatility_shift > anomaly_threshold:
    print("Error: VolatilityShiftAD выявил слишком много аномалий")
