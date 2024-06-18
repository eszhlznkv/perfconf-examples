import pandas as pd
from scipy.stats import mannwhitneyu
from itertools import product


# Функция для выполнения Mann-Whitney U-теста и вывода результатов
def perform_mannwhitneyu_test(data1, data2):
    statistic, pvalue = mannwhitneyu(data1, data2, alternative='two-sided')
    result = {
        'statistic': statistic,
        'pvalue': pvalue,
        'significant': pvalue < 0.05
    }
    return result


if __name__ == '__main__':
    # Загрузка данных
    df1 = pd.read_csv('data/baseline/raw_data.csv')
    df2 = pd.read_csv('data/new_release/raw_data.csv')
    data1 = df1['responseTime']
    data2 = df2['responseTime']
    # Получение уникальных значений sampleLabel из обоих датафреймов
    unique_labels = set(df1['sampleLabel']).intersection(set(df2['sampleLabel']))

    # Итерация по уникальным значениям sampleLabel
    results = {}
    for label in unique_labels:
        # Фильтрация данных по текущему значению sampleLabel
        data1_filtered = df1[df1['sampleLabel'] == label]['responseTime']
        data2_filtered = df2[df2['sampleLabel'] == label]['responseTime']

        # Выполнение Mann-Whitney U-теста
        test_result = perform_mannwhitneyu_test(data1_filtered, data2_filtered)

        # Сохранение результатов
        results[label] = test_result

    # Вывод результатов (можно изменить формат вывода по вашему усмотрению)
    for label, result in results.items():
        print(f"Comparison for sampleLabel '{label}':")
        print(f"Mann-Whitney U-статистика: {result['statistic']}")
        print(f"P-значение: {result['pvalue']}")
        if result['significant']:
            print("Различия статистически значимы.")
        else:
            print("Нет статистически значимых различий.")
        print()
