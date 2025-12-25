import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Winter_Fashion_Trends_Dataset (1).csv')

print('Первые 5 строк:')
print(df.head())
print('Последние 5 строк:')
print(df.tail())

print('Общая информация (структура, типы, пропуски):')
print(df.info())

print('Типы данных по столбцам:')
print(df.dtypes)

print('Пропущенные значения по столбцам:')
missing = df.isnull().sum()
print(missing)

df = df.dropna().reset_index(drop=True)

duplicates_count = df.duplicated().sum()
print(f'Количество дублирующихся строк: {duplicates_count}')
df = df.drop_duplicates().reset_index(drop=True)

df['price_zscore'] = np.abs((df['Price(USD)'] - df['Price(USD)'].mean()) / df['Price(USD)'].std())
outliers = df[df['price_zscore'] > 3]

print('Описательная статистика числовых признаков:')
print(df.describe())

print('Распределение Trend_Status:')
print(df['Trend_Status'].value_counts())

print('Корреляционная матрица числовых столбцов:')
numeric_cols = df.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()
print(corr_matrix)