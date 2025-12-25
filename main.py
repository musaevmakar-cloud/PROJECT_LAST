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

plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
plt.hist(df['Price(USD)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Распределение цен')
plt.xlabel('Цена (USD)')
plt.ylabel('Частота')

plt.subplot(2, 3, 2)
df['Trend_Status'].value_counts().plot(kind='bar', color='lightcoral')
plt.title('Количество товаров по трендам')
plt.xticks(rotation=45)

plt.subplot(2, 3, 4)
season_pop = df.groupby('Season')['Popularity_Score'].mean().reset_index()
plt.plot(season_pop['Season'], season_pop['Popularity_Score'], marker='o', color='green')
plt.title('Средняя популярность по сезонам')
plt.xticks(rotation=45)

plt.subplot(2, 3, 5)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляция числовых признаков')

plt.subplot(2, 3, 3)
plt.hist(df['Customer_Rating'], bins=10, color='gold', edgecolor='black')
plt.title('Распределение рейтинга')
plt.xlabel('Рейтинг')
plt.ylabel('Частота')

brand_stats = df.groupby('Brand').agg({
    'Price(USD)': 'mean',
    'Customer_Rating': 'median'
}).round(2)
print('Средняя цена и медианный рейтинг по брендам:')
print(brand_stats.head(10))

count_by_season_gender = df.groupby(['Season', 'Gender']).size()
print('Количество товаров по сезону и гендеру:')
print(count_by_season_gender)

pivot = pd.pivot_table(
    df,
    values='Popularity_Score',
    index='Category',
    columns='Trend_Status',
    fill_value=0
).round(2)
print('Сводная таблица: средняя популярность по категории и тренду:')
print(pivot)