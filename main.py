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