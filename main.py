import matplotlib
matplotlib.use('TkAgg')  # Используем бэкенд, который поддерживает GUI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# Загрузка данных
data = pd.read_csv('diabetes.csv')

# Просмотр первых строк данных
print(data.head())

# Основная информация о данных
print(data.info())
print(data.describe())

# Проверка на пропущенные значения
print(data.isnull().sum())

# Пример обработки пропущенных значений (если есть)
# data = data.dropna()  # Удаление пропущенных значений
# data = data.fillna(data.mean())  # Заполнение пропущенных значений средним значением

# Преобразование данных в массив NumPy
data_array = data.to_numpy()

# Основные статистические характеристики с использованием NumPy
mean_values = np.mean(data_array, axis=0)
variance_values = np.var(data_array, axis=0)
correlation_matrix = np.corrcoef(data_array, rowvar=False)
min_values = np.min(data_array, axis=0)
max_values = np.max(data_array, axis=0)
quartiles = np.percentile(data_array, [25, 50, 75], axis=0)

print("Mean values:\n", mean_values)
print("Variance values:\n", variance_values)
print("Correlation matrix:\n", correlation_matrix)
print("Min values:\n", min_values)
print("Max values:\n", max_values)
print("Quartiles:\n", quartiles)

# Гистограммы с графиком функции распределения
data.hist(figsize=(10, 10), bins=20)
plt.show()

# Боксплоты
plt.figure(figsize=(10, 10))
sns.boxplot(data=data)
plt.show()

# Диаграммы рассеивания
sns.pairplot(data)
plt.show()

# Поиск выбросов с помощью межквартильного размаха
Q1 = np.percentile(data_array, 25, axis=0)
Q3 = np.percentile(data_array, 75, axis=0)
IQR = Q3 - Q1

outliers = (data_array < (Q1 - 1.5 * IQR)) | (data_array > (Q3 + 1.5 * IQR))
print("Number of outliers in each column:\n", np.sum(outliers, axis=0))

# Проверка нормальности распределения
for i, column in enumerate(data.columns):
    k2, p = stats.normaltest(data_array[:, i])
    alpha = 1e-3
    print(f'{column}: p-value = {p}')
    if p < alpha:
        print(f"{column} does not follow a normal distribution")
    else:
        print(f"{column} follows a normal distribution")

# Построение линейной регрессии
X = data[['Glucose', 'BloodPressure']].to_numpy()  # Пример двух признаков
y = data['Outcome'].to_numpy()  # Целевая переменная

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

predictions = model.predict(X)

plt.scatter(X[:, 0], y, color='blue')
plt.plot(X[:, 0], predictions, color='red')
plt.xlabel('Glucose')
plt.ylabel('Outcome')
plt.show()
