#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'HW4\Scripts04\1'))
	print(os.getcwd())
except:
	pass

#%%
import numpy as np
import pandas as pd

#%% [markdown]
# #### Работа со встроенными наборами данных в scikit-learn
# 
# Набор данных Boston House Prices

#%%
from sklearn.datasets import load_boston


#%%
boston = load_boston()


#%%
boston.keys()

#%% [markdown]
# Данные о недвижимости

#%%
data = boston.data


#%%
data.shape


#%%
data

#%% [markdown]
# Target - величина, которую требуется предсказать (цена на недвижимость)

#%%
target = boston.target

#%% [markdown]
# Названия признаков

#%%
feature_names = boston.feature_names


#%%
feature_names

#%% [markdown]
# Описание датасета

#%%
for line in boston.DESCR.split('\n'):
    print(line)

#%% [markdown]
# Создадим два датафрейма

#%%
X = pd.DataFrame(data, columns=feature_names)


#%%
X.head()


#%%
X.shape


#%%
X.info()


#%%
y = pd.DataFrame(target, columns=['price'])


#%%
y.head()


#%%
y.info()

#%% [markdown]
# #### Разбиение данных на тренировочный и тестовый датасеты

#%%
from sklearn.model_selection import train_test_split


#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#%% [markdown]
# #### Построение модели линейной регрессии

#%%
from sklearn.linear_model import LinearRegression


#%%
lr = LinearRegression()

#%% [markdown]
# Задача линейной регрессии - найти подходящие коэффициенты w 
# 
# при признаках x для вычисления целевой переменной y,
# 
# минимизируя ошибку e:
# 
# ### $y = w_0 + w_1 * x_1 + w_2 * x_2 + ... + w_m * x_m + e$

#%%
lr.fit(X_train, y_train)


#%%
y_pred = lr.predict(X_test)


#%%
check_test = pd.DataFrame({'y_test': y_test['price'], 
                           'y_pred': y_pred.flatten()}, 
                          columns=['y_test', 'y_pred'])


#%%
check_test.head(10)

#%% [markdown]
# #### Метрики оценки качества регрессионных моделей
#%% [markdown]
# Средняя квадратичная ошибка

#%%
check_test['error'] = check_test['y_pred'] - check_test['y_test']


#%%
check_test.head()


#%%
initial_mse = (check_test['error'] ** 2).mean()
initial_mse


#%%
from sklearn.metrics import mean_squared_error


#%%
initial_mse = mean_squared_error(y_test, y_pred)
initial_mse

#%% [markdown]
# Средняя абсолютная ошибка

#%%
(np.abs(check_test['error'])).mean()


#%%
from sklearn.metrics import mean_absolute_error


#%%
mean_absolute_error(y_test, y_pred)

#%% [markdown]
# $R^2$

#%%
from sklearn.metrics import r2_score


#%%
r2_score(y_test, y_pred)

#%% [markdown]
# #### Просмотр коэффициентов линейной регрессии
#%% [markdown]
# $w_0$

#%%
lr.intercept_

#%% [markdown]
# $w_1 ... w_m$

#%%
lr.coef_


#%%
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


#%%
plt.barh(X_train.columns, lr.coef_.flatten())
plt.xlabel('Вес признака')
plt.ylabel('Признак')


#%%
X_train.describe()

#%% [markdown]
# #### Стандартизация признаков
#%% [markdown]
# При стандартизации от признака нужно отнять
# 
# среднее и поделить на среднеквадратичное отклонение:
# 
# ## $x_{scaled} = \frac{x - x_{mean}}{\sigma_x}$
#%% [markdown]
# Применим стандартизацию

#%%
from sklearn.preprocessing import StandardScaler


#%%
scaler = StandardScaler()


#%%
X_train_scaled = scaler.fit_transform(X_train)


#%%
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)


#%%
X_test_scaled = scaler.transform(X_test)


#%%
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


#%%
lr.fit(X_train_scaled, y_train)


#%%
plt.barh(X_train.columns, lr.coef_.flatten())
plt.xlabel('Вес признака')
plt.ylabel('Признак')


#%%
feats = ['CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']


#%%
def create_model(X_train, y_train, X_test, y_test, feats, model):
    model.fit(X_train.loc[:, feats], y_train)
    y_pred=model.predict(X_test.loc[:, feats])
    mse = mean_squared_error(y_test, y_pred)
    return mse


#%%
create_model(X_train_scaled, y_train, X_test_scaled, y_test, feats, LinearRegression())


#%%
# Сверяем с исходной ошибкой
initial_mse

#%% [markdown]
# #### Модели линейной регрессии с регуляризацией

#%%
from sklearn.linear_model import Lasso, Ridge

#%% [markdown]
# Lasso (линейная регрессия с L1-регуляризацией)

#%%
# Параметр alpha отвечает за регуляризацию
model = Lasso(alpha=0.003)


#%%
create_model(X_train_scaled, y_train, X_test_scaled, y_test, feats, model)

#%% [markdown]
# Ridge (линейная регрессия с L2-регуляризацией)

#%%
# Параметр alpha отвечает за регуляризацию
model = Ridge(alpha=0.001)


#%%
create_model(X_train_scaled, y_train, X_test_scaled, y_test, feats, model)


#%%
n = 21
coeffs = np.zeros((n, len(feats)))
alpha_list = np.logspace(-3, 1, n)
for i, val in enumerate(alpha_list):
    lasso = Lasso(alpha = val)
    lasso.fit(X_train_scaled.loc[:, feats], y_train)
    coeffs[i, :] = lasso.coef_.flatten()

for i in range(len(feats)):
    plt.plot(alpha_list, coeffs[:, i])
plt.title('Убывание абсолютных значений весов признаков\n при увеличении коэффициента регуляризации alpha (Lasso)')
plt.xlabel('alpha')
plt.ylabel('Вес признака')
plt.legend(feats)


#%%
n=66
coeffs = np.zeros((n, len(feats)))
alpha_list = np.logspace(-3, 3.5, n)
for i, val in enumerate(alpha_list):
    ridge = Ridge(alpha = val)
    ridge.fit(X_train_scaled.loc[:, feats], y_train)
    coeffs[i, :] = ridge.coef_.flatten()

for i in range(len(feats)):
    plt.plot(alpha_list, coeffs[:, i])
plt.title('Убывание абсолютных значений весов признаков\n при увеличении коэффициента регуляризации alpha (Ridge)')
plt.xlabel('alpha')
plt.ylabel('Вес признака')
plt.legend(feats)


#%%



