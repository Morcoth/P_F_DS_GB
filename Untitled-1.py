# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split 
pd.options.display.max_columns = 100
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import mean_absolute_error


#%%
data = pd.read_csv('./P_F_DS/data_v1-12.csv')

#%% [markdown]
# #градиентный спуск

#%%
X = data['x']


#%%
y = data['y']


#%%
tets = [1, 0.5, 0.1]


#%%
count_ = y.count()


#%%
def mse_(tets):
    for t in range(len(tets)):
        plt.scatter (X,y)
        W1 = np.sum((y-t*X)**2) / (count_)
        plt.scatter(X, y)
        plt.plot(X, X*W1, marker='o', linestyle='dashed',   linewidth=2, markersize=1)


#%%
mse_(tets)

#%% [markdown]
# Построить графики зависимости выхода модели от x, наложенные на диаграмму рассеяния, при  = 0.5 для случаев: а) слишком простой модели; б) переобучения; в) хорошей обобщающей способности.

#%%
plt.scatter(X, y, s=20, facecolors='none', edgecolors='r')
plt.plot (X, X*0.5)


#%%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
X_train1, X_test1, y_train1, y_test1 = train_test_split (X, y, test_size=0.01, random_state=15)


#%%
X_train1 = X_train1.values.reshape(-1,1)
y_train1 = y_train1.values.reshape(-1,1)
lr.fit(X_train1, y_train1)


#%%
X_test1 = X_test1.values.reshape(-1, 1)


#%%
y_pred1 = lr.predict(X_test1)


#%%
mse1=mseib(y_test1, y_pred1)
mse1


#%%
X_train2, X_test2, y_train2, y_test2 = train_test_split(X,y, test_size=0.1, random_state=15)
X_train2 = X_train2.values.reshape(-1,1)
y_train2 = y_train2.values.reshape(-1,1)
lr.fit(X_train2, y_train2)
X_test2 = X_test2.values.reshape(-1, 1)
y_pred2 = lr.predict(X_test2)
mse2=mseib(y_test2, y_pred2)
mse2


#%%
X_train3, X_test3, y_train3, y_test3 = train_test_split(X,y, test_size=1, random_state=15)
X_train3 = X_train3.values.reshape(-1,1)
y_train3 = y_train3.values.reshape(-1,1)
lr.fit(X_train3, y_train3)
X_test3 = X_test3.values.reshape(-1, 1)
y_pred3 = lr.predict(X_test3)
mse3=mseib(y_test3, y_pred3)
mse3


#%%
from sklearn.metrics import mean_squared_error
mseib=mean_squared_error


#%%
mse1=mseib(y_test1, y_pred1)
mse2=mseib(y_test2, y_pred2)
mse3=mseib(y_test3, y_pred3)
plt.scatter(X, y, s=20, facecolors='none', edgecolors='r')
plt.plot (X, X*mse1)
plt.plot (X, X*mse2)
plt.plot (X, X*mse3)


#%%


