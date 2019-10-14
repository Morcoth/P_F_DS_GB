#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'HW4\Scripts04\3'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np

#%% [markdown]
# <center><h4>Support Vector Machine</h1></center>

#%%
from sklearn.svm import SVC


#%%
X_train = pd.read_pickle('X_train.pkl')
y_train = pd.read_pickle('y_train.pkl')


#%%
X_valid = pd.read_pickle('X_valid.pkl')
y_valid = pd.read_pickle('y_valid.pkl')

#%% [markdown]
# #### Нормализация признаков

#%%
from sklearn.preprocessing import MinMaxScaler


#%%
scaler = MinMaxScaler()


#%%
X_train.head(10)


#%%
X_train.describe()


#%%
cols_for_scale = ['Age', 'SibSp', 'Parch', 'Fare']


#%%
X_train[cols_for_scale] = scaler.fit_transform(X_train[cols_for_scale])


#%%
X_train.head(10)


#%%
X_train.describe()


#%%
X_valid[cols_for_scale] = scaler.transform(X_valid[cols_for_scale])


#%%
X_valid.describe()

#%% [markdown]
# #### Сохранение и загрузка модели нормализации

#%%
from sklearn.externals import joblib


#%%
joblib.dump(scaler, 'min_max_scaler.pkl')


#%%
scaler = joblib.load('min_max_scaler.pkl')

#%% [markdown]
# #### SVC

#%%
clf = SVC()


#%%
clf.fit(X_train, y_train)


#%%
y_pred = clf.predict(X_valid)


#%%
y_pred_train = clf.predict(X_train)

#%% [markdown]
# Оценим Accuracy

#%%
from sklearn.metrics import accuracy_score


#%%
accuracy_score(y_valid, y_pred)


#%%
accuracy_score(y_train, y_pred_train)

#%% [markdown]
# #### Тюнинг модели SVC

#%%
c_vals = np.logspace(-2, 5, 29)
c_vals


#%%
accuracy_valid = []
accuracy_train = []
for val in c_vals:
    clf = SVC(C=val)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    y_pred_train = clf.predict(X_train)
    acc_valid = accuracy_score(y_valid, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)
    accuracy_valid.append(acc_valid)
    accuracy_train.append(acc_train)
    print('C = {} \n\tacc_valid = {} \n\tacc_train = {}\n'.format(val, acc_valid, acc_train))


#%%
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
plt.plot(c_vals, accuracy_valid)
plt.plot(c_vals, accuracy_train)
plt.xlabel('Значение параметра C')
plt.ylabel('Accuracy')
plt.legend(['valid', 'train'])


#%%



