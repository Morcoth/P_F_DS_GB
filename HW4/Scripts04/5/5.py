#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'HW4\Scripts04\5'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pylab import rcParams

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# <center><h2>Интерпретируемость моделей машинного обучения</h2></center>
#%% [markdown]
# Пример интерпретируемости:
#     
# в модели линейной регрессии по прогнозированию цен на недвижимость
# 
# увеличение площеди квартиры приводит к увеличению цены.
#%% [markdown]
# <center><h2>Применение деревьев решений</h2></center>
#%% [markdown]
# ```
# Пример использования деревьев решения для задач классификации - 
# 
# определение кредитоспособности заемщика и принятие решения о выдаче кредита.
# ```
#%% [markdown]
# #### Решение задачи классификации 
# 
# #### пассажиров Титаника с использованием деревьев решений

#%%
from sklearn.tree import DecisionTreeClassifier


#%%
X_train = pd.read_pickle('X_train.pkl')
y_train = pd.read_pickle('y_train.pkl')


#%%
X_valid = pd.read_pickle('X_valid.pkl')
y_valid = pd.read_pickle('y_valid.pkl')


#%%
max_depth_arr = np.arange(2, 20)
max_depth_arr


#%%
accuracy_valid = []
accuracy_train = []
for val in max_depth_arr:
    clf = DecisionTreeClassifier(max_depth=val, random_state=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    y_pred_train = clf.predict(X_train)
    acc_valid = accuracy_score(y_valid, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)
    accuracy_valid.append(acc_valid)
    accuracy_train.append(acc_train)
    print('max_depth = {} \n\tacc_valid = {} \n\tacc_train = {}\n'.format(val, acc_valid, acc_train))


#%%
rcParams['figure.figsize'] = 8, 5
plt.plot(max_depth_arr, accuracy_valid)
plt.plot(max_depth_arr, accuracy_train)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend(['valid', 'train'])
plt.xlim(2, 19)


#%%
clf = DecisionTreeClassifier(max_depth=7, random_state=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)


#%%
accuracy_score(y_valid, y_pred)

#%% [markdown]
# #### RandomForest

#%%
from sklearn.ensemble import RandomForestClassifier


#%%
from sklearn.model_selection import GridSearchCV


#%%
parameters = [{'n_estimators': [150, 200, 250], 
               'max_features': np.arange(5, 9),
               'max_depth': np.arange(5, 10)}]


#%%
clf = GridSearchCV(estimator=RandomForestClassifier(random_state=100), 
                   param_grid=parameters,
                   scoring='accuracy',
                   cv=5)

#%% [markdown]
# <center><h4>Кросс-валидация</h4></center>

#%%
clf.fit(X_train, y_train)


#%%
clf.best_params_


#%%
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


#%%
y_pred = clf.predict(X_valid)
accuracy_score(y_valid, y_pred)

#%% [markdown]
# Полученный классификатор clf равносилен такой модели:

#%%
clf = RandomForestClassifier(max_depth = 5, max_features = 5, n_estimators = 200, random_state=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)


#%%
accuracy_score(y_valid, y_pred)


#%%



