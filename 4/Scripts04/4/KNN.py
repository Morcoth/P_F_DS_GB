#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'HW4\Scripts04\4'))
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


#%%
from sklearn.neighbors import KNeighborsClassifier


#%%
X_train = pd.read_pickle('X_train.pkl')
y_train = pd.read_pickle('y_train.pkl')


#%%
X_valid = pd.read_pickle('X_valid.pkl')
y_valid = pd.read_pickle('y_valid.pkl')

#%% [markdown]
# #### Масштабирование признаков с использованием RobustScaler

#%%
from sklearn.preprocessing import RobustScaler


#%%
cols_for_scale = ['Age', 'SibSp', 'Parch', 'Fare']


#%%
scaler = RobustScaler()


#%%
X_train[cols_for_scale] = scaler.fit_transform(X_train[cols_for_scale])


#%%
X_train.describe()


#%%
X_valid[cols_for_scale] = scaler.transform(X_valid[cols_for_scale])

#%% [markdown]
# #### Классификация с помощью KNN

#%%
k_vals = np.arange(2, 10)


#%%
accuracy_valid = []
accuracy_train = []
for val in k_vals:
    clf = KNeighborsClassifier(n_neighbors=val)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    y_pred_train = clf.predict(X_train)
    acc_valid = accuracy_score(y_valid, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)
    accuracy_valid.append(acc_valid)
    accuracy_train.append(acc_train)
    print('n_neighbors = {} \n\tacc_valid = {} \n\tacc_train = {}\n'.format(val, acc_valid, acc_train))


#%%
plt.plot(k_vals, accuracy_valid)
plt.plot(k_vals, accuracy_train)
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.legend(['valid', 'train'])


#%%
clf = KNeighborsClassifier(n_neighbors=6)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_valid)


#%%
accuracy_score(y_valid, y_pred)

#%% [markdown]
# #### Точность и полнота

#%%
y_valid.value_counts(normalize=True)


#%%
from sklearn.metrics import confusion_matrix


#%%
confusion_matrix(y_valid, y_pred)

#%% [markdown]
# True Negative

#%%
TN = ((y_valid == 0) & (y_pred == 0)).sum()
TN

#%% [markdown]
# False Positive

#%%
FP = ((y_valid == 0) & (y_pred == 1)).sum()
FP

#%% [markdown]
# False Negative

#%%
FN = ((y_valid == 1) & (y_pred == 0)).sum()
FN

#%% [markdown]
# True Positive

#%%
TP = ((y_valid == 1) & (y_pred == 1)).sum()
TP


#%%
# Normalized confusion matrix
cm = confusion_matrix(y_valid, y_pred) / y_valid.shape[0]
cm

#%% [markdown]
# Точность

#%%
Precision = TP / (TP + FP)
Precision


#%%
from sklearn.metrics import precision_score


#%%
precision_score(y_valid, y_pred)

#%% [markdown]
# Полнота

#%%
Recall = TP / (TP + FN)
Recall


#%%
from sklearn.metrics import recall_score


#%%
recall_score(y_valid, y_pred)

#%% [markdown]
# #### F1 score

#%%
F1 = 2 * (Precision * Recall) / (Precision + Recall)
F1


#%%
from sklearn.metrics import f1_score


#%%
f1_score(y_valid, y_pred)

#%% [markdown]
# #### Метрика AUC

#%%
y_pred_proba = clf.predict_proba(X_valid)


#%%
y_pred_proba


#%%
y_pred_proba = y_pred_proba[:, 1]


#%%
y_pred_proba


#%%
plt.hist(y_pred_proba[y_valid == 1], bins = 7)
plt.xlabel('Probability')
plt.ylabel('Количество')


#%%
plt.hist(y_pred_proba[y_valid == 0], bins = 7)
plt.xlabel('Probability')
plt.ylabel('Количество')


#%%
plt.hist(y_pred_proba[y_valid == 0], bins = 7, density = True, alpha=0.5)
plt.hist(y_pred_proba[y_valid == 1], bins = 7, density = True, alpha=0.5)
plt.legend(['not survived', 'survived'])
plt.xlabel('Probability')
plt.ylabel('Density')


#%%
from sklearn.metrics import roc_curve


#%%
fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba, pos_label = 1)

#%% [markdown]
# False Positive Rate (fall-out)
# ### $FPR = \frac{FP}{N} = \frac{FP}{FP + TN}$
#%% [markdown]
# True Positive Rate (recall, sensitivity, hit rate)
# ### $TPR = \frac{TP}{P} = \frac{TP}{TP + FN}$

#%%
fpr


#%%
tpr


#%%
thresholds


#%%
# FPR для порога, равного 1
((y_valid == 0) & (y_pred == 1) & (y_pred_proba >= 1)).sum() / (y_valid == 0).sum()


#%%
# TPR для порога, равного 1
((y_valid == 1) & (y_pred == 1) & (y_pred_proba >= 1)).sum() / (y_valid == 1).sum()


#%%
# FPR для порога, равного 0.83333333
((y_valid == 0) & (y_pred == 1) & (y_pred_proba >= 0.83333333)).sum() / (y_valid == 0).sum()


#%%
# TPR для порога, равного 0.83333333
((y_valid == 1) & (y_pred == 1) & (y_pred_proba >= 0.83333333)).sum() / (y_valid == 1).sum()


#%%
rcParams['figure.figsize'] = 5, 4.5
plt.plot(fpr, tpr, color='green', label='ROC curve')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')


#%%
from sklearn.metrics import roc_auc_score


#%%
roc_auc_score(y_valid, y_pred_proba)


#%%



