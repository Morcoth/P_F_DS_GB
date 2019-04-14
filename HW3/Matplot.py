#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'HW3'))
	print(os.getcwd())
except:
	pass

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


#%%
pd.options.display.max_columns = 100


#%%
data = pd.read_csv('input/train.csv')


#%%
data.shape


#%%
data.head()

#%% [markdown]
# #### Scatter Plot

#%%
data = data.loc[data['Square'] < 200, :]


#%%
plt.scatter(data['Square'], data['Price'])


#%%
data.shape

#%% [markdown]
# #### Histogram

#%%
plt.hist(data['Price'], bins=10)


#%%
data.loc[(data['Price'] >= 59174.77802758) & (data['Price'] < 116580.64688182), :].shape


#%%
data.loc[(data['Price'] >= 116580.64688182) & (data['Price'] < 173986.51573605), :].shape


#%%
data.loc[(data['Price'] >= 575827.59771571) & (data['Price'] < 633233.46656995), :].shape


#%%
data['Price'].min()


#%%
data['Price'].max()

#%% [markdown]
# #### Jointplot

#%%
sns.jointplot(data['Square'], data['Price'])


#%%
get_ipython().run_line_magic('pinfo', 'sns.jointplot')


#%%



#%%
sns.jointplot(data['Square'], data['Price'], kind='reg')


#%%
sns.jointplot(data['Square'], data['Price'], kind='kde')

#%% [markdown]
# #### Boxplot

#%%
sns.boxplot(data['Price'], orient='v')


#%%
get_ipython().run_line_magic('pinfo', 'sns.boxplot')


#%%
2-й квартиль (медиана), IQR - интерквартильный размах, 1.5 IQR


#%%
data['Price'].describe()


#%%
249109.948055 - 153887.002294


#%%
1.5*(249109.948055 - 153887.002294)


#%%
sns.boxplot(data['Rooms'], data['Price'])

#%% [markdown]
# #### Pairplot

#%%
sns.pairplot(data.loc[:, ['Rooms', 'Square', 'KitchenSquare', 'Price']])


#%%
sns.pairplot(data.loc[:, ['Rooms', 'Square', 'KitchenSquare', 'Price']], kind='reg')


#%%
get_ipython().run_line_magic('pinfo', 'sns.pairplot')


#%%



