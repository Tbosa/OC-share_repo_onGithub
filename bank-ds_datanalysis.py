# %%
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%connect_info

%matplotlib inline

print('\n', sys.path, '\n')
print(sys.argv, '\n')
print(os.getcwd(), '\n')

print(sys.version, '\n')
# %%
# dataset description : https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
catfeatures_dict = {'job': 'category'}

df_raw = pd.read_csv(os.path.join(os.getcwd(), 'Datasets', 'Supervised Learning', 'bank-additional-full.csv'), sep=";", header=0,
                     keep_default_na=True, dtype=catfeatures_dict)

print(df_raw.info(), '\n')  # n_raw = 41 188   |   p = 20 + 1 (y output) | no missing values

df_raw['y'] = df_raw['y'].apply(lambda x: 1 if x == 'yes' else 0)
print(
    f"Target 'y' - negative labels prop (%) : {round((df_raw['y'].value_counts()[0]/df_raw.shape[0])*100,2)}%", '\n')
print(
    f"Target 'y' - positive labels prop (%) : {round((df_raw['y'].value_counts()[1]/df_raw.shape[0])*100,2)}%")

# %% 'emp.var.rate' column analysis
fig, ax = plt.subplots(ncols=2)
ax[0].set_title(label = 'emp.var.rate variable histogram', fontdict = {'fontsize' : 10})
ax[1].set_title(label = 'emp.var.rate variable boxplot', fontdict = {'fontsize' : 10})

df_raw['emp.var.rate'].plot(kind='hist', ax = ax[0], grid=True, xticks=[-3, -2, -1, 0, 1])
df_raw['emp.var.rate'].plot(kind='box', ax = ax[1])

df_raw['emp.var.rate'].describe()
