
# %% Import packages

import sys
# third party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV

    
# %% Load data

df_train = pd.read_csv("./Patients_train.csv", decimal='.')
df_test = pd.read_csv("./Patients_test.csv", decimal='.')

new_df_train = pd.DataFrame()
new_df_test = pd.DataFrame()
for case_id in df_train['caseid'].unique():
    print(case_id)
    Patient_df = df_train[df_train['caseid'] == case_id]
    Patient_df = Patient_df.copy()
    # find the start of ETCO2 value
    Induction_duration = 5*60
    Patient_df = Patient_df.loc[:Induction_duration, :]

    new_df_train = pd.concat([new_df_train, Patient_df], ignore_index=True)

for case_id in df_test['caseid'].unique():
    print(case_id)
    Patient_df = df_test[df_test['caseid'] == case_id]
    Patient_df = Patient_df.copy()
    # find the start of ETCO2 value
    Induction_duration = 5*60
    Patient_df = Patient_df.loc[:Induction_duration, :]

    new_df_test = pd.concat([new_df_test, Patient_df], ignore_index=True)


new_df_train = new_df_train[new_df_train['full_BIS'] == 0]
new_df_test = new_df_test[new_df_test['full_BIS'] == 0]

cov = ['age', 'sex', 'height', 'weight', 'bmi', 'lbm']
induction_info = ['min_BIS', 'max_BIS', 'max_c_propo', 'max_c_remin']
hemo = ['mean_HR']
Ce_bis_eleveld = ['Ce_Prop_Eleveld', 'Ce_Rem_Eleveld']
Ce_map_eleveld = ['Ce_Prop_MAP_Eleveld', 'Ce_Rem_MAP_Eleveld']
Cplasma_eleveld = ['Cp_Prop_Eleveld', 'Cp_Rem_Eleveld']

features = cov + hemo + Ce_bis_eleveld + Ce_map_eleveld + Cplasma_eleveld

ts = 1
new_df_train = new_df_train[features + ['caseid', 'BIS', 'train_set']].dropna()
new_df_train = new_df_train[::ts]
X_train = new_df_train[features]
y_train = new_df_train['BIS']

new_df_test = new_df_test[features + ['caseid', 'BIS']].dropna()
X_test = new_df_test[features]
y_test = new_df_test['BIS']

#  Scale data

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Polynomial features

poly = PolynomialFeatures(2)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

ps = PredefinedSplit(new_df_train['train_set'].values)
# %% Grid search

param_grid = {'kernel': ['rbf'], 'C': [0.1], 'gamma': np.logspace(-1, 3, 5), 'epsilon': np.logspace(-3, 1, 5)}
param_grid = {'alpha': np.logspace(-4, 0, 2), 'l1_ratio': np.linspace(0, 1, 3)}
grid = GridSearchCV(ElasticNet(), param_grid, refit=True, verbose=3, n_jobs=8)
grid.fit(X_train, y_train)

# %% Evaluate model

y_pred = grid.predict(X_test)
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
print('MDAPE: %.2f' % (np.mean(np.abs(y_pred - y_test) / y_test) * 100))

# %% Plot outputs

plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot(y_test, y_test, color='red', linewidth=1)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.grid()
plt.show()

# %%
