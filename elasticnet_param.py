"""Get parameter of Elastic Net regression model and plot the coedfficients
Created on Wed May 18 08:47:45 2022"""


# %% import libraries

import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# %% Load regression model
filename = "./saved_reg/reg_ElasticNet_induction.pkl"
regressors = pickle.load(open(filename, 'rb'))
rg = regressors['BIS']

# %% Get parameters
coef = rg.coef_
cov = ['age', 'gender', 'height', 'weight']
Ce_bis_eleveld = ['$x_{p4}$', '$x_{r4}$']
Ce_map_eleveld = ['$x_{p5}$', '$x_{r5}$']
Cplasma_eleveld = ['$x_{p1}$', '$x_{r1}$']

X_col = cov + ['bmi', 'lbm', 'mean(HR)'] + Ce_map_eleveld + Ce_bis_eleveld + Cplasma_eleveld

# %% Plot coefficients
plt.figure(1, figsize=(6, 2))
plt.bar(X_col, coef, width=1)
plt.grid()
plt.xticks(rotation=90)
plt.ylabel('Coefficients')
plt.title('Elastic Net coefficients')
savepath = f'./outputs/figs/elasticnet_coeff.pdf'
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()
