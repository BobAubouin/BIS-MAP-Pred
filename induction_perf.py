"""Get peformance on Induction (5 first minutes)"""

# %% Imports

# standard imports
import numpy as np
import pandas as pd
# local imports
from metrics_functions import compute_metrics

# %% Load data
standard_data = pd.read_csv("./outputs/standard_model.csv")
reg_data = pd.read_csv("./outputs/all_reg.csv")

# Merge the data on standard_data
reg_data = pd.merge(standard_data, reg_data, on=['caseid', 'Time'], how='left')
# select induction phase
reg_data = reg_data[reg_data['Time'] <= 5*60]


# %% Compute metrics
results_df = pd.DataFrame()

for standard_name in ['Eleveld', 'Schnider-Minto', 'Marsh-Minto']:
    print(f"***{standard_name:-^30s}***\n")
    temp_df = reg_data[['caseid', 'true_BIS', f'pred_BIS_{standard_name}']].copy()
    temp_df.dropna(inplace=True)
    _, _, df_bis = compute_metrics(temp_df)
    df = pd.concat([pd.DataFrame({'name_rg': standard_name}, index=[0]), df_bis], axis=1)
    results_df = pd.concat([results_df, df], axis=0)


for name_rg in ['ElasticNet', 'KNeighborsRegressor', 'KernelRidge', 'SVR', 'MLPRegressor']:
    print(f"***{name_rg:-^30s}***\n")
    temp_df = reg_data[['caseid', 'true_BIS', f'BIS_{name_rg}']].copy()
    temp_df.dropna(inplace=True)
    temp_df.rename(columns={f'BIS_{name_rg}': 'pred_BIS'}, inplace=True)
    _, _, df_bis = compute_metrics(temp_df)
    df = pd.concat([pd.DataFrame({'name_rg': name_rg}, index=[0]), df_bis], axis=1)
    results_df = pd.concat([results_df, df], axis=0)

print('\n')
styler = results_df[['name_rg', 'MDPE', 'MDAPE']].style
styler.hide(axis='index')
# styler.format(precision=2)
print(styler.to_latex())
