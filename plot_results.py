# %% import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# %% load data

standard_data = pd.read_csv("./outputs/standard_model.csv")
all_reg_data = pd.read_csv("./outputs/all_reg.csv")
# delay_data = pd.read_csv("./outputs/delay.csv")

# %% merge the data on standard_data

df = pd.merge(standard_data, all_reg_data, on=['caseid', 'Time'], how='left')
# df = pd.merge(df, delay_data, on=['case_id', 'Time'], how='left')

# %% plot the results

case_id = 2943
df_case = df[df['caseid'] == case_id].copy()
df_case.fillna(method='ffill', inplace=True)

fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

ax[0].plot(df_case['Time'], df_case['true_BIS'], label='data')
ax[0].plot(df_case['Time'], df_case['pred_BIS_Eleveld'], label='Eleveld model')
ax[0].plot(df_case['Time'], df_case['BIS_SVR'], label='SVR')
ax[0].plot(df_case['Time'], df_case['BIS_MLPRegressor'], label='RNN')
ax[0].plot(df_case['Time'], df_case['BIS_KNeighborsRegressor'], label='KNeighbors')
# ax[0].plot(df_case['Time'], df_case['pred_BIS_delay'], label='delayed SVR')
ax[0].set_ylabel('BIS')
ax[0].legend()
ax[0].grid()

ax[1].plot(df_case['Time'], df_case['true_MAP'], label='true MAP')
ax[1].plot(df_case['Time'], df_case['pred_MAP_Eleveld'], label='Eleveld model')
ax[1].plot(df_case['Time'], df_case['MAP_SVR'], label='SVR')
ax[1].plot(df_case['Time'], df_case['MAP_MLPRegressor'], label='RNN')
ax[1].plot(df_case['Time'], df_case['MAP_KNeighborsRegressor'], label='KNeighbors')

ax[1].set_ylabel('MAP')
ax[1].set_xlabel('Time (min)')
ax[1].legend()
ax[1].grid()
plt.show()
# %%
