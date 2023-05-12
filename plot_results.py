# %% import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from bokeh.plotting import figure, show
from bokeh.layouts import row, column

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# %% load data

standard_data = pd.read_csv("./outputs/standard_model.csv")
all_reg_data = pd.read_csv("./outputs/all_reg.csv")
delay_data = pd.read_csv("./outputs/delay.csv")
lee_data = pd.read_csv("./outputs/output_lee.csv")
induction_data = pd.read_csv('outputs/induction.csv')

lee_data.rename(columns={'pred_BIS': 'pred_BIS_Lee', 'case_id': 'caseid'}, inplace=True)
induction_data.rename(columns={'BIS_ElasticNet': 'BIS_ElasticNet_induction',
                               'BIS_KNeighborsRegressor': 'BIS_KNeighborsRegressor_induction',
                               'BIS_KernelRidge': 'BIS_KernelRidge_induction',
                               'BIS_SVR': 'BIS_SVR_induction',
                               'BIS_MLPRegressor': 'BIS_MLPRegressor_induction'}, inplace=True)

# merge the data on standard_data

df = pd.merge(standard_data, all_reg_data, on=['caseid', 'Time'], how='left')
df = pd.merge(df, delay_data, on=['caseid', 'Time'], how='left')
df = pd.merge(df, lee_data[['caseid', 'Time', 'pred_BIS_Lee']], on=['caseid', 'Time'], how='left')
df = pd.merge(df, induction_data[['caseid', 'Time', 'BIS_ElasticNet_induction',
                                  'BIS_KNeighborsRegressor_induction', 'BIS_KernelRidge_induction',
                                  'BIS_SVR_induction', 'BIS_MLPRegressor_induction']], on=['caseid', 'Time'], how='left')

# %% plot the results

case_id = 2628
df_case = df[df['caseid'] == case_id].copy()
df_case.fillna(method='ffill', inplace=True)

plt.figure(figsize=(6, 2))
plt.plot(df_case['Time']/60, df_case['true_BIS'], label='data')
plt.plot(df_case['Time']/60, df_case['pred_BIS_Eleveld'], label='Eleveld model')
plt.plot(df_case['Time']/60, df_case['pred_BIS_Lee'], label='deep-learning')
plt.plot(df_case['Time']/60, df_case['BIS_SVR'], label='SVR')
# plt.plot(df_case['Time'], df_case['BIS_MLPRegressor'], label='RNN')
# plt.plot(df_case['Time'], df_case['BIS_KNeighborsRegressor_x'], label='KNeighbors')
# plt.plot(df_case['Time'], df_case['pred_BIS_delay'], label='delayed SVR')
plt.ylabel('BIS')
plt.xlabel('Time (min)')
plt.legend()
plt.grid()
savepath = f'./outputs/figs/BIS_case_{case_id}.pdf'
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()

plt.figure(figsize=(6, 2))
plt.plot(df_case['Time']/60, df_case['true_MAP'], label='data')
plt.plot(df_case['Time']/60, df_case['pred_MAP_Eleveld'], label='Eleveld model')
plt.plot(df_case['Time']/60, df_case['MAP_SVR'], label='SVR', color='#d62728')
# plt.plot(df_case['Time'], df_case['MAP_MLPRegressor'], label='RNN')
# ax[1].plot(df_case['Time'], df_case['MAP_KNeighborsRegressor_x'], label='KNeighbors')

plt.ylabel('MAP')
plt.xlabel('Time (min)')
plt.legend()
plt.grid()
savepath = f'./outputs/figs/MAP_case_{case_id}.pdf'
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()
# %% plot different delay

case_id = 29
df_case = df[df['caseid'] == case_id].copy()
df_case.fillna(method='ffill', inplace=True)

fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)


ax[0].plot(df_case['Time'], df_case['BIS_KNeighborsRegressor_x'], label='no delay', alpha=0.7)
ax[0].plot(df_case['Time'], df_case['BIS_plus_30_KNeighborsRegressor'].shift(30), label='+30s', alpha=0.7)
ax[0].plot(df_case['Time'], df_case['BIS_plus_120_KNeighborsRegressor'].shift(120), label='+2min', alpha=0.7)
ax[0].plot(df_case['Time'], df_case['BIS_plus_300_KNeighborsRegressor'].shift(300), label='+5min', alpha=0.7)
ax[0].plot(df_case['Time'], df_case['BIS_plus_600_KNeighborsRegressor'].shift(600), label='+10min', alpha=0.7)
ax[0].plot(df_case['Time'], df_case['true_BIS'], 'b', label='data')
ax[0].set_ylabel('BIS')
ax[0].legend()
ax[0].grid()


ax[1].plot(df_case['Time'], df_case['MAP_KNeighborsRegressor_x'], label='no delay', alpha=0.7)
ax[1].plot(df_case['Time'], df_case['MAP_plus_30_KNeighborsRegressor'].shift(30), label='+30s', alpha=0.7)
ax[1].plot(df_case['Time'], df_case['MAP_plus_120_KNeighborsRegressor'].shift(120), label='+2min', alpha=0.7)
ax[1].plot(df_case['Time'], df_case['MAP_plus_300_KNeighborsRegressor'].shift(300), label='+5min', alpha=0.7)
ax[1].plot(df_case['Time'], df_case['MAP_plus_600_KNeighborsRegressor'].shift(600), label='+10min', alpha=0.7)
ax[1].plot(df_case['Time'], df_case['true_MAP'], 'b', label='data')
ax[1].set_ylabel('MAP')
ax[1].set_xlabel('Time (min)')
ax[1].legend()
ax[1].grid()
plt.title('KNeighborsRegressor')
plt.show()

# %% same plot with bokeh

case_id = 2943
df_case = df[df['caseid'] == case_id].copy()
df_case.fillna(method='ffill', inplace=True)

p1 = figure(width=1500, height=500, title="BIS")
p1.line(df_case['Time'], df_case['BIS_KNeighborsRegressor_x'], legend_label='no delay', line_color='red')
p1.line(df_case['Time'], df_case['BIS_plus_30_KNeighborsRegressor'].shift(30), legend_label='+30s', line_color='blue')
p1.line(df_case['Time'], df_case['BIS_plus_120_KNeighborsRegressor'].shift(
    120), legend_label='+2min', line_color='green')
p1.line(df_case['Time'], df_case['BIS_plus_300_KNeighborsRegressor'].shift(
    300), legend_label='+5min', line_color='orange')
p1.line(df_case['Time'], df_case['BIS_plus_600_KNeighborsRegressor'].shift(
    600), legend_label='+10min', line_color='purple')
p1.line(df_case['Time'], df_case['true_BIS'], legend_label='data', line_color='black')
p1.legend.location = "top_left"
p1.xaxis.axis_label = 'Time (min)'
p1.yaxis.axis_label = 'BIS'

p2 = figure(width=1500, height=500, title="MAP")
p2.line(df_case['Time'], df_case['MAP_KNeighborsRegressor_x'], legend_label='no delay', line_color='red')
p2.line(df_case['Time'], df_case['MAP_plus_30_KNeighborsRegressor'].shift(30), legend_label='+30s', line_color='blue')
p2.line(df_case['Time'], df_case['MAP_plus_120_KNeighborsRegressor'].shift(
    120), legend_label='+2min', line_color='green')
p2.line(df_case['Time'], df_case['MAP_plus_300_KNeighborsRegressor'].shift(
    300), legend_label='+5min', line_color='orange')
p2.line(df_case['Time'], df_case['MAP_plus_600_KNeighborsRegressor'].shift(
    600), legend_label='+10min', line_color='purple')
p2.line(df_case['Time'], df_case['true_MAP'], legend_label='data', line_color='black')
p2.legend.location = "top_left"
p2.xaxis.axis_label = 'Time (min)'
p2.yaxis.axis_label = 'MAP'

show(column(p1, p2))

# %% plot induction phase with bokeh for all cases

df_induction = df[df['Time'] < 5*60].copy()
df_induction.fillna(method='ffill', inplace=True)
p = figure(width=1500, height=500, title="BIS")
p.xaxis.axis_label = 'Time (min)'
p.yaxis.axis_label = 'BIS error'

for case_id in df_induction['caseid'].unique():
    df_case = df_induction[df_induction['caseid'] == case_id]
    p.line(df_case['Time'], df_case['true_BIS'] - df_case['pred_BIS_Eleveld'],
           legend_label='Eleveld', line_color='red')
    p.line(df_case['Time'], df_case['true_BIS'] - df_case['BIS_KNeighborsRegressor'],
           legend_label='MLP reg', line_color='blue')

show(p)

p = figure(width=1500, height=500, title="BIS")
p.xaxis.axis_label = 'Time (min)'
p.yaxis.axis_label = 'BIS error'


show(p)

# %% same in matplotlib

df_induction = df[df['Time'] < 6*60]
# df_induction = df_induction[df_induction['Time'] > 1*60]
df_induction.fillna(method='bfill', inplace=True)
fig, ax = plt.subplots(1, 1, figsize=(6, 2))
flag = True
for case_id in df_induction['caseid'].unique():

    df_case = df_induction[df_induction['caseid'] == case_id]
    if flag:
        plt.plot(df_case['Time']/60, df_case['true_BIS'] - df_case['pred_BIS_Eleveld'],
                 color='red', alpha=1, label='Eleveld')
        plt.plot(df_case['Time']/60, df_case['true_BIS'] - df_case['BIS_ElasticNet_induction'],
                 color='blue', alpha=1, label='ElasticNet')
        flag = False
    else:
        plt.plot(df_case['Time']/60, df_case['true_BIS'] - df_case['pred_BIS_Eleveld'],
                 color='red', alpha=1)
        plt.plot(df_case['Time']/60, df_case['true_BIS'] - df_case['BIS_ElasticNet_induction'],
                 color='blue', alpha=1)

plt.legend()
plt.xlabel('Time (min)')
plt.ylabel('BIS error')
plt.grid()
savepath = f'./outputs/figs/induction.pdf'
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()
