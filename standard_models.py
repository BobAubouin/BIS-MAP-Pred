#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:46:15 2022

@author: aubouinb
"""

import numpy as np
import pandas as pd
from metrics_functions import compute_metrics
import matplotlib.pyplot as plt
import python_anesthesia_simulator as pas


# %% load data
Patients_test = pd.read_csv("./data/Patients_test.csv", index_col=0)
Patients_test['MAP'].fillna(0, inplace=True)

# %% Perform simulation
model_name_list = ['Eleveld', 'Schnider-Minto', 'Marsh-Minto']
column = ['case_id', 'full_BIS', 'full_MAP', 'True_BIS', 'True_MAP']
for model_name in model_name_list:
    column.append('pred_BIS_' + model_name)
    column.append('pred_MAP_' + model_name)

Output_df = pd.DataFrame(columns=column)

i = 0
plot_flag = False
for model_name in model_name_list:
    if 'pred_BIS_' + model_name not in Patients_test.columns:
        Patients_test.insert(len(Patients_test.columns), 'pred_BIS_'+model_name, 0)
    if 'pred_MAP_' + model_name not in Patients_test.columns:
        Patients_test.insert(len(Patients_test.columns), 'pred_MAP_'+model_name, 0)


for caseid, Patient_df in Patients_test.groupby('caseid'):
    print(caseid)

    Patient_df.reset_index(inplace=True)
    Output_df_temp = pd.DataFrame(columns=column)
    Output_df_temp['case_id'] = [caseid]*len(Patient_df)

    # create model
    MAP_base_case = Patient_df['MAP_base_case'].iloc[0]
    age = int(Patient_df['age'].iloc[0])
    height = int(Patient_df['height'].iloc[0])
    weight = int(Patient_df['weight'].iloc[0])
    sex = int(Patient_df['sex'].iloc[0])

    Patient_simu_Schnider = pas.Patient([age, height, weight, sex],
                                        model_propo='Schnider', model_remi='Minto',
                                        map_base=MAP_base_case)
    Patient_simu_Marsh = pas.Patient([age, height, weight, sex],
                                     model_propo='Marsh_modified', model_remi='Minto',
                                     map_base=MAP_base_case)
    Patient_simu_Eleveld = pas.Patient([age, height, weight, sex],
                                       model_propo='Eleveld', model_remi='Eleveld',
                                       map_base=MAP_base_case)

    df_schnider = Patient_simu_Schnider.full_sim(u_propo=Patient_df['Propofol']*20/3600,
                                                 u_remi=Patient_df['Remifentanil']*20/3600)
    df_marsh = Patient_simu_Marsh.full_sim(u_propo=Patient_df['Propofol']*20/3600,
                                           u_remi=Patient_df['Remifentanil']*20/3600)
    df_eleveld = Patient_simu_Eleveld.full_sim(u_propo=Patient_df['Propofol']*20/3600,
                                               u_remi=Patient_df['Remifentanil']*20/3600)

    Output_df_temp['pred_BIS_Schnider-Minto'] = df_schnider['BIS']
    Output_df_temp['pred_MAP_Schnider-Minto'] = df_schnider['MAP']
    Output_df_temp['pred_BIS_Marsh-Minto'] = df_marsh['BIS']
    Output_df_temp['pred_MAP_Marsh-Minto'] = df_marsh['MAP']
    Output_df_temp['pred_BIS_Eleveld'] = df_eleveld['BIS']
    Output_df_temp['pred_MAP_Eleveld'] = df_eleveld['MAP']

    Output_df_temp['full_BIS'] = Patient_df['full_BIS']
    Output_df_temp['full_MAP'] = Patient_df['full_MAP']
    Output_df_temp['true_BIS'] = Patient_df['BIS']
    Output_df_temp['true_MAP'] = Patient_df['MAP']
    Output_df = pd.concat([Output_df, Output_df_temp], ignore_index=True)

    # Output_df_temp['pred_BIS'] = Output_df_temp['pred_BIS'].diff()
    # Output_df_temp['true_BIS'] = Output_df_temp['true_BIS'].diff()
    # Output_df_temp.loc[0, 'true_BIS'] = 0
    # Output_df_temp.loc[0, 'pred_BIS'] = 0
    for model_name in model_name_list:
        Patients_test.loc[Patients_test["caseid"] == caseid, 'pred_BIS_' +
                          model_name] = Output_df_temp['pred_BIS_'+model_name]
        Patients_test.loc[Patients_test["caseid"] == caseid, 'pred_MAP_' +
                          model_name] = Output_df_temp['pred_MAP_'+model_name].values

    if i % 5 == 0 and plot_flag:
        fig, ax = plt.subplots(2)
        ax[1].set_xlabel('caseid : ' + str(caseid))
        ax[0].plot(Patient_df['BIS'], label='true_BIS')
        ax[0].plot(Output_df_temp['pred_BIS_Schnider-Minto'], label='Schnider-Minto')
        ax[0].plot(Output_df_temp['pred_BIS_Marsh-Minto'], label='Marsh-Minto')
        ax[0].plot(Output_df_temp['pred_BIS_Eleveld'], label='Eleveld')
        ax[1].plot(Patient_df['MAP'], label='true_MAP')
        ax[1].plot(Output_df_temp['pred_MAP_Schnider-Minto'], label='Schnider-Minto')
        ax[1].plot(Output_df_temp['pred_MAP_Marsh-Minto'], label='Marsh-Minto')
        ax[1].plot(Output_df_temp['pred_MAP_Eleveld'], label='Eleveld')
        ax[0].legend()
        ax[1].legend()
        plt.show()
        plt.pause(0.05)
    i += 1

Output_df_BIS = Output_df[Output_df['full_BIS'] == 0]
Output_df_BIS_Schnider = Output_df_BIS[['pred_BIS_Schnider-Minto', 'true_BIS', 'case_id']]
Output_df_BIS_Marsh = Output_df_BIS[['pred_BIS_Marsh-Minto', 'true_BIS', 'case_id']]
Output_df_BIS_Eleveld = Output_df_BIS[['pred_BIS_Eleveld', 'true_BIS', 'case_id']]
Output_df_MAP = Output_df[Output_df['full_MAP'] == 0]
Output_df_MAP_Schnider = Output_df_MAP[['pred_MAP_Schnider-Minto', 'true_MAP', 'case_id']]
Output_df_MAP_Marsh = Output_df_MAP[['pred_MAP_Marsh-Minto', 'true_MAP', 'case_id']]
Output_df_MAP_Eleveld = Output_df_MAP[['pred_MAP_Eleveld', 'true_MAP', 'case_id']]

Output_df_BIS_Schnider = Output_df_BIS_Schnider[Output_df_BIS_Schnider['true_BIS'] != 0]
Output_df_BIS_Marsh = Output_df_BIS_Marsh[Output_df_BIS_Marsh['true_BIS'] != 0]
Output_df_BIS_Eleveld = Output_df_BIS_Eleveld[Output_df_BIS_Eleveld['true_BIS'] != 0]
Output_df_MAP_Schnider = Output_df_MAP_Schnider[Output_df_MAP_Schnider['true_MAP'] != 0]
Output_df_MAP_Marsh = Output_df_MAP_Marsh[Output_df_MAP_Marsh['true_MAP'] != 0]
Output_df_MAP_Eleveld = Output_df_MAP_Eleveld[Output_df_MAP_Eleveld['true_MAP'] != 0]

Output_df_BIS_Schnider.rename(columns={'pred_BIS_Schnider-Minto': 'pred_BIS'},
                              inplace=True)
Output_df_BIS_Marsh.rename(columns={'pred_BIS_Marsh-Minto': 'pred_BIS'},
                           inplace=True)
Output_df_BIS_Eleveld.rename(columns={'pred_BIS_Eleveld': 'pred_BIS'},
                             inplace=True)
Output_df_MAP_Schnider.rename(columns={'pred_MAP_Schnider-Minto': 'pred_MAP'},
                              inplace=True)
Output_df_MAP_Marsh.rename(columns={'pred_MAP_Marsh-Minto': 'pred_MAP'},
                           inplace=True)
Output_df_MAP_Eleveld.rename(columns={'pred_MAP_Eleveld': 'pred_MAP'},
                             inplace=True)


# Patients_test.to_csv("./Patients_test.csv")
# %% Analyse results

result_df = pd.DataFrame()
print("-------------------Schnider-Minto-------------------")
print("-------------------BIS-------------------")
_, _, df_bis = compute_metrics(Output_df_BIS_Schnider)
df_bis.rename(columns={'MDPE': 'MDPE_BIS',
                       'MDAPE': 'MDAPE_BIS',
                       'RMSE': 'RMSE_BIS'}, inplace=True)
_, _, df_map = compute_metrics(Output_df_MAP_Schnider)
df_map.rename(columns={'MDPE': 'MDPE_MAP',
                       'MDAPE': 'MDAPE_MAP',
                       'RMSE': 'RMSE_MAP'}, inplace=True)
df = pd.concat([pd.DataFrame({'model': 'Schnider-Minto'}, index=[0]), df_bis, df_map], axis=1)
result_df = pd.concat([result_df, df], axis=0)

print("-------------------Marsh-Minto-------------------")
print("-------------------BIS-------------------")
_, _, df_bis = compute_metrics(Output_df_BIS_Marsh)
df_bis.rename(columns={'MDPE': 'MDPE_BIS',
                       'MDAPE': 'MDAPE_BIS',
                       'RMSE': 'RMSE_BIS'}, inplace=True)
_, _, df_map = compute_metrics(Output_df_MAP_Marsh)
df_map.rename(columns={'MDPE': 'MDPE_MAP',
                       'MDAPE': 'MDAPE_MAP',
                       'RMSE': 'RMSE_MAP'}, inplace=True)
df = pd.concat([pd.DataFrame({'model': 'Marsh-Minto'}, index=[0]), df_bis, df_map], axis=1)
result_df = pd.concat([result_df, df], axis=0)

print("-------------------Eleveld-------------------")
print("-------------------BIS-------------------")
_, _, df_bis = compute_metrics(Output_df_BIS_Eleveld)
df_bis.rename(columns={'MDPE': 'MDPE_BIS',
                       'MDAPE': 'MDAPE_BIS',
                       'RMSE': 'RMSE_BIS'}, inplace=True)
_, _, df_map = compute_metrics(Output_df_MAP_Eleveld)
df_map.rename(columns={'MDPE': 'MDPE_MAP',
                       'MDAPE': 'MDAPE_MAP',
                       'RMSE': 'RMSE_MAP'}, inplace=True)
df = pd.concat([pd.DataFrame({'model': 'Eleveld'}, index=[0]), df_bis, df_map], axis=1)
result_df = pd.concat([result_df, df], axis=0)

print('\n')
styler = result_df.style
styler.hide(axis='index')
# styler.format(precision=2)
print(styler.to_latex())

# %%
