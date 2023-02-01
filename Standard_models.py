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
import model

# Model to test

model_name = 'Eleveld'
# model_name = 'Schnider - Minto'
# model_name = 'Marsh - Minto'


# %% load data
Patients_test = pd.read_csv("./Patients_test.csv", index_col=0)

# %% Perform simulation

Output_df = pd.DataFrame(columns=['case_id', 'true_BIS', 'pred_BIS', 'true_MAP', 'pred_MAP', 'full_BIS', 'full_MAP'])

i = 0
if 'pred_BIS_' + model_name not in Patients_test.columns:
    Patients_test.insert(len(Patients_test.columns), 'pred_BIS_'+model_name, 0)

for caseid in Patients_test['caseid'].unique():
    print(caseid)

    Patient_df = Patients_test[Patients_test['caseid'] == caseid].copy().reset_index()
    Output_df_temp = pd.DataFrame(columns=['case_id', 'true_BIS', 'pred_BIS',
                                  'true_MAP', 'pred_MAP', 'full_BIS', 'full_MAP'])

    Patient_df['MAP'] = Patient_df['MAP'].fillna(0)

    # create model
    MAP_base_case = Patient_df['MAP_base_case'][0]
    age = int(Patient_df['age'][0])
    height = int(Patient_df['height'][0])
    weight = int(Patient_df['weight'][0])
    sex = int(Patient_df['sex'][0])

    v1_p, Ap = model.PropoModel(model_name, age, sex, weight, height)
    v1_r, Ar = model.RemiModel(model_name, age, sex, weight, height)

    Bp = np.zeros((6, 1))
    Bp[0, 0] = 1 / v1_p
    Br = np.zeros((5, 1))
    Br[0, 0] = 1 / v1_r

    Adp, Bdp = model.discretize(Ap, Bp, 1)
    Adr, Bdr = model.discretize(Ar, Br, 1)

    Ncase = len(Patient_df['BIS'])
    Output_df_temp['true_BIS'] = Patient_df['BIS']
    Output_df_temp['true_MAP'] = Patient_df['MAP']
    Output_df_temp['full_BIS'] = Patient_df['full_BIS']
    Output_df_temp['full_MAP'] = Patient_df['full_MAP']
    Output_df_temp['case_id'] = np.ones((Ncase)) * caseid

    Output_df_temp['pred_BIS'] = np.zeros((Ncase))
    Output_df_temp['pred_MAP'] = np.zeros((Ncase))

    x_p = np.zeros((6, 1))
    x_r = np.zeros((5, 1))
    Output_df_temp.loc[0, 'pred_BIS'], Output_df_temp.loc[0, 'pred_MAP'] = model.surface_model(x_p, x_r, MAP_base_case)
    for j in range(Ncase-1):
        x_p = Adp @ x_p + Bdp * Patient_df['Propofol'][j]*20/3600
        x_r = Adr @ x_r + Bdr * Patient_df['Remifentanil'][j]*20/3600

        Bis, Map = model.surface_model(x_p, x_r, MAP_base_case)
        Output_df_temp.loc[j+1, 'pred_BIS'] = Bis
        Output_df_temp.loc[j+1, 'pred_MAP'] = Map

    Output_df = pd.concat([Output_df, Output_df_temp], ignore_index=True)

    # Output_df_temp['pred_BIS'] = Output_df_temp['pred_BIS'].diff()
    # Output_df_temp['true_BIS'] = Output_df_temp['true_BIS'].diff()
    # Output_df_temp.loc[0, 'true_BIS'] = 0
    # Output_df_temp.loc[0, 'pred_BIS'] = 0
    Patients_test.loc[Patients_test["caseid"] == caseid, 'pred_BIS_'+model_name] = Output_df_temp['pred_BIS'].values
    Patients_test.loc[Patients_test["caseid"] == caseid, 'pred_MAP_'+model_name] = Output_df_temp['pred_MAP'].values

    if i % 5 == 0:
        fig, ax = plt.subplots(2)
        ax[1].set_xlabel('caseid : ' + str(caseid))
        ax[0].plot(Patient_df['BIS'])
        ax[0].plot(Output_df_temp['pred_BIS'])
        ax[1].plot(Patient_df['MAP'])
        ax[1].plot(Output_df_temp['pred_MAP'])
        plt.pause(0.05)
    i += 1

Output_df_BIS = Output_df[Output_df['full_BIS'] == 0]
Output_df_BIS = Output_df_BIS[['pred_BIS', 'true_BIS', 'case_id']]
Output_df_MAP = Output_df[Output_df['full_MAP'] == 0]
Output_df_MAP = Output_df_MAP[['pred_MAP', 'true_MAP', 'case_id']]

Output_df_BIS = Output_df_BIS[Output_df_BIS['true_BIS'] != 0]
Output_df_MAP = Output_df_MAP[Output_df_MAP['true_MAP'] != 0]

# Patients_test.to_csv("./Patients_test.csv")
# %% Analyse results


compute_metrics(Output_df_BIS)
compute_metrics(Output_df_MAP)
# plot_results(Output_df)
