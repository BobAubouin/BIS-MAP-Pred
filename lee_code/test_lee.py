#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:56:47 2022

@author: aubouinb

Use the code of Hyung-Chul Lee to test it on our own cases

"""
"""
Predicting bispectral index from propofol, remifentanil infusion history.

This program demonstrates how to build an estimation model of drug effect using deep learning techniques.
It runs on python 3.5.2 with keras 1.2.2 and tensorflow 1.2.1.

Developed by Hyung-Chul Lee (lucid80@gmail.com) in Aug 2016 and
licensed to the public domain for academic advancement.
"""



import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model


#%% functions
def compute_metrics(data):

    """compute the metrics MDPE +/- SD, MDAPE +/- SD and RMSE +/- SD for a prediction signals 
    over a patient population.
    Inputs:     - data is a panda dataframe with the fiels case_id, true_*, pred_*
    Output:     -  caseid of the best and worst RMSE
    print also the results to copy them in a latex table"""
    
    
    MDPE = 0
    MDAPE = 0
    RMSE = 0
    
    SD_MDPE = 0
    SD_MDAPE = 0
    
    RMSE_BIS_list = []
    RMSE_list = []
    case_length_list = []
    rmse_max = 0
    rmse_min = 1e10
    
    for col_name in data.columns:
        if 'true' in col_name:
            true_col = col_name
        elif 'pred' in col_name:
            pred_col = col_name
    print(pred_col)
    print(true_col)
    
    for case in data['case_id'].unique():
        case_data = data[data['case_id']==case]
        case_length = len(case_data)
        PE = 100 * (case_data[true_col].values - case_data[pred_col].values)/case_data[true_col].values

        MDPE += case_length * np.median(PE)
        
        MDAPE += case_length *  np.median(np.abs(PE))

        efficiency_case = 4*int(case_length/2)/(np.pi * case_length)
        
        SD_MDPE += case_length * np.var(PE) / efficiency_case 
        SD_MDAPE += case_length * np.var(np.abs(PE)) / efficiency_case

        
        rmse = np.sqrt(np.mean((case_data[true_col].values - case_data[pred_col].values)**2))
     
        RMSE += case_length * rmse
        
        RMSE_list.append(rmse)
        case_length_list.append(case_length)
        
        if rmse>rmse_max:
            case_rmse_max = case
            rmse_max = rmse
        if rmse<rmse_min:
            case_rmse_min = case
            rmse_min = rmse
    sample_nb = len(data)
    
    MDPE /= sample_nb
    MDAPE /= sample_nb
    RMSE /= sample_nb


    
    SD_MDPE = np.sqrt(SD_MDPE / sample_nb)
    SD_MDAPE = np.sqrt(SD_MDAPE / sample_nb)
    

    SD_RMSE = np.sqrt(np.sum([(RMSE_list[i] - RMSE)**2 * case_length_list[i] for i in range(len(RMSE_list)) ]) / sample_nb)

    
    col_name = pred_col[5:]
    print("                 ______   "+col_name+" results   ______")
    print( "     MDPE \t & \t MDAPE \t & \t RMSE       ")

    print( "$" + str(round(MDPE,1)) + " \pm " + str(round(SD_MDPE,1) ) 
          + "$ & $" + str(round(MDAPE,1)) + " \pm " + str(round(SD_MDAPE,1)) 
          + "$ & $" + str(round(RMSE,1)) + " \pm " + str(round(SD_RMSE,1))+ "$")


    return case_rmse_max, case_rmse_min


#%% parameters
timepoints = 60
lnode = 8
fnode = 16
batch_size = 64
cache_path = "cache.var"
cache_path_test = "cache_perso.var"
output_dir = "."
weight_path = output_dir + "/weights.hdf5"

Data = pd.read_csv("data.csv", index_col=0)

test_p, test_r, test_c, test_y = pickle.load(open(cache_path_test, "rb"))




# normalize data from the training dataset of Lee et al.
mean_c = [ 56.50928621,   0.49630326,  61.13655517, 162.0409402 ]
std_c = [ 56.50928621,   0.49630326,  61.13655517, 162.0409402 ]


#%% load model
model = load_model(weight_path)

#%% test model
sample_nb = 0

Ytrue = np.zeros(1)
Ypred = np.zeros(1)

Output_df = pd.DataFrame(columns=['case_id','true_BIS','pred_BIS'])


print('id\ttesting_err')
for id in test_p.keys():
    
    case_p = np.array(test_p[id])
    case_r = np.array(test_r[id])
    case_c = np.array(test_c[id])
    true_y = np.array(test_y[id])
    true_y = np.array(true_y)
    
    try:
        case_p = case_p.reshape(case_p.shape[0], case_p.shape[1], 1)
        case_r = case_r.reshape(case_r.shape[0], case_r.shape[1], 1)
    except:
        print('problem case:', id)
        continue
    case_c = (case_c - mean_c) / std_c
    case_p /= 120.0
    case_r /= 120.0

    case_x = [case_p, case_r, case_c]
    pred_y = model.predict(case_x)

    true_y = true_y.T
    pred_y = pred_y[:, 0].T
    case_len = len(pred_y)
    
    Output_df_temp = pd.DataFrame(columns=['case_id','true_BIS','pred_BIS'])
    Output_df_temp['case_id'] = np.ones((case_len)) * float(id)
    Output_df_temp['true_BIS'] = true_y*100
    Output_df_temp['pred_MAP']  = pred_y*100
    Output_df = pd.concat([Output_df, Output_df_temp], ignore_index=True)
    
compute_metrics(Output_df)

#%%plot function
from bokeh.plotting import figure, show
from bokeh.palettes import Viridis256 as colors  # just make sure to import a palette that centers on white (-ish)
from bokeh.layouts import gridplot, row, column
from bokeh.models import ColorBar, LinearColorMapper, Range1d

def plot_results(y, y_predicted, name, step=1):
    fig1 = figure(plot_width=900, plot_height=450, title = "Data (blue) Vs Fitted data (red)")
    fig1.line(range(0,len(y)*step,step), y, line_color='navy', legend_label="y")
    fig1.line(range(0,len(y_predicted)*step,step), y_predicted, line_color='red', legend_label="y predicted")
    fig1.xaxis.axis_label = "time [s]"
    fig1.yaxis.axis_label = "y"

    fig2 = figure(plot_width=900, plot_height=450, title = "Fitting error")
    fig2.line(range(0,len(y)*step,step), y-y_predicted, line_color='navy', legend_label="y")
    fig2.xaxis.axis_label = "time [s]"
    fig2.yaxis.axis_label = "y"
    
    plt.figure()
    error = y-y_predicted
    plt.hist(error, bins=21)

    y_filt = y
    mytitle = "Fitting test dataset " + name
    fig3 = figure(plot_width=900, plot_height=450, title=mytitle)
    fig3.circle(y_filt, y_predicted, legend_label='y',size=2, color="navy", alpha=0.1)
    fig3.line(np.array([y_filt.min(), y_filt.max()]), np.array([y_filt.min(), y_filt.max()]),
             line_color="red")
    fig3.legend.location = "top_left"
    fig3.xaxis.axis_label = "y measured"
    fig3.yaxis.axis_label = "y predicted"
    fig3.x_range = Range1d(y_filt.min(), y_filt.max())
    fig3.y_range = Range1d(y_filt.min(), y_filt.max())
    
    layout = row(column(fig1,fig2),fig3)
    show(layout)
    
plot_results(Ytrue, Ypred,'BIS')