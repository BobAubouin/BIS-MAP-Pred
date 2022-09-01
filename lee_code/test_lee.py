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


# parameters
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

MDPE_BIS = 0
MDAPE_BIS = 0

SD_MDPE_BIS = 0
SD_MDAPE_BIS = 0
    
RMSE_list_BIS = []
case_length_list = []


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

    PE = 100*(true_y - pred_y)/true_y
    
    MDPE_BIS += case_len * np.median(PE)
    MDAPE_BIS += case_len *  np.median(np.abs(PE))
    print(str(id) + '\t' + str(np.median(np.abs(PE))))
    efficiency_case = 4*int(case_len/2)/(np.pi * case_len)
    
    SD_MDPE_BIS += case_len * np.var(PE) / efficiency_case
    SD_MDAPE_BIS += case_len * np.var(np.abs(PE)) / efficiency_case
    
    RMSE_BIS = np.sqrt(np.mean((100*(true_y - pred_y))**2))
    RMSE_list_BIS.append(RMSE_BIS)
    case_length_list.append(case_len)
    
    Ytrue = np.concatenate((Ytrue, true_y*100), axis=0)
    Ypred = np.concatenate((Ypred, (pred_y)*100), axis=0)
    true_y = true_y.tolist()
    pred_y = pred_y.tolist()

    
    sample_nb += case_len


Ytrue = Ytrue[1:]
Ypred = Ypred[1:]


MDPE_BIS /= sample_nb
MDAPE_BIS /= sample_nb
RMSE_BIS = np.sum([RMSE_list_BIS[i] * case_length_list[i] for i in range(len(RMSE_list_BIS))]) / sample_nb

SD_MDPE_BIS = np.sqrt(SD_MDPE_BIS / sample_nb)
SD_MDAPE_BIS = np.sqrt(SD_MDAPE_BIS / sample_nb)
SD_RMSE_BIS = np.sqrt(np.sum([(RMSE_list_BIS[i] - RMSE_BIS)**2 * case_length_list[i] for i in range(len(RMSE_list_BIS)) ]) / sample_nb)

print("                 ______   BIS results   ______")
print( "     MDPE      &       MDAPE      &       RMSE       ")

print( "$" + str(round(MDPE_BIS, 1)) + " \pm " + str(round(SD_MDPE_BIS, 1) ) 
      + "$ & $" + str(round(MDAPE_BIS, 1)) + " \pm " + str(round(SD_MDAPE_BIS, 1)) 
      + "$ & $" + str(round(RMSE_BIS, 1)) + " \pm " + str(round(SD_RMSE_BIS, 1))+ "$")

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