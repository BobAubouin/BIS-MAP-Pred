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



import os
import csv
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from  tensorflow.keras.optimizers import Adam
from  tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
# parameters
timepoints = 60
lnode = 8
fnode = 16
batch_size = 64
cache_path = "cache.var"
cache_path_test = "cache_perso.var"
output_dir = "."
weight_path = output_dir + "/weight matrix.hdf5"

Data = pd.read_csv("data.csv", index_col=0)

train_p, train_r, train_c, train_y, val_p, val_r, val_c, val_y, test_p, test_r, test_c, test_y = pickle.load(open(cache_path, "rb"))
test_p, test_r, test_c, test_y = pickle.load(open(cache_path_test, "rb"))


train_val_c = train_c + val_c

# convert data to numpy array

train_c = np.array(train_c)

val_c = np.array(val_c)

# normalize data
mean_c = np.mean(train_val_c, axis=0)
std_c = np.std(train_val_c, axis=0)


#%% load model
model = load_model(weight_path)

#%% test model
sum_err = 0
cnt_err = 0
PE_sum = np.zeros(1)
error = np.zeros(1)
Ytrue = np.zeros(1)
Ypred = np.zeros(1)


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

    err = np.mean(np.abs(np.subtract(true_y, pred_y)))
    PE_sum = np.concatenate((PE_sum, (true_y - pred_y)/true_y), axis=0)
    error = np.concatenate((error, (true_y - pred_y)*100), axis=0)
    Ytrue = np.concatenate((Ytrue, true_y*100), axis=0)
    Ypred = np.concatenate((Ypred, (pred_y)*100), axis=0)
    true_y = true_y.tolist()
    pred_y = pred_y.tolist()

    case_len = len(pred_y)
    sum_err += err * case_len
    cnt_err += case_len

    print('{}\t{}'.format(id, err))
    try:
        fig,ax = plt.subplots(2)
        ax[0].plot(true_y)
        ax[0].plot(pred_y)
        ax[0].set_title('case='+str(id))
        
        ax[1].plot(case_p[:,0],label='propo')
        ax[1].plot(case_r[:,0],label='remi')
        plt.legend()
        plt.show()
    except:
        print('case '+ str(id) +' empty ')

PE_sum = PE_sum[1:]
error = error[1:]
Ytrue = Ytrue[1:]
Ypred = Ypred[1:]
print('RMSE:', np.sqrt(np.nansum(np.power(error,2))/len(error)))
print('MDAPE (%):', np.median(np.abs(PE_sum))*100)
print('MDPE (%):', np.median(PE_sum)*100)
print('std (%):', np.std(PE_sum)*100)

if cnt_err > 0:
    print("Mean test error: {}".format(sum_err / cnt_err))

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