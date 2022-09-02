#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:09:38 2022

@author: aubouinb
"""
import numpy as np
import pandas as pd
import bokeh
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import Range1d
from bokeh.io import export_svg
from matplotlib import pyplot as plt
from matplotlib import cm


def compute_metrics(data):

    """compute the metrics MDPE +/- SD, MDAPE +/- SD and RMSE +/- SD for a prediction signals 
    over a patient population.
    Inputs:     - data is a panda dataframe with the fiels case_id, true_BIS, true_MAP, pred_BIS, pred_MAP
    Output:     - 2 lists, for BIS and MAP with the structure: 
                                    [MDPE,SD_MDPE, MDAPE, SD_MDAPE, RMSE, SD_RMSE]
    print also the results to copy them in a latex table"""
    
    
    MDPE_BIS = 0
    MDAPE_BIS = 0
    MDPE_MAP = 0
    MDAPE_MAP = 0
    RMSE_BIS = 0
    RMSE_MAP = 0
    
    SD_MDPE_BIS = 0
    SD_MDAPE_BIS = 0
    SD_MDPE_MAP = 0
    SD_MDAPE_MAP = 0
    
    RMSE_BIS_list = []
    RMSE_MAP_list = []
    case_length_list = []
    rmse_max_bis = 0
    rmse_min_bis = 1000
    rmse_max_map = 0
    rmse_min_map = 1000
    for case in data['case_id'].unique():
        case_data = data[data['case_id']==case]
        case_length = len(case_data)
        PE_BIS = 100 * (case_data['true_BIS'].values - case_data['pred_BIS'].values)/case_data['true_BIS'].values
        PE_MAP = 100 * (case_data['true_MAP'].values - case_data['pred_MAP'].values)/case_data['true_MAP'].values
        
        MDPE_BIS += case_length * np.median(PE_BIS)
        MDPE_MAP += case_length *  np.median(PE_MAP)
        
        MDAPE_BIS += case_length *  np.median(np.abs(PE_BIS))
        # print(str(case) + ' :  ' + str(np.median(np.abs(PE_BIS))))
        MDAPE_MAP += case_length *  np.median(np.abs(PE_MAP))
        
        efficiency_case = 4*int(case_length/2)/(np.pi * case_length)
        
        SD_MDPE_BIS += case_length * np.var(PE_BIS) / efficiency_case 
        SD_MDPE_MAP += case_length * np.var(PE_MAP) / efficiency_case
        SD_MDAPE_BIS += case_length * np.var(np.abs(PE_BIS)) / efficiency_case
        SD_MDAPE_MAP += case_length * np.var(np.abs(PE_MAP)) / efficiency_case
        
        
        RMSE_bis = np.sqrt(np.mean((case_data['true_BIS'].values - case_data['pred_BIS'].values)**2))
        RMSE_map = np.sqrt(np.mean((case_data['true_MAP'].values - case_data['pred_MAP'].values)**2))
        
        RMSE_BIS += case_length * RMSE_bis
        RMSE_MAP += case_length * RMSE_map
        
        RMSE_BIS_list.append(RMSE_bis)
        RMSE_MAP_list.append(RMSE_map)
        case_length_list.append(case_length)
        
        if RMSE_bis>rmse_max_bis:
            case_rmse_max_bis = case
            rmse_max_bis = RMSE_bis
        if RMSE_bis<rmse_min_bis:
            case_rmse_min_bis = case
            rmse_min_bis = RMSE_bis
        if RMSE_map>rmse_max_map:
            case_rmse_max_map = case
            rmse_max_map = RMSE_map
        if RMSE_map<rmse_min_map:
            case_rmse_min_map = case
            rmse_min_map = RMSE_map
    sample_nb = len(data)
    
    MDPE_BIS /= sample_nb
    MDPE_MAP /= sample_nb
    MDAPE_BIS /= sample_nb
    MDAPE_MAP /= sample_nb
    RMSE_BIS /= sample_nb
    RMSE_MAP /= sample_nb


    
    SD_MDPE_BIS = np.sqrt(SD_MDPE_BIS / sample_nb)
    SD_MDPE_MAP = np.sqrt(SD_MDPE_MAP / sample_nb)
    SD_MDAPE_BIS = np.sqrt(SD_MDAPE_BIS / sample_nb)
    SD_MDAPE_MAP = np.sqrt(SD_MDAPE_MAP / sample_nb)
    

    SD_RMSE_BIS = np.sqrt(np.sum([(RMSE_BIS_list[i] - RMSE_BIS)**2 * case_length_list[i] for i in range(len(RMSE_BIS_list)) ]) / sample_nb)
    SD_RMSE_MAP = np.sqrt(np.sum([(RMSE_MAP_list[i] - RMSE_MAP)**2 * case_length_list[i] for i in range(len(RMSE_MAP_list)) ]) / sample_nb)
    
    
    print("                 ______   BIS results   ______")
    print( "     MDPE \t & \t MDAPE \t & \t RMSE       ")

    print( "$" + str(round(MDPE_BIS,1)) + " \pm " + str(round(SD_MDPE_BIS,1) ) 
          + "$ & $" + str(round(MDAPE_BIS,1)) + " \pm " + str(round(SD_MDAPE_BIS,1)) 
          + "$ & $" + str(round(RMSE_BIS,1)) + " \pm " + str(round(SD_RMSE_BIS,1))+ "$")

    print("\n               ______   MAP results   ______")
    print( "     MDPE      &       MDAPE      &       RMSE       ")

    print("$" + str(round(MDPE_MAP,1)) + " \pm " + str(round(SD_MDPE_MAP,1)) 
          + "$ & $" + str(round(MDAPE_MAP,1)) + " \pm " + str(round(SD_MDAPE_MAP,1)) 
          + "$ & $" + str(round(RMSE_MAP,1)) + " \pm " + str(round(SD_RMSE_MAP,1))+ "$")

    
    return case_rmse_max_bis, case_rmse_min_bis, case_rmse_max_map, case_rmse_min_map



def plot_results(data, data_train = pd.DataFrame()):
    """plot the result of the prediction with bokeh module"""
    y_true_test = data["true_BIS"].values
    y_pred_test = data["pred_BIS"].values
    if not data_train.empty:
        train = True
        y_true_train = data_train["true_BIS"].values
        y_pred_train = data_train["pred_BIS"].values
    else:
        train = False
                
    fig1 = figure(plot_width=900, plot_height=450, title = "Data (blue) Vs Fitted data (red)")
    fig1.line(range(0,len(y_true_test)), y_true_test, line_color='navy', legend_label="y")
    fig1.line(range(0,len(y_pred_test)), y_pred_test, line_color='red', legend_label="y predicted")
    fig1.xaxis.axis_label = "time [s]"
    fig1.yaxis.axis_label = "y"

    fig2 = figure(plot_width=900, plot_height=450, title = "Fitting error")
    fig2.line(range(0,len(y_true_test)), y_true_test-y_pred_test, line_color='navy', legend_label="y")
    fig2.xaxis.axis_label = "time [s]"
    fig2.yaxis.axis_label = "y"


    fig3 = figure(plot_width=900, plot_height=450, title="BIS")
    fig3.circle(y_true_test, y_pred_test, legend_label='y',size=2, color="navy", alpha=0.1)
    if train:
        fig3.circle(y_true_train, y_pred_train, legend_label='train',size=2, color="green", alpha=0.1)
        
    fig3.line(np.array([y_true_test.min(), y_true_test.max()]), np.array([y_true_test.min(), y_true_test.max()]),
             line_color="red")
    fig3.legend.location = "top_left"
    fig3.xaxis.axis_label = "y measured"
    fig3.yaxis.axis_label = "y predicted"
    fig3.x_range = Range1d(y_true_test.min(), y_true_test.max())
    fig3.y_range = Range1d(y_true_test.min(), y_true_test.max())
    
    
    fig4 = figure(plot_width=900, plot_height=450, title="Trained fit")
    if train:
        fig4 = figure(plot_width=900, plot_height=450, title = "Data (blue) Vs Fitted data (red)")
        fig4.line(range(0,len(y_true_train)), y_true_train, line_color='navy', legend_label="y traint rue")
        fig4.line(range(0,len(y_pred_train)), y_pred_train, line_color='red', legend_label="y train predicted")
        fig4.xaxis.axis_label = "time [s]"
        fig4.yaxis.axis_label = "y"
    
    layout = row(column(fig1,fig2),column(fig3, fig4))
    show(layout)
    
    y_true_test = data["true_MAP"].values
    y_pred_test = data["pred_MAP"].values
    if train:
        y_true_train = data_train["true_MAP"].values
        y_pred_train = data_train["pred_MAP"].values


    fig1 = figure(plot_width=900, plot_height=450, title = "Data (blue) Vs Fitted data (red)")
    fig1.line(range(0,len(y_true_test)), y_true_test, line_color='navy', legend_label="y")
    fig1.line(range(0,len(y_pred_test)), y_pred_test, line_color='red', legend_label="y predicted")
    fig1.xaxis.axis_label = "time [s]"
    fig1.yaxis.axis_label = "y"

    fig2 = figure(plot_width=900, plot_height=450, title = "Fitting error")
    fig2.line(range(0,len(y_true_test)), y_true_test-y_pred_test, line_color='navy', legend_label="y")
    fig2.xaxis.axis_label = "time [s]"
    fig2.yaxis.axis_label = "y"

    
    fig3 = figure(plot_width=900, plot_height=450, title="MAP")
    fig3.circle(y_true_test, y_pred_test, legend_label='y',size=2, color="navy", alpha=0.1)
    if train:
        fig3.circle(y_true_train, y_pred_train, legend_label='train',size=2, color="green", alpha=0.1)
        
    fig3.line(np.array([y_true_test.min(), y_true_test.max()]), np.array([y_true_test.min(), y_true_test.max()]),
             line_color="red")
    fig3.legend.location = "top_left"
    fig3.xaxis.axis_label = "y measured"
    fig3.yaxis.axis_label = "y predicted"
    fig3.x_range = Range1d(y_true_test.min(), y_true_test.max())
    fig3.y_range = Range1d(y_true_test.min(), y_true_test.max())
    
    fig4 = figure(plot_width=900, plot_height=450, title="Trained fit")
    if train:
        fig4 = figure(plot_width=900, plot_height=450, title = "Data (blue) Vs Fitted data (red)")
        fig4.line(range(0,len(y_true_train)), y_true_train, line_color='navy', legend_label="y traint rue")
        fig4.line(range(0,len(y_pred_train)), y_pred_train, line_color='red', legend_label="y train predicted")
        fig4.xaxis.axis_label = "time [s]"
        fig4.yaxis.axis_label = "MAP (mmHg)"
    
    layout = row(column(fig1,fig2),column(fig3, fig4))
    show(layout)
    

def plot_one_fig(case_full, case_pred, output, columns_pred, columns_pred_full):
    
    color = list(bokeh.palettes.brewer['Dark2'][max(3,len(columns_pred)+1+len(columns_pred_full))])
    
    fig = figure(plot_width=900, plot_height=450, title = output + " for best case")
    i=0
    fig.line(case_full['Time'].values, case_full[output].values, line_color = color[i], legend_label="True " + output, line_width=2)
    
    for name in columns_pred:
        i+=1
        fig.line(case_pred['Time'].values, case_pred[name], legend_label=name[9:], line_color = color[i], line_width=2)
        
    for name in columns_pred_full:
        i+=1
        fig.line(case_full['Time'].values, case_full[name], legend_label=name[9:], line_color = color[i], line_width=2)
        
    fig.xaxis.axis_label = "time (s)"
    fig.yaxis.axis_label = output + '(%)'*(output=='BIS') + '(mmHg)'*(output=='MAP')
    return fig

def plot_case(Patient_pred, Patient_full, caseid_min_bis, case_min_map, caseid_max_bis, caseid_max_map):
    #case minimum
    case_pred = Patient_pred[Patient_pred['caseid']==caseid_min_bis]
    case_full = Patient_full[Patient_full['caseid']==caseid_min_bis]

   
    columns_pred_BIS = [name for name in case_pred.columns if name[:8]=='pred_BIS']
    columns_pred_MAP = [name for name in case_pred.columns if name[:8]=='pred_MAP']
    
    columns_pred_BIS_full = [name for name in case_full.columns if name[:8]=='pred_BIS']
    columns_pred_MAP_full = [name for name in case_full.columns if name[:8]=='pred_MAP']
    
    fig1 = plot_one_fig(case_full, case_pred, 'BIS', columns_pred_BIS, columns_pred_BIS_full)
    
    case_pred = Patient_pred[Patient_pred['caseid']==case_min_map]
    case_full = Patient_full[Patient_full['caseid']==case_min_map]

    fig2 = plot_one_fig(case_full, case_pred, 'MAP', columns_pred_MAP, columns_pred_MAP_full)

    case_pred = Patient_pred[Patient_pred['caseid']==caseid_max_bis]
    case_full = Patient_full[Patient_full['caseid']==caseid_max_bis]

    
    fig3 = plot_one_fig(case_full, case_pred, 'BIS', columns_pred_BIS, columns_pred_BIS_full)

    case_pred = Patient_pred[Patient_pred['caseid']==caseid_max_map]
    case_full = Patient_full[Patient_full['caseid']==caseid_max_map]

    fig4 = plot_one_fig(case_full, case_pred, 'MAP', columns_pred_MAP, columns_pred_MAP_full)
    
    layout = row(column(fig1,fig2), column(fig3,fig4))
    show(layout)
    fig1.output_backend="svg"
    export_svg(fig1, filename="fig1.svg")
    fig2.output_backend="svg"
    export_svg(fig2, filename="fig2.svg")
    fig3.output_backend="svg"
    export_svg(fig3, filename="fig3.svg")
    fig4.output_backend="svg"
    export_svg(fig4, filename="fig4.svg")



def plot_surface(reg, scaler):
    """Plot the 3D surface of the BIS related to Propofol and Remifentanil effect site concentration"""
    age = 35
    weight = 70
    height = 170
    sex = 1
    bmi = weight / (height/100)**2
    if sex == 1: # homme
        lbm = 1.1 * weight - 128 * (weight / height) ** 2
    else : #femme
        lbm = 1.07 * weight - 148 * (weight / height) ** 2
    MAP_base=100
    
    cer = np.linspace(1, 8, 50)
    cep = np.linspace(1, 8, 50)
    output = np.zeros((50,50))
    X_p = np.zeros((50,50))
    Y_r = np.zeros((50,50))
    for i in range(len(cep)):
        for j in range(len(cer)):
            input = np.array([age, sex, height, weight, bmi, lbm, MAP_base, cep[i], cer[j],cep[i], cer[j]]).reshape(1, -1) 
            input = scaler.transform(input)
            output[i,j] = reg.predict(input)
            X_p[i,j] = cep[i]
            Y_r[i,j] = cer[j]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X_p, Y_r, output, cmap=cm.jet, linewidth=0.1)
    ax.set_xlabel('Remifentanil')
    ax.set_ylabel('Propofol')
    ax.set_zlabel('BIS')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


























    
    
    