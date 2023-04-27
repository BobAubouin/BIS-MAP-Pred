#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 10:09:38 2022

@author: aubouinb
"""
import numpy as np
import pandas as pd
import torch
import bokeh
import matplotlib
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from bokeh.models import Range1d
from bokeh.io import export_svg
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def compute_metrics(data):
    """compute the metrics MDPE +/- SD, MDAPE +/- SD and RMSE +/- SD for a prediction signals 
    over a patient population.
    Inputs:     - data is a panda dataframe with the fiels caseid, true_*, pred_*
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
    rmse_max = -1
    rmse_min = 1e10

    for col_name in data.columns:
        if 'true' in col_name:
            true_col = col_name
        elif 'pred' in col_name:
            pred_col = col_name

    for case in data['caseid'].unique():
        case_data = data[data['caseid'] == case]
        case_length = len(case_data)
        PE = 100 * (case_data[true_col].values - case_data[pred_col].values)/case_data[true_col].values

        MDPE += case_length * np.median(PE)

        MDAPE += case_length * np.median(np.abs(PE))

        efficiency_case = 4*int(case_length/2)/(np.pi * case_length)

        SD_MDPE += case_length * np.var(PE) / efficiency_case
        SD_MDAPE += case_length * np.var(np.abs(PE)) / efficiency_case

        rmse = np.sqrt(np.mean((case_data[true_col].values - case_data[pred_col].values)**2))

        RMSE += case_length * rmse

        RMSE_list.append(rmse)
        case_length_list.append(case_length)

        if rmse > rmse_max:
            case_rmse_max = case
            rmse_max = rmse
        if rmse < rmse_min:
            case_rmse_min = case
            rmse_min = rmse
    sample_nb = len(data)

    MDPE /= sample_nb
    MDAPE /= sample_nb
    RMSE /= sample_nb

    SD_MDPE = np.sqrt(SD_MDPE / sample_nb)
    SD_MDAPE = np.sqrt(SD_MDAPE / sample_nb)

    SD_RMSE = np.sqrt(np.sum([(RMSE_list[i] - RMSE)**2 * case_length_list[i]
                      for i in range(len(RMSE_list))]) / sample_nb)

    col_name = pred_col[5:]
    print("                 ______   "+col_name+" results   ______")
    print(f'MDPE: {round(MDPE, 1)} +/- {round(SD_MDPE, 1)}')
    print(f'MDAPE: {round(MDAPE, 1)} +/- {round(SD_MDAPE, 1)}')
    print(f'RMSE: {round(RMSE, 1)} +/- {round(SD_RMSE, 1)}')

    df = pd.DataFrame({'MDPE': f"${round(MDPE, 1)} \pm {round(SD_MDPE, 1)}$",
                       'MDAPE': f"${round(MDAPE, 1)} \pm {round(SD_MDAPE, 1)}$",
                       'RMSE': f"${round(RMSE, 1)} \pm {round(SD_RMSE, 1)}$"},
                      index=[0])
    return case_rmse_max, case_rmse_min, df


def plot_results(data_BIS, data_MAP, data_train_BIS=None, data_train_MAP=None):
    """plot the result of the prediction with bokeh module."""
    # first the BIS plot
    # get the data of the column with 'true_BIS' and 'pred_BIS'
    y_true_test = data_BIS[[
        col_name for col_name in data_BIS.columns if col_name.startswith('true_BIS')]].values
    print(y_true_test)
    y_pred_test = data_BIS[[
        col_name for col_name in data_BIS.columns if col_name.startswith('pred_BIS')]].values

    if data_train_BIS is not None:
        train = True
        y_true_train = data_train_BIS[[
            col_name for col_name in data_train_BIS.columns if col_name.startswith('true_BIS')]].values
        y_pred_train = data_train_BIS[[
            col_name for col_name in data_train_BIS.columns if col_name.startswith('pred_BIS')]].values
    else:
        train = False

    fig1 = figure(width=900, height=450, title="Data (blue) Vs Fitted data (red)")
    fig1.line(np.arange(0, len(y_true_test)), y_true_test, line_color='navy', legend_label="y")
    fig1.line(np.arange(0, len(y_pred_test)), y_pred_test, line_color='red', legend_label="y predicted")
    fig1.xaxis.axis_label = "time [s]"
    fig1.yaxis.axis_label = "y"

    fig2 = figure(width=900, height=450, title="Fitting error")
    fig2.line(np.arange(0, len(y_true_test)), y_true_test-y_pred_test, line_color='navy', legend_label="y")
    fig2.xaxis.axis_label = "time [s]"
    fig2.yaxis.axis_label = "y"

    fig3 = figure(width=900, height=450, title="BIS")
    fig3.circle(y_true_test, y_pred_test, legend_label='y', size=2, color="navy", alpha=0.1)
    if train:
        fig3.circle(y_true_train, y_pred_train, legend_label='train', size=2, color="green", alpha=0.1)

    fig3.line(np.array([y_true_test.min(), y_true_test.max()]), np.array([y_true_test.min(), y_true_test.max()]),
              line_color="red")
    fig3.legend.location = "top_left"
    fig3.xaxis.axis_label = "y measured"
    fig3.yaxis.axis_label = "y predicted"
    fig3.x_range = Range1d(y_true_test.min(), y_true_test.max())
    fig3.y_range = Range1d(y_true_test.min(), y_true_test.max())

    fig4 = figure(width=900, height=450, title="Trained fit")
    if train:
        fig4 = figure(width=900, height=450, title="Data (blue) Vs Fitted data (red)")
        fig4.line(np.arange(0, len(y_true_train)), y_true_train, line_color='navy', legend_label="y traint rue")
        fig4.line(np.arange(0, len(y_pred_train)), y_pred_train, line_color='red', legend_label="y train predicted")
        fig4.xaxis.axis_label = "time [s]"
        fig4.yaxis.axis_label = "y"

    layout = row(column(fig1, fig2), column(fig3, fig4))
    show(layout)

    # then the MAP plot
    y_true_test = data_MAP[[
        col_name for col_name in data_MAP.columns if col_name.startswith('true_MAP')]].values
    y_pred_test = data_MAP[[
        col_name for col_name in data_MAP.columns if col_name.startswith('pred_MAP')]].values
    if train:
        y_true_train = data_train_MAP[[
            col_name for col_name in data_train_MAP.columns if col_name.startswith('true_BIS')]].values
        y_pred_train = data_train_MAP[[
            col_name for col_name in data_train_MAP.columns if col_name.startswith('pred_BIS')]].values

    fig1 = figure(width=900, height=450, title="Data (blue) Vs Fitted data (red)")
    fig1.line(np.arange(0, len(y_true_test)), y_true_test, line_color='navy', legend_label="y")
    fig1.line(np.arange(0, len(y_pred_test)), y_pred_test, line_color='red', legend_label="y predicted")
    fig1.xaxis.axis_label = "time [s]"
    fig1.yaxis.axis_label = "y"

    fig2 = figure(width=900, height=450, title="Fitting error")
    fig2.line(np.arange(0, len(y_true_test)), y_true_test-y_pred_test, line_color='navy', legend_label="y")
    fig2.xaxis.axis_label = "time [s]"
    fig2.yaxis.axis_label = "y"

    fig3 = figure(width=900, height=450, title="MAP")
    fig3.circle(y_true_test, y_pred_test, legend_label='y', size=2, color="navy", alpha=0.1)
    if train:
        fig3.circle(y_true_train, y_pred_train, legend_label='train', size=2, color="green", alpha=0.1)

    fig3.line(np.array([y_true_test.min(), y_true_test.max()]), np.array([y_true_test.min(), y_true_test.max()]),
              line_color="red")
    fig3.legend.location = "top_left"
    fig3.xaxis.axis_label = "y measured"
    fig3.yaxis.axis_label = "y predicted"
    fig3.x_range = Range1d(y_true_test.min(), y_true_test.max())
    fig3.y_range = Range1d(y_true_test.min(), y_true_test.max())

    fig4 = figure(width=900, height=450, title="Trained fit")
    if train:
        fig4 = figure(width=900, height=450, title="Data (blue) Vs Fitted data (red)")
        fig4.line(np.arange(0, len(y_true_train)), y_true_train, line_color='navy', legend_label="y traint rue")
        fig4.line(np.arange(0, len(y_pred_train)), y_pred_train, line_color='red', legend_label="y train predicted")
        fig4.xaxis.axis_label = "time [s]"
        fig4.yaxis.axis_label = "MAP (mmHg)"

    layout = row(column(fig1, fig2), column(fig3, fig4))
    show(layout)


def plot_one_fig(case_full, case_pred, output, columns_pred, columns_pred_full):

    color = list(bokeh.palettes.Category10[max(3, len(columns_pred)+1+len(columns_pred_full))])

    fig = figure(width=900, height=250)
    i = 0
    fig.line(case_full['Time'].values/60, case_full[output].values,
             line_color=color[i], legend_label="True " + output, line_width=3)

    for name in columns_pred_full:
        i += 1
        fig.line(case_full['Time'].values/60, case_full[name], legend_label=name[9:], line_color=color[i], line_width=2)

    for name in columns_pred:
        i += 1
        fig.line(case_pred['Time'].values/60, case_pred[name], legend_label=name[9:], line_color=color[i], line_width=2)

    fig.xaxis.axis_label = "time (min)"
    fig.yaxis.axis_label = output + '(%)'*(output == 'BIS') + '(mmHg)'*(output == 'MAP')
    return fig


def plot_case(Patient_pred_BIS, Patient_pred_MAP, Patient_full, caseid_min_bis, case_min_map, caseid_max_bis, caseid_max_map):
    # case minimum
    case_pred = Patient_pred_BIS[Patient_pred_BIS['caseid'] == caseid_min_bis]
    case_full = Patient_full[Patient_full['caseid'] == caseid_min_bis]

    columns_pred_BIS = [name for name in case_pred.columns if name[:8] == 'pred_BIS']
    columns_pred_BIS_full = [name for name in case_full.columns if name[:8] == 'pred_BIS']

    fig1 = plot_one_fig(case_full, case_pred, 'BIS', columns_pred_BIS, columns_pred_BIS_full)
    fig1.title.text = 'BIS best case'

    case_pred = Patient_pred_MAP[Patient_pred_MAP['caseid'] == case_min_map]
    case_full = Patient_full[Patient_full['caseid'] == case_min_map]

    columns_pred_MAP = [name for name in case_pred.columns if name[:8] == 'pred_MAP']
    columns_pred_MAP_full = [name for name in case_full.columns if name[:8] == 'pred_MAP']

    fig2 = plot_one_fig(case_full, case_pred, 'MAP', columns_pred_MAP, columns_pred_MAP_full)
    fig2.title.text = 'MAP best case'

    case_pred = Patient_pred_BIS[Patient_pred_BIS['caseid'] == caseid_max_bis]
    case_full = Patient_full[Patient_full['caseid'] == caseid_max_bis]

    fig3 = plot_one_fig(case_full, case_pred, 'BIS', columns_pred_BIS, columns_pred_BIS_full)
    fig3.title.text = 'BIS worst case'

    case_pred = Patient_pred_MAP[Patient_pred_MAP['caseid'] == caseid_max_map]
    case_full = Patient_full[Patient_full['caseid'] == caseid_max_map]

    fig4 = plot_one_fig(case_full, case_pred, 'MAP', columns_pred_MAP, columns_pred_MAP_full)
    fig4.title.text = 'MAP worst case'

    layout = row(column(fig1, fig2), column(fig3, fig4))
    show(layout)
    fig1.output_backend = "svg"
    export_svg(fig1, filename="fig1.svg")
    fig2.output_backend = "svg"
    export_svg(fig2, filename="fig2.svg")
    fig3.output_backend = "svg"
    export_svg(fig3, filename="fig3.svg")
    fig4.output_backend = "svg"
    export_svg(fig4, filename="fig4.svg")


def plot_surface(reg, scaler, feature, pca=None):
    """Plot the 3D surface of the BIS related to Propofol and Remifentanil effect site concentration"""
    age = 35
    weight = 70
    height = 170
    sex = 1
    bmi = weight / (height/100)**2
    if sex == 1:  # homme
        lbm = 1.1 * weight - 128 * (weight / height) ** 2
    else:  # femme
        lbm = 1.07 * weight - 148 * (weight / height) ** 2
    MAP_base = 100
    HR_base = 80
    cer = np.linspace(1, 6, 50)
    cep = np.linspace(1, 6, 50)
    output = np.zeros((50, 50))
    X_p = np.zeros((50, 50))
    Y_r = np.zeros((50, 50))
    for i in range(len(cep)):
        for j in range(len(cer)):
            if feature == '-Cplasma' or feature == '-Cmap' or feature == '-Cbis':
                input = np.array([age, sex, height, weight, bmi, lbm, HR_base,
                                 cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            if feature == '-hr':
                input = np.array([age, sex, height, weight, bmi, lbm, cep[i], cer[j],
                                 cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            if feature == 'All':
                input = np.array([age, sex, height, weight, bmi, lbm, HR_base, cep[i],
                                 cer[j], cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            if feature == '-bmi':
                input = np.array([age, sex, height, weight, lbm, HR_base, cep[i],
                                 cer[j], cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            if feature == '-lbm':
                input = np.array([age, sex, height, weight, bmi, HR_base, cep[i],
                                 cer[j], cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            if feature == '-map':
                input = np.array([age, sex, height, weight, bmi, lbm, HR_base, cep[i],
                                 cer[j], cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            input = PolynomialFeatures(degree=1, include_bias=False).fit_transform(input)
            input = scaler.transform(input)
            # input = pca.transform(input)
            # input = input[:,:20]
            output[i, j] = reg.predict(input)
            X_p[i, j] = cep[i]
            Y_r[i, j] = cer[j]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X_p, Y_r, output, cmap=cm.jet, linewidth=0.1)
    ax.set_xlabel('$C_{Remifentanil}$')
    ax.set_ylabel('$C_{Propofol}$')
    ax.set_zlabel('BIS')
    fig.colorbar(surf, shrink=0.5, aspect=20)
    ax.view_init(30, 17)
    savepath = "/home/aubouinb/ownCloud/Anesthesie/Science/Bob/Article/Images/surf1.pdf"
    fig.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()


def plot_surface_tensor(reg, scaler, feature, pca=None):
    """Plot the 3D surface of the BIS related to Propofol and Remifentanil effect site concentration"""
    age = 35
    weight = 70
    height = 170
    sex = 1
    bmi = weight / (height/100)**2
    if sex == 1:  # homme
        lbm = 1.1 * weight - 128 * (weight / height) ** 2
    else:  # femme
        lbm = 1.07 * weight - 148 * (weight / height) ** 2
    MAP_base = 100
    HR_base = 80
    cer = np.linspace(1, 6, 50)
    cep = np.linspace(1, 6, 50)
    output = np.zeros((50, 50))
    X_p = np.zeros((50, 50))
    Y_r = np.zeros((50, 50))
    for i in range(len(cep)):
        for j in range(len(cer)):
            if feature == '-Cplasma' or feature == '-Cmap' or feature == '-Cbis':
                input = np.array([age, sex, height, weight, bmi, lbm, HR_base,
                                 cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            if feature == '-hr':
                input = np.array([age, sex, height, weight, bmi, lbm, cep[i], cer[j],
                                 cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            if feature == 'All':
                input = np.array([age, sex, height, weight, bmi, lbm, HR_base, cep[i],
                                 cer[j], cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            if feature == '-bmi':
                input = np.array([age, sex, height, weight, lbm, HR_base, cep[i],
                                 cer[j], cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            if feature == '-lbm':
                input = np.array([age, sex, height, weight, bmi, HR_base, cep[i],
                                 cer[j], cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            if feature == '-map':
                input = np.array([age, sex, height, weight, bmi, lbm, HR_base, cep[i],
                                 cer[j], cep[i], cer[j], cep[i], cer[j]]).reshape(1, -1)
            input = PolynomialFeatures(degree=1, include_bias=False).fit_transform(input)
            input = scaler.transform(input)
            # input = pca.transform(input)
            # input = input[:,:20]
            input = torch.tensor(input.astype(np.float32))
            output[i, j] = reg.predict(input)
            X_p[i, j] = cep[i]
            Y_r[i, j] = cer[j]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X_p, Y_r, (output+1)*50, cmap=cm.jet, linewidth=0.1)
    ax.set_xlabel('$C_{Remifentanil}$')
    ax.set_ylabel('$C_{Propofol}$')
    ax.set_zlabel('BIS')
    fig.colorbar(surf, shrink=0.5, aspect=20)
    ax.view_init(30, 17)
    savepath = "/home/aubouinb/ownCloud/Anesthesie/Science/Bob/Article/Images/surf1.pdf"
    fig.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()


def plot_dresults(data_BIS, data_MAP, data_train_BIS=pd.DataFrame(), data_train_MAP=pd.DataFrame()):
    """plot the result of the prediction with bokeh module"""
    y_true_test = data_BIS["true_dBIS"].values
    y_pred_test = data_BIS["pred_dBIS"].values
    if not data_train_BIS.empty:
        train = True
        y_true_train = data_train_BIS["true_dBIS"].values
        y_pred_train = data_train_BIS["pred_dBIS"].values
    else:
        train = False

    fig1 = figure(width=900, height=450, title="Data (blue) Vs Fitted data (red)")
    fig1.line(np.arange(0, len(y_true_test)), y_true_test, line_color='navy', legend_label="y")
    fig1.line(np.arange(0, len(y_pred_test)), y_pred_test, line_color='red', legend_label="y predicted")
    fig1.xaxis.axis_label = "time [s]"
    fig1.yaxis.axis_label = "y"

    fig2 = figure(width=900, height=450, title="Fitting error")
    fig2.line(np.arange(0, len(y_true_test)), y_true_test-y_pred_test, line_color='navy', legend_label="y")
    fig2.xaxis.axis_label = "time [s]"
    fig2.yaxis.axis_label = "y"

    fig3 = figure(width=900, height=450, title="BIS")
    fig3.circle(y_true_test, y_pred_test, legend_label='y', size=2, color="navy", alpha=0.1)
    if train:
        fig3.circle(y_true_train, y_pred_train, legend_label='train', size=2, color="green", alpha=0.1)

    fig3.line(np.array([y_true_test.min(), y_true_test.max()]), np.array([y_true_test.min(), y_true_test.max()]),
              line_color="red")
    fig3.legend.location = "top_left"
    fig3.xaxis.axis_label = "y measured"
    fig3.yaxis.axis_label = "y predicted"
    fig3.x_range = Range1d(y_true_test.min(), y_true_test.max())
    fig3.y_range = Range1d(y_true_test.min(), y_true_test.max())

    fig4 = figure(width=900, height=450, title="Trained fit")
    if train:
        fig4 = figure(width=900, height=450, title="Data (blue) Vs Fitted data (red)")
        fig4.line(np.arange(0, len(y_true_train)), y_true_train, line_color='navy', legend_label="y traint rue")
        fig4.line(np.arange(0, len(y_pred_train)), y_pred_train, line_color='red', legend_label="y train predicted")
        fig4.xaxis.axis_label = "time [s]"
        fig4.yaxis.axis_label = "y"

    layout = row(column(fig1, fig2), column(fig3, fig4))
    show(layout)

    y_true_test = data_MAP["true_dMAP"].values
    y_pred_test = data_MAP["pred_dMAP"].values
    if train:
        y_true_train = data_train_MAP["true_dMAP"].values
        y_pred_train = data_train_MAP["pred_dMAP"].values

    fig1 = figure(width=900, height=450, title="Data (blue) Vs Fitted data (red)")
    fig1.line(np.arange(0, len(y_true_test)), y_true_test, line_color='navy', legend_label="y")
    fig1.line(np.arange(0, len(y_pred_test)), y_pred_test, line_color='red', legend_label="y predicted")
    fig1.xaxis.axis_label = "time [s]"
    fig1.yaxis.axis_label = "y"

    fig2 = figure(width=900, height=450, title="Fitting error")
    fig2.line(np.arange(0, len(y_true_test)), y_true_test-y_pred_test, line_color='navy', legend_label="y")
    fig2.xaxis.axis_label = "time [s]"
    fig2.yaxis.axis_label = "y"

    fig3 = figure(width=900, height=450, title="MAP")
    fig3.circle(y_true_test, y_pred_test, legend_label='y', size=2, color="navy", alpha=0.1)
    if train:
        fig3.circle(y_true_train, y_pred_train, legend_label='train', size=2, color="green", alpha=0.1)

    fig3.line(np.array([y_true_test.min(), y_true_test.max()]), np.array([y_true_test.min(), y_true_test.max()]),
              line_color="red")
    fig3.legend.location = "top_left"
    fig3.xaxis.axis_label = "y measured"
    fig3.yaxis.axis_label = "y predicted"
    fig3.x_range = Range1d(y_true_test.min(), y_true_test.max())
    fig3.y_range = Range1d(y_true_test.min(), y_true_test.max())

    fig4 = figure(width=900, height=450, title="Trained fit")
    if train:
        fig4 = figure(width=900, height=450, title="Data (blue) Vs Fitted data (red)")
        fig4.line(np.arange(0, len(y_true_train)), y_true_train, line_color='navy', legend_label="y traint rue")
        fig4.line(np.arange(0, len(y_pred_train)), y_pred_train, line_color='red', legend_label="y train predicted")
        fig4.xaxis.axis_label = "time [s]"
        fig4.yaxis.axis_label = "MAP (mmHg)"

    layout = row(column(fig1, fig2), column(fig3, fig4))
    show(layout)
