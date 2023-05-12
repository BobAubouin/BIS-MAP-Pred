"""Try to use regression techniques to predict BIS and MAP in the future
Created on Wed May 18 08:47:45 2022

@author: aubouinb
"""


import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, TheilSenRegressor, BayesianRidge, HuberRegressor, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from metrics_functions import compute_metrics, plot_results, plot_case, plot_surface


# %% Load dataset
Patients_train = pd.read_csv("./data/Patients_train.csv", index_col=0)
Patients_test = pd.read_csv("./data/Patients_test.csv", index_col=0)

# %% Undersample data

step = 60  # Undersampling step


Patients_test_full = Patients_test.copy()

Patients_train_BIS_init = Patients_train[Patients_train['full_BIS'] == 0]
Patients_test_BIS_init = Patients_test[Patients_test['full_BIS'] == 0]
Patients_train_MAP_init = Patients_train[Patients_train['full_MAP'] == 0]
Patients_test_MAP_init = Patients_test[Patients_test['full_MAP'] == 0]

Patients_train_BIS_init = Patients_train_BIS_init[::step]
Patients_test_BIS_init = Patients_test_BIS_init[::step]
Patients_train_MAP_init = Patients_train_MAP_init[::step]
Patients_test_MAP_init = Patients_test_MAP_init[::step]

# %% Model based Regressions

feature = 'All'
cov = ['age', 'gender', 'height', 'weight']
Ce_bis_eleveld = ['Ce_Prop_Eleveld', 'Ce_Rem_Eleveld']
Ce_map_eleveld = ['Ce_Prop_MAP_Eleveld', 'Ce_Rem_MAP_Eleveld']
Cplasma_eleveld = ['Cp_Prop_Eleveld', 'Cp_Rem_Eleveld']
name_rg = 'SVR'
results_df = pd.DataFrame()
output_df = Patients_test[['caseid', 'Time']]
for delay in [0, 30, 120, 300, 600]:  # Delay in seconds

    if delay == 0:
        output = ['BIS', 'MAP']
    else:
        output = [f'BIS_plus_{delay}', f'MAP_plus_{delay}']

    X_col = cov + ['bmi', 'lbm', 'mean_HR'] + Ce_map_eleveld + Ce_bis_eleveld + Cplasma_eleveld

    Patients_train_BIS = Patients_train_BIS_init[X_col + ['caseid', output[0], 'train_set']].dropna()
    Patients_test_BIS = Patients_test_BIS_init[X_col + ['caseid', output[0], 'Time']].dropna()
    Patients_train_MAP = Patients_train_MAP_init[X_col + ['caseid', output[1], 'train_set']].dropna()
    Patients_test_MAP = Patients_test_MAP_init[X_col + ['caseid', output[1], 'Time']].dropna()

    # , 'ElasticNet', 'KNeighborsRegressor', 'KernelRidge'

    if delay > 0:
        filename = './saved_reg/reg_' + name_rg + '_feat_' + feature + '_delay_' + str(delay) + '.pkl'
    else:
        filename = './saved_reg/reg_' + name_rg + '_feat_' + feature + '.pkl'
    poly_degree = 1
    pca_bool = False
    regressors = {}

    Train_data_BIS = pd.DataFrame()
    Test_data_BIS = pd.DataFrame()
    Train_data_BIS['caseid'] = Patients_train_BIS['caseid']
    Test_data_BIS['caseid'] = Patients_test_BIS['caseid']
    Test_data_BIS['Time'] = Patients_test_BIS['Time']

    Train_data_MAP = pd.DataFrame()
    Test_data_MAP = pd.DataFrame()
    Train_data_MAP['caseid'] = Patients_train_MAP['caseid']
    Test_data_MAP['caseid'] = Patients_test_MAP['caseid']
    Test_data_MAP['Time'] = Patients_test_MAP['Time']

    i = 0

    for y_col in output:
        # --------------Training-------------
        if 'BIS' in y_col:
            Patients_train = Patients_train_BIS
            Patients_test = Patients_test_BIS
        elif 'MAP' in y_col:
            Patients_train = Patients_train_MAP
            Patients_test = Patients_test_MAP
        try:  # Try to load trained regressor
            regressors = pickle.load(open(filename, 'rb'))
            rg = regressors[y_col]
            print("load ok")

        except:  # Otherwhise train the regressors and save it
            Y_train = Patients_train[y_col]

            X_train = Patients_train[X_col].values
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            ps = PredefinedSplit(Patients_train['train_set'].values)

            # ---SVR----
            rg = SVR(verbose=0, shrinking=False, cache_size=1000)  # kernel = 'poly', 'rbf'; 'linear', 'sigmoid'
            Gridsearch = GridSearchCV(rg, {'kernel': ['rbf'], 'C': [0.1], 'degree': [2],
                                           'gamma': np.logspace(-1, 3, 5), 'epsilon': np.logspace(-3, 1, 5)},  # np.logspace(-2,1,3)
                                      n_jobs=8, cv=ps, scoring='r2', verbose=0)

            Gridsearch.fit(X_train[:], Y_train[:])

            print(Gridsearch.best_score_)
            print(Gridsearch.best_params_)

            rg = Gridsearch.best_estimator_
            regressors[y_col] = rg
            pickle.dump(regressors, open(filename, 'wb'))

        # --------------test performances on test cases-------------

        X_train = Patients_train[X_col].values
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_test = Patients_test[X_col].values
        X_test = scaler.transform(X_test)
        y_predicted = rg.predict(X_test)

        col_name = 'pred_' + y_col + '_' + name_rg
        if 'BIS' in y_col:
            Test_data_BIS[f'true_{y_col}'] = Patients_test[y_col]
            Test_data_BIS[f'pred_{y_col}'] = y_predicted
            temp = Test_data_BIS[['caseid', 'Time', f'pred_{y_col}']].copy()
        else:
            Test_data_MAP[f'true_{y_col}'] = Patients_test[y_col]
            Test_data_MAP[f'pred_{y_col}'] = y_predicted
            temp = Test_data_MAP[['caseid', 'Time', 'pred_' + y_col]].copy()
        temp.rename(columns={'pred_' + y_col: f'{y_col}_{name_rg}_{delay}'}, inplace=True)
        output_df = pd.merge(output_df, temp,
                             on=['caseid', 'Time'], how='left')
        # -----------------test performances on train cases--------------------

        X_train = Patients_train[X_col].values
        X_train = scaler.transform(X_train)

        y_predicted_train = rg.predict(X_train)

        if 'BIS' in y_col:
            Train_data_BIS[f'true_{y_col}'] = Patients_train[y_col]
            Train_data_BIS[f'pred_{y_col}'] = y_predicted_train
        else:
            Train_data_MAP[f'true_{y_col}'] = Patients_train[y_col]
            Train_data_MAP[f'pred_{y_col}'] = y_predicted_train

        # plot_surface(rg, scaler, feature)

    # results_BIS.to_csv("./results_BIS.csv")
    # results_MAP.to_csv("./results_MAP.csv")

    print(
        f"***{delay:-^30}***\n"
        f"***{' Test Results ':-^30}***")
    max_case_bis, min_case_bis, df_bis = compute_metrics(Test_data_BIS)
    df_bis.rename(columns={'MDPE': 'MDPE_BIS',
                           'MDAPE': 'MDAPE_BIS',
                           'RMSE': 'RMSE_BIS'}, inplace=True)
    max_case_map, min_case_map, df_map = compute_metrics(Test_data_MAP)
    df_map.rename(columns={'MDPE': 'MDPE_MAP',
                           'MDAPE': 'MDAPE_MAP',
                           'RMSE': 'RMSE_MAP'}, inplace=True)
    df = pd.concat([pd.DataFrame({'delay': delay}, index=[0]), df_bis, df_map], axis=1)
    results_df = pd.concat([results_df, df], axis=0)

output_df.to_csv("./outputs/delay.csv")

print('\n')
styler = results_df[['delay', 'MDAPE_BIS', 'MDAPE_MAP']].style
styler.hide(axis='index')
# styler.format(precision=2)
print(styler.to_latex())
# print("\n\n                 ------ Train Results ------")
# compute_metrics(Train_data_BIS)
# compute_metrics(Train_data_MAP)
plot_results(Test_data_BIS, Test_data_MAP, Train_data_BIS, Train_data_MAP)

# plot_case(results_BIS, results_MAP, Patients_test_full, min_case_bis, min_case_map, max_case_bis, max_case_map)
