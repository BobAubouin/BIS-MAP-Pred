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

Patients_train_BIS = Patients_train[Patients_train['full_BIS'] == 0]
Patients_test_BIS = Patients_test[Patients_test['full_BIS'] == 0]
Patients_train_MAP = Patients_train[Patients_train['full_MAP'] == 0]
Patients_test_MAP = Patients_test[Patients_test['full_MAP'] == 0]

Patients_train_BIS = Patients_train_BIS[::step]
Patients_test_BIS = Patients_test_BIS[::step]
Patients_train_MAP = Patients_train_MAP[::step]
Patients_test_MAP = Patients_test_MAP[::step]

# %% Model based Regressions

feature = 'All'
cov = ['age', 'gender', 'height', 'weight']
Ce_bis_eleveld = ['Ce_Prop_Eleveld', 'Ce_Rem_Eleveld']
Ce_map_eleveld = ['Ce_Prop_MAP_Eleveld', 'Ce_Rem_MAP_Eleveld']
Cplasma_eleveld = ['Cp_Prop_Eleveld', 'Cp_Rem_Eleveld']

delay = 0  # Delay in seconds
kernel = 'linear'
poly_degree = 2
output = ['BIS', 'MAP']

X_col = cov + ['bmi', 'lbm', 'mean_HR'] + Ce_map_eleveld + Ce_bis_eleveld + Cplasma_eleveld


Patients_train_BIS = Patients_train_BIS[X_col + ['caseid', output[0], 'train_set']].dropna()
Patients_test_BIS = Patients_test_BIS[X_col + ['caseid', output[0], 'Time']].dropna()
Patients_train_MAP = Patients_train_MAP[X_col + ['caseid', output[1], 'train_set']].dropna()
Patients_test_MAP = Patients_test_MAP[X_col + ['caseid', output[1], 'Time']].dropna()

# , 'ElasticNet', 'KNeighborsRegressor', 'KernelRidge'
results_df = pd.DataFrame()
for name_rg in ['SVR']:
    if poly_degree > 1:
        filename = f'./saved_reg/reg_{name_rg}_kernek_poly.pkl'
    else:
        filename = f'./saved_reg/reg_{name_rg}_kernek_{kernel}.pkl'
    pca_bool = False
    regressors = {}

    try:
        results_BIS = pd.read_csv("./data/results_"+output[0]+".csv", index_col=0)
        results_MAP = pd.read_csv("./data/results_"+output[1]+".csv", index_col=0)
    except:
        results_BIS = Patients_test_BIS[['Time', 'caseid', output[0]]].copy()
        results_MAP = Patients_test_MAP[['Time', 'caseid', output[1]]].copy()

    Train_data_BIS = pd.DataFrame()
    Test_data_BIS = pd.DataFrame()
    Train_data_BIS['case_id'] = Patients_train_BIS['caseid']
    Test_data_BIS['case_id'] = Patients_test_BIS['caseid']

    Train_data_MAP = pd.DataFrame()
    Test_data_MAP = pd.DataFrame()
    Train_data_MAP['case_id'] = Patients_train_MAP['caseid']
    Test_data_MAP['case_id'] = Patients_test_MAP['caseid']

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
            poly = PolynomialFeatures(degree=poly_degree)
            X_train = poly.fit_transform(X_train)

            ps = PredefinedSplit(Patients_train['train_set'].values)

            # ---SVR----
            rg = SVR(verbose=0, shrinking=False, cache_size=1000)  # kernel = 'poly', 'rbf'; 'linear', 'sigmoid'
            Gridsearch = GridSearchCV(rg, {'kernel': ['linear'], 'C': [0.1], 'degree': [2],
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
        poly = PolynomialFeatures(degree=poly_degree)
        poly.fit(X_train)

        X_test = Patients_test[X_col].values
        X_test = scaler.transform(X_test)
        X_test = poly.transform(X_test)
        y_predicted = rg.predict(X_test)

        col_name = 'pred_' + y_col + '_' + name_rg
        if 'BIS' in y_col:
            Test_data_BIS['true_' + y_col] = Patients_test[y_col]
            Test_data_BIS['pred_' + y_col] = y_predicted
            results_BIS.loc[:, col_name] = y_predicted
        else:
            Test_data_MAP['true_' + y_col] = Patients_test[y_col]
            Test_data_MAP['pred_' + y_col] = y_predicted
            results_MAP.loc[:, col_name] = y_predicted
        # -----------------test performances on train cases--------------------

        X_train = Patients_train[X_col].values
        X_train = scaler.transform(X_train)
        X_train = poly.transform(X_train)

        y_predicted_train = rg.predict(X_train)

        if 'BIS' in y_col:
            Train_data_BIS['true_' + y_col] = Patients_train[y_col]
            Train_data_BIS['pred_' + y_col] = y_predicted_train
        else:
            Train_data_MAP['true_' + y_col] = Patients_train[y_col]
            Train_data_MAP['pred_' + y_col] = y_predicted_train

        # plot_surface(rg, scaler, feature)

    # results_BIS.to_csv("./results_BIS.csv")
    # results_MAP.to_csv("./results_MAP.csv")

    print('     ***-----------------' + name_rg + '-----------------***')
    print("\n                 ------ Test Results ------")
    print('     ***-----------------' + name_rg + '-----------------***')
    print("\n                 ------ Test Results ------")
    max_case_bis, min_case_bis, df_bis = compute_metrics(Test_data_BIS)
    df_bis.rename(columns={'MDPE': 'MDPE_BIS',
                           'MDAPE': 'MDAPE_BIS',
                           'RMSE': 'RMSE_BIS'}, inplace=True)
    max_case_map, min_case_map, df_map = compute_metrics(Test_data_MAP)
    df_map.rename(columns={'MDPE': 'MDPE_MAP',
                           'MDAPE': 'MDAPE_MAP',
                           'RMSE': 'RMSE_MAP'}, inplace=True)
    df = pd.concat([pd.DataFrame({'name_rg': name_rg}, index=[0]), df_bis, df_map], axis=1)
    results_df = pd.concat([results_df, df], axis=0)
print('\n')
styler = results_df.style
styler.hide(axis='index')
# styler.format(precision=2)
print(styler.to_latex())
# print("\n\n                 ------ Train Results ------")
# compute_metrics(Train_data_BIS)
# compute_metrics(Train_data_MAP)
# plot_results(Test_data_BIS, Test_data_MAP, Train_data_BIS, Train_data_MAP)

# plot_case(results_BIS, results_MAP, Patients_test_full, min_case_bis, min_case_map, max_case_bis, max_case_map)

# %%
