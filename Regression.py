#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 08:47:45 2022

@author: aubouinb
"""
"""Script to use regression technique in order to fit the pharmacodynamic of the Propofol/Remifentanil effect"""

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

import bisect
import pickle
import numpy as np
import pandas as pd
from math import pi
from itertools import chain
from joblib import dump, load
from collections import OrderedDict
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show
from bokeh.palettes import Viridis256 as colors  # just make sure to import a palette that centers on white (-ish)
from bokeh.layouts import gridplot, row, column
from bokeh.models import ColorBar, LinearColorMapper

#%% Load dataset
Patients_train = pd.read_csv("./Patients_train.csv", index_col=0)
Patients_test = pd.read_csv("./Patients_test.csv", index_col=0)



#%% correlation matrix
# X1 = Patients_test[X_col + ['BIS']]
# corr_matrix_X1 = X1.corr()
# labels = X1.columns
# nlabels = len(labels)
# ## SETTING UP THE PLOT
# def get_bounds(n):
#     """Gets bounds for quads with n features"""
#     bottom = list(chain.from_iterable([[ii]*nlabels for ii in range(nlabels)]))
#     top = list(chain.from_iterable([[ii+1]*nlabels for ii in range(nlabels)]))
#     left = list(chain.from_iterable([list(range(nlabels)) for ii in range(nlabels)]))
#     right = list(chain.from_iterable([list(range(1,nlabels+1)) for ii in range(nlabels)]))
#     return top, bottom, left, right

# def get_colors(corr_array, colors):
#     """Aligns color values from palette with the correlation coefficient values"""
#     ccorr = np.arange(-1, 1, 1/(len(colors)/2))
#     color = []
#     for value in corr_array:
#         ind = bisect.bisect_left(ccorr, value)
#         color.append(colors[ind-1])
#     return color

# p = figure(plot_width=600, plot_height=600,
#             x_range=(0,nlabels), y_range=(0,nlabels),
#             title="Correlation Coefficient Heatmap",
#             toolbar_location=None, tools='')

# p.xgrid.grid_line_color = None
# p.ygrid.grid_line_color = None
# p.xaxis.major_label_orientation = pi/4
# p.yaxis.major_label_orientation = pi/4

# top, bottom, left, right = get_bounds(nlabels)  # creates sqaures for plot
# color_list = get_colors(np.abs(corr_matrix_X1.values.flatten()), colors)

# p.quad(top=top, bottom=bottom, left=left,
#         right=right, line_color='white',
#         color=color_list)

# # Set ticks with labels
# ticks = [tick+0.5 for tick in list(range(nlabels))]
# tick_dict = OrderedDict([[tick, labels[ii]] for ii, tick in enumerate(ticks)])
# # Create the correct number of ticks for each axis 
# p.xaxis.ticker = ticks
# p.yaxis.ticker = ticks
# # Override the labels 
# p.xaxis.major_label_overrides = tick_dict
# p.yaxis.major_label_overrides = tick_dict

# # Setup color bar
# mapper = LinearColorMapper(palette="Viridis256", low=0, high=1)
# color_bar = ColorBar(color_mapper=mapper, location=(0, 0))
# p.add_layout(color_bar, 'right')

# show(p)



#%% Undersample data

step = 60 # Undersampling step



Patients_test_full = Patients_test.copy()

Patients_train_BIS = Patients_train[Patients_train['full_BIS']==0]
Patients_test_BIS = Patients_test[Patients_test['full_BIS']==0]
Patients_train_MAP = Patients_train[Patients_train['full_MAP']==0]
Patients_test_MAP = Patients_test[Patients_test['full_MAP']==0]

Patients_train_BIS = Patients_train_BIS[::step]
Patients_test_BIS = Patients_test_BIS[::step]
Patients_train_MAP = Patients_train_MAP[::step]
Patients_test_MAP = Patients_test_MAP[::step]

# Patients_train.insert(len(Patients_train.columns),"train_set", 0)

# id_train = Patients_train['caseid'].unique()
# id_train = np.random.permutation(id_train)
# id_split = np.array_split(id_train, 5)

# set_int = 0
# for i in  id_train:
#     for j in range(len(id_split)):
#         if i in id_split[j]:
#             set_int = j
#             break
#     Patients_train.loc[Patients_train['caseid']==i, "train_set"] = set_int * np.ones((len(Patients_train[Patients_train['caseid']==i])))

#%% Model based Regressions

feature = 'All'
cov = ['age', 'sex', 'height', 'weight']
Ce_bis = ['Ce_Prop', 'Ce_Rem']
Ce_map = ['Ce_Prop_MAP', 'Ce_Rem_MAP']
Cplasma = ['Cp_Prop', 'Cp_Rem']
Ce_bis_eleveld = ['Ce_Prop_Eleveld', 'Ce_Rem_Eleveld']
Ce_map_eleveld = ['Ce_Prop_MAP_Eleveld', 'Ce_Rem_MAP_Eleveld']
Cplasma_eleveld = ['Cp_Prop_Eleveld', 'Cp_Rem_Eleveld']
median = ['med_BIS', 'med_MAP']
hr = []

#feat_A
if feature=='All':
     X_col = cov + ['bmi','lbm', 'mean_HR'] + Ce_map_eleveld + Ce_bis_eleveld + Cplasma_eleveld
elif feature == '-bmi':    
     X_col = cov + ['lbm', 'mean_HR'] + Ce_bis_eleveld + Ce_map_eleveld + Cplasma_eleveld
elif feature=='-lbm':  
     X_col = cov + ['bmi', 'mean_HR'] + Ce_bis_eleveld + Ce_map_eleveld + Cplasma_eleveld
elif feature=='-map':     
     X_col = cov + ['bmi','lbm', 'mean_HR'] + Ce_bis_eleveld + Ce_map_eleveld + Cplasma_eleveld
elif feature=='-hr':     
     X_col = cov + ['bmi','lbm', 'mean_HR'] + Ce_bis_eleveld + Ce_map_eleveld + Cplasma_eleveld
elif feature=='-Cplasma':
    X_col = cov + ['bmi','lbm', 'mean_HR'] + Ce_bis_eleveld + Ce_map_eleveld
elif feature=='-Cmap':
    X_col = cov + ['bmi','lbm', 'mean_HR'] + Ce_bis_eleveld + Cplasma_eleveld
elif feature=='-Cbis':
    X_col = cov + ['bmi','lbm'] + Ce_map_eleveld + Cplasma_eleveld 


     
Patients_train_BIS = Patients_train_BIS[X_col + ['caseid','BIS','train_set']].dropna()
Patients_test_BIS = Patients_test_BIS[X_col + ['caseid','BIS','Time']].dropna()
Patients_train_MAP = Patients_train_MAP[X_col + ['caseid','MAP','train_set']].dropna()
Patients_test_MAP = Patients_test_MAP[X_col + ['caseid','MAP','Time']].dropna()


name_rg = 'SVR'
filename = './saved_reg/reg_' + name_rg + '_feat_' + feature + '.pkl'
poly_degree = 1
pca_bool = False
regressors = {}

try:
    results_BIS = pd.read_csv("./results_BIS.csv", index_col=0)
    results_MAP = pd.read_csv("./results_MAP.csv", index_col=0)
except:
    results_BIS = Patients_test_BIS[['Time','caseid','BIS']].copy()
    results_MAP = Patients_test_MAP[['Time','caseid','MAP']].copy()
    
Train_data_BIS = pd.DataFrame()
Test_data_BIS = pd.DataFrame()
Train_data_BIS['case_id'] = Patients_train_BIS['caseid']
Test_data_BIS['case_id'] = Patients_test_BIS['caseid']  

Train_data_MAP = pd.DataFrame()
Test_data_MAP = pd.DataFrame()
Train_data_MAP['case_id'] = Patients_train_MAP['caseid']
Test_data_MAP['case_id'] = Patients_test_MAP['caseid']


i = 0

for y_col in ['BIS', 'MAP']: #
    if y_col=='BIS':
        Patients_train = Patients_train_BIS
        Patients_test = Patients_test_BIS
    elif y_col=='MAP':
        Patients_train = Patients_train_MAP
        Patients_test = Patients_test_MAP
        
    #--------------Training-------------
 
    try: #Try to load trained regressor
        regressors = pickle.load(open(filename, 'rb'))
        rg = regressors[y_col]
        print("load ok")
        
    except: #Otherwhise train the regressors and save it
        Y_train = Patients_train[y_col]
        if name_rg in ['ElasticNet','TheilSenRegressor','BayesianRidge','KNeighborsRegressor','HuberRegressor','SGDRegressor']:
            X_train = PolynomialFeatures(degree=poly_degree, include_bias=False).fit_transform(Patients_train[X_col].values)   
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train) 
            if pca_bool:
                pca = PCA()#n_components=4)
                X_train = pca.fit_transform(X_train)
                plt.plot(pca.explained_variance_ratio_,'-o')
                X_train = X_train[:,0:20]
        elif name_rg=='KernelRidge'  or name_rg=='SVR' or name_rg=='MLPRegressor':
            X_train = Patients_train[X_col].values
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train) 
        
        ps = PredefinedSplit(Patients_train['train_set'].values)
        # ---ElasticNet----
        if name_rg=='ElasticNet':
            rg = ElasticNet(max_iter=100000)
            Gridsearch = GridSearchCV(rg, {'alpha': np.logspace(-4,0,5), 'l1_ratio' : np.linspace(0,1,11)},
                                      n_jobs = 8, cv = ps, scoring='r2', verbose=0)
            Gridsearch.fit(X_train,Y_train)
            
        # ---KernelRidge----
        elif name_rg=='KernelRidge':
            parameters = {'kernel':('linear', 'rbf', 'polynomial'), 'alpha':np.logspace(-3,1,5)}
            rg = KernelRidge()
            #kmeans = KMeans(n_clusters=5000, random_state=0, verbose=4).fit(np.concatenate((X_train,np.expand_dims(Y_train,axis=1)),axis=1))
            Gridsearch = GridSearchCV(rg, parameters, cv=ps, n_jobs = 8,
                scoring='neg_mean_squared_error', return_train_score=True, verbose=4)
            Gridsearch.fit(X_train, Y_train)
            
        # ---SVR----
        elif name_rg=='SVR':
            rg = SVR(verbose=0, shrinking=False, cache_size=1000)# kernel = 'poly', 'rbf'; 'linear'
            Gridsearch = GridSearchCV(rg, {'kernel': ['rbf'], 'C' : [0.1],
                                           'gamma' : np.logspace(-1,3,5) , 'epsilon' : np.logspace(-3,1,5)}, #np.logspace(-2,1,3)
                                      n_jobs = 8, cv = ps, scoring='r2', verbose=0)

            Gridsearch.fit(X_train[:],Y_train[:])
        
        elif name_rg=='MLPRegressor':
            rg = MLPRegressor(verbose=0, learning_rate = 'adaptive', max_iter = 1000, random_state=8)
            Gridsearch = GridSearchCV(rg, {'hidden_layer_sizes': [512, 1024], 'alpha': 10.0 ** -np.arange(1, 4),
                                           'activation': ('tanh','relu','logistic','identity')}, cv = ps, n_jobs = 6) #,'relu','logistic','identity'v
            if y_col=='BIS':
                Gridsearch.fit(X_train, Y_train/100)
            elif y_col=='MAP':
                Gridsearch.fit(X_train, Y_train/150)
                
        elif name_rg=='TheilSenRegressor':
            rg = TheilSenRegressor()
            Gridsearch = GridSearchCV(rg, {'max_subpopulation': [1e4, 1e5]}, n_jobs = 6, cv = ps)
            Gridsearch.fit(X_train, Y_train)
        elif name_rg=='BayesianRidge':
            rg = BayesianRidge(n_iter=1000)
            Gridsearch = GridSearchCV(rg, {'alpha_1': [1e-6,1e-7,1e-5], 'alpha_2':[1e-6,1e-7,1e-5], 'lambda_1': [1e-6,1e-7,1e-5],
                                           'lambda_2':[1e-6,1e-7,1e-5]}, n_jobs = 8, cv = ps)
            Gridsearch.fit(X_train, Y_train)
        elif name_rg=='KNeighborsRegressor':
            rg = KNeighborsRegressor(n_jobs=8)
            Gridsearch = GridSearchCV(rg, {'n_neighbors': [500, 1000, 2000, 3000], 'weights':('uniform', 'distance')}, n_jobs = 8, cv = ps)
            Gridsearch.fit(X_train, Y_train)
        elif name_rg=='HuberRegressor':
            rg = HuberRegressor(max_iter=1000)
            Gridsearch = GridSearchCV(rg, {'epsilon': [1.2, 1.35, 1.5, 2], 'alpha': [1e-5, 1e-4, 1e-3]}, n_jobs = 8, cv = ps)
            Gridsearch.fit(X_train, Y_train)
        elif name_rg=='SGDRegressor':
            rg = SGDRegressor()
            Gridsearch = GridSearchCV(rg, {'loss': ('squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'), 
                                           'alpha':[1e-6,1e-7,1e-5]}, n_jobs = 8, cv = ps)
            Gridsearch.fit(X_train, Y_train)
        print(Gridsearch.best_params_)
        
        rg = Gridsearch.best_estimator_
        regressors[y_col] = rg
        pickle.dump(regressors, open(filename, 'wb'))
    
    #--------------test performances on test cases-------------
    
    if name_rg in ['ElasticNet','TheilSenRegressor','BayesianRidge','KNeighborsRegressor','HuberRegressor','SGDRegressor']:
        X_train = PolynomialFeatures(degree=poly_degree, include_bias=False).fit_transform(Patients_train[X_col].values)
        scaler = StandardScaler()
        scaler.fit(X_train) 
        pca = PCA()#n_components=4)
        pca.fit(X_train)
        
        X_test = PolynomialFeatures(degree=poly_degree, include_bias=False).fit_transform(Patients_test[X_col].values)
        X_test = scaler.transform(X_test)
        if pca_bool:
            X_test = pca.transform(X_test)
            X_test = X_test[:,0:20]
        
        y_predicted = rg.predict(X_test)
        
    elif name_rg=='KernelRidge'  or name_rg=='SVR':
        X_train = Patients_train[X_col].values
        scaler = StandardScaler()
        scaler.fit(X_train) 
        
        X_test = Patients_test[X_col].values
        X_test = scaler.transform(X_test)
        y_predicted = rg.predict(X_test)
        
    elif name_rg=='MLPRegressor':
        X_train = Patients_train[X_col].values
        scaler = StandardScaler()
        scaler.fit(X_train) 
        
        X_test = Patients_test[X_col].values
        X_test = scaler.transform(X_test)
        
        if y_col=='BIS':
            y_predicted = rg.predict(X_test)*100
        elif y_col=='MAP':
            y_predicted = rg.predict(X_test)*150
    
    
    col_name = 'pred_' + y_col + '_' + name_rg
    if y_col=='BIS':
        Test_data_BIS['true_' + y_col] = Patients_test[y_col]
        Test_data_BIS['pred_' + y_col] = y_predicted
        results_BIS.loc[:,col_name] = y_predicted
    else:
        Test_data_MAP['true_' + y_col] = Patients_test[y_col]
        Test_data_MAP['pred_' + y_col] = y_predicted
        results_MAP.loc[:,col_name] = y_predicted
    #-----------------test performances on train cases--------------------
    
    if name_rg in ['ElasticNet','TheilSenRegressor','BayesianRidge','KNeighborsRegressor','HuberRegressor','SGDRegressor']:
        X_train = PolynomialFeatures(degree=poly_degree, include_bias=False).fit_transform(Patients_train[X_col].values)
        X_train = scaler.transform(X_train)
        if pca_bool:
            X_train = pca.transform(X_train)
            X_train = X_train[:,0:20]
        
        y_predicted_train = rg.predict(X_train)
        
    elif name_rg=='KernelRidge'  or name_rg=='SVR':
        X_train = Patients_train[X_col].values
        X_train = scaler.transform(X_train)

        y_predicted_train = rg.predict(X_train)
        
    elif name_rg=='MLPRegressor':
        X_train = Patients_train[X_col].values
        X_train = scaler.transform(X_train)
        
        if y_col=='BIS':
            y_predicted_train = rg.predict(X_train)*100
        elif y_col=='MAP':
            y_predicted_train = rg.predict(X_train)*150
    
    if y_col=='BIS':
        Train_data_BIS['true_' + y_col] = Patients_train[y_col]
        Train_data_BIS['pred_' + y_col] = y_predicted_train 
    else:
        Train_data_MAP['true_' + y_col] = Patients_train[y_col]
        Train_data_MAP['pred_' + y_col] = y_predicted_train 
        
    plot_surface(rg, scaler, feature)
    
# results_BIS.to_csv("./results_BIS.csv")
# results_MAP.to_csv("./results_MAP.csv")

print('     ***-----------------' + name_rg + '-----------------***')
print("\n                 ------ Test Results ------")
max_case_bis, min_case_bis= compute_metrics(Test_data_BIS)
max_case_map, min_case_map = compute_metrics(Test_data_MAP)
print("\n\n                 ------ Train Results ------")
# compute_metrics(Train_data_BIS)
# compute_metrics(Train_data_MAP)
# plot_results(Test_data_BIS, Test_data_MAP, Train_data_BIS, Train_data_MAP)

# plot_case(results_BIS, results_MAP, Patients_test_full, min_case_bis, min_case_map, max_case_bis, max_case_map)


