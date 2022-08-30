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
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from metrics_functions import compute_metrics, plot_results


#%% Load dataset
Patients_train = pd.read_csv("./Patients_train.csv", index_col=0)
Patients_test = pd.read_csv("./Patients_test.csv", index_col=0)

#%% Undersample data

step = 30 # Undersampling step

Patients_train = Patients_train[::step]

Patients_test = Patients_test[::step]

Patients_train = Patients_train[Patients_train['full']==0]
Patients_test = Patients_test[Patients_test['full']==0]

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
X_col = ['age', 'sex', 'height', 'weight', 'bmi', 'MAP_base_case', 'Ce_Prop', 'Ce_Rem'] + [('Cp_Prop_' + str(i+1)) for i in range(10)] + [('Cp_Rem_' + str(i+1)) for i in range(10)]

Patients_train = Patients_train[X_col + ['caseid','BIS','MAP','train_set']].dropna()
Patients_test = Patients_test[X_col + ['caseid','BIS','MAP']].dropna()


name_rg = 'SVR'
regressors = {}


Train_data = pd.DataFrame()
Test_data = pd.DataFrame()
Train_data['case_id'] = Patients_train['caseid']
Test_data['case_id'] = Patients_test['caseid']

i = 0
for y_col in ['BIS', 'MAP']: #

    #--------------Training-------------
 
    try: #Try to load trained regressor
        filename = './saved_reg/reg_' + name_rg + '.pkl'
        regressors = pickle.load(open(filename, 'rb'))
        rg = regressors[y_col]
        print("load ok")
        
    except: #Otherwhise train the regressors and save it
        Y_train = Patients_train[y_col]
        if name_rg=='ElasticNet':
            poly_degree = 1
            pca_bool = False
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
            rg = SVR(verbose=3, shrinking=False)# kernel = 'poly', 'rbf'; 'linear'
            Gridsearch = GridSearchCV(rg, {'kernel': ['rbf'], 'C' : np.logspace(1,3,3),
                                           'gamma' : np.logspace(-1,1,3) , 'epsilon' : np.logspace(-2,1,3)}, #np.logspace(-2,1,3)
                                      n_jobs = 8, cv = ps, scoring='r2', verbose=4)

            Gridsearch.fit(X_train[:],Y_train[:])
        
        elif name_rg=='MLPRegressor':
            rg = MLPRegressor(learning_rate = 'adaptive', max_iter = 1000)
            Gridsearch = GridSearchCV(rg, {'hidden_layer_sizes': [4, 8, 16, 32, 64], 'alpha': np.logspace(-4,-2,3),
                                           'activation': ('tanh','relu')})
            Gridsearch.fit(X_train, Y_train/100)
            
        print(Gridsearch.best_params_)
        
        rg = Gridsearch.best_estimator_
        name_rg = str(rg)
        name_rg = name_rg[:name_rg.find('(')]
        filename = './saved_reg/reg_' + name_rg + '.pkl'
        regressors[y_col] = rg
        pickle.dump(regressors, open(filename, 'wb'))
    
    #--------------test performances on test cases-------------
    
    if name_rg=='ElasticNet':
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
        
        y_predicted = rg.predict(X_test)*100
        
        
    if y_col=='BIS':
        Test_data['true_BIS'] = Patients_test[y_col]
        Test_data['pred_BIS'] = y_predicted
    elif y_col=='MAP':
        Test_data['true_MAP'] = Patients_test[y_col]
        Test_data['pred_MAP'] = y_predicted  
    
    #-----------------test performances on train cases--------------------
    if name_rg=='ElasticNet':
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
        
        y_predicted_train = rg.predict(X_train)*100
        
    if y_col=='BIS':
        Train_data['true_BIS'] = Patients_train[y_col]
        Train_data['pred_BIS'] = y_predicted_train
    elif y_col=='MAP':
        Train_data['true_MAP'] = Patients_train[y_col]
        Train_data['pred_MAP'] = y_predicted_train  

print('     ***-----------------' + name_rg + '-----------------***')
print("\n                 ------ Test Results ------")
compute_metrics(Test_data)
print("\n\n                 ------ Train Results ------")
compute_metrics(Train_data)
plot_results(Test_data, Train_data)




