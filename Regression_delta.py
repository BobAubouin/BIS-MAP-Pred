#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:29:23 2022

@author: aubouinb
"""
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
from metrics_functions import compute_metrics, plot_dresults, plot_case, plot_surface


#%% Load dataset
Patients_train = pd.read_csv("./Patients_train.csv", index_col=0)
Patients_test = pd.read_csv("./Patients_test.csv", index_col=0)

#%% Undersample data

step = 1 # Undersampling step



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


cov = ['age', 'sex', 'height', 'weight', 'bmi', 'lbm']
Ce = ['Ce_Prop', 'Ce_Rem', 'Ce_Prop_MAP', 'Ce_Rem_MAP']
dCe = ['dCe_Prop', 'dCe_Rem', 'dCe_Prop_MAP', 'dCe_Rem_MAP']
Cp = [('Cp_Prop_' + str(i+1)) for i in range(10)] + [('Cp_Rem_' + str(i+1)) for i in range(10)]

feature = 'A'

if feature=='A':
    X_col = cov + Ce + dCe
elif feature == 'B':
    X_col = cov + Ce + ['MAP_base_case'] + Cp

Patients_train_BIS = Patients_train_BIS[X_col + ['caseid','dBIS','train_set']].dropna()
Patients_test_BIS = Patients_test_BIS[X_col + ['caseid','dBIS','Time']].dropna()
Patients_train_MAP = Patients_train_MAP[X_col + ['caseid','dMAP','train_set']].dropna()
Patients_test_MAP = Patients_test_MAP[X_col + ['caseid','dMAP','Time']].dropna()


name_rg = 'MLPRegressor'
filename = './saved_reg/reg_' + name_rg + '_delta.pkl'
poly_degree = 1
pca_bool = False
regressors = {}

try:
    results_BIS = pd.read_csv("./results_BIS.csv", index_col=0)
    results_MAP = pd.read_csv("./results_MAP.csv", index_col=0)
except:
    results_BIS = Patients_test_BIS[['Time','caseid','dBIS']]
    results_MAP = Patients_test_MAP[['Time','caseid','dMAP']]
    
Train_data_BIS = pd.DataFrame()
Test_data_BIS = pd.DataFrame()
Train_data_BIS['case_id'] = Patients_train_BIS['caseid']
Test_data_BIS['case_id'] = Patients_test_BIS['caseid']

Train_data_MAP = pd.DataFrame()
Test_data_MAP = pd.DataFrame()
Train_data_MAP['case_id'] = Patients_train_MAP['caseid']
Test_data_MAP['case_id'] = Patients_test_MAP['caseid']


i = 0

for y_col in ['dBIS', 'dMAP']: #
    if y_col=='dBIS':
        Patients_train = Patients_train_BIS
        Patients_test = Patients_test_BIS
    elif y_col =='dMAP':
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
            Gridsearch = GridSearchCV(rg, {'alpha': np.logspace(-4,0,5), 'l1_ratio' : np.linspace(0,1,5)},
                                      scoring='neg_median_absolute_error',n_jobs = 8, cv = ps, verbose=0)
            Gridsearch.fit(X_train,Y_train)
            
        # ---KernelRidge----
        elif name_rg=='KernelRidge':
            parameters = {'kernel':('linear', 'rbf', 'polynomial'), 'alpha':np.logspace(-3,1,5)}
            rg = KernelRidge()
            #kmeans = KMeans(n_clusters=5000, random_state=0, verbose=4).fit(np.concatenate((X_train,np.expand_dims(Y_train,axis=1)),axis=1))
            Gridsearch = GridSearchCV(rg, parameters, cv=ps, n_jobs = 6,
                scoring='neg_median_absolute_error', return_train_score=True, verbose=4)
            Gridsearch.fit(X_train, Y_train)
            
        # ---SVR----
        elif name_rg=='SVR':
            rg = SVR(verbose=3, shrinking=False, cache_size=1000)# kernel = 'poly', 'rbf'; 'linear'
            Gridsearch = GridSearchCV(rg, {'kernel': ['rbf'], 'C' : [0.1],
                                           'gamma' : np.logspace(-1,3,5) , 'epsilon' : np.logspace(-3,1,5)}, #np.logspace(-2,1,3)
                                      n_jobs = 8, cv = ps, scoring='neg_median_absolute_error', verbose=4)

            Gridsearch.fit(X_train[:],Y_train[:])
        
        elif name_rg=='MLPRegressor':
            rg = MLPRegressor(learning_rate = 'adaptive', max_iter = 1000)
            Gridsearch = GridSearchCV(rg, {'hidden_layer_sizes': [128, 256, 512], 'alpha': 10.0 ** -np.arange(1, 7),
                                           'activation': ('tanh','relu','logistic','identity')}, scoring='neg_median_absolute_error', cv = ps)
            if y_col=='dBIS':
                Gridsearch.fit(X_train, Y_train/100)
            elif y_col=='dMAP':
                Gridsearch.fit(X_train, Y_train/150)
                
        elif name_rg=='TheilSenRegressor':
            rg = TheilSenRegressor()
            Gridsearch = GridSearchCV(rg, {'max_subpopulation': [1e4, 1e5]}, n_jobs = 6, scoring='neg_median_absolute_error', cv = ps)
            Gridsearch.fit(X_train, Y_train)
        elif name_rg=='BayesianRidge':
            rg = BayesianRidge(n_iter=1000)
            Gridsearch = GridSearchCV(rg, {'alpha_1': [1e-6,1e-7,1e-5], 'alpha_2':[1e-6,1e-7,1e-5], 'lambda_1': [1e-6,1e-7,1e-5],
                                           'lambda_2':[1e-6,1e-7,1e-5]}, scoring='neg_median_absolute_error', n_jobs = 8, cv = ps)
            Gridsearch.fit(X_train, Y_train)
        elif name_rg=='KNeighborsRegressor':
            rg = KNeighborsRegressor(n_jobs=8)
            Gridsearch = GridSearchCV(rg, {'n_neighbors': [5,10,20,50,100, 500, 1000]}, scoring='neg_median_absolute_error', n_jobs = 8, cv = ps)
            Gridsearch.fit(X_train, Y_train)
        elif name_rg=='HuberRegressor':
            rg = HuberRegressor(max_iter=1000)
            Gridsearch = GridSearchCV(rg, {'epsilon': [1.2, 1.35, 1.5, 2], 'alpha': [1e-5, 1e-4, 1e-3]}, scoring='neg_median_absolute_error', n_jobs = 8, cv = ps)
            Gridsearch.fit(X_train, Y_train)
        elif name_rg=='SGDRegressor':
            rg = SGDRegressor()
            Gridsearch = GridSearchCV(rg, {'loss': ('squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'), 
                                           'alpha':[1e-6,1e-7,1e-5]}, n_jobs = 8, cv = ps, scoring='neg_median_absolute_error')
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
        
        if y_col=='dBIS':
            y_predicted = rg.predict(X_test)*100
        elif y_col=='dMAP':
            y_predicted = rg.predict(X_test)*150
    
    
    col_name = 'pred_' + y_col + '_' + name_rg
    if y_col=='dBIS':
        Test_data_BIS['true_' + y_col] = Patients_test[y_col]
        Test_data_BIS['pred_' + y_col] = y_predicted
        results_BIS[col_name] = y_predicted
    else:
        Test_data_MAP['true_' + y_col] = Patients_test[y_col]
        Test_data_MAP['pred_' + y_col] = y_predicted
        results_MAP[col_name] = y_predicted
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
        
        if y_col=='dBIS':
            y_predicted_train = rg.predict(X_train)*100
        elif y_col=='dMAP':
            y_predicted_train = rg.predict(X_train)*150
    
    if y_col=='dBIS':
        Train_data_BIS['true_' + y_col] = Patients_train[y_col]
        Train_data_BIS['pred_' + y_col] = y_predicted_train 
    else:
        Train_data_MAP['true_' + y_col] = Patients_train[y_col]
        Train_data_MAP['pred_' + y_col] = y_predicted_train 
        
    # plot_surface(rg, scaler)
    
results_BIS.to_csv("./results_BIS.csv")
results_MAP.to_csv("./results_MAP.csv")

print('     ***-----------------' + name_rg + '-----------------***')
print("\n                 ------ Test Results ------")
max_case_bis, min_case_bis= compute_metrics(Test_data_BIS)
max_case_map, min_case_map = compute_metrics(Test_data_MAP)
print("\n\n                 ------ Train Results ------")
compute_metrics(Train_data_BIS)
compute_metrics(Train_data_MAP)
plot_dresults(Test_data_BIS, Test_data_MAP, Train_data_BIS, Train_data_MAP)

# plot_case(results_BIS, results_MAP, Patients_test_full, min_case_bis, min_case_map, max_case_bis, max_case_map)


