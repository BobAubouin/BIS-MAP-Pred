#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 18:03:46 2022

@author: aubouinb
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from metrics_functions import compute_metrics, plot_results, plot_case, plot_surface_tensor


class RNNreg(nn.Module):
    def __init__(self, input_size, RNN_hidden_layer = 512):
        super(RNNreg, self).__init__()
        
        self.input_size = input_size
        self.RNN_hidden_layer = RNN_hidden_layer
        
        #linear
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, RNN_hidden_layer),
            nn.LeakyReLU(),
            # nn.Linear(RNN_hidden_layer, RNN_hidden_layer),
            # nn.ReLU(),
            nn.Linear(RNN_hidden_layer, 1),
            nn.Tanh())
        
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



#%% Load dataset
Patients_train = pd.read_csv("./Patients_train.csv", index_col=0)
Patients_test = pd.read_csv("./Patients_test.csv", index_col=0)

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


#%%


filename = './saved_reg/reg_' + 'RNN' + '_feat_' + feature + '.pkl'
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
    elif y_col =='MAP':
        Patients_train = Patients_train_MAP
        Patients_test = Patients_test_MAP
        
    #--------------Training-------------
 
    try: #Try to load trained regressor
        regressors = pickle.load(open(filename, 'rb'))
        rg = regressors[y_col]
        print("load ok")
        
    except: #Otherwhise train the regressors and save it
        Y_train = torch.tensor(Patients_train[y_col].to_numpy()).reshape(-1, 1)
        X_train = Patients_train[X_col].to_numpy()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 
        X_train = torch.tensor(X_train.astype(np.float32))
        ps = PredefinedSplit(Patients_train['train_set'].values)
        
        net = NeuralNetRegressor(RNNreg,
                                 module__input_size = len(X_col),
                                 module__RNN_hidden_layer = 512,
                                 max_epochs=500,
                                 optimizer=optim.Adam,
                                 lr=1e-4,                      
                                 verbose=0)

        params = {'module__RNN_hidden_layer': [1024, 2048], 'max_epochs' : [1000], 'lr' : 10**np.linspace(-8,-6,2)}
        
        Gridsearch = GridSearchCV(net, params, cv = ps, n_jobs = 8) #,'relu','logistic','identity'
        
        if y_col=='BIS':
            Gridsearch.fit(X_train, Y_train.float()/50 - 1 )
        elif y_col=='MAP':
            Gridsearch.fit(X_train, Y_train.float()/75 - 1)

        print(Gridsearch.best_params_)
        
        rg = Gridsearch.best_estimator_
        regressors[y_col] = rg
        pickle.dump(regressors, open(filename, 'wb'))
    
    #--------------test performances on test cases-------------
    
    X_train = Patients_train[X_col].values
    scaler = StandardScaler()
    scaler.fit(X_train) 
    
    X_test = Patients_test[X_col].values
    X_test = scaler.transform(X_test)
    X_test = torch.tensor(X_test.astype(np.float32))
    if y_col=='BIS':
        y_predicted = (rg.predict(X_test)+1)*50
    elif y_col=='MAP':
        y_predicted = (rg.predict(X_test)+1)*75
    
    
    col_name = 'pred_' + y_col + '_' + 'RNN'
    if y_col=='BIS':
        Test_data_BIS['true_' + y_col] = Patients_test[y_col]
        Test_data_BIS['pred_' + y_col] = y_predicted
        results_BIS.loc[:,col_name] = y_predicted
    else:
        Test_data_MAP['true_' + y_col] = Patients_test[y_col]
        Test_data_MAP['pred_' + y_col] = y_predicted
        results_MAP.loc[:,col_name] = y_predicted
    #-----------------test performances on train cases--------------------
    
    X_train = Patients_train[X_col].values
    X_train = scaler.transform(X_train)
    X_train = torch.tensor(X_train.astype(np.float32))
    if y_col=='BIS':
        y_predicted_train = (rg.predict(X_train)+1)*50
    elif y_col=='MAP':
        y_predicted_train = (rg.predict(X_train)+1)*75
    
    if y_col=='BIS':
        Train_data_BIS['true_' + y_col] = Patients_train[y_col]
        Train_data_BIS['pred_' + y_col] = y_predicted_train 
    else:
        Train_data_MAP['true_' + y_col] = Patients_train[y_col]
        Train_data_MAP['pred_' + y_col] = y_predicted_train 
        
    plot_surface_tensor(rg, scaler, feature)
    
# results_BIS.to_csv("./results_BIS.csv")
# results_MAP.to_csv("./results_MAP.csv")

print('     ***-----------------' + 'RNN' + '-----------------***')
print("\n                 ------ Test Results ------")
max_case_bis, min_case_bis= compute_metrics(Test_data_BIS)
max_case_map, min_case_map = compute_metrics(Test_data_MAP)
print("\n\n                 ------ Train Results ------")
# compute_metrics(Train_data_BIS)
# compute_metrics(Train_data_MAP)
# plot_results(Test_data_BIS, Test_data_MAP, Train_data_BIS, Train_data_MAP)

# plot_case(results_BIS, results_MAP, Patients_test_full, min_case_bis, min_case_map, max_case_bis, max_case_map)



