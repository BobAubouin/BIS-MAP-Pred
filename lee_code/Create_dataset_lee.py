#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:10:13 2022

@author: aubouinb
"""

import csv
import pickle
import sys
import os

# Path = os.path.dirname(__file__)
# print(Path)
# Path = Path[:-9]
# print(Path)
# print(sys.path)
# sys.path.append(Path)
# print(sys.path)
from vitaldb_local import load_cases
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#%% Import data from vitalDB

caselist = [46, 70, 101, 167, 172, 218, 221, 247, 268, 345, 353, 405, 447, 533, 537, 544, 545, 585,
            593, 636, 663, 671, 672, 685, 711, 734, 751, 812, 827, 831, 835, 847, 866, 872, 894, 
            926, 940, 952, 963, 1029, 1044, 1047, 1154, 1176, 1201, 1212, 1213, 1237, 1238, 1239, 1267,
            1376, 1392, 1396, 1398, 1404, 1416, 1440, 1602, 1611, 1613, 1657, 1658, 1662, 
            1687, 1690, 1918, 1925, 1994, 2000, 2014, 2029, 2049, 2051, 2072, 2074, 2139, 2148, 
            2157, 2196, 2229, 2238, 2309, 2379, 2382, 2392, 2409, 2442, 2479, 2480, 2500, 2511, 
            2527, 2528, 2542, 2562, 2569, 2572, 2949, 2955, 2956, 2975, 3027, 3042, 3047, 3050, 
            3065, 3070, 3073, 3092, 3315, 3366, 3367, 3376, 3379, 3398, 3407, 3435, 3458, 3710, 3729,
            3791,3859, 4050, 4091, 4098, 4122, 4146, 4172, 4173, 4177, 4195, 4202, 4212, 4253,
            4277, 4292, 4350, 4375, 4387, 4432, 4472, 4547, 4673, 4678, 4716, 4741, 4745, 4789, 4803] #4768


id_train, id_test = train_test_split(caselist, test_size=0.3, random_state=4) #split between test and train

perso_data = pd.read_csv("../info_clinic_vitalDB.csv", decimal='.')


cases = load_cases(['BIS/BIS','Orchestra/PPF20_RATE','Orchestra/RFTN20_RATE',
                    'Orchestra/PPF20_CE','Orchestra/RFTN20_CE','Solar8000/ART_MBP',
                    'BIS/SQI','Solar8000/PLETH_HR','Orchestra/PPF20_CP',
                    'Orchestra/RFTN20_CP','Orchestra/RFTN20_VOL',
                    'Orchestra/PPF20_VOL','Solar8000/NIBP_MBP'], caseids=(caselist)) #load the case from vitalDB

cases.rename(columns = {'BIS/BIS':'BIS',
                        'Orchestra/PPF20_RATE':'Propofol',
                        'Orchestra/RFTN20_RATE':"Remifentanil",
                        'Orchestra/PPF20_CE':"Ce_Prop",
                        'Orchestra/RFTN20_CE':"Ce_Rem",
                        'Solar8000/ART_MBP':"MAP",
                        'BIS/SQI':"SQI",
                        'Solar8000/PLETH_HR':"HR",
                        'Orchestra/PPF20_CP':"Cp_Prop",
                        'Orchestra/RFTN20_CP':"Cp_Rem",
                        'Orchestra/RFTN20_VOL': 'Vol_Rem',
                        'Orchestra/PPF20_VOL': 'Vol_Prop',
                        'Solar8000/NIBP_MBP': 'NI_MAP'}, inplace = True)
cases = cases.reset_index()

cases['Propofol'] = cases['Propofol']*20/360
cases['Remifentanil'] = cases['Remifentanil']*20/360


cols = ['BIS','MAP','HR','Propofol','Remifentanil', "Ce_Prop", "Ce_Rem", "SQI", 'age', 'sex', 'height', 'weight', 'bmi']

min_val= {'BIS': 20,'MAP': 50, 'Propofol':0, 'Remifentanil': 0, "Ce_Prop": 0, "Ce_Rem": 0, "SQI": 50, 'HR':20} 
max_val= {'BIS': 100,'MAP': 130, 'Propofol':1e3, 'Remifentanil': 1e3, "Ce_Prop": 1e3, "Ce_Rem": 1e3, "SQI": 100, 'HR':150}



cases.pop('index')
Patients_test = pd.DataFrame()
nb_points = 0
for caseid in id_test:
    print(caseid)
    Patient_df = cases[cases['caseid']==caseid].copy().reset_index()
    Patient_df = Patient_df.copy()
    
    #replace nan by 0 in drug rates
    Patient_df['Propofol'].fillna(method='bfill', inplace=True)
    Patient_df['Remifentanil'].fillna(method='bfill', inplace=True)

    
    #find first drug injection
    istart = 0
    for i in range(len(Patient_df)):
        if Patient_df.loc[i, 'Propofol'] != 0 or Patient_df.loc[i, 'Remifentanil'] != 0:
            istart = i
            break
    #removed before strating of anesthesia
    Patient_df = Patient_df[istart:]
    Patient_df.reset_index(inplace=True)
    
    Patient_df['BIS'].replace(0, np.nan, inplace=True)
    Patient_df['MAP'].replace(0, np.nan, inplace=True)
    Patient_df['HR'].replace(0, np.nan, inplace=True)
    
    #remove artefact in map measure
    Patient_df.loc[abs(Patient_df['MAP']-np.nanmean(Patient_df['MAP'].values)) > 50, 'MAP'] = np.nan * np.ones((len(Patient_df.loc[abs(Patient_df['MAP']-np.nanmean(Patient_df['MAP'].values)) > 50, 'MAP'])))
    
    #remove bad quality point for BIS
    Patient_df.loc[Patient_df['SQI'] < 50, 'BIS'] = np.nan * np.ones((len(Patient_df.loc[Patient_df['SQI'] < 50, 'BIS'])))
    

    window_size = 500 # Mean window

    L = Patient_df['BIS'].to_numpy()
    for i in range(len(L)):
        if not np.isnan(L[i]):
            i_first_non_nan = i
            break
    
    L = np.concatenate((Patient_df.loc[i_first_non_nan,'BIS']*np.ones(500),L))
    L = pd.DataFrame(L)
    L = L.ewm(span=20, min_periods=1).mean()
    
    Patient_df.loc[:,'BIS'] = L[500:].to_numpy()
    
    Patient_df = Patient_df.fillna(method='ffill')
    Patient_df['Propofol'].fillna(0,inplace=True)
    Patient_df['Remifentanil'].fillna(0,inplace=True)
    
    Patient_df.insert(len(Patient_df.columns),"full", 0)
    Patient_df.loc[(Patient_df['BIS']<=min_val['BIS']) | (Patient_df['BIS']>=max_val['BIS']), 'full'] = np.ones((len(Patient_df.loc[(Patient_df['BIS']<=min_val['BIS']) | (Patient_df['BIS']>=max_val['BIS']), 'full'])))
    Patient_df.loc[Patient_df['BIS'].isna(), 'full'] = np.ones((len(Patient_df.loc[Patient_df['BIS'].isna(), 'full'])))
    
    
    nb_points += len(Patient_df['BIS'])
    Patient_df.insert(1,"Time", np.arange(0,len(Patient_df['BIS'])))
    Patient_df.insert(len(Patient_df.columns),"age", float(perso_data[perso_data['caseid']==str(caseid)]['age']))
    sex = int(perso_data[perso_data['caseid']==str(caseid)]['sex']=='F') # F = 0, M = 1
    Patient_df.insert(len(Patient_df.columns),"sex", sex)
    Patient_df.insert(len(Patient_df.columns),"weight", float(perso_data[perso_data['caseid']==str(caseid)]['weight']))
    Patient_df.insert(len(Patient_df.columns),"height", float(perso_data[perso_data['caseid']==str(caseid)]['height']))    
    
    Patient_df = Patient_df.dropna().reset_index(drop=True) # Remove Nan
    
    Patient_df.loc[(Patient_df['BIS'].isna()), 'full'] = np.ones((len(Patient_df.loc[(Patient_df['BIS'].isna()), 'full'])))
    
    
    Patients_test = pd.concat([Patients_test, Patient_df], ignore_index=True)
    
Patients_test = Patients_test[['caseid','Time','BIS','SQI','Propofol','Remifentanil','age','sex','weight','height','full']]
print("nb point tot: " +str(nb_points))
print("nb point test: " +str(len(Patients_test['BIS'])))
# Save Patients DataFrame    
Patients_test.to_csv("./data.csv", index = False)


#%% Preprocessing the data
# parameters
timepoints = 180
cache_path = "./cache_perso.var"

# generate sequence data

test_p = {} # by case
test_r = {}
test_c = {}
test_y = {}

# load raw data
f = open('data.csv', 'rt')
raw_data = {}
first = True
for row in csv.reader(f):
    id = row[0]
    if first:
        first = False
        continue
    if id not in raw_data:
        raw_data[id] = []
    rowvals = []
    for j in range(1, len(row)):
        rowvals.append(float(row[j]))
    raw_data[id].append(rowvals)

# make sequence
gaps = np.arange(0, 10 * (timepoints + 1), 10)
for id, case in raw_data.items():  # for each case
    print(id)
    ifirst = -1
    ilast = -1
    ppf_seq = []
    rft_seq = []

    is_test = True
    if is_test:
        case_p= []
        case_r = []
        case_c = []
        case_y = []

    for isamp in range(len(case)):
        row = case[isamp]

        bis = row[1] / 100  # normalize bis
        sqi = row[2]
        ppf_dose = row[3]
        rft_dose = row[4]
        age = row[5]
        sex = row[6]
        wt = row[7]
        ht = row[8]
        ppf_seq.append(ppf_dose)  # make time sequence
        rft_seq.append(rft_dose)

        if ifirst is None:  # before started
            if ppf_dose < 1:
                
                continue
            ifirst = isamp
        else:  # started
            if ppf_dose > 0.05:  # restarted
                ilast = isamp
        # if ilast is not None and isamp - ilast > gaps[-1]:
        #     count_ppf+=1
        #     break  # case finished
        full = row[9]
        if full:
            continue

        pvals = []
        rvals = []
        for i in reversed(range(timepoints)):
            istart = isamp + 1 - gaps[i + 1]
            iend = isamp + 1 - gaps[i]
            pvals.append(sum(ppf_seq[max(0, istart):max(0, iend)]))
            rvals.append(sum(rft_seq[max(0, istart):max(0, iend)]))

        
        case_p.append(pvals)
        case_r.append(rvals)
        case_c.append([age, sex, wt, ht])
        case_y.append(bis)

    test_p[id] = case_p
    test_r[id] = case_r
    test_c[id] = case_c
    test_y[id] = case_y
    
# save cache file
pickle.dump((test_p, test_r, test_c, test_y), open(cache_path, "wb"), protocol=4)

                                                     
                                                    