#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:10:13 2022

@author: aubouinb
"""

import csv
import pickle
from vitaldb_local import load_cases
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#%% Import data from vitalDB

caselist = [46, 70, 101, 167, 172, 218, 221, 247, 268, 345, 353, 405, 447, 533, 537, 544, 545, 585,
            593, 636, 663, 671, 672, 685, 711, 734, 751, 812, 827, 831, 835, 847, 866, 872, 894, 925, 
            926, 940, 952, 963, 1029, 1044, 1047, 1154, 1176, 1201, 1212, 1213, 1237, 1238, 1239, 1267,
            1339, 1376, 1392, 1396, 1398, 1404, 1416, 1440, 1602, 1611, 1613, 1657, 1658, 1662, 
            1687, 1690, 1918, 1925, 1933, 1994, 2000, 2014, 2029, 2049, 2051, 2072, 2074, 2139, 2148, 
            2157, 2196, 2229, 2238, 2309, 2379, 2382, 2392, 2409, 2442, 2479, 2480, 2497, 2500, 2511, 
            2527, 2528, 2542, 2562, 2569, 2572]

id_train, id_test = train_test_split(caselist, test_size=0.3, random_state=14) #split between test and train

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
max_val= {'BIS': 70,'MAP': 130, 'Propofol':1e3, 'Remifentanil': 1e3, "Ce_Prop": 1e3, "Ce_Rem": 1e3, "SQI": 100, 'HR':150}



cases.pop('index')
Patients_test = pd.DataFrame()
nb_points = 0
for caseid in id_test:
    print(caseid)
    Patient_df = cases[cases['caseid']==caseid]
    window_size = 100 # Mean window
    Patient_df = Patient_df.copy()
    Patient_df['BIS'] = Patient_df['BIS'].rolling(window_size, min_periods=5, center=True).mean().dropna()
    Patient_df['MAP'] = Patient_df['MAP'].rolling(window_size, min_periods=5, center=True).mean().dropna()
    Patient_df['HR'] = Patient_df['HR'].rolling(window_size, min_periods=5, center=True).mean().dropna()
    
    nb_points += len(Patient_df['BIS'])
    Patient_df.insert(1,"Time", np.arange(0,len(Patient_df['BIS'])))
    Patient_df.insert(len(Patient_df.columns),"age", float(perso_data[perso_data['caseid']==str(caseid)]['age']))
    sex = int(perso_data[perso_data['caseid']==str(caseid)]['sex']=='F') # F = 0, M = 1
    Patient_df.insert(len(Patient_df.columns),"sex", sex)
    Patient_df.insert(len(Patient_df.columns),"weight", float(perso_data[perso_data['caseid']==str(caseid)]['weight']))
    Patient_df.insert(len(Patient_df.columns),"height", float(perso_data[perso_data['caseid']==str(caseid)]['height']))    
    
    Patient_df = Patient_df.fillna(method='ffill')
    Patient_df = Patient_df.dropna().reset_index(drop=True) # Remove Nan
    
    Patient_df.insert(len(Patient_df.columns),"full", 0)
    for col in cols[:8]: 
        Patient_df.loc[(Patient_df[col]<=min_val[col]) | (Patient_df[col]>=max_val[col]), 'full'] = np.ones((len(Patient_df.loc[(Patient_df[col]<=min_val[col]) | (Patient_df[col]>=max_val[col]), 'full'])))

    Patient_df.loc[(Patient_df['BIS'].isna()) | (Patient_df['MAP'].isna()), 'full'] = np.ones((len(Patient_df.loc[(Patient_df['BIS'].isna()) | (Patient_df['MAP'].isna()), 'full'])))
    
    
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

                                                     
                                                    