#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:57:35 2022

@author: aubouinb
"""

"""This script is used to create a simple dataset from vitalDB"""

import pandas as pd
import numpy as np
from prepare_data_functions import formate_patient_data
from vitaldb_local import load_cases
from scipy import signal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


perso_data = pd.read_csv("./info_clinic_vitalDB.csv", decimal='.')
perso_data = formate_patient_data(perso_data)


# id_list


caselist = [46, 70, 101, 167, 172, 218, 221, 247, 268, 345, 353, 405, 447, 533, 537, 544, 545, 585,
            593, 636, 663, 671, 672, 685, 711, 734, 751, 812, 827, 831, 835, 847, 866, 872, 894, 925, 
            926, 940, 952, 963, 1029, 1044, 1047, 1154, 1176, 1201, 1212, 1213, 1237, 1238, 1239, 1267,
            1339, 1376, 1392, 1396, 1398, 1404, 1416, 1440, 1602, 1611, 1613, 1657, 1658, 1662, 
            1687, 1690, 1918, 1925, 1933, 1994, 2000, 2014, 2029, 2049, 2051, 2072, 2074, 2139, 2148, 
            2157, 2196, 2229, 2238, 2309, 2379, 2382, 2392, 2409, 2442, 2479, 2480, 2497, 2500, 2511, 
            2527, 2528, 2542, 2562, 2569, 2572]

id_train, id_test = train_test_split(caselist, test_size=0.3, random_state=14) #split between test and train
id_train = np.random.permutation(id_train)
id_split = np.array_split(id_train, 3) #split train set in 5 set for cross validation



#import the cases
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
#define bound for the values
cols = ['BIS','MAP','HR','Propofol','Remifentanil', "Ce_Prop", "Ce_Rem", "SQI", 'age', 'sex', 'height', 'weight', 'bmi']

min_val= {'BIS': 20,'MAP': 50, 'Propofol':0, 'Remifentanil': 0, "Ce_Prop": 0, "Ce_Rem": 0, "SQI": 50, 'HR':20} 
max_val= {'BIS': 70,'MAP': 130, 'Propofol':1e3, 'Remifentanil': 1e3, "Ce_Prop": 1e3, "Ce_Rem": 1e3, "SQI": 100, 'HR':150}

#%%
Patients_train = pd.DataFrame()
Patients_test = pd.DataFrame()

nb_points = 0
hist_Cp = 10*60
windows_Cp = 30
win_vec = np.ones(windows_Cp)

for caseid in cases['caseid'].unique():
    print(caseid)
    Patient_df = cases[cases['caseid']==caseid]
    Patient_df = Patient_df.copy()
    Patient_df['BIS'].replace(0, np.nan, inplace=True)
    Patient_df['MAP'].replace(0, np.nan, inplace=True)
    Patient_df['HR'].replace(0, np.nan, inplace=True)
    Patient_df = Patient_df.fillna(method='ffill')
    window_size = 300 # Mean window
    
    fig, ax = plt.subplots()
    Patient_df['BIS'].plot(ax = ax)
    Patient_df.loc[:,'BIS'] = Patient_df['BIS'].rolling(window_size, min_periods=5, center=True).mean().dropna()
    Patient_df['BIS'].plot(ax = ax)
    plt.title('case = ' + str(caseid))
    plt.show()
    Patient_df.loc[:,'MAP'] = Patient_df['MAP'].rolling(window_size, min_periods=5, center=True).mean().dropna()
    Patient_df.loc[:,'HR'] = Patient_df['HR'].rolling(window_size, min_periods=5, center=True).mean().dropna()
    
    nb_points += len(Patient_df['BIS'])
    Patient_df.insert(1,"Time", np.arange(0,len(Patient_df['BIS'])))
    Patient_df.insert(len(Patient_df.columns),"age", float(perso_data[perso_data['caseid']==str(caseid)]['age']))
    sex = int(perso_data[perso_data['caseid']==str(caseid)]['sex']=='M') # F = 0, M = 1
    Patient_df.insert(len(Patient_df.columns),"sex", sex)
    weight = float(perso_data[perso_data['caseid']==str(caseid)]['weight'])
    Patient_df.insert(len(Patient_df.columns),"weight", weight)
    height = float(perso_data[perso_data['caseid']==str(caseid)]['height'])
    Patient_df.insert(len(Patient_df.columns),"height", height)
    Patient_df.insert(len(Patient_df.columns),"bmi", float(perso_data[perso_data['caseid']==str(caseid)]['bmi']))
    
    if sex == 1: # homme
        lbm = 1.1 * weight - 128 * (weight / height) ** 2
    else : #femme
        lbm = 1.07 * weight - 148 * (weight / height) ** 2
    Patient_df.insert(len(Patient_df.columns),"lbm", lbm)
    
    Map_base_case = Patient_df['NI_MAP'].fillna(method='bfill')[0]
    Patient_df.insert(len(Patient_df.columns),"MAP_base_case", Map_base_case)
    
    
    
    Ncase = len(Patient_df['BIS'])

    Ce_Prop_MAP = np.zeros(Ncase)
    Ce_Rem_MAP = np.zeros(Ncase)
    ke_prop =  (0.0540 + 0.0695)/2 / 60 
    ke_rem = 0.81 / 60 
    dt = 1
    for j in range(Ncase-1):
        if not np.isnan(Patient_df.loc[j,"Cp_Prop"]):
            Ce_Prop_MAP[j+1] = Ce_Prop_MAP[j] + dt * ke_prop * (Patient_df.loc[j,"Cp_Prop"] - Ce_Prop_MAP[j])
        else:
            Ce_Prop_MAP[j+1] = Ce_Prop_MAP[j] + dt * ke_prop * (0 - Ce_Prop_MAP[j])
        if not np.isnan(Patient_df.loc[j,"Cp_Prop"]):    
            Ce_Rem_MAP[j+1] = Ce_Rem_MAP[j] + dt * ke_rem * (Patient_df.loc[j,"Cp_Rem"] - Ce_Rem_MAP[j])
        else:
            Ce_Rem_MAP[j+1] = Ce_Rem_MAP[j] + dt * ke_rem * (0 - Ce_Rem_MAP[j])
   
    

    Patient_df.insert(len(Patient_df.columns), "Ce_Prop_MAP", Ce_Prop_MAP)
    Patient_df.insert(len(Patient_df.columns), "Ce_Rem_MAP", Ce_Prop_MAP)
    
    Patient_df.loc[:,"Ce_Prop"] = Patient_df["Ce_Prop"].fillna(value = 0)
    Patient_df.loc[:,"Ce_Rem"] = Patient_df["Ce_Rem"].fillna(value = 0)
    
    Cp_Prop = Patient_df["Cp_Prop"].fillna(value = 0)
    Cp_Rem = Patient_df["Cp_Rem"].fillna(value = 0)
    
    Cp_Prop = np.concatenate((np.zeros(hist_Cp), Cp_Prop))
    Cp_Rem = np.concatenate((np.zeros(hist_Cp), Cp_Rem))
    
    Cp_Prop_filtered = signal.convolve(Cp_Prop, win_vec, mode='same') / windows_Cp
    Cp_Rem_filtered = signal.convolve(Cp_Rem, win_vec, mode='same') / windows_Cp
    
    for i in range(int(hist_Cp/windows_Cp)):
        Patient_df.insert(len(Patient_df.columns),"Cp_Prop_" + str(i+1), Cp_Prop_filtered[hist_Cp-2-i*windows_Cp:-2-i*windows_Cp])
        Patient_df.insert(len(Patient_df.columns),"Cp_Rem_" + str(i+1), Cp_Rem_filtered[hist_Cp-2-i*windows_Cp:-2-i*windows_Cp])
    
    Patient_df.insert(len(Patient_df.columns),"full", 0)
    for col in cols[:8]: 
        Patient_df.loc[(Patient_df[col]<=min_val[col]) | (Patient_df[col]>=max_val[col]), 'full'] = np.ones((len(Patient_df.loc[(Patient_df[col]<=min_val[col]) | (Patient_df[col]>=max_val[col]), 'full'])))

    Patient_df.loc[(Patient_df['BIS'].isna()) | (Patient_df['MAP'].isna()), 'full'] = np.ones((len(Patient_df.loc[(Patient_df['BIS'].isna()) | (Patient_df['MAP'].isna()), 'full'])))
    
    if caseid in id_test:
        Patients_test = pd.concat([Patients_test, Patient_df], ignore_index=True)
    else:
        for i in range(len(id_split)):
            if caseid in id_split[i]:
                set_int = i
                break
        Patient_df.insert(len(Patient_df.columns),"train_set", set_int)
        Patients_train = pd.concat([Patients_train, Patient_df], ignore_index=True)



# Save Patients DataFrame
Patients_train.to_csv("./Patients_train.csv")
Patients_test.to_csv("./Patients_test.csv")

#Print stats
print("nb point tot: " +str(nb_points))
print("nb point train: " +str(len(Patients_train['BIS'])))
print("nb point test: " +str(len(Patients_test['BIS'])))


print_dist = True
if print_dist:
    fig, axs = plt.subplots(2,2)
    Patients_test['age'].hist(label = "test", ax=axs[0,0])
    Patients_train['age'].hist(label = "train", alpha=0.5, ax=axs[0,0])
    axs[0,0].set_title('age')
    axs[0,0].legend()
    
    Patients_test['weight'].hist(label = "test", ax=axs[0,1])
    Patients_train['weight'].hist(label = "train", alpha=0.5, ax=axs[0,1])
    axs[0,1].set_title('weight')

    
    Patients_test['height'].hist(label = "test", ax=axs[1,0])
    Patients_train['height'].hist(label = "train", alpha=0.5, ax=axs[1,0])
    axs[1,0].set_title('height')

    data = []
    for i in range(len(Patients_test['sex'])):
        data.append(('M'*Patients_test['sex'][i] + 'F'*(1 - Patients_test['sex'][i])))
    print_df = pd.DataFrame(data, columns=['sex'])
    print_df['sex'].hist(label = "test", ax=axs[1,1])
    data = []
    for i in range(len(Patients_train['sex'])):
        data.append(('M'*Patients_train['sex'][i] + 'F'*(1 - Patients_train['sex'][i])))
    print_df = pd.DataFrame(data, columns=['sex'])
    print_df['sex'].hist(label = "train", alpha=0.5, ax=axs[1,1])
    axs[1,1].set_title('sex')
    fig.tight_layout()
    plt.show()
