"""
Created on Mon Jul  4 11:10:13 2022

@author: aubouinb
"""

import csv
import pickle

import pandas as pd
import numpy as np

Patients_test = pd.read_csv("./data/Patients_test.csv")
Patients_test = Patients_test[['caseid', 'Time', 'BIS', 'SQI', 'Propofol',
                              'Remifentanil', 'age', 'sex', 'weight', 'height', 'full_BIS']]
Patients_test.rename({'full_BIS': 'full'}, axis=1, inplace=True)
# scale the inputs
Patients_test['Propofol'] = Patients_test['Propofol']*20/360
Patients_test['Remifentanil'] = Patients_test['Remifentanil']*20/360
Patients_test['SQI'].fillna(0, inplace=True)
Patients_test.to_csv("./lee_code/data.csv", index=False)


# %% Preprocessing the data
# parameters
timepoints = 180
cache_path = "./lee_code/cache_perso.var"

# generate sequence data

test_p = {}  # by case
test_r = {}
test_c = {}
test_y = {}

# load raw data
f = open('./lee_code/data.csv', 'rt')
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
        case_p = []
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
