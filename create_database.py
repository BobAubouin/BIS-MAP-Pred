"""Create a simple dataset from vitalDB."""

# %% Import
import sys
import pandas as pd
import numpy as np
import python_anesthesia_simulator as pas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
# local
try:
    sys.path.append('/home/aubouinb/ownCloud/Anesthesie/Science/Bob/Code/utilities')
    from vitaldb_local import load_cases

except:
    print('Could not import vitaldb_local, import online version')
    import vitaldb as vdb

    def load_cases(track_names: list, caseids: list):
        """Import a list of cases from vitaldb in a dataframe format."""
        dataframe_final = pd.DataFrame()
        for caseid in caseids:
            cases = vdb.VitalFile(caseid, track_names)
            dataframe_temp = cases.to_pandas(track_names, 1)
            dataframe_temp.insert(0, 'caseid', caseid)
            dataframe_final = pd.concat([dataframe_final, dataframe_temp], ignore_index=True)
        return dataframe_final

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# %% Load data
perso_data = pd.read_csv("./data/info_clinic_vitalDB.csv", decimal='.')  # personal data


# id_list


with open('./data/caseid_list.txt', 'r') as f:
    caselist = f.read().splitlines()

caselist = [int(i) for i in caselist]

# import the cases
cases = load_cases(['BIS/BIS', 'Orchestra/PPF20_RATE', 'Orchestra/RFTN20_RATE',
                    'Orchestra/PPF20_CE', 'Orchestra/RFTN20_CE', 'Solar8000/ART_MBP',
                    'BIS/SQI', 'Solar8000/PLETH_HR', 'Orchestra/PPF20_CP',
                    'Orchestra/RFTN20_CP', 'Orchestra/RFTN20_VOL',
                    'Orchestra/PPF20_VOL', 'Solar8000/NIBP_MBP',
                    'Orchestra/PPF20_CT', 'Orchestra/RFTN20_CT'], caseids=caselist)  # load the case from vitalDB

cases.rename(columns={'BIS/BIS': 'BIS',
                      'Orchestra/PPF20_RATE': 'Propofol',
                      'Orchestra/RFTN20_RATE': "Remifentanil",
                      'Orchestra/PPF20_CE': "Ce_Prop",
                      'Orchestra/RFTN20_CE': "Ce_Rem",
                      'Solar8000/ART_MBP': "MAP",
                      'BIS/SQI': "SQI",
                      'Solar8000/PLETH_HR': "HR",
                      'Orchestra/PPF20_CP': "Cp_Prop",
                      'Orchestra/RFTN20_CP': "Cp_Rem",
                      'Orchestra/RFTN20_VOL': 'Vol_Rem',
                      'Orchestra/PPF20_VOL': 'Vol_Prop',
                      'Orchestra/PPF20_CT': 'Target_Propo',
                      'Orchestra/RFTN20_CT': 'Target_Remi',
                      'Solar8000/NIBP_MBP': 'NI_MAP'}, inplace=True)
# define bound for the values
cols = ['BIS', 'MAP', 'HR', 'Propofol', 'Remifentanil', "Ce_Prop",
        "Ce_Rem", "SQI", 'age', 'gender', 'height', 'weight', 'bmi']

min_val = {'BIS': 10, 'MAP': 50, 'Propofol': 0, 'Remifentanil': 0, "Ce_Prop": 0, "Ce_Rem": 0, "SQI": 50}
max_val = {'BIS': 100, 'MAP': 160, 'Propofol': 1e3, 'Remifentanil': 1e3, "Ce_Prop": 1e3, "Ce_Rem": 1e3, "SQI": 100}

# %%
Full_data = pd.DataFrame()

for caseid, Patient_df in cases.groupby('caseid'):
    print(caseid)
    # find MAP baseline
    Patient_df = Patient_df.copy()
    Patient_df.reset_index(inplace=True)
    Patient_df.drop(columns=['index'], inplace=True)
    Map_base_case = Patient_df['NI_MAP'].fillna(method='bfill')[0]
    Patient_df.insert(len(Patient_df.columns), "MAP_base_case", Map_base_case)

    # compute median HR
    median_window = 600
    Patient_df.loc[:, 'mean_HR'] = Patient_df.loc[:, 'HR'].rolling(
        median_window, min_periods=1, center=False).apply(np.nanmedian)

    # replace nan by previous value in drug rates
    Patient_df['Propofol'].fillna(method='bfill', inplace=True)
    Patient_df['Remifentanil'].fillna(method='bfill', inplace=True)

    # find first drug injection
    istart = 0
    for i in range(len(Patient_df)):
        if Patient_df.loc[i, 'Propofol'] != 0 or Patient_df.loc[i, 'Remifentanil'] != 0:
            istart = max(0, i-60)  # start 1 min before
            break
    # removed before starting of anesthesia
    Patient_df = Patient_df[istart:]
    Patient_df.reset_index(inplace=True)

    # replace 0 by nan in BIS, MAP and HR
    Patient_df['BIS'].replace(0, np.nan, inplace=True)
    Patient_df['MAP'].replace(0, np.nan, inplace=True)
    Patient_df['HR'].replace(0, np.nan, inplace=True)

    # remove artefact in map measure
    Patient_df.loc[abs(Patient_df['MAP']-np.nanmean(Patient_df['MAP'].values)) > 50, 'MAP'] = np.nan * \
        np.ones((len(Patient_df.loc[abs(Patient_df['MAP']-np.nanmean(Patient_df['MAP'].values)) > 50, 'MAP'])))

    # remove bad quality point for BIS
    Patient_df.loc[Patient_df['SQI'] < 50, 'BIS'] = np.nan * \
        np.ones((len(Patient_df.loc[Patient_df['SQI'] < 50, 'BIS'])))

    window_size = 30  # Mean window

    # fig, ax = plt.subplots()
    # Patient_df['BIS'].plot(ax = ax)

    # fig2, ax2 = plt.subplots()
    # Patient_df.loc[1000:1500,'BIS'].plot(ax = ax2)

    L = Patient_df['BIS'].to_numpy()
    for i in range(len(L)):
        if not np.isnan(L[i]):
            i_first_non_nan = i
            break

    L = np.concatenate((Patient_df.loc[i_first_non_nan, 'BIS']*np.ones(500), L))
    L = pd.DataFrame(L)
    L = L.ewm(span=20, min_periods=1).mean()

    Patient_df.loc[:, 'BIS'] = L[500:].to_numpy()

    Patient_df.loc[:, 'MAP'] = Patient_df['MAP'].ewm(span=20, min_periods=1).mean()

    Patient_df = Patient_df.fillna(method='ffill')

    Patient_df.insert(len(Patient_df.columns), "full_BIS", 0)
    Patient_df.insert(len(Patient_df.columns), "full_MAP", 0)

    Patient_df.loc[(Patient_df['BIS'] <= min_val['BIS']) | (Patient_df['BIS'] >= max_val['BIS']), 'full_BIS'] = np.ones(
        (len(Patient_df.loc[(Patient_df['BIS'] <= min_val['BIS']) | (Patient_df['BIS'] >= max_val['BIS']), 'full_BIS'])))

    Patient_df.loc[(Patient_df['MAP'] <= min_val['MAP']) | (Patient_df['MAP'] >= max_val['MAP']), 'full_MAP'] = np.ones(
        (len(Patient_df.loc[(Patient_df['MAP'] <= min_val['MAP']) | (Patient_df['MAP'] >= max_val['MAP']), 'full_MAP'])))

    Patient_df.loc[Patient_df['BIS'].isna(), 'full_BIS'] = np.ones(
        (len(Patient_df.loc[Patient_df['BIS'].isna(), 'full_BIS'])))
    Patient_df.loc[Patient_df['MAP'].isna(), 'full_MAP'] = np.ones(
        (len(Patient_df.loc[Patient_df['MAP'].isna(), 'full_MAP'])))

    # Add time column
    Patient_df.insert(1, "Time", np.arange(0, len(Patient_df['BIS'])))

    # Add personnal information to the dataframe
    age = perso_data.loc[perso_data['caseid'] == str(caseid), 'age'].astype(float).item()
    Patient_df.insert(len(Patient_df.columns), "age", age)
    gender = (perso_data[perso_data['caseid'] == str(caseid)]['sex'] == 'M').astype(int).item()  # F = 0, M = 1
    Patient_df.insert(len(Patient_df.columns), "gender", gender)
    weight = perso_data.loc[perso_data['caseid'] == str(caseid), 'weight'].astype(float).item()
    Patient_df.insert(len(Patient_df.columns), "weight", weight)
    height = perso_data.loc[perso_data['caseid'] == str(caseid), 'height'].astype(float).item()
    Patient_df.insert(len(Patient_df.columns), "height", height)
    bmi = perso_data.loc[perso_data['caseid'] == str(caseid), 'bmi'].astype(float).item()
    Patient_df.insert(len(Patient_df.columns), "bmi", bmi)

    # Use the Python Anesthesia Simulator (PAS) to simulate the pharmacokinetics of propofol and remifentanil
    model = "Eleveld"  # Eleveld model
    Patient_simu = pas.Patient([age, height, weight, gender], model_propo=model, model_remi=model)
    Patient_df.insert(len(Patient_df.columns), "lbm", Patient_simu.lbm)
    df_simu = Patient_simu.full_sim(u_propo=Patient_df['Propofol']*20/3600,
                                    u_remi=Patient_df['Remifentanil']*20/3600)

    Patient_df["Cp_Prop_Eleveld"] = df_simu['x_propo_1']
    Patient_df["Cp_Rem_Eleveld"] = df_simu['x_remi_1']
    Patient_df["Ce_Prop_Eleveld"] = df_simu['x_propo_4']
    Patient_df["Ce_Rem_Eleveld"] = df_simu['x_remi_4']
    Patient_df["Ce_Prop_MAP_Eleveld"] = (df_simu['x_propo_5'] + df_simu['x_propo_6'])/2
    Patient_df["Ce_Rem_MAP_Eleveld"] = df_simu['x_remi_5']

    # add delayed output
    Patient_df.insert(len(Patient_df.columns), "BIS_plus_30", Patient_df['BIS'].shift(-30))
    Patient_df.insert(len(Patient_df.columns), "BIS_plus_120", Patient_df['BIS'].shift(-120))
    Patient_df.insert(len(Patient_df.columns), "BIS_plus_300", Patient_df['BIS'].shift(-300))
    Patient_df.insert(len(Patient_df.columns), "BIS_plus_600", Patient_df['BIS'].shift(-600))
    Patient_df.insert(len(Patient_df.columns), "MAP_plus_30", Patient_df['MAP'].shift(-30))
    Patient_df.insert(len(Patient_df.columns), "MAP_plus_120", Patient_df['MAP'].shift(-120))
    Patient_df.insert(len(Patient_df.columns), "MAP_plus_300", Patient_df['MAP'].shift(-300))
    Patient_df.insert(len(Patient_df.columns), "MAP_plus_600", Patient_df['MAP'].shift(-600))

    Full_data = pd.concat([Full_data, Patient_df], ignore_index=True)


# Save Patients DataFrame
Full_data.drop(columns=['index'], inplace=True)
Full_data.to_csv("./data/full_data.csv")
