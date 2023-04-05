"""Create a simple dataset from vitalDB."""


import pandas as pd
import numpy as np
from model import PropoModel, RemiModel, discretize
import vitaldb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def formate_patient_data(data):
    """Format the data extracted from the .csv file to be able to read them as integers."""
    for s in ['age', 'height', 'weight']:
        if s == 'age':
            data[s] = data[s].str.replace(r'.', '0', regex=False)
        else:
            data[s] = data[s].str.replace(r'. ', '0', regex=False)
        data[s] = data[s].str.replace(r' %', '0', regex=False)
        data[s] = data[s].str.replace(r'0%', '0', regex=False)
        data[s] = data[s].astype(float)
    return data


def load_cases(track_names: list, caseids: list):
    """Import a list of cases from vitaldb in a dataframe format."""
    dataframe_final = pd.DataFrame()
    for caseid in caseids:
        cases = vitaldb.VitalFile(caseid, track_names)
        dataframe_temp = cases.to_pandas(track_names, 1)
        dataframe_temp.insert(0, 'caseid', caseid)
        dataframe_final = pd.concat([dataframe_final, dataframe_temp], ignore_index=True)
    return dataframe_final


# %% Load data
perso_data = pd.read_csv("./info_clinic_vitalDB.csv", decimal='.')
perso_data = formate_patient_data(perso_data)


# id_list


caselist = [46, 70, 101, 167, 172, 218, 221, 247, 268, 345, 353, 405, 447, 533, 537, 544, 545, 585,
            593, 636, 663, 671, 672, 685, 711, 734, 751, 812, 827, 831, 835, 847, 866, 872, 894,
            926, 940, 952, 963, 1029, 1044, 1047, 1154, 1176, 1201, 1212, 1213, 1237, 1238, 1239, 1267,
            1376, 1392, 1396, 1398, 1404, 1416, 1440, 1602, 1611, 1613, 1657, 1658, 1662,
            1687, 1690, 1918, 1925, 1994, 2000, 2014, 2029, 2049, 2051, 2072, 2074, 2139, 2148,
            2157, 2196, 2229, 2238, 2309, 2379, 2382, 2392, 2409, 2442, 2479, 2480, 2500, 2511,
            2527, 2528, 2542, 2562, 2569, 2572, 2949, 2955, 2956, 2975, 3027, 3042, 3047, 3050,
            3065, 3070, 3073, 3092, 3315, 3366, 3367, 3376, 3379, 3398, 3407, 3435, 3458, 3710, 3729,
            3791, 3859, 4050, 4091, 4098, 4122, 4146, 4172, 4173, 4177, 4195, 4202, 4212, 4253,
            4277, 4292, 4350, 4375, 4387, 4432, 4472, 4547, 4673, 4678, 4716, 4741, 4745, 4789, 4803]  # 4768


id_train, id_test = train_test_split(caselist, test_size=0.3, random_state=4)  # split between test and train
id_train = np.random.permutation(id_train)
id_split = np.array_split(id_train, 3)  # split train set in 5 set for cross validation


# import the cases
cases = load_cases(['BIS/BIS', 'Orchestra/PPF20_RATE', 'Orchestra/RFTN20_RATE',
                    'Orchestra/PPF20_CE', 'Orchestra/RFTN20_CE', 'Solar8000/ART_MBP',
                    'BIS/SQI', 'Solar8000/PLETH_HR', 'Orchestra/PPF20_CP',
                    'Orchestra/RFTN20_CP', 'Orchestra/RFTN20_VOL',
                    'Orchestra/PPF20_VOL', 'Solar8000/NIBP_MBP'], caseids=caselist)  # load the case from vitalDB

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
                      'Solar8000/NIBP_MBP': 'NI_MAP'}, inplace=True)
# define bound for the values
cols = ['BIS', 'MAP', 'HR', 'Propofol', 'Remifentanil', "Ce_Prop",
        "Ce_Rem", "SQI", 'age', 'sex', 'height', 'weight', 'bmi']

min_val = {'BIS': 10, 'MAP': 50, 'Propofol': 0, 'Remifentanil': 0, "Ce_Prop": 0, "Ce_Rem": 0, "SQI": 50}
max_val = {'BIS': 100, 'MAP': 160, 'Propofol': 1e3, 'Remifentanil': 1e3, "Ce_Prop": 1e3, "Ce_Rem": 1e3, "SQI": 100}

# %%
Patients_train = pd.DataFrame()
Patients_test = pd.DataFrame()

nb_points = 0
hist_Cp = 10*60
windows_Cp = 30
win_vec = np.ones(windows_Cp)

for caseid in cases['caseid'].unique():
    print(caseid)
    Patient_df = cases[cases['caseid'] == caseid]
    Patient_df = Patient_df.copy()

    # find MAP baseline
    Map_base_case = Patient_df['NI_MAP'].fillna(method='bfill')[0]
    Patient_df.insert(len(Patient_df.columns), "MAP_base_case", Map_base_case)

    # compute median HR
    median_window = 600
    Patient_df.loc[:, 'mean_HR'] = Patient_df.loc[:, 'HR'].rolling(
        median_window, min_periods=1, center=False).apply(np.nanmedian)

    # replace nan by 0 in drug rates
    Patient_df['Propofol'].fillna(method='bfill', inplace=True)
    Patient_df['Remifentanil'].fillna(method='bfill', inplace=True)

    # find first drug injection
    istart = 0
    for i in range(len(Patient_df)):
        if Patient_df.loc[i, 'Propofol'] != 0 or Patient_df.loc[i, 'Remifentanil'] != 0:
            istart = i
            break
    # removed before strating of anesthesia
    Patient_df = Patient_df[istart:]
    Patient_df.reset_index(inplace=True)

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

    # Patient_df.loc[1000:1500,'BIS'].plot(ax = ax2)
    # plt.title('case = ' + str(caseid))
    # plt.show()

    # Patient_df['BIS'].plot(ax = ax)
    # plt.title('case = ' + str(caseid))
    # plt.show()

    Patient_df.loc[:, 'HR'] = Patient_df['HR'].rolling(window_size, min_periods=1, center=True).apply(np.nanmean)

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

    Patient_df.insert(len(Patient_df.columns), "med_BIS", np.nan)
    Patient_df.insert(len(Patient_df.columns), "med_MAP", np.nan)

    Patient_df.loc[:, 'med_BIS'] = Patient_df.loc[:, 'BIS'].rolling(median_window, center=False).median()
    Patient_df.loc[:, 'med_MAP'] = Patient_df.loc[:, 'MAP'].rolling(median_window, center=False).median()

    # Patient_df.insert(len(Patient_df.columns),"mean_HR", np.nanmedian(Patient_df.loc[:15*60, 'HR']))

    # find first MAP non Nan
    # for i in range(len(Patient_df)):
    #     if not np.isnan(Patient_df.loc[i,"MAP"]):
    #         first_map = i
    #         break
    # # find first BIS non Nan
    # for i in range(len(Patient_df)):
    #     if not np.isnan(Patient_df.loc[i,"BIS"]):
    #         first_bis = i
    #         break
    # median_window = 600

    # Patient_df.insert(len(Patient_df.columns),"med_BIS", np.nanmedian(Patient_df.loc[first_bis:median_window + first_bis,'BIS']))
    # Patient_df.insert(len(Patient_df.columns),"med_MAP", np.nanmedian(Patient_df.loc[first_map :median_window + first_map,'MAP']))

    # Patient_df.loc[:median_window + first_bis,'med_BIS'] = np.nan*np.ones(median_window + first_bis + 1)
    # Patient_df.loc[:median_window + first_map,'med_MAP'] = np.nan*np.ones(median_window + first_map + 1)

    nb_points += len(Patient_df['BIS'])
    Patient_df.insert(1, "Time", np.arange(0, len(Patient_df['BIS'])))
    age = float(perso_data[perso_data['caseid'] == str(caseid)]['age'])
    Patient_df.insert(len(Patient_df.columns), "age", age)
    sex = int(perso_data[perso_data['caseid'] == str(caseid)]['sex'] == 'M')  # F = 0, M = 1
    Patient_df.insert(len(Patient_df.columns), "sex", sex)
    weight = float(perso_data[perso_data['caseid'] == str(caseid)]['weight'])
    Patient_df.insert(len(Patient_df.columns), "weight", weight)
    height = float(perso_data[perso_data['caseid'] == str(caseid)]['height'])
    Patient_df.insert(len(Patient_df.columns), "height", height)
    Patient_df.insert(len(Patient_df.columns), "bmi", float(perso_data[perso_data['caseid'] == str(caseid)]['bmi']))

    if sex == 1:  # homme
        lbm = 1.1 * weight - 128 * (weight / height) ** 2
    else:  # femme
        lbm = 1.07 * weight - 148 * (weight / height) ** 2
    Patient_df.insert(len(Patient_df.columns), "lbm", lbm)

    model = "Eleveld"

    v1_p, Ap = PropoModel(model, age, sex, weight, height)
    v1_r, Ar = RemiModel(model, age, sex, weight, height)

    Bp = np.zeros((6, 1))
    Bp[0, 0] = 1 / v1_p
    Br = np.zeros((5, 1))
    Br[0, 0] = 1 / v1_r

    Adp, Bdp = discretize(Ap, Bp, 1)
    Adr, Bdr = discretize(Ar, Br, 1)

    x_p = np.zeros((6, 1))
    x_r = np.zeros((5, 1))
    Patient_df.insert(len(Patient_df.columns), "Ce_Prop_MAP", 0)
    Patient_df.insert(len(Patient_df.columns), "Ce_Rem_MAP", 0)

    Ncase = len(Patient_df['BIS'])
    Cp_Prop = np.zeros(Ncase)
    Cp_Rem = np.zeros(Ncase)
    Ce_Prop = np.zeros(Ncase)
    Ce_Rem = np.zeros(Ncase)
    Ce_Prop_MAP = np.zeros(Ncase)
    Ce_Rem_MAP = np.zeros(Ncase)
    for j in range(Ncase-1):
        x_p = Adp @ x_p + Bdp * Patient_df['Propofol'][j]*20/3600
        x_r = Adr @ x_r + Bdr * Patient_df['Remifentanil'][j]*20/3600

        Cp_Prop[j+1] = x_p[0]
        Cp_Rem[j+1] = x_r[0]
        Ce_Prop[j+1] = x_p[3]
        Ce_Rem[j+1] = x_r[3]
        Ce_Prop_MAP[j+1] = (x_p[4] + x_p[5])/2
        Ce_Rem_MAP[j+1] = x_r[4]

    Patient_df["Cp_Prop_Eleveld"] = Cp_Prop
    Patient_df["Cp_Rem_Eleveld"] = Cp_Rem
    Patient_df["Ce_Prop_Eleveld"] = Ce_Prop
    Patient_df["Ce_Rem_Eleveld"] = Ce_Rem
    Patient_df["Ce_Prop_MAP_Eleveld"] = Ce_Prop_MAP
    Patient_df["Ce_Rem_MAP_Eleveld"] = Ce_Rem_MAP

    if caseid in id_test:
        Patients_test = pd.concat([Patients_test, Patient_df], ignore_index=True)
    else:
        for i in range(len(id_split)):
            if caseid in id_split[i]:
                set_int = i
                break
        Patient_df.insert(len(Patient_df.columns), "train_set", set_int)
        Patients_train = pd.concat([Patients_train, Patient_df], ignore_index=True)


# Save Patients DataFrame
# Patients_train.to_csv("./Patients_train.csv")
# Patients_test.to_csv("./Patients_test.csv")

# Print stats
print("nb point tot: " + str(nb_points))
print("nb point train: " + str(len(Patients_train['BIS'])))
print("nb point test: " + str(len(Patients_test['BIS'])))


print_dist = True

Patients_train = pd.to_csv("./Patients_train.csv", index_col=0)
Patients_test = pd.to_csv("./Patients_test.csv", index_col=0)
# %%
if print_dist:

    Data_train = []
    Data_test = []

    for case in id_train:
        p = Patients_train[Patients_train['caseid'] == case]
        temp = p[['age', 'height', 'weight', 'sex']].to_numpy()
        Data_train.append(temp[0])
    for case in id_test:
        p = Patients_test[Patients_test['caseid'] == case]
        temp = p[['age', 'height', 'weight', 'sex']].to_numpy()
        Data_test.append(temp[0])

    Data_plot_train = pd.DataFrame(data=Data_train, columns=['age', 'height', 'weight', 'sex'])
    Data_plot_test = pd.DataFrame(data=Data_test, columns=['age', 'height', 'weight', 'sex'])

    fig, axs = plt.subplots(2, 2)
    Data_plot_test['age'].hist(label="test", ax=axs[0, 0])
    Data_plot_train['age'].hist(label="train", alpha=0.5, ax=axs[0, 0])
    axs[0, 0].set_title('Age')
    axs[0, 0].legend()

    Data_plot_test['weight'].hist(label="test", ax=axs[0, 1])
    Data_plot_train['weight'].hist(label="train", alpha=0.5, ax=axs[0, 1])
    axs[0, 1].set_title('Weight')

    Data_plot_test['height'].hist(label="test", ax=axs[1, 0])
    Data_plot_train['height'].hist(label="train", alpha=0.5, ax=axs[1, 0])
    axs[1, 0].set_title('Height')

    data = []
    for i in range(len(Data_plot_test['sex'])):
        data.append(('M'*int(Data_plot_test['sex'][i]) + 'F'*(1 - int(Data_plot_test['sex'][i]))))
    print_df = pd.DataFrame(data, columns=['sex'])
    print_df['sex'].hist(label="test", ax=axs[1, 1])
    data = []
    for i in range(len(Data_plot_train['sex'])):
        data.append(('M'*int(Data_plot_train['sex'][i]) + 'F'*(1 - int(Data_plot_train['sex'][i]))))
    print_df = pd.DataFrame(data, columns=['sex'])
    print_df['sex'].hist(label="train", alpha=0.5, ax=axs[1, 1])
    axs[1, 1].set_title('Sex')
    fig.tight_layout()
    savepath = "/home/aubouinb/ownCloud/Anesthesie/Science/Bob/Article/Images/dataset_info.pdf"
    fig.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()
