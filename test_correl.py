"""Small script to create a dataset and test few correlation between variables."""

# %%  Import
# standard library
import sys

# third party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# local
try:
    sys.path.append('/home/aubouinb/ownCloud/Anesthesie/Science/Bob/Code/utilities')
    import vitaldb_local as vdb

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


# %%  get dataset
try:
    final_df = pd.read_csv('./data/data_respi.csv')
except:  # if the file does not exist, create it
    caselist = [46, 70, 101, 167, 172, 218, 221, 247, 268, 345, 353, 405, 447, 533, 537, 544, 545, 585,
                593, 636, 663, 671, 672, 685, 711, 734, 751, 812, 827, 831, 835, 847, 866, 872, 894,
                926, 940, 952, 963, 1029, 1044, 1047, 1154, 1176, 1201, 1212, 1213, 1237, 1238, 1239, 1267,
                1376, 1392, 1396, 1398, 1404, 1416, 1440, 1602, 1611, 1613, 1657, 1658, 1662,
                1687, 1690, 1918, 1925, 1994, 2000, 2014, 2029, 2049, 2051, 2072, 2074, 2139, 2148,
                2157, 2196, 2229, 2238, 2309, 2379, 2382, 2392, 2409, 2442, 2479, 2480, 2500, 2511,
                2527, 2528, 2542, 2562, 2569, 2572, 2949, 2955, 2956, 2975, 3027, 3042, 3047, 3050,
                3065, 3070, 3073, 3092, 3315, 3366, 3367, 3376, 3379, 3398, 3407, 3435, 3458, 3710, 3729,
                3791, 3859, 4050, 4091, 4098, 4122, 4146, 4172, 4173, 4177, 4195, 4202, 4212, 4253,
                4277, 4292, 4350, 4375, 4387, 4432, 4472, 4547, 4673, 4678, 4716, 4741, 4745, 4789, 4803]

    track_names = ['BIS/BIS', 'Orchestra/PPF20_RATE', 'Orchestra/RFTN20_RATE',
                   'Orchestra/PPF20_CE', 'Orchestra/RFTN20_CE', 'Solar8000/ART_MBP',
                   'BIS/SQI', 'Solar8000/PLETH_HR', 'Orchestra/PPF20_CP',
                   'Orchestra/RFTN20_CP', 'Orchestra/RFTN20_VOL', 'Solar8000/PLETH_SPO2',
                   'Orchestra/PPF20_VOL', 'Solar8000/NIBP_MBP', 'Primus/ETCO2',
                   'Primus/PIP_MBAR', 'Primus/TV', 'Primus/RR_CO2', 'Primus/SET_RR_IPPV']

    dataframe = vdb.load_cases(track_names, caselist)
    dataframe.rename(columns={'BIS/BIS': 'BIS',
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
                              'Solar8000/NIBP_MBP': 'NI_MAP',
                              'Solar8000/PLETH_SPO2': 'SPO2',
                              'Primus/ETCO2': 'ETCO2',
                              'Primus/PIP_MBAR': 'PIP',
                              'Primus/TV': 'TV',
                              'Primus/RR_CO2': 'RR',
                              'Primus/SET_RR_IPPV': 'SET_RR'}, inplace=True)

    perso_data = pd.read_csv("./info_clinic_vitalDB.csv", decimal='.')

    final_df = pd.DataFrame()   # final dataframe
    for caseid in dataframe['caseid'].unique():
        print(caseid)
        Patient_df = dataframe[dataframe['caseid'] == caseid]
        Patient_df = Patient_df.copy()
        # find the start of ETCO2 value
        for index in range(len(Patient_df['SET_RR'])):
            if Patient_df['SET_RR'][index] > 0:
                start = index
                break
        end = len(Patient_df['SET_RR']) - (len(Patient_df['SET_RR']) - start) % 10
        Patient_df = Patient_df[start:end]
        # add patient information
        Patient_df.insert(1, "Time", np.arange(0, len(Patient_df['BIS'])))
        age = perso_data.loc[perso_data['caseid'] == str(caseid), 'age'].astype(float).item()
        Patient_df.insert(len(Patient_df.columns), "age", age)
        sex = int(perso_data[perso_data['caseid'] == str(caseid)]['sex'] == 'M')  # F = 0, M = 1
        Patient_df.insert(len(Patient_df.columns), "sex", sex)
        weight = perso_data.loc[perso_data['caseid'] == str(caseid), 'weight'].astype(float).item()
        Patient_df.insert(len(Patient_df.columns), "weight", weight)
        height = perso_data.loc[perso_data['caseid'] == str(caseid), 'height'].astype(float).item()
        Patient_df.insert(len(Patient_df.columns), "height", height)
        bmi = perso_data.loc[perso_data['caseid'] == str(caseid), 'bmi'].astype(float).item()
        Patient_df.insert(len(Patient_df.columns), "bmi", bmi)

        if sex == 1:  # homme
            lbm = 1.1 * weight - 128 * (weight / height) ** 2
        else:  # femme
            lbm = 1.07 * weight - 148 * (weight / height) ** 2
        Patient_df.insert(len(Patient_df.columns), "lbm", lbm)

        # fill missing values
        Patient_df.fillna(method='ffill', inplace=True)
        # save dataframe
        final_df = pd.concat([final_df, Patient_df], ignore_index=True)

    final_df.to_csv("./data_respi.csv", index=False)

# %% Process data

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV

step = 7  # Undersampling step
poly_degree = 2  # Polynomial degree
final_df = final_df[::step]

cov = ['age', 'weight', 'height', 'sex', 'bmi', 'lbm']
features = ['RR', 'TV', 'PIP', 'HR', 'MAP', 'SPO2'] + cov
output = 'ETCO2'
final_df = final_df[features + [output]].dropna()
X = final_df[features]
y = final_df['ETCO2']

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
poly_feat = PolynomialFeatures(degree=poly_degree, include_bias=False)
scaler.fit(X_train)
poly_feat.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train = poly_feat.transform(X_train)
X_test = poly_feat.transform(X_test)

# Create linear regression object
regr = ElasticNet()
Gridsearch = GridSearchCV(regr, {'alpha': np.logspace(-4, 0, 3), 'l1_ratio': np.linspace(0.01, 0.99, 3)},
                          n_jobs=8, cv=3, scoring='r2', verbose=0)
# Train the model using the training sets
Gridsearch.fit(X_train, y_train)

regr = Gridsearch.best_estimator_
# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(y_test, y_pred,  color='blue', alpha=0.5)
plt.plot([min(y_test) - 5, max(y_test) + 5], [min(y_test) - 5, max(y_test) + 5], color='red', linewidth=1)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.grid()
plt.show()


# %%
