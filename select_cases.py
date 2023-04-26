"""Select cases for the database"""
# %% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vitaldb
from os.path import exists

perso_data = pd.read_csv("./data/info_clinic_vitalDB.csv")
perso_data.dropna(subset=["caseid"], inplace=True)
print(f"Initial number of cases: {perso_data.count()[0]}")

# Select cases with at least 2 hours of data
perso_data = perso_data[perso_data["casedur"] >= 120]

print(f"Number of cases with at least 2 hours of data: {perso_data.count()[0]}")

# Select cases without bolus drugs impacting BIS and MAP
perso_data = perso_data[perso_data["intraop_mdz"] == 0]
perso_data = perso_data[perso_data["intraop_ftn"] == 0]
perso_data = perso_data[perso_data["intraop_eph"] == 0]
perso_data = perso_data[perso_data["intraop_phe"] == 0]
perso_data = perso_data[perso_data["intraop_epi"] == 0]

print(f"Number of cases without bolus drugs impacting BIS and MAP: {perso_data.count()[0]}")

# select cases including the main signals
Main_signals = ['Solar8000/ART_MBP', 'Solar8000/NIBP_MBP',
                'Solar8000/PLETH_HR', 'BIS/BIS']

caseid_with_signals = vitaldb.find_cases(Main_signals)

perso_data = perso_data[perso_data["caseid"].astype(int).isin(caseid_with_signals)]

print(f"Number of cases including the main signals: {perso_data.count()[0]}")

# Select TIVA cases

caseid_tiva = list(vitaldb.caseids_tiva)
perso_data = perso_data[perso_data["caseid"].astype(int).isin(caseid_tiva)]

print(f"Number of TIVA cases: {perso_data.count()[0]}")

# select cases without gazes
caseid_n2o = list(vitaldb.caseids_n2o)
casei_sevo = list(vitaldb.caseids_sevo)
caseid_des = list(vitaldb.caseids_des)
caseid_gazes = caseid_n2o + casei_sevo + caseid_des
perso_data = perso_data[~perso_data["caseid"].astype(int).isin(caseid_gazes)]

print(f"Number of cases without gazes: {perso_data.count()[0]}")

# %% Check the signals for the selected cases
number_of_cases_with_more_drugs = 0
number_of_cases_with_gazes = 0
number_of_cases_with_missing_volume = 0
final_caseid_list = []
count = 1
for caseid in perso_data["caseid"].astype(int):
    print(caseid)
    print(f"Case {count}/{perso_data.count()[0]}")
    count += 1
    gaz_signals = ['Primus/INSP_DES', 'Primus/INSP_SEVO', 'Primus/FIN2O']
    intra_drug_signals = ['AMD_RATE', 'DEX2_RATE', 'DEX4_RATE', 'DOBU_RATE', 'DOPA_RATE', 'DTZ_RATE', 'EPI_RATE',
                          'FUT_RATE', 'MRN_RATE', 'NEPI_RATE', 'NPS_RATE', 'NTG_RATE', 'OXY_RATE', 'PGE1_RATE',
                          'PHEN_RATE', 'PPF20_RATE', 'RFTN20_RATE', 'RFTN50_RATE', 'ROC_RATE', 'VASO_RATE', 'VEC_RATE']
    intra_drug_signals = ['Orchestra/' + el for el in intra_drug_signals]
    intra_drug_signals += [string.replace('RATE', 'VOL') for string in intra_drug_signals]
    signals = Main_signals + gaz_signals + intra_drug_signals
    if exists(f'../Vital_DB_database/Case_{caseid}.csv'):
        dataframe_case = pd.read_csv(f'../Vital_DB_database/Case_{caseid}.csv')
        dataframe_case = dataframe_case[[signal for signal in signals if signal in dataframe_case.columns]]
    else:
        case = vitaldb.VitalFile(caseid, signals)
        dataframe_case = case.to_pandas(signals, 1)

    # Check that there is only Propofol and Remifentanil
    count_drugs = dataframe_case[[col_name for col_name in dataframe_case.columns if col_name.startswith(
        'Orchestra/')]].sum(axis=0).astype(bool).sum()

    if count_drugs > 4:
        number_of_cases_with_more_drugs += 1
        continue

    # Check that there is no gaz usage
    total_gaz_usage = dataframe_case[[
        col_name for col_name in dataframe_case.columns if col_name.startswith('Primus/')]].sum(axis=0).sum()
    if total_gaz_usage > 10:
        number_of_cases_with_gazes += 1
        continue
    # check that the volume of Propofol and Remifentanil is 0 at the first non nan value of the case
    if dataframe_case.loc[dataframe_case['Orchestra/PPF20_RATE'].first_valid_index(), 'Orchestra/PPF20_RATE'] > 0:
        number_of_cases_with_missing_volume += 1
        continue
    if dataframe_case.loc[dataframe_case['Orchestra/RFTN20_RATE'].first_valid_index(), 'Orchestra/RFTN20_RATE'] > 0:
        number_of_cases_with_missing_volume += 1
        continue

    # Check signal quality of the output signals
    # check that there is at least 90% of BIS data
    Ncase = len(dataframe_case)
    if dataframe_case['BIS/BIS'].fillna(0).eq(0).sum()/Ncase > 0.1:
        continue
    # check there is at least 80% of MAP data (2s step time so there is already 50% of missing data)
    if dataframe_case['Solar8000/ART_MBP'].fillna(0).eq(0).sum()/Ncase > 0.6:
        continue
    if len(dataframe_case[dataframe_case['Solar8000/ART_MBP'] < 50])/Ncase > 0.2:
        continue
    if dataframe_case['Solar8000/PLETH_HR'].fillna(0).eq(0).sum()/Ncase > 0.6:
        continue
    if len(dataframe_case[dataframe_case['Solar8000/PLETH_HR'] < 50])/Ncase > 0.2:
        continue

    final_caseid_list.append(caseid)

    # plot BIS, MAP, HR and Drug rates on one figure and save it as png
    dataframe_case.fillna(method='ffill', inplace=True)
    fig, axs = plt.subplots(3, 1, figsize=(20, 10))
    axs[0].plot(dataframe_case['BIS/BIS'], color='red')
    axs[0].set_ylabel('BIS')
    axs[0].set_ylim(0, 100)
    axs[1].plot(dataframe_case['Solar8000/ART_MBP'], color='blue', label='MAP')
    axs[1].plot(dataframe_case['Solar8000/PLETH_HR'], color='green', label='HR')
    axs[1].set_ylabel('MAP and HR')
    axs[1].legend()
    axs[1].set_ylim(0, 200)
    axs[2].plot(dataframe_case['Orchestra/PPF20_RATE'], color='red', label='Propofol')
    axs[2].plot(dataframe_case['Orchestra/RFTN20_RATE'], color='blue', label='Remifentanil')
    axs[2].set_ylabel('Drug rates')
    axs[2].legend()

    fig.savefig(f'../Vitaldb_database_2/figures/{caseid}.png')
    plt.close(fig)


print(f"Number of cases with more than 2 drugs: {number_of_cases_with_more_drugs}")
print(f"Number of cases with gazes: {number_of_cases_with_gazes}")
print(f"Number of cases with missing volume: {number_of_cases_with_missing_volume}")
print(f"Number of cases with good signal quality: {len(final_caseid_list)}")

perso_data = perso_data[perso_data["caseid"].astype(int).isin(final_caseid_list)]

bad_visual_inspection = [70,  # ??
                         77,
                         672,
                         1212,
                         1404,
                         1762,
                         1894,
                         2238,  # ??
                         3130,  # ??
                         3611,
                         4286,  # ??
                         4522,
                         4872,  # ??
                         4913,
                         5140,
                         5150,
                         5781]

# remove cases with bad visual inspection
final_caseid_list = [caseid for caseid in final_caseid_list if caseid not in bad_visual_inspection]

print(f"Number of cases with good visual inspection: {len(final_caseid_list)}")

# save the list of caseid
with open('./data/caseid_list.txt', 'w') as f:
    for item in final_caseid_list:
        f.write("%s\n" % item)


# Plot on selected data


# Check the number of cases per operation name
perso_data_by_surgeon = perso_data.groupby("opname")
perso_data_by_surgeon.count().plot(kind="bar", y="caseid")

# %% Plot Patient Characteristics

fig, ax = plt.subplots(2, 2, figsize=(15, 5))
# Age
perso_data["age"].astype(float).plot(kind="hist", bins=20, ax=ax[0, 0])
ax[0, 0].set_title('Age')
# Weight
perso_data["weight"].astype(float).plot(kind="hist", bins=20, ax=ax[0, 1])
ax[0, 1].set_title('Weight')
# Height
perso_data["height"].astype(float).plot(kind="hist", bins=20, ax=ax[1, 0])
ax[1, 0].set_title('Height')
# count 'F' and 'M' in sex

sex_df = pd.DataFrame({'F': perso_data.sex.eq('F').sum(),
                       'M': perso_data.sex.eq('M').sum()}, index=[0, 1])

sex_df.plot(kind="hist", ax=ax[1, 1])
ax[1, 1].set_title('Gender')

# %%
