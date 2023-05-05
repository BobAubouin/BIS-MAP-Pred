"""Separate train and test set considering the frequency of the target variable
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("./data/full_data.csv")

# dataframe.Target_Propo.fillna(method="ffill", inplace=True)
# dataframe.Target_Remi.fillna(method="ffill", inplace=True)

# # Explore data
# dataframe_by_caseid = dataframe.groupby("caseid")
# Propo_freq = dataframe_by_caseid.Target_Propo.count()/60/60
# Remi_freq = dataframe_by_caseid.Target_Remi.count()/60/60
# for caseid, df_case in dataframe_by_caseid:
#     Propo_freq[caseid] = df_case.Target_Propo.diff().astype(bool).sum() / \
#         (df_case.Target_Propo.count()/60/60)
#     Remi_freq[caseid] = df_case.Target_Remi.diff().astype(bool).sum() / \
#         (df_case.Target_Remi.count()/60/60)
# Propo_freq.plot(kind="hist", bins=20)
# plt.title("Propofol frequency of target change")
# plt.show()

# Remi_freq.plot(kind="hist", bins=20)
# plt.title("Remifentanil frequency of target change")
# plt.show()

# # plot case with max and min frequency
# caseid_max = Propo_freq.idxmax()
# caseid_min = Propo_freq.idxmin()
# dataframe_by_caseid.get_group(caseid_max).BIS.plot()
# dataframe_by_caseid.get_group(caseid_max).MAP.plot()
# plt.show()


# dataframe_by_caseid.get_group(caseid_min).BIS.plot()
# dataframe_by_caseid.get_group(caseid_min).MAP.plot()
# plt.show()

# df_concat = pd.concat([Propo_freq, Remi_freq], axis=1)
# df_concat.insert(2, 'mean', df_concat.mean(axis=1))
# df_concat.sort_values(by='mean', inplace=True, ascending=False)
# train_list = df_concat.index[int(0.3*len(df_concat)):].tolist()


train_list = train_test_split(dataframe.caseid.unique(), test_size=0.3, random_state=1)[0]

# save train list
with open("./data/train_list.txt", "w") as f:
    for caseid in train_list:
        f.write(str(caseid) + "\n")


# separe train and test set

id_split = np.array_split(train_list, 3)  # split train set in 5 set for cross validation
Patients_train = dataframe[dataframe['caseid'].isin(train_list)]
Patients_test = dataframe[~dataframe['caseid'].isin(train_list)]
# for each patient in Patient_train, add a column with the set number
Patients_train.insert(0, 'train_set', 0)
for i in range(len(id_split)):
    Patients_train.loc[Patients_train['caseid'].isin(id_split[i]), 'train_set'] = i

# save train and test set
Patients_train.to_csv("./data/Patients_train.csv")
Patients_test.to_csv("./data/Patients_test.csv")


# Print stats
print("nb point train: " + str(len(Patients_train['BIS'])))
print("nb point test: " + str(len(Patients_test['BIS'])))


# %% plot

print_dist = True

Patients_train = pd.read_csv("./data/Patients_train.csv", index_col=0)
Patients_test = pd.read_csv("./data/Patients_test.csv", index_col=0)
# %% plot
if print_dist:

    Data_train = []
    Data_test = []

    for case in Patients_train.caseid.unique():
        p = Patients_train[Patients_train['caseid'] == case]
        temp = p[['age', 'height', 'weight', 'gender']].to_numpy()
        Data_train.append(temp[0])
    for case in Patients_test.caseid.unique():
        p = Patients_test[Patients_test['caseid'] == case]
        temp = p[['age', 'height', 'weight', 'gender']].to_numpy()
        Data_test.append(temp[0])

    Data_plot_train = pd.DataFrame(data=Data_train, columns=['age', 'height', 'weight', 'gender'])
    Data_plot_test = pd.DataFrame(data=Data_test, columns=['age', 'height', 'weight', 'gender'])

    fig, axs = plt.subplots(2, 2)
    Data_plot_train['age'].hist(label="train", alpha=0.5, ax=axs[0, 0])
    Data_plot_test['age'].hist(label="test", ax=axs[0, 0])
    axs[0, 0].set_title('Age')
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].legend()

    Data_plot_train['weight'].hist(label="train", alpha=0.5, ax=axs[0, 1])
    Data_plot_test['weight'].hist(label="test", ax=axs[0, 1])
    axs[0, 1].set_title('Weight')
    axs[0, 1].set_axisbelow(True)

    Data_plot_train['height'].hist(label="train", alpha=0.5, ax=axs[1, 0])
    Data_plot_test['height'].hist(label="test", ax=axs[1, 0])
    axs[1, 0].set_title('Height')
    axs[1, 0].set_axisbelow(True)

    Data_plot_train['gender'].hist(label="train", alpha=0.5, ax=axs[1, 1], rwidth=0.3,
                                   align='mid', bins=[-0.5, 0.5, 1.5])
    Data_plot_test['gender'].hist(label="test", ax=axs[1, 1], rwidth=0.3,
                                  align='mid', bins=[-0.5, 0.5, 1.5])
    axs[1, 1].set_xlim(-1, 2)
    axs[1, 1].set_xticks([0, 1], ['F', 'M'])
    axs[1, 1].set_title('Gender')
    axs[1, 1].set_axisbelow(True)
    fig.tight_layout()
    savepath = "/home/aubouinb/ownCloud/Anesthesie/Science/Bob/Journal_model/Images/dataset_info.pdf"
    fig.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()
