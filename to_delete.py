from vitaldb_local import load_cases
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# add a path to python
import sys
sys.path.append('/home/aubouinb/ownCloud/Anesthesie/Science/Bob/Code/utilities')
# local


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# %% import a case from vitaldb
caseid = 671
case = load_cases(['BIS/BIS', 'Solar8000/ART_MBP'], [caseid])
# replace value with MAP >150 by nan
case.loc[case['Solar8000/ART_MBP'] > 130, 'Solar8000/ART_MBP'] = np.nan

case.fillna(method='ffill', inplace=True)
# smooth the signals
case.loc[case.index[:400], 'BIS/BIS'] = 96
case['BIS/BIS'] = case['BIS/BIS'].rolling(100).mean()
case['Solar8000/ART_MBP'] = case['Solar8000/ART_MBP'].rolling(100).mean()
# replace first 100 BIS value by 96


# plot BIS and MAP
Time = np.arange(0, len(case['BIS/BIS']))/60/60
plt.figure(figsize=(8, 4))
plt.plot(Time, case['BIS/BIS'], label='BIS')
plt.plot(Time, case['Solar8000/ART_MBP'], label='MAP')
plt.legend(loc='lower center')

plt.xlabel('Time (h)')
plt.ylim([0, 130])
plt.grid()
plt.savefig('./BIS_MAP.pdf', bbox_inches='tight')
plt.show()
