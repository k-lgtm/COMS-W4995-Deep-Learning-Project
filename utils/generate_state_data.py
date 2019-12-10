import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# load data
gg = pd.read_excel("google.xlsx")
am = pd.read_excel("amazon.xlsx")
ms = pd.read_excel("microsoft.xlsx")

# set indices
gg = gg.set_index("Dates")
am = am.set_index("Dates")
ms = ms.set_index("Dates")

# calculate differences
gg.Close = gg.Close - gg.Open
gg.High = gg.High - gg.Open
gg.Low = gg.Low - gg.Open
am.Close = am.Close - am.Open
am.High = am.High - am.Open
am.Low = am.Low - am.Open
ms.Close = ms.Close - ms.Open
ms.High = ms.High - ms.Open
ms.Low = ms.Low - ms.Open

# clean dfs
gg.drop(['Value', 'Number Ticks'], axis=1, inplace=True)
am.drop(['Value', 'Number Ticks'], axis=1, inplace=True)
ms.drop(['Value', 'Number Ticks'], axis=1, inplace=True)

# scale data
scaler = StandardScaler()

ggs = pd.DataFrame(scaler.fit_transform(gg), index=gg.index, columns=gg.columns)
ams = pd.DataFrame(scaler.fit_transform(am), index=am.index, columns=am.columns)
mss = pd.DataFrame(scaler.fit_transform(ms), index=ms.index, columns=ms.columns)

# clean ticker names
ggs.columns = [f.lower() for f in ggs.columns]
ams.columns = [f.lower() for f in ams.columns]
mss.columns = [f.lower() for f in mss.columns]

# merge data
temp = pd.merge(ggs, ams, on="Dates", how="inner", suffixes=['_gg', '_am'])
mss.columns = [f+"_ms" for f in mss.columns]
state = pd.merge(temp, mss, on="Dates", how="inner")

# Don't trade in the last 20 minutes
state = state[:-20]   

# output generated state data
state.to_csv("state.csv")
