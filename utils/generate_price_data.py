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

# adjust ticker names
gg.columns = [f.lower() for f in gg.columns]
am.columns = [f.lower() for f in am.columns]
ms.columns = [f.lower() for f in ms.columns]

# merge dfs
temp = pd.merge(gg.loc[:,["open", "close"]], am.loc[:,["open", "close"]], on="Dates", how="inner", suffixes=['_gg', '_am'])
ms.columns = [f+"_ms" for f in ms.columns]
price = pd.merge(temp, ms.loc[:,["open_ms", "close_ms"]], on="Dates", how="inner")

# output price data
price.to_csv("price.csv")
