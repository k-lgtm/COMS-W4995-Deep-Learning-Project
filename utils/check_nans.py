def check_nan(df):
    nans = []
    positions = []
    names = []
    for i in range(1, df.shape[1]):
        stock = df.iloc[:,i]
        if np.isnan(stock).any():
            positions.append(i)
            nans.append((i, df.columns[i], np.isnan(stock).sum()))
            names.append(df.columns[i])
    return nans, len(nans), positions, names
