# put all preprocessing things here
# run this from dqn-trader
import os
import numpy as np
import pandas as pd
import datetime
from time import time

def main():
    assert os.path.exists('data'), 'create data file first'
    startTime = time()

    print('Loading data')
    data_file_path = 'data/SNP500_PRICE.csv'
    df = pd.read_csv(data_file_path)

    print('Transform timestamp into proper form')
    df.index = range(len(df))
    df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'))

    # drop syms without all timestamps
    print('Identify syms to drop')
    syms_to_drop = []
    for s in df.Sym.unique():
        count = sum(df.Sym == s)
        if (count != 3200):
            syms_to_drop.append(s)
            print(s, count)

    df2 = df[df.Sym.apply(lambda x: x not in syms_to_drop)].copy()

    # arrange data into a (timestamp, sym, values) shaped array for fast data traversal
    print('Syms dropped, now creating new data array')
    print('Time since start:', time() - startTime)
    timestamps = df2['Timestamp'].unique()
    syms       = df2['Sym'].unique()
    col_names  = ['Open', 'Close', 'High', 'Low', 'Adj Close', 'Volume']

    df3 = np.zeros((len(timestamps), len(syms), len(col_names)))

    for i, t in enumerate(timestamps):
        temp = df2[df2['Timestamp'] == t]
        for j, s in enumerate(syms):
            df3[i,j] = temp.loc[temp['Sym'] == s, col_names]
        print(t)

    # drop syms with nan data
    # a data exploration shows that there are only a few assets with large counts of nan data
    # makes sense to drop these assets as they would most likely be treated as outliers
    print('Data array created, dropping nan syms')
    print('Time since start:', time() - startTime)

    syms_to_keep = []
    for i in range(len(syms)):
        nb_nan = np.sum(np.isnan(data[:,i,:]))
        if nb_nan == 0:
            syms_to_keep.append(i)

    syms = syms[syms_to_keep]
    df4 = df3[:,syms_to_keep,:].copy()

    print('Done dropping nan syms')
    print('Time since start:', time() - startTime)

    df_f = df4.copy()

    print('\ntimestamps.shape, syms.shape, col_names.shape')
    print(timestamps.shape, syms.shape, col_names.shape)

    print('\ndf_f.shape')
    print(df_f.shape)

    print('\nNow saving')
    col_names = ['open', 'close', 'high', 'low', 'adjclose', 'volume']
    # comment out when not in use just in case this file is ran by accident
    np.savez('data/snp500_preprocessed.npz', timestamps=timestamps, syms=syms, col_names=col_names, data=df3)
    print('Data array saved')

if __name__ == '__main__':
    print('preprocess.py started')
    main()
    print('preprocess.py finished')
