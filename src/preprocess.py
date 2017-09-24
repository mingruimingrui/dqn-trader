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
    data_file_path = '/mnt/filesystem1/datasets/SNP500_PRICE.csv'
    df = pd.read_csv(data_file_path)

    print('Transform timestamp into proper form')
    df.index = range(len(df))
    df['Timestamp'] = df['Timestamp'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'))

    print('Identify syms to drop')
    syms_to_drop = []
    for s in df.Sym.unique():
        count = sum(df.Sym == s)
        if (count != 3200):
            syms_to_drop.append(s)
            print(s, count)

    df2 = df[df.Sym.apply(lambda x: x not in syms_to_drop)]

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

    print('Data array created')
    print('Time since start:', time() - startTime)

    print('\ntimestamps.shape, syms.shape, col_names.shape')
    print(timestamps.shape, syms.shape, col_names.shape)

    print('\ndf3.shape')
    print(df3.shape)

    print('\nNow saving')
    col_names = ['open', 'close', 'high', 'low', 'adjclose', 'volume']
    # comment out when not in use just in case this file is ran by accident
    #np.savez('data/snp500_preprocessed.npz', timestamps=timestamps, syms=syms, col_names=col_names, data=df3)
    print('Data array saved')

if __name__ == '__main__':
    print('preprocess.py started')
    main()
    print('preprocess.py finished')
