# feature engineering and normalization will be conducted here
# run this from dqn-trader
import os
import numpy as np
import pandas as pd
import datetime

def main():
    data_file_path = 'data/snp500_preprocessed.npz'
    assert os.path.exists('data/snp500_preprocessed.npz'), 'preprocessing.py not yet ran'

    # load them params into enviornment
    file = np.load(data_file_path)
    timestamps = file['timestamps']
    syms       = file['syms']
    col_names  = file['col_names']
    data       = file['data']

    # add MKT risk-free asset which remains at one with no interest rate
    data = np.concatenate((np.ones((data.shape[0],1,data.shape[2])), data), 1)
    syms = np.array(['MKT'] + list(syms))

    # attempt to find places with stock split
    data2 = data.copy()
    for j, s in enumerate(syms):
        for i in range(len(timestamps) - 1):
            open_1 = data[i,j,col_names == 'open']
            open_2 = data[i+1,j,col_names == 'open']
            if ((open_1 / open_2) > 1.9) | ((open_2 / open_1) > 1.9) :
                data2[i+1:,j,col_names == 'open'] *= open_1 / open_2

    #**************************************************************************#
    #*********  reserved for feature engineering stuff in the future  *********#
    #**************************************************************************#



    #**************************************************************************#
    #**************************************************************************#
    #**************************************************************************#

    dataf = data2.copy()
    print(timestamps.shape, syms.shape, col_names.shape)
    print(dataf.shape)

    # comment out when not in use just in case this file is ran by accident
    #np.savez('data/snp500_transformed.npz', timestamps=timestamps, syms=syms, col_names=col_names, data=dataf)

if __name__ == '__main__':
    print('data_transform.py started')
    main()
    print('data_transform.py finished')
