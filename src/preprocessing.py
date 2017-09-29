# feature engineering and normalization will be conducted here
# run this from dqn-trader
import os
import numpy as np
import pandas as pd
import datetime

def main():
    data_file_path = 'data/snp500_raw.npz'
    export_file_path = 'data/snp500_transformed.npz'
    assert os.path.exists('data/snp500_preprocessed.npz'), 'preprocessing.py not yet ran'

    # load them params into enviornment
    file = np.load(data_file_path)
    timestamps = file['timestamps']
    syms       = file['syms']
    col_names  = file['col_names']
    data       = file['data']

    # add MKT risk-free asset which remains at one with no interest rate
    MKT_data = np.ones((data.shape[0],1,data.shape[2]))
    data2 = np.concatenate((MKT_data, data), 1)
    syms = np.array(['MKT'] + list(syms))

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
    np.savez(export_file_path, timestamps=timestamps, syms=syms, col_names=col_names, data=dataf)

if __name__ == '__main__':
    print('data_transform.py started')
    main()
    print('data_transform.py finished')
