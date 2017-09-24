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

    # we're really just interested in the multiple from the previous day's open price
    # since we're trading based on monetary values rather than number of stocks
    close_price = data[:,:,col_names == 'open']
    data2 = data[1:len(timestamps)] / close_price[:len(timestamps) - 1]
    timestamps = timestamps[1:len(timestamps)]

    # fill na with 1s
    data2[np.isnan(data2)] = 1

    # attempt to find places with stock split
    stock_split_index = np.squeeze(data2[:,:,col_names == 'open']) > 1.99
    data3 = data2
    data3[stock_split_index] = 1

    #**************************************************************************#
    #*********  reserved for feature engineering stuff in the future  *********#
    #**************************************************************************#



    #**************************************************************************#
    #**************************************************************************#
    #**************************************************************************#

    dataf = data2
    print(timestamps.shape, syms.shape, col_names.shape)
    print(dataf.shape)

    # comment out when not in use just in case this file is ran by accident
    #np.savez('data/snp500_transformed.npz', timestamps=timestamps, syms=syms, col_names=col_names, data=dataf)

if __name__ == '__main__':
    print('data_transform.py started')
    main()
    print('data_transform.py finished')
