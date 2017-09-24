# Feature engineering and normalization will be conducted here
# run this from dqn-trader
import numpy as np
import pandas as pd
import datetime
import os

def main():
    data_file_path = 'data/snp500_preprocessed.npz'
    assert os.path.exists('data/snp500_preprocessed.npz')

    file = np.load(data_file_path)
    timestamps = file['timestamps']
    syms       = file['syms']
    col_names  = file['col_names']
    data       = file['data']

    close_price = data[:,:,col_names == 'close']
    data2 = data[1:len(timestamps)] / close_price[:len(timestamps) - 1]
    timestamps = timestamps[1:len(timestamps)]

    print(timestamps.shape, syms.shape, col_names.shape)
    print(data2.shape)

    # comment out when not in use just in case this file is ran by accident
    np.savez('../data/snp500_transformed.npz', timestamps=timestamps, syms=syms, col_names=col_names, data=data2)

if __name__ == '__main__':
    main()
