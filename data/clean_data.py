import os
import sys

from datetime import datetime
from time import time

import numpy as np
import pandas as pd

import pickle

from global_const import *

TEMP_PATH = 'temp/'

# initialize script
t = time()
print('clean_data.py started')
print('This can take a few minutes')

bitcoin_data_file = open(TEMP_PATH + BITCOIN_DATA_FILE_NAME, 'rb')
clean_data_file = open(BITCOIN_DATA_FILE_NAME, 'wb')

bitcoin_data_file_reader = pickle.Unpickler(bitcoin_data_file)
clean_data_file_writer = pickle.Pickler(clean_data_file)

time_interval = bitcoin_data_file_reader.load()
column_names = bitcoin_data_file_reader.load()

print('Time interval is', time_interval)
print('Column names are', column_names)

clean_data_file_writer.dump(time_interval)
clean_data_file_writer.dump(column_names)

done = False
row_count = 0

# fill nan_data
while not done:
    try:
        data_chunk = bitcoin_data_file_reader.load()

        r_means = np.nanmean(data_chunk[:,1:], axis=1)

        for i in range(len(data_chunk)):
            for j in range(1,len(column_names)):
                if np.isnan(data_chunk[i,j]):
                    data_chunk[i,j] = r_means[i]

        clean_data_file_writer.dump(data_chunk)

        row_count += len(data_chunk)
        sys.stdout.write('\r' + str(row_count))
        sys.stdout.write(' time elapsed ' + str(time() - t))
    except:
        done = True

        bitcoin_data_file.close()
        clean_data_file.close()

        print()
        print('done', time() - t)
