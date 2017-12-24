"""
meant to subset the dataset into sizable chunks for training
"""

import os
import sys
import pickle

from datetime import datetime
from time import time

import numpy as np
import pandas as pd

from global_const import *

assert len(sys.argv) >= 4, 'please provide an output_filename start and end'

output_file_name = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])

t = time()
print('create_subset.py started')

bitcoin_data_file = open(BITCOIN_DATA_FILE_NAME, 'rb')
subset_data_file = open(output_file_name, 'wb')

bitcoin_data_reader = pickle.Unpickler(bitcoin_data_file)
subset_data_writer = pickle.Pickler(subset_data_file)

time_interval = bitcoin_data_reader.load()
column_names = bitcoin_data_reader.load()

time_interval[0] = max(time_interval[0], start)
time_interval[1] = min(time_interval[1], end)

subset_data_writer.dump(time_interval)
subset_data_writer.dump(column_names)

done = False

output_data = np.array([]).reshape(0,len(column_names))

while not done:
    try:
        data_chunk = bitcoin_data_reader.load()
        data_kept = data_chunk[(data_chunk[:,0] >= start) & (data_chunk[:,0] <= end)]
        output_data = np.concatenate((output_data, data_kept))

        sys.stdout.write('\r' + str(data_chunk[-1,0]))
        sys.stdout.write(' time elapsed ' + str(time() - t))
    except:
        done = True

        subset_data_writer.dump(output_data)

        bitcoin_data_file.close()
        subset_data_file.close()

        print()
        print('done', time() - t)
