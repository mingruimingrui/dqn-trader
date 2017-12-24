"""
Not the cleanest way to do things but it works

EDA shows that most timestamps are covered by the dataset. However there are
still a number of timestamps where there are no recorded trades (either due to
the market being not very active or the market being down)

We simply fill the gaps with the most recent prices except for the case where
the gap is too wide, (> 60 seconds), then in that case we fill with NaN

clean_data.py will handle the NaN cases
"""

import os
import sys

from datetime import datetime
from time import time

import numpy as np
import pandas as pd

import pickle

from global_const import *

def chunk(l, n):
    """
    Yield successive n-sized chunks from l
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def dumpChunk(o, file_name, permission='ab'):
    with open(file_name, permission) as f:
        writer = pickle.Pickler(f)
        writer.dump(o)

# script level constants
TEMP_PATH = 'temp/'

FILE_NAMES = [file_name for file_name in os.listdir(TEMP_PATH) if 'TEMP.npz' in file_name]
COLUMN_NAMES = ['timestamp'] + [file_name.replace('TEMP.npz', '') for file_name in FILE_NAMES]

TEMP_DATA_FILE_NAME = 'bitcoin_price_temp.pkl'
CHUNK_SIZE = 1000000

# initialize script
t = time()
print('preprocess_data.py started')
print('This can take a few minutes')

# latest_min and earliest_max represents the maximum period that we can
# set our data table to be
# from an EDA we determined the dates to be from 1st feb 2015 to 28th sep 2017
latest_min   = datetime(2015,2,1).timestamp()
earliest_max = datetime(2017,9,28).timestamp()

print('Total rows to generate for each table:', round(earliest_max - latest_min + 1))
print('Files to process', FILE_NAMES)

# create new file to store data sequentially
print('Creating new empty file to store data')
row_count = 0

with open(TEMP_PATH + BITCOIN_DATA_FILE_NAME, 'wb') as bitcoin_data_file:
    bitcoin_data_writer = pickle.Pickler(bitcoin_data_file)
    bitcoin_data_writer.dump([int(latest_min), int(earliest_max)])
    bitcoin_data_writer.dump(COLUMN_NAMES)

for timestamps in chunk(np.arange(latest_min, earliest_max + 1), CHUNK_SIZE):
    data_chunk = np.zeros((len(timestamps), len(COLUMN_NAMES))) * np.nan
    data_chunk[:,0] = timestamps.astype('int')
    dumpChunk(data_chunk, TEMP_PATH + BITCOIN_DATA_FILE_NAME)
    # bitcoin_data_writer.dump(data_chunk)

    row_count += len(timestamps)
    sys.stdout.write('\r' + str(row_count) + ' rows generated')

bitcoin_data_file.close()

print('\nBase table generated, now processing files to fill data')

# loop through each file
for i, file_name in enumerate(FILE_NAMES):
    with np.load(TEMP_PATH + file_name) as file:

        # read from bitcoin_data_file and write to a temp_data_file
        bitcoin_data_file = open(TEMP_PATH + BITCOIN_DATA_FILE_NAME, 'rb')
        bitcoin_data_reader = pickle.Unpickler(bitcoin_data_file)
        temp_data_file = open(TEMP_PATH + TEMP_DATA_FILE_NAME, 'wb')
        temp_data_writer = pickle.Pickler(temp_data_file)

        # don't forget that first 2 rows are time interval and column names
        bitcoin_data_reader.load()
        bitcoin_data_reader.load()
        temp_data_writer.dump([int(latest_min), int(earliest_max)])
        temp_data_writer.dump(COLUMN_NAMES)

        temp_data_file.close()

        # also keep track of rows appended for verbose
        row_count = 0
        nan_count = 0

        # get data
        data = file['data']

        # create lists to store timestamps and prices
        timestamps = []
        prices = []

        cur_time  = data[0,0]
        cur_price = data[0,1]
        if (cur_time >= latest_min) & (cur_time <= earliest_max):
            timestamps.append(cur_time)
            prices.append(cur_price)

        for cur_index in range(len(data) - 1):
            cur_time = data[cur_index,0] + 1
            cur_price = data[cur_index,1]

            next_time = data[cur_index+1,0]
            next_price = data[cur_index+1,1]

            # rep_count to keep track of how many repeated prices
            rep_count = 1

            # generate prices for all 'missing' timestamps
            while cur_time < next_time:

                if rep_count < 60:
                    rep_count += 1
                    if (cur_time >= latest_min) & (cur_time <= earliest_max):
                        timestamps.append(cur_time)
                        prices.append(cur_price)
                else:
                    if (cur_time >= latest_min) & (cur_time <= earliest_max):
                        timestamps.append(cur_time)
                        # prices.append(np.nan)
                        prices.append(cur_price)

                cur_time += 1

            if (next_time >= latest_min) & (next_time <= earliest_max):
                timestamps.append(next_time)
                prices.append(next_price)

            # append prices when we get a batch of size > CHUNK_SIZE
            if len(prices) >= CHUNK_SIZE:

                data_chunk = bitcoin_data_reader.load()
                data_chunk[:,i+1] = prices[:CHUNK_SIZE]
                dumpChunk(data_chunk, TEMP_PATH + TEMP_DATA_FILE_NAME)
                # temp_data_writer.dump(data_chunk)

                nan_count += np.sum(np.isnan(prices[CHUNK_SIZE:]))
                row_count += len(prices)
                prices = prices[CHUNK_SIZE:]
                row_count -= len(prices)

                sys.stdout.write('\r' + str(row_count) + ' rows generated for ' + COLUMN_NAMES[i+1])

        # append remaining prices
        if len(prices) > 0:

            data_chunk = bitcoin_data_reader.load()
            data_chunk[:,i+1] = prices
            dumpChunk(data_chunk, TEMP_PATH + TEMP_DATA_FILE_NAME)
            # temp_data_writer.dump(data_chunk)

            nan_count += np.sum(np.isnan(prices))
            row_count += len(prices)
            prices = prices

            sys.stdout.write('\r' + str(row_count) + ' rows generated for ' + COLUMN_NAMES[i+1])

        bitcoin_data_file.close()
        # temp_data_file.close()

        os.remove(TEMP_PATH + BITCOIN_DATA_FILE_NAME)
        os.rename(TEMP_PATH + TEMP_DATA_FILE_NAME, TEMP_PATH + BITCOIN_DATA_FILE_NAME)

        print()
        print(file_name, 'preprocessed, time_elpased: {}s'.format(round(time() - t)))
        print('Number of nan rows', nan_count)
