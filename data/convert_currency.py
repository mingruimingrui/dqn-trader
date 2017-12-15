"""
Purpose of this file is to summarise all similar timestamps as well as to convert the currency into USD
"""

import os
import sys
import re

from datetime import datetime
from time import time

import numpy as np
import pandas as pd

from global_const import *

def returnOnlyCaps(x):
    """
    Return only captial letters in a string
    """
    result = ""
    for c in x:
        if c.isupper():
            result += c

    return result

def findClosestIndex(value, series):
    """
    Finds index of closest value in a series
    """
    abs_value_difference = np.abs(series)
    closest_index = np.argmin(abs_value_difference)

    return closest_index

def convertCurrency(data_row, currency_ticker, exchange_rates_table):
    """
    This is going to be weirdly specific
    Basically I need a function that returns the currency
    in USD given a data_row and an exchange_rates_table
    """
    closest_index = findClosestIndex(data_row['timestamp'], exchange_rates_table['timestamp'])

    return data_row['price'] / exchange_rates_table.loc[closest_index, currency_ticker]

FILE_NAMES = [file_name for file_name in os.listdir(DATA_PATH + 'raw/') if '.npz' in file_name]
FILE_NAMES = FILE_NAMES[:3]
OUTPUT_FILE_NAMES = [re.sub('([A-Z])+', lambda x: 'TEMP', file_name) for file_name in FILE_NAMES]

# processing entire datasets will take up uneccessary time
# start/end values are taken from an EDA of the dataset
# seems like coinbase only begins to have consistent trades from around Feb 2015
# here we choose Jan 2015 as the start date
# also the last recorded trading dates of btcn is the 30th of Sep 2017
# we pick 1st Oct 2017 as the end date
START = int(datetime(2015, 1, 1).timestamp())
END   = int(datetime(2017, 10, 1).timestamp())

print('Starting convert_currency.py')
print('This can take a few hours')

exchange_rates_table = pd.read_csv(EXCHANGE_RATES_PATH + EXCHANGE_RATES_FILE_NAME)

for i, file_name in enumerate(FILE_NAMES):
    with np.load(RAW_PATH + file_name) as file:
        t = time()

        print('Processing', file_name)

        out_file_path = TEMP_PATH + OUTPUT_FILE_NAMES[i]

        if os.path.isfile(out_file_path):
            print(out_file_path)
            print('File has already been processed')
            continue

        data = file['data']
        data = pd.DataFrame(data, columns=['timestamp', 'price', 'volume'])

        data = data.loc[(data['timestamp'] >= START) & (data['timestamp'] <= END)]
        data['monetary_volume'] = data['price'] * data['volume']

        volume = data.groupby('timestamp')['volume'].sum()
        monetary_volume = data.groupby('timestamp')['monetary_volume'].sum()

        data2 = pd.DataFrame([])
        data2['timestamp'] = volume.index
        data2['price'] = (monetary_volume / volume).values

        del data
        del volume, monetary_volume

        print('Time taken to summarise', file_name, round(time() - t), 'seconds')

        currency_ticker = returnOnlyCaps(file_name)

        if currency_ticker != 'USD':
            convertCurrency_x = lambda x : convertCurrency(x, currency_ticker, exchange_rates_table)
            data2['price'] = data2.apply(convertCurrency_x , axis=1)
            print('\nTime taken to convert', file_name, round(time() - t), 'seconds')

        np.savez(out_file_path, data=data2)
        print(OUTPUT_FILE_NAMES[i],'saved')

        del data2
