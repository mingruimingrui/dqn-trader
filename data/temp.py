import os
import sys

from datetime import datetime
from time import time

import numpy as np
import pandas as pd

import pickle

from global_const import *

with open('bitcoin_prices_test.pkl', 'wb') as f:
    pickler = pickle.Pickler(f)
    pickler.dump([100, 956])
    pickler.dump(['timestamp', 'marketA', 'marketB', 'marketC', 'marketD'])

    for i in range(8):
        temp = np.random.randn(100,5) * 5 + 1000
        temp[:,0] = np.arange((i + 1) * 100, (i + 2) * 100)
        pickler.dump(temp)

    temp = np.random.randn(57,5) * 5 + 1000
    temp[:,0] = np.arange(900,957)
    pickler.dump(temp)

with open('bitcoin_prices_test.pkl', 'rb') as f:
    unpickler = pickle.Unpickler(f)

    done = False
    while not done:
        try:
            temp = unpickler.load()
        except:
            done = True
            print('done')

fees = pd.DataFrame(
    index=['marketA','marketB','marketC','marketD'],
    columns=['pair','BC_withdraw','BC_deposit','USD_withdraw','USD_deposit']
)

fees.loc['marketA'] = [0.0025,     0, 0,   0.02, 0.02]
fees.loc['marketB'] = [     0,     0, 0, 0.0038,    0]
fees.loc['marketC'] = [0.002 , 0.001, 0,   0.01,    0]
fees.loc['marketD'] = [  0.01,     0, 0,   0.01, 0.01]

fees['BC_deposit'] = 0.00001

fees.to_csv('bitcoin_fees_test.csv')
