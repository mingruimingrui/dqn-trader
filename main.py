import os
import sys
import pickle
import warnings
from optparse import OptionParser

from time import time
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from src.Env import Env

from data.global_const import *


# we want to avoid division by zero and not just throw an error
warnings.filterwarnings('error')


# Script options
parser = OptionParser()
parser.add_option("-f", "--file", dest="BITCOIN_DATA_FILE", default=DATA_PATH + BITCOIN_DATA_FILE_NAME,
                  help="change BITCOIN_DATA_FILE to use", metavar="BITCOIN_DATA_FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

(options, args) = parser.parse_args()


def main(argv):
    options,_ = parser.parse_args(argv)
    fees = pd.read_csv(DATA_PATH + BITCOIN_FEES_FILE_NAME, index_col=0)
    env = Env(options.BITCOIN_DATA_FILE, fees, lookback=300)

    done = False

    while not done:
        cur_price_state, cur_acc_state, trans_acc_state, reward, done = env.step(np.zeros(env.action_shape))

    print()
    print('done')


if __name__ == '__main__':
    print('main.py started')
    main(sys.argv[1:])
    print('main.py finished')
