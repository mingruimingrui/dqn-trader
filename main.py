import os
import datetime
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
# import seaborn as sns
from src.env import Env

# some params if ran as main
DATA_FILE_PATH = 'data/snp500_transformed.npz'
SYMS_TO_USE = ['GOOGL', 'AAPL', 'BRK-B']  # None for all assets [syms] for multiple assets
START = datetime.datetime(2005,1,1)
END   = datetime.datetime(2017,9,16)
LOOKBACK = 5

def load_data():
    file = np.load(DATA_FILE_PATH)
    return pd.DatetimeIndex(file['timestamps']), file['syms'], file['col_names'], file['data']

def load_data_make_env(syms_to_use=None, start=None, end=None, lookback=5):
    timestamps, syms, col_names, data = load_data()
    return Env(timestamps, syms, col_names, data, syms_to_use, start, end, lookback)

def main(syms_to_use=None, start=None, end=None, lookback=None):
    assert os.path.exists(DATA_FILE_PATH), 'Have you ran preprocess.py and data_transform.py yet?'

    # simple override if called from another file
    SYMS_TO_USE_ = SYMS_TO_USE
    START_ = START
    END_ = END
    LOOKBACK_ = LOOKBACK

    if syms_to_use != None:
        SYMS_TO_USE_ = syms_to_use
    if start != None:
        START_ = start
    if end != None:
        END_ = end
    if lookback != None:
        LOOKBACK_ = lookback

    # load up data and create enviornment
    env = load_data_make_env(syms_to_use=SYMS_TO_USE_, start=START_, end=END_, lookback=LOOKBACK_)
    action = np.ones(env.action_shape)  # we just give weights of 1 to every asset for now
    val = 1

    # these are totally optional but I put them here for visualization later on
    rewards = []
    sharpe = []

    # typically we start off with a random action just to get an initial state
    # take note that state is noramlized according to the current open prices at env.cur_time
    # I do it this way as all I'm insterested in is how each asset changed over the course of lookback
    state, time, reward, done = env.step(env.random_action())

    while not done:
        # you can build your model here
        # mainly you will have to make use of state time and previous prewards to create an action
        # which should be a numpy array of shape env.action_shape
        # all raw datas are in env.data but I trust that you won't use it
        #*************************************************************************#
        #*********************  reserved for model building  *********************#
        #*************************************************************************#



        #*************************************************************************#
        #*************************************************************************#
        #*************************************************************************#

        state, time, reward, done = env.step(action)
        val *= reward

        # I append rewards and sharpe here for visualization later
        rewards.append(reward)
        sharpe.append((np.sum(rewards) - len(rewards)) / np.sqrt(np.var(rewards) * len(rewards)))

        if done:
            # once done, you can visualize you results
            print('Final value:', val)
            print('Yearly reward:', val**(252 / len(env.timestamps)))
            print('Sharpe ratio:', (np.sum(rewards) - len(rewards)) / np.sqrt(np.var(rewards) * len(rewards)))
            plt.plot(range(len(rewards))[50:], np.cumprod(rewards)[50:])
            plt.title('Value over time')
            plt.figure()
            plt.plot(range(len(rewards))[50:], rewards[50:])
            plt.title('Rewards over time')
            plt.figure()
            plt.plot(range(len(sharpe))[50:], sharpe[50:])
            plt.title('Sharpe over time')
            plt.show()

if __name__ == '__main__':
    print('main.py started')
    main()
    print('main.py finished')
