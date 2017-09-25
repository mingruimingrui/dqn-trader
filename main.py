import os
import datetime
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
# import seaborn as sns
from src.env import env_make

# some params if ran as main
DATA_FILE_PATH = 'data/snp500_transformed.npz'
SYMS_TO_USE = ['AAP', 'GOOGL']  # None for all assets [syms] for multiple assets
START = datetime.datetime(2005,1,1)
END   = datetime.datetime(2017,9,16)
LOOKBACK = 5

def load_data():
    file = np.load(DATA_FILE_PATH)
    return file['timestamps'], file['syms'], file['col_names'], file['data']

def load_data_make_env(syms_to_use=None, start=None, end=None, lookback=5):
    timestamps, syms, col_names, data = load_data()
    return env_make(timestamps, syms, col_names, data, syms_to_use, start, end, lookback)

def main(syms_to_use=None, start=None, end=None, lookback=None):
    assert os.path.exists(DATA_FILE_PATH), 'Have you ran preprocess.py and data_transform.py yet?'

    # simple override if called from another file
    if syms_to_use != None:
        SYMS_TO_USE = syms_to_use
    if start != None:
        START = start
    if end != None:
        END = end
    if lookback != None:
        LOOKBACK = lookback

    # load up data and create enviornment
    env = load_data_make_env(syms_to_use=SYMS_TO_USE)
    # we just give weights of 1 to every asset
    action = np.ones(env.action_shape)
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
            plt.plot(range(len(rewards)), np.cumprod(rewards))
            plt.title('Value over time')
            plt.figure()
            plt.plot(range(len(rewards)), rewards)
            plt.title('Rewards over time')
            plt.figure()
            plt.plot(range(len(sharpe)), sharpe)
            plt.title('Sharpe over time')
            plt.show()

if __name__ == '__main__':
    print('main.py started')
    main()
    print('main.py finished')
