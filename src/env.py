# Took from one of my other projects
# https://github.com/mingruimingrui/myEnv/blob/master/Env.py
# Here we have to have a stack of incoming transactions and data has to be
# read through a stream

import os
import types

import numpy as np

import pickle

class PriceStateHandler:
    """
    Home made class meant to facilitate with the getting of price states from
    correct timeframes

    data_file_reader: pickle.Unpickler
    on next load gets the first chunk of data

    time_col: the position where the timestamp column is located

    step_size, lookback:
    same from Env
    """
    def __init__(self, data_file_reader, time_col, step_size, lookback, start, end):
        self.data_file_reader = data_file_reader
        self.time_col = time_col
        self.step_size = step_size
        self.lookback = lookback
        self.start = start
        self.end = end
        # data_store is something to be topped up
        self.data_store = data_file_reader.load()
        self.data_store = self.data_store[self.data_store[:,0] >= start]
        self.data_store = self.data_store[self.data_store[:,0] <= end]
        self.isOutOfStates = False

        while(len(self.data_store) < lookback):
            self.data_store = np.concatenate(
                self.data_store,
                self.data_file_reader.load()
            )
            self.data_store = self.data_store[self.data_store[:,0] >= start]
            self.data_store = self.data_store[self.data_store[:,0] <= end]

    def getNextState(self):
        """
        returns next_price_state, isOutOfStates
        """

        assert self.isOutOfStates is False, 'no more price states to fetch please re_initialize'

        next_price_state = self.data_store[:self.lookback]

        try:
            while(len(self.data_store) < self.lookback + self.step_size):
                new_data = self.data_file_reader.load()
                self.data_store = np.concatenate((self.data_store, new_data))
                self.data_store = self.data_store[self.data_store[:,0] >= self.start]
                self.data_store = self.data_store[self.data_store[:,0] <= self.end]
        except:
            self.done = True

        self.data_store = self.data_store[self.step_size:]

        return next_price_state, self.isOutOfStates

class Env:
    """
    Home made Env, works like OpenAI gym but adjusted for bitcoin trading in
    multiple markets

    data_path: string
        Should point to the path of a pkl file storing table like data

        - Data file's first row should be a list [earliest_time, latest_time]
        - Data file's second row should be column names with the first column
        being 'timestamp' rest of which are market names
        eg. ['timestamp', 'marketA', 'marketB', ...]
        - Rest of data file should be data tables of class np.ndarray and
        of shape N * len(column_names)

    start, end: int (optional)
        Set start and end time, inclusive.

    step_size: int (Default: 1)
        With every step, how many timesteps to move forward to next state
        Minimum 1

    lookback: int (Default: 5)
        Signifies the number of timesteps to look back when returning state
        Try not to have lookback < step_size for obvious reasons
        Minimum 1

    init_acc_state: np.ndarray (optional)
        Initial account position of USD and BITCOIN of each market
        Array should be of size 2 * num_markets
        Columns of this array should represent each market in order as defined
        in the colum names of your data file
        Rows of this array should represent the money and Bitcoin account
        eg. [[  100,   100,   100],
             [0.010, 0.009, 0.013]]
            for column names of ['timestamp', 'marketA', 'marketB', 'marketC']

    """

    def __init__(self, data_path, start=None, end=None,
        step_size=1, lookback=5, init_acc_state=None):

        assert isinstance(data_path, str), 'data_path must be a string'
        assert os.path.isfile(data_path), 'data_path must point to a file'

        if start is not None:
            assert isinstance(start, int), 'start must be int'

        if end is not None:
            assert isinstance(end, int), 'end must be int'

        assert isinstance(step_size, int), 'step_size must be an int'
        assert step_size >= 1, 'step_size must be 1 or greater'

        assert isinstance(lookback, int), 'lookback must be an int'
        assert lookback >= 1, 'lookback must be 1 or greater'

        self.data_path = data_path
        self.data_file = open(data_path, 'rb')
        self.data_file_reader = pickle.Unpickler(self.data_file)

        time_interval = self.data_file_reader.load()
        self.earliest_time = max(time_interval[0], start)
        self.latest_time = min(time_interval[1], end)
        self.column_names = self.data_file_reader.load()

        # now that we got the time period, we also check if it is valid
        is_time_period_valid = self.latest_time - self.earliest_time + 1 >= lookback + step_size
        assert is_time_period_valid, 'not enough timesteps'

        self.step_size = step_size
        self.lookback = lookback
        self.time_col = column_names.index('timestamp')

        self.state_handler = PriceStateHandler(self.data_file_reader,
            self.time_col, step_size, lookback, start, end)

        self.cur_state, _ = self.state_handler.getNextState()
        self.incoming_transactions = []


        self.cur_time = self.cur_state[-1,self.time_col]
        self.next_time = self.cur_time + self.step_size

        print('New Env initiated')
        print('Timestamps from', self.earliest_time, 'to', self.latest_time)
        print('Step size:', step_size)
        print('Lookback:', lookback)

    def step(self, action):
        """
        action: <your condition in getReward>
        """

        assert self.next_time is not None, 'no more steps left in env, please reset'

        # generate new state
        next_state, isOutOfStates = self.state_handler.getNextState()

        # calculate reward
        reward = self.getReward(self.cur_state, next_state, action)

        # check if done
        done = self.cur_time_index + self.step_size >= len(self.timestamps)

        # update state
        self.cur_time_index = self.cur_time_index + self.step_size
        self.cur_data_index = self.cur_data_index + self.step_size
        self.cur_time = timestamps[self.cur_time_index]
        self.cur_state = next_state

        if done:
            self.next_time = None
        else:
            self.next_time = timestamps[self.cur_time_index + self.step_size]

        return next_state, reward, done

    def reset():
        """
        resets your env with same init values
        """
        print('Env reset')
        self.cur_time_index = lookback - 1
        self.cur_data_index = 0
        self.cur_time = timestamps[self.cur_time_index]
        self.next_time = timestamps[self.cur_time_index + self.step_size]
        self.cur_state = self.data[self.cur_data_index:(self.cur_data_index + self.lookback)]

    def close():
        """
        closes all opened files and turns env off
        """
