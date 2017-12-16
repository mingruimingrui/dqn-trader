# Took from one of my other projects
# https://github.com/mingruimingrui/myEnv/blob/master/Env.py
# Here we have to have a stack of incoming transactions and data has to be
# read through a stream

import os
import types

import numpy as np

import pickle

def isListLike(x):
    return isinstance(x, np.ndarray) | isinstance(x, list) | isinstance(x, tuple)

class PriceStateHandler:
    """
    Home made class meant to facilitate with the getting of price states from
    correct timeframes

    data_file_reader: pickle.Unpickler
    on next load gets the first chunk of data

    step_size, lookback:
    same from Env
    """
    def __init__(self, data_file_reader, step_size, lookback, start, end):
        self.data_file_reader = data_file_reader
        self.step_size = step_size
        self.lookback = lookback
        self.start = start
        self.end = end
        # data_store is something to be topped up
        self.data_store = data_file_reader.load()
        self.data_store = self.data_store[self.data_store[:,0] >= start]
        self.data_store = self.data_store[self.data_store[:,0] <= end]
        self.is_out_of_states = False

        while(len(self.data_store) < lookback):
            self.data_store = np.concatenate(
                self.data_store,
                self.data_file_reader.load()
            )
            self.data_store = self.data_store[self.data_store[:,0] >= start]
            self.data_store = self.data_store[self.data_store[:,0] <= end]

    def getNextState(self):
        """
        returns next_price_state, is_out_of_states
        """

        assert self.is_out_of_states is False, 'no more price states to fetch please re_initialize'

        next_timestamps = self.data_store[:self.lookback,0]
        next_price_state = self.data_store[:self.lookback,1:]

        try:
            while(len(self.data_store) < self.lookback + self.step_size):
                new_data = self.data_file_reader.load()
                self.data_store = np.concatenate((self.data_store, new_data))
                self.data_store = self.data_store[self.data_store[:,0] >= self.start]
                self.data_store = self.data_store[self.data_store[:,0] <= self.end]
        except:
            self.is_out_of_states = True

        self.data_store = self.data_store[self.step_size:]

        return next_timestamps, next_price_state, self.is_out_of_states

class Env:
    """
    Home made Env, works like OpenAI gym but adjusted for bitcoin trading in
    multiple markets
    Here we assume that on every trading platform we don't import private keys
    (it's dangerous AF) and we transfer USD using the fastest means possible
    (typically debit/credit transfer).
    We also hard code the transaction duration of
    - 1 hour for bitcoin transactions
    - 10 minutes for cash transfer
    -

    data_path: string
        Should point to the path of a pkl file storing data in the following
        batch format
        - Data file's first batch should be a list [earliest_time, latest_time]
        - Data file's second batch should be column names with the first column
        being 'timestamp' rest of which are market names
        eg. ['timestamp', 'marketA', 'marketB', ...]
        - Rest of batches should be data tables of class np.ndarray and
        of shape N * len(column_names)

    fees: pd.DataFrame
        Should be a table of transaction fees with the following properties
        - Rows are market names
        - Columns are ['pair', 'BC_withdraw', 'BC_deposit', 'USD_withdraw', 'USD_deposit']

    start, end: int (optional)
        Set start and end time, inclusive
        If not set, then all timestamps taken

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
        Rows of this array should represent the USD and Bitcoin account
        eg. [[  100,   100,   100],
             [0.010, 0.009, 0.013]]
            for column names of ['timestamp', 'marketA', 'marketB', 'marketC']
        If not set, then assume initial equal allocation

    """

    def __init__(self, data_path, fees, start=None, end=None,
        step_size=1, lookback=5, init_acc_state=None):

        # Assess validity of data_path, step_size, and lookback
        assert isinstance(data_path, str), 'data_path must be a string'
        assert os.path.isfile(data_path), 'data_path must point to a file'

        assert isinstance(fees, pd.DataFrame), 'fees must be a pd.DataFrame object'

        assert isinstance(step_size, int), 'step_size must be an int'
        assert step_size >= 1, 'step_size must be 1 or greater'

        assert isinstance(lookback, int), 'lookback must be an int'
        assert lookback >= 1, 'lookback must be 1 or greater'

        if init_acc_state is not None:
            assert isinstance(init_acc_state, np.ndarray), 'init_acc_state must be an np.ndarray'

        # Open file and store reader
        self.data_path = data_path
        self.data_file = open(self.data_path, 'rb')
        self.data_file_reader = pickle.Unpickler(self.data_file)

        # Extract time_interval and column names plus other info
        time_interval = self.data_file_reader.load()
        self.column_names = self.data_file_reader.load()
        self.num_markets = len(self.column_names) - 1

        # Assess validity of start, end, timeframe, and init_acc_state
        if start is not None:
            assert isinstance(start, int), 'start must be int'
            self.earliest_time = max(time_interval[0], start)
        else:
            self.earliest_time = time_interval[0]

        if end is not None:
            assert isinstance(end, int), 'end must be int'
            self.latest_time = min(time_interval[1], end)
        else:
            self.latest_time = time_interval[1]

        is_time_period_valid = self.latest_time - self.earliest_time + 1 >= lookback + step_size
        assert is_time_period_valid, 'not enough timesteps'

        if init_acc_state is not None:
            assert isinstance(init_acc_state, np.ndarray), 'init_acc_state must be an np.ndarray'

            is_acc_state_right_shape = init_acc_state.shape == (2, len(self.column_names) - 1)
            assert is_acc_state_right_shape, 'acc states are not of correct shape'

            self.cur_acc_state = init_acc_state

        # Define action label and shape
        market_names = self.column_names[1:]
        self.action_label = [market_name + 'USD_' + market_name + 'BC'
            for market_name in market_names]

        for i, market_name1 in enumerate(market_names):
            self.action_label += [market_name1 + 'USD_' + market_name2 + 'USD'
                for market_name2 in market_names[i+1:]]
            self.action_label += [market_name1 + 'BC_' + market_name2 + 'BC'
                for market_name2 in market_names[i+1:]]

        self.action_shape = (len(self.action_label),)

        # Store rest of env parameters
        self.fees = fees
        self.step_size = step_size
        self.lookback = lookback

        self.state_handler = PriceStateHandler(self.data_file_reader,
            self.step_size, self.lookback,
            self.earliest_time, self.latest_time)

        self.incoming_transactions = []
        self.cur_timestamps, self.cur_price_state, _ = self.state_handler.getNextState()
        self.init_acc_state = init_acc_state

        # if init_acc_state is None then we do equal allocation
        if self.init_acc_state is None:
            self.cur_acc_state = np.ones((2, self.num_markets))
            self.cur_acc_state = self.cur_acc_state / 2 / self.num_markets
            self.cur_acc_state[1,:] = self.cur_acc_state[1,:] / self.cur_price_state[-1]

        self.cur_time = self.cur_timestamps[-1]
        self.next_time = self.cur_time + self.step_size

        self.is_env_open = True

        print('New Env initiated')
        print('Timestamps from', self.earliest_time, 'to', self.latest_time)
        print('Step size:', step_size)
        print('Lookback:', lookback)

    def step(self, action):
        """
        action: list-like
        """

        assert self.is_env_open, 'Enviornment is closed'
        assert self.next_time is not None, 'no more steps left in env, please reset'

        assert isListLike(action), 'action must be like-like'
        assert len(action) == self.action_shape, 'action must be correct shaped'

        # carry out action


        # generate new state
        next_timestamps, next_price_state, is_out_of_states = self.state_handler.getNextState()

        # calculate reward

        # check if done

        # update states


        return next_price_state, next_acc_state, reward, done

    def reset(self):
        """
        resets your env with same init values
        """

        assert self.is_env_open, 'Enviornment is closed'

        self.data_file.close()
        self.data_file = open(self.data_path, 'rb')
        self.data_file_reader = pickle.Unpickler(self.data_file)

        # remove batches containing time_interval and column names
        self.data_file_reader.load()
        self.data_file_reader.load()

        self.state_handler = PriceStateHandler(self.data_file_reader,
            self.step_size, self.lookback,
            self.earliest_time, self.latest_time)

        self.incoming_transactions = []
        self.cur_timestamps, self.cur_price_state, _ = self.state_handler.getNextState()

        if self.init_acc_state is None:
            self.cur_acc_state = np.ones((2, self.num_markets))
            self.cur_acc_state = self.cur_acc_state / 2 / self.num_markets
            self.cur_acc_state[1,:] = self.cur_acc_state[1,:] / self.cur_price_state[-1]

        self.cur_time = self.cur_timestamps[-1]
        self.next_time = self.cur_time + self.step_size

        print('Env reset')


    def close(self):
        """
        closes all opened files and turns env off
        """
        self.state_handler.is_out_of_states = True
        self.data_file.close()
        self.is_env_open = False

        print('Env closed')
