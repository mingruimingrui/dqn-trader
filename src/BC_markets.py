# Took from one of my other projects
# https://github.com/mingruimingrui/myEnv/blob/master/Env.py
# Here we have to have a stack of incoming transactions and data has to be
# read through a stream

import os
import types
import copy
import pickle
import warnings
from functools import reduce

import numpy as np
import pandas as pd

# warnings.filterwarnings('error')

def isListLike(x):
    return isinstance(x, np.ndarray) | isinstance(x, list) | isinstance(x, tuple)

def calcStateValue(acc_state, trans_acc_state, bc_price_state):
    """
    calcStateValue does not need to be a visible function
    it will take into account the current bc_prices
    to calculate the total value of an account state and incoming transactions
    """
    total_state = acc_state + trans_acc_state

    state_value = np.sum(total_state[0])
    state_value += np.sum(total_state[1] * bc_price_state)

    return state_value

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
        self.data_store = self.data_store[self.data_store[:,0] >= self.start]
        self.data_store = self.data_store[self.data_store[:,0] <= self.end]
        self.data_store = self.data_store[(self.data_store[:,0] - self.start) % self.step_size == 0]
        self.is_out_of_states = False

        while(len(self.data_store) < lookback):
            new_data = self.data_file_reader.load()
            new_data = new_data[new_data[:,0] >= self.start]
            new_data = new_data[new_data[:,0] <= self.end]
            new_data = new_data[(new_data[:,0] - self.start) % self.step_size == 0]

            self.data_store = np.concatenate((self.data_store, new_data))

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
                new_data = new_data[new_data[:,0] >= self.start]
                new_data = new_data[new_data[:,0] <= self.end]
                new_data = new_data[(new_data[:,0] - self.start) % self.step_size == 0]

                self.data_store = np.concatenate((self.data_store, new_data))
        except:
            self.is_out_of_states = True

        self.data_store = self.data_store[1:]

        return next_timestamps, next_price_state, self.is_out_of_states

class Env:
    """
    Home made Env, works like OpenAI gym but adjusted for bitcoin trading in
    multiple markets
    Here we assume that on every trading platform we don't import private keys
    (it's dangerous AF) and we transfer USD using the fastest means possible
    (typically debit/credit transfer).
    We also hard code the transaction duration of
    - 1 hour for bitcoin transfers
    - 1 day for cash transfers
    - 5 seconds for pair trades

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

    start, end: int (Optional)
        Set start and end time, inclusive
        If not set, then all timestamps taken

    step_size: int (Default: 1)
        With every step, how many timesteps to move forward to next state
        Minimum 1

    lookback: int (Default: 5)
        Signifies the number of timesteps to look back when returning state
        Try not to have lookback < step_size for obvious reasons
        Minimum 1

    init_acc_state: np.ndarray (Optional)
        Initial account position of USD and BITCOIN of each market
        Array should be of size 2 * num_markets
        Columns of this array should represent each market in order as defined
        in the colum names of your data file
        Rows of this array should represent the USD and Bitcoin account
        eg. [[  100,   100,   100],
             [0.010, 0.009, 0.013]]
            for column names of ['timestamp', 'marketA', 'marketB', 'marketC']
        If not set, then assume initial equal allocation

    minimum_action_limit: int (Default: 0.1)
        Actions smaller than this size should be ignored
        Read about actions under the step function

    """

    def __init__(self, data_path, fees, start=None, end=None,
        step_size=1, lookback=5, init_acc_state=None, minimum_action_limit=0.1):

        # Assess validity of data_path, step_size, and lookback
        assert isinstance(data_path, str), 'data_path must be a string'
        assert os.path.isfile(data_path), 'data_path must point to a file'

        assert isinstance(fees, pd.DataFrame), 'fees must be a pd.DataFrame object'

        assert isinstance(step_size, int), 'step_size must be an int'
        assert step_size >= 1, 'step_size must be 1 or greater'

        assert isinstance(lookback, int), 'lookback must be an int'
        assert lookback >= 1, 'lookback must be 1 or greater'

        # Open file and store reader
        self.data_path = data_path
        self.data_file = open(self.data_path, 'rb')
        self.data_file_reader = pickle.Unpickler(self.data_file)

        # Extract time_interval and column names plus other info
        time_interval = self.data_file_reader.load()
        self.column_names = self.data_file_reader.load()
        self.market_names = self.column_names[1:]
        self.currency_list = ['USD', 'BC']

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

        is_time_period_valid = self.latest_time - self.earliest_time + 1 >= (lookback + 1) * step_size
        assert is_time_period_valid, 'not enough timesteps'

        # Define action info and shape
        self.action_info = [{
            'type': 'pair',
            'from': market_name, 'from_currency': 'USD', 'from_x': 0, 'from_y': i,
            'to': market_name, 'to_currency': 'BC', 'to_x': 1, 'to_y': i,
            'fee_long': fees.loc[market_name, 'pair'], 'fee_short': fees.loc[market_name, 'pair']
        } for i, market_name in enumerate(self.market_names)]

        for i, market_name1 in enumerate(self.market_names):
            self.action_info += [{
                'type': 'USD',
                'from': market_name1, 'from_currency': 'USD', 'from_x': 0, 'from_y': i,
                'to': market_name2, 'to_currency': 'USD', 'to_x': 0, 'to_y': i + j + 1,
                'fee_long':  fees.loc[market_name1, 'USD_withdraw'] + fees.loc[market_name2, 'USD_deposit'],
                'fee_short': fees.loc[market_name2, 'USD_withdraw'] + fees.loc[market_name1, 'USD_deposit']
            } for j, market_name2 in enumerate(self.market_names[i+1:])]

        for i, market_name1 in enumerate(self.market_names):
            self.action_info += [{
                'type': 'BC',
                'from': market_name1, 'from_currency': 'BC', 'from_x': 1, 'from_y': i,
                'to': market_name2, 'to_currency': 'BC', 'to_x': 1, 'to_y': i + j + 1,
                'fee_long':  fees.loc[market_name1, 'BC_withdraw'] + fees.loc[market_name2, 'BC_deposit'],
                'fee_short': fees.loc[market_name2, 'BC_withdraw'] + fees.loc[market_name1, 'BC_deposit']
            } for j, market_name2 in enumerate(self.market_names[i+1:])]

        # simply for reference by user does not serve any further purpose
        self.action_label = [
            info['from'] + '_' + info['from_currency'] + ' to ' +
            info['to'] + '_' + info['to_currency']
            for info in self.action_info
        ]

        self.action_shape = (len(self.action_info),)

        # when performing actions we need to regularize each action according to
        # the available funds in each acc
        # we create an action_acc_outflow helper to assist in this process
        self.action_acc_outflow = {}
        for market_name in self.market_names:
            self.action_acc_outflow[market_name] = {
                'USD':[], 'BC':[]
            }

        # Store rest of env parameters
        self.step_size = step_size
        self.lookback = lookback
        self.minimum_action_limit = minimum_action_limit

        # initialize price_state_handler to help retrieve prices at correct time points
        self.price_state_handler = PriceStateHandler(self.data_file_reader,
            self.step_size, self.lookback,
            self.earliest_time, self.latest_time)

        # initialize incoming_transactions,
        # cur_timestamps, cur_time
        # cur_price_state, cur_acc_state
        # also store init_acc_state for reset
        self.cur_timestamps, self.cur_price_state, _ = self.price_state_handler.getNextState()
        self.cur_time = self.cur_timestamps[-1]

        # now that we finally got the price_states, we can initialize the acc states
        acc_state_shape = (2, len(self.market_names))
        if init_acc_state is not None:
            assert isinstance(init_acc_state, np.ndarray), 'init_acc_state must be an np.ndarray'

            is_acc_state_right_shape = init_acc_state.shape == acc_state_shape
            assert is_acc_state_right_shape, 'acc states are not of correct shape'

            self.init_acc_state = init_acc_state
        else:
            # if init_acc_state is None then we do equal allocation
            self.init_acc_state = np.ones(acc_state_shape)
            self.init_acc_state = self.init_acc_state / 2 / len(self.market_names)
            self.init_acc_state[1,:] /= self.cur_price_state[-1]

        self.cur_acc_state = self.init_acc_state

        self.incoming_transactions = []
        # we name the total of all incoming transactions the trans_acc_state
        self.trans_acc_state = np.zeros(acc_state_shape)

        self.cur_state_value = calcStateValue(self.cur_acc_state,
            self.trans_acc_state, self.cur_price_state[-1])

        self.is_env_open = True

        print('New Env initiated')
        print('Timestamps from', self.earliest_time, 'to', self.latest_time)
        print('Step size:', step_size)
        print('Lookback:', lookback, '\n')

    def getRandomAction(self):
        """
        we got to improve this
        """
        return np.random.rand(self.action_shape[0]) * 2 - 1

    def step(self, action):
        """
        action: list-like
        """

        assert self.is_env_open, 'Enviornment has been closed'
        assert self.cur_time + self.step_size < self.latest_time, 'No more time steps left'

        assert isListLike(action), 'action must be list-like'
        assert len(action) == self.action_shape[0], 'action must be correct shaped'

        # carry out incoming transactions
        kept_transactions = []

        for incoming_transaction in self.incoming_transactions:
            if self.cur_time + self.step_size >= incoming_transaction['timestamp']:
                self.cur_acc_state[
                    incoming_transaction['x'], incoming_transaction['y']
                ] += incoming_transaction['volume']

                self.trans_acc_state[
                    incoming_transaction['x'], incoming_transaction['y']
                ] -= incoming_transaction['volume']

            else:
                kept_transactions.append(incoming_transaction)

        self.incoming_transactions = kept_transactions

        # generate next price state
        next_timestamps, next_price_state, is_out_of_states = self.price_state_handler.getNextState()

        # new_state_has_nan = np.sum(np.isnan(next_price_state)) > 0
        new_state_has_nan = np.isnan(next_price_state).all()

        # if there are nans in the new state then we skip this state
        if new_state_has_nan:
            reward = None

            # update states and check if done
            self.cur_price_state = next_price_state
            self.cur_timestamps = next_timestamps
            self.cur_time = self.cur_timestamps[-1]

            if self.cur_time + self.step_size > self.latest_time:
                done = True
            else:
                done = False

            return self.cur_price_state, self.cur_acc_state, reward, done

        # if there are no nans in the new state then we cont as per normal
        # starting with the normalization and performing of actions

        # if action is below the min action_limit, then we do not perform it
        action = [(0 if a < self.minimum_action_limit else a) for a in action]

        # perform normalization of actions starting by grouping accoriding
        # to accounts that we short from
        action_acc_outflow = copy.deepcopy(self.action_acc_outflow)
        for i, a in enumerate(action):
            action_info = copy.deepcopy(self.action_info[i])

            if a > 0:
                # is long
                action_info['a'] = a
                action_acc_outflow[action_info['from']][action_info['from_currency']] += [action_info]
            else:
                # is short
                action_info['a'] = a
                action_acc_outflow[action_info['to']][action_info['to_currency']] += [action_info]

        for i, market_name in enumerate(self.market_names):
            for j, currency in enumerate(['USD', 'BC']):
                actions_w_info = action_acc_outflow[market_name][currency]
                # normalize the actions
                action_mult = 1 / max(reduce(lambda x,y: x + abs(y['a']), actions_w_info, 0), 1)
                available_funds = self.cur_acc_state[j,i]
                available_funds_value = available_funds * (1 if currency == 'USD' else self.cur_price_state[-1, i])

                for action_w_info in actions_w_info:
                    if action_w_info['type'] == 'pair':
                        # is pair trade
                        mult = next_price_state[-1,action_w_info['to_y']]
                        delay = 5
                    else:
                        mult = 1
                        if action_w_info['type'] == 'USD':
                            # is cash trade
                            delay = 60 * 60 *24
                        else:
                            # is BC trade
                            delay = 60 * 60

                    if action_w_info['a'] > 0:
                        # is long
                        # available_funds = self.cur_acc_state[action_w_info['from_x'],action_w_info['from_y']]
                        # available_funds_value = available_funds * (
                        #     1 if action_w_info['from_currency'] == 'USD' else self.cur_price_state[-1,action_w_info['from_y']])

                        if available_funds_value < self.cur_state_value * 0.01:
                            transaction_volume = available_funds
                            available_funds = 0
                        else:
                            transaction_volume = available_funds * action_w_info['a'] * action_mult

                        self.cur_acc_state[action_w_info['from_x'],action_w_info['from_y']] -= transaction_volume
                        self.incoming_transactions.append({
                            'volume': transaction_volume * (1-action_w_info['fee_long']) / mult,
                            'timestamp': self.cur_time + delay,
                            'x': action_w_info['to_x'],
                            'y': action_w_info['to_y']
                        })

                        self.trans_acc_state[action_w_info['to_x'],action_w_info['to_y']] += (
                            transaction_volume * (1-action_w_info['fee_long']) / mult)

                    else:
                        # is short
                        # available_funds = self.cur_acc_state[action_w_info['to_x'],action_w_info['to_y']]
                        # available_funds_value = available_funds * (
                        #     1 if action_w_info['to_currency'] == 'USD' else self.cur_price_state[-1,action_w_info['to_y']])

                        if available_funds_value < self.cur_state_value * 0.01:
                            transaction_volume = available_funds
                            available_funds = 0
                        else:
                            transaction_volume = available_funds * -action_w_info['a'] * action_mult

                        self.cur_acc_state[action_w_info['to_x'],action_w_info['to_y']] -= transaction_volume
                        self.incoming_transactions.append({
                            'volume': transaction_volume * (1-action_w_info['fee_short']) / mult,
                            'timestamp': self.cur_time + delay,
                            'x': action_w_info['from_x'],
                            'y': action_w_info['from_y']
                        })

                        self.trans_acc_state[action_w_info['from_x'],action_w_info['from_y']] += (
                            transaction_volume * (1-action_w_info['fee_short']) / mult)


        for i, a in enumerate(action):
            action_info = self.action_info[i]

            if action_info['type'] == 'pair':
                # is pair trade
                mult = next_price_state[-1,action_info['to_y']]
                delay = 5
            else:
                multi = 1
                if action_info['type'] == 'USD':
                    # is cash trade
                    delay = 60 * 60 *24
                else:
                    # is BC trade
                    delay = 60 * 60


        # calculate reward
        next_state_value = calcStateValue(self.cur_acc_state,
            self.trans_acc_state, next_price_state[-1])
        reward = np.log(next_state_value / self.cur_state_value)

        # update states and check if done
        self.cur_state_value = next_state_value
        self.cur_price_state = next_price_state
        self.cur_timestamps = next_timestamps
        self.cur_time = self.cur_timestamps[-1]

        if self.cur_time + self.step_size > self.latest_time:
            done = True
        else:
            done = False

        return self.cur_price_state, self.cur_acc_state, self.trans_acc_state, reward, done

    def reset(self):
        """
        resets your env with same init values
        """

        assert self.is_env_open, 'Enviornment is closed'

        # reopen data file
        self.data_file.close()
        self.data_file = open(self.data_path, 'rb')
        self.data_file_reader = pickle.Unpickler(self.data_file)

        # remove batches containing time_interval and column names
        self.data_file_reader.load()
        self.data_file_reader.load()

        self.price_state_handler = PriceStateHandler(self.data_file_reader,
            self.step_size, self.lookback,
            self.earliest_time, self.latest_time)

        self.incoming_transactions = []
        self.trans_acc_state = np.zeros(self.cur_acc_state.shape)

        self.cur_timestamps, self.cur_price_state, _ = self.price_state_handler.getNextState()
        self.cur_time = self.cur_timestamps[-1]

        self.cur_acc_state = self.init_acc_state

        print('Env reset')

    def close(self):
        """
        closes all opened files and turns env off
        """
        self.price_state_handler.is_out_of_states = True
        self.data_file.close()
        self.is_env_open = False

        print('Env closed')
