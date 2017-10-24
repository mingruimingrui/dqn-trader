# home made env maker made to behave like openai gym
import os
from datetime import datetime, timedelta
import numpy as np

# helper function to calculate difference in weeks
def diff_week(d1, d2):
    monday1 = d1 - timedelta(days=d1.weekday())
    monday2 = d2 - timedelta(days=d2.weekday())

    return round((monday2 - monday1).days / 7)

# syms_to_use (list-like)[optional]: sets env to contain only syms in list
# start, end (datetime)[optional]: sets time period, inclusive
# lookback (int)[default 5]: sets lookback period
class Env:

    def __init__(self, timestamps, syms, col_names, data,
        syms_to_use=None, start=None, end=None, step_size='day', lookback=5):

        if syms_to_use is not None:
            assert isinstance(syms_to_use, list) | isinstance(syms_to_use, np.ndarray), 'syms_to_use must be list-like'
            if isinstance(syms_to_use, np.ndarray):
                assert len(syms_to_use, 1), 'syms_to_use must be vector'
            data = data[:, list(map(lambda x: x in syms_to_use, syms)), :]
            syms = syms_to_use

        if start is not None:
            assert isinstance(start, datetime), 'start must be datetime'
            data = data[timestamps >= start]
            timestamps = timestamps[timestamps >= start]

        if end is not None:
            assert isinstance(end, datetime), 'end must be datetime'
            data = data[timestamps <= end]
            timestamps = timestamps[timestamps <= end]

        assert step_size in ['day', 'week', 'month'], 'step_size must be "day", "week" or "month"'
        assert isinstance(lookback, int), 'lookback must be an integer'

        # state_ts_dict is basically a dictionary linking state number to the timestamp index
        # we want to figure out exactly which days to initiate trades
        state_ts_dict = []
        lookback_mult_dict = {'day': 1, 'week': 5, 'month': 20}

        # day is the simplest, we trade on every opening date and lookback the previous few open days
        if step_size is 'day':
            state_ts_dict = np.arange(lookback * lookback_mult_dict['day'], len(timestamps))

        # week is more complex, we want to trade on every mid week
        if step_size in ['week', 'month']:
            ts_week_number = list(map(lambda x: diff_week(x, timestamps[0]), timestamps))
            ts_week_number = np.array(ts_week_number)
            ts_index = np.arange(len(ts_week_number))

            # this while look will config state_timestamps_dict to a list of all timestamps for mid-week
            # will be usually a wed, sometimes tues or thurs, rarely mon or fri
            while len(ts_week_number) > 0:
                cur_week = ts_week_number[0]
                index_cur_week = ts_week_number == cur_week

                to_add_temp = ts_index[index_cur_week]
                state_ts_dict.append(to_add_temp[int(np.floor(len(to_add_temp)/2))])

                ts_index = ts_index[~index_cur_week]
                ts_week_number = ts_week_number[~index_cur_week]

            state_ts_dict = np.unique(state_ts_dict)

            if step_size is 'week':
                state_ts_dict = state_ts_dict[
                    state_ts_dict >= lookback * lookback_mult_dict['week']
                ]

            # for months we want to trade on the first mid-week of every month
            if step_size is 'month':
                state_ts_months = list(map(lambda x: x.month, timestamps[state_ts_dict]))

                prev_month = -1
                temp = []
                while len(state_ts_months) > 0:
                    if prev_month is not state_ts_months[0]:
                        prev_month = state_ts_months[0]
                        temp.append(state_ts_dict[0])
                    state_ts_months = state_ts_months[1:]
                    state_ts_dict = state_ts_dict[1:]

                state_ts_dict = np.array(temp)
                state_ts_dict = state_ts_dict[
                    state_ts_dict >= lookback * lookback_mult_dict['month']
                ]

        assert len(state_ts_dict) > 0 , 'not enough timestamps for your env'
        assert data.shape[1]      > 0 , 'not enough assets in your env'
        assert data.shape[2]      > 0 , 'not enough cols in your env'

        self.timestamps = timestamps
        self.syms = syms
        self.col_names = col_names
        self.data = data
        self.step_size = step_size
        self.lookback = lookback
        self.lookback_mult = lookback_mult_dict[self.step_size]
        self.action_shape = (len(self.syms),)
        self.state_ts_dict = state_ts_dict
        self.state_nb = 0
        self.max_state_nb = len(self.state_ts_dict) - 1
        self.cur_time = timestamps[self.state_ts_dict[self.state_nb]]
        self.next_time = timestamps[self.state_ts_dict[self.state_nb + 1]]

        print('env initiated')
        print(min(self.timestamps).strftime('%d %b %y'), 'to', max(self.timestamps).strftime('%d %b %y'))
        print('step size:', self.step_size)
        print('lookback :', self.lookback)

    def random_action(self):
        """
        random_action()
        randomly generates an action
        """
        action = np.random.random(self.action_shape)
        while np.sum(action) == 0:
            action = np.random.random(self.action_shape)
        action /= np.sum(action)

        return action

    def step(self, action):
        """
        step(action)
        action must be an array of size env.action_shape
        """
        assert self.state_nb <= self.max_state_nb, 'no more steps left in env, please reset'
        assert action.shape == self.action_shape, 'action is wrong shape'
        assert np.sum(action) > 0, 'action sum must be greater than 0'
        # if np.sum(abs(action[1:] / np.sum(action)) > 0.17):
        #     print('UserWarning: Asset has more than 0.17% allocation')

        # generate new state and normalize
        state_ts_index = self.state_ts_dict[self.state_nb]
        state = self.data[state_ts_index - self.lookback * self.lookback_mult : state_ts_index, :, :].copy()
        mult = self.data[state_ts_index, :, self.col_names == 'open']
        mult = np.dot(np.ones((self.lookback * self.lookback_mult,1)), mult)
        mult = np.dot(np.expand_dims(mult, 2), np.ones((1, len(self.col_names))))
        state /= mult

        # calculate the timestamps for the state
        time = self.timestamps[state_ts_index - self.lookback * self.lookback_mult : state_ts_index]

        # normalize action and calculate reward
        action = action / np.sum(action)
        action = np.squeeze(action)
        open_p = np.squeeze(self.data[self.state_ts_dict[self.state_nb - 1], :, self.col_names == 'open'])
        open_n = np.squeeze(self.data[state_ts_index                       , :, self.col_names == 'open'])

        reward = np.sum(open_n / open_p * action)

        # check if done
        done = self.state_nb + 1 >= self.max_state_nb

        # update state_nbs
        self.state_nb += 1
        self.cur_time = self.timestamps[self.state_ts_dict[self.state_nb]]
        if not done:
            self.next_time = self.timestamps[self.state_ts_dict[self.state_nb + 1]]
        else:
            self.next_time = None

        return state, time, reward, done

    def reset(self):
        """
        reset()
        resets your env with same init values
        """
        print('env reset')
        self.state_nb = 0
        self.cur_time = timestamps[timestamps[self.state_ts_dict[self.state_nb]]]
        self.next_time = timestamps[timestamps[self.state_ts_dict[self.state_nb + 1]]]
