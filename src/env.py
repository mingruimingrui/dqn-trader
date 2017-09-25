# home made env maker made to behave like openai gym
import os
import numpy as np
import datetime

# syms_to_use (list-like)[optional]: sets env to contain only syms in list
# start, end (datetime)[optional]: sets time period, inclusive
# lookback (int)[default 5]: sets lookback period
class env_make:

    def __init__(self, timestamps, syms, col_names, data, syms_to_use=None, start=None, end=None, lookback=5):
        if syms_to_use != None:
            assert isinstance(syms_to_use, list) | isinstance(syms_to_use, np.ndarray), 'syms_to_use must be list-like'
            if isinstance(syms_to_use, np.ndarray):
                assert len(syms_to_use, 1), 'syms_to_use must be vector'
            data = data[:, list(map(lambda x: x in syms_to_use, syms)), :]
            syms = syms_to_use

        if start != None:
            assert isinstance(start, datetime.datetime), 'start must be datetime'
            data = data[timestamps >= start]
            timestamps = timestamps[timestamps >= start]

        if end != None:
            assert isinstance(end, datetime.datetime), 'end must be datetime'
            data = data[timestamps <= end]
            timestamps = timestamps[timestamps <= end]

        assert data.shape[0] > lookback, 'not enough timesteps for your env'
        assert data.shape[1] > 0       , 'not enough assets in your env'
        assert data.shape[2] > 0       , 'not enough cols in your env'

        self.timestamps = timestamps
        self.syms = syms
        self.col_names = col_names
        self.data = data
        self.lookback = lookback
        self.count = 0
        self.max_count = len(self.data) - self.lookback - 1
        self.cur_time = timestamps[self.count + self.lookback]
        self.next_time = timestamps[self.count + self.lookback + 1]
        self.action_shape = (len(self.syms),)

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
        assert self.count <= self.max_count, 'no more steps left in env, please reset'
        assert action.shape == self.action_shape, 'action is wrong shape'
        assert np.sum(action) != 0, 'action must not sum to 0'

        # generate new state and normalize
        state = self.data[self.count : self.count + self.lookback, :, :].copy()
        mult = self.data[self.count + self.lookback, :, self.col_names == 'open']
        mult = np.dot(np.ones((self.lookback,1)), mult)
        mult = np.dot(np.expand_dims(mult, 2), np.ones((1, len(self.col_names))))
        state /= mult

        # calculate the timestamps for the state
        time = self.timestamps[self.count : self.count + self.lookback]

        # normalize action and calculate reward
        action /= np.sum(action)
        action = np.squeeze(action)
        open_p = np.squeeze(self.data[self.count + self.lookback - 1, :, self.col_names == 'open'])
        open_n = np.squeeze(self.data[self.count + self.lookback, :, self.col_names == 'open'])

        reward = np.sum(open_n / open_p * action)

        # check if done
        done = self.count >= self.max_count

        # update counts
        self.cur_time = self.timestamps[self.count + self.lookback - 1]
        if not done:
            self.next_time = self.timestamps[self.count + self.lookback]
        self.count += 1

        return state, time, reward, done

    def reset(self):
        """
        reset()
        resets your env with same init values
        """
        print('env reset')
        self.count = 0
        self.cur_time = timestamps[self.count + self.lookback]
        self.next_time = timestamps[self.count + self.lookback + 1]
