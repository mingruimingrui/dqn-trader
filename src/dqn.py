# dqn to do predict action from state
import os
from datetime import datetime

import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.layers import GRU, LSTM


class Dqn:
