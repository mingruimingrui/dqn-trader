import numpy as np
import pandas as pd
import tensorflow as tf

from env import env_make

data_file_path = '/mnt/filesystem1/datasets/snp500_preprocessed.npz'

def main():
    print('Hello world')

    env = env_make(data_file_path)

if __name__ == '__main__':
    main()
