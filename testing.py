# This file's just for testing purposes to make sure that all files are working properly
# I know I should probably write my test scripts and save it somewhere but fk it
import numpy as np
import pandas as pd
import tensorflow as tf

from env import env_make

data_file_path = '/mnt/filesystem1/datasets/snp500_preprocessed.npz'

def main():
    env = env_make(data_file_path)

if __name__ == '__main__':
    main()
