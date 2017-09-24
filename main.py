import os
import numpy as np
import pandas as pd
# import tensorflow as tf
from src.env import env_make

data_file_path = 'data/snp500_transformed.npz'

def load_data():
    file = np.load(data_file_path)
    return file['timestamps'], file['syms'], file['col_names'], file['data']

def main():
    assert os.path.exists(data_file_path), 'Have you ran preprocess.py and data_transform.py yet?'

    timestamps, syms, col_names, data = load_data()

if __name__ == '__main__':
    main()
