# Jai, Shah
# 1001-380-311
# 2017-10-08
# Assignment_03_05

import numpy as np


def read_csv_as_matrix(file_name):
    # Each row of data in the file becomes a row in the matrix
    # So the resulting matrix has dimension [num_samples x sample_dimension]
    data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
    return data

print(read_csv_as_matrix('stock_data.csv'))
