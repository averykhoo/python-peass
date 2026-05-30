"""
PEASS Parameter Converter: MAT to NPZ
Converts legacy MATLAB 5.0 MAT-files to native compressed NumPy archives.
Adjusts 1-based indexing arrays to 0-based indexing for Python.
"""

import os
from pathlib import Path

import scipy.io as sio
import numpy as np

BASE_DIR = Path('../peass_master_22c7fc4e/v2.0.1')
OUT_DIR = Path('')

def convert_mat_to_npz():
    # Loop through all 4 task parameter files
    for nTask in range(4):
        mat_filename = f"paramTask{nTask + 1}.mat"
        mat_path = BASE_DIR / mat_filename
        assert mat_path.is_file()

        # Load the MATLAB MAT-file
        mat_data = sio.loadmat(mat_path)

        # Extract neural network weights and biases
        W = mat_data['W']
        b = mat_data['b']
        v = mat_data['v']
        a = mat_data['a']

        # Convert 1-based MATLAB indices to 0-based Python indices permanently
        selec = mat_data['selec'].flatten() - 1

        # Save variables into a single zipped NumPy archive (.npz)
        npz_path = OUT_DIR / mat_path.with_suffix('.npz').name
        np.savez_compressed(npz_path, W=W, b=b, v=v, a=a, selec=selec)
        print(f"Successfully converted: {mat_path} -> {npz_path}")


if __name__ == '__main__':
    convert_mat_to_npz()