"""
PEASS Toolkit - Python Port
Equivalent of myMapping.m (and myLogSig.m functionality)
"""
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (2.0 + np.exp(x) + np.exp(-x))


def myMapping(x: np.ndarray, W: np.ndarray, b: np.ndarray, v: np.ndarray, a: np.ndarray):
    # %%% MATLAB Code %%%
    if len(x.shape) == 1:
        x = x[:, np.newaxis]

    # s1=W*x+b*ones(1,ndata); o1=sigmoid(s1); s2=v'*o1+a; y=100*sigmoid(s2);
    s1 = W @ x + b
    o1 = sigmoid(s1)
    s2 = v.T @ o1 + a
    y = 100.0 * sigmoid(s2)

    return float(y[0, 0])
