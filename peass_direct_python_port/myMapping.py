# %%% myMapping.m %%%
import numpy as np

def sigmoid(x):
    # y=1./(1+exp(-x));
    return 1.0 / (1.0 + np.exp(-x))

# function [y,dW,db,dv,da] = myMapping(x,W,b,v,a)
def myMapping(x, W, b, v, a):
    # [nin,ndata]=size(x);
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    nin, ndata = x.shape
    nhid = len(v)

    # s1=W*x+b*ones(1,ndata);
    # Note: Ensure shape alignment for the bias vector addition
    s1 = W @ x + b
    # o1=sigmoid(s1);
    o1 = sigmoid(s1)
    # s2=v'*o1+a;
    s2 = v.T @ o1 + a
    # y=100*sigmoid(s2);
    y = 100.0 * sigmoid(s2)

    # return
    return float(y[0, 0])