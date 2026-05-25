# %%% map2SubjScale.m %%%
import numpy as np
import scipy.io as sio
from myMapping import myMapping


# function [OPS, TPS, IPS, APS] = map2SubjScale(qTarget, qInterf, qArtif, qGlobal)
def map2SubjScale(qTarget, qInterf, qArtif, qGlobal):
    # q = [qGlobal; qTarget; qInterf; qArtif];
    q = np.array([qGlobal, qTarget, qInterf, qArtif])

    # % Log-mapping
    # q=max(min(log((1+q)./(1-q)),5.5),-5.5);
    q_mapped = np.maximum(np.minimum(np.log((1.0 + q) / (1.0 - q)), 5.5), -5.5)

    taskQ = np.zeros(4)
    # for nTask = 1:4
    for nTask in range(4):
        # load(sprintf('paramTask%d.mat',nTask));
        # Note: Using scipy.io.loadmat to dynamically load the neural parameters
        mat_data = sio.loadmat(f"paramTask{nTask + 1}.mat")

        # Extract variables from mat structure
        # W, b, v, a and selec
        W = mat_data['W']
        b = mat_data['b']
        v = mat_data['v']
        a = mat_data['a']
        # Note: MATLAB arrays are 1-based, we must subtract 1 from the feature indices
        selec = mat_data['selec'].flatten() - 1

        # taskQ(nTask) = myMapping(q(selec),W,b,v,a);
        taskQ[nTask] = myMapping(q_mapped[selec], W, b, v, a)

    # OPS = taskQ(1);
    # TPS = taskQ(2);
    # IPS = taskQ(3);
    # APS = taskQ(4);
    OPS = taskQ[0]
    TPS = taskQ[1]
    IPS = taskQ[2]
    APS = taskQ[3]

    return OPS, TPS, IPS, APS