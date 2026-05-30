"""
PEASS Toolkit - Python Port
Equivalent of map2SubjScale.m
Loads native `.npz` parameter definitions converted from legacy MATLAB .mat archives.
"""
import numpy as np

from myMapping import myMapping


def map2SubjScale(qTarget: float, qInterf: float, qArtif: float, qGlobal: float) -> tuple:
    # %%% MATLAB Code %%%
    q = np.array([qGlobal, qTarget, qInterf, qArtif])

    # % Log-mapping: q=max(min(log((1+q)./(1-q)),5.5),-5.5);
    q_mapped = np.clip(np.log((1.0 + q) / (1.0 - q)), -5.5, 5.5)

    taskQ = np.zeros(4)
    for nTask in range(4):
        # load(sprintf('paramTask%d.mat',nTask));
        mat_data = np.load(f"paramTask{nTask + 1}.npz")

        W = mat_data['W']
        b = mat_data['b']
        v = mat_data['v']
        a = mat_data['a']

        # Note: 'selec' has been permanently re-indexed to Python's 0-based layout during .npz conversion
        selec = mat_data['selec']

        # taskQ(nTask) = myMapping(q(selec),W,b,v,a);
        taskQ[nTask] = myMapping(q_mapped[selec], W, b, v, a)

    # OPS = taskQ(1); TPS = taskQ(2); IPS = taskQ(3); APS = taskQ(4);
    return taskQ[0], taskQ[1], taskQ[2], taskQ[3]
