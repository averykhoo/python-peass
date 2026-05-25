# %%% erbBW.m %%%
# function bw = erbBW(fc)
def erbBW(fc):
    # bw = 24.7*(.00437*fc+1);
    bw = 24.7 * (0.00437 * fc + 1)
    # return
    return bw