# %%% LSDecompose.m %%%
import numpy as np
import scipy.linalg


# function proj = LSDecompose(se,s,flen2,wa)
def LSDecompose(se, s, flen2, wa):
    # flen = 2*flen2+1;
    flen = 2 * flen2 + 1
    # J = size(s,2);
    J = s.shape[1]

    # S = zeros(size(se,1),J*flen);
    L = se.shape[0]
    S = np.zeros((L, J * flen), dtype=s.dtype)

    # for j=1:J
    for j in range(J):
        # try
        try:
            # S(:,(j-1)*flen+1:j*flen) = toeplitzC(s(flen:end,j),s(flen:-1:1,j));
            # Note: We substitute the C/MEX toeplitzC with scipy.linalg.toeplitz.
            # s(flen:end, j) is s[flen-1:, j] and s(flen:-1:1, j) is s[flen-1::-1, j]
            col = s[flen - 1:, j]
            row = s[flen - 1::-1, j]
            S[:, j * flen: (j + 1) * flen] = scipy.linalg.toeplitz(col, row)
        # catch
        except Exception:
            # S(:,(j-1)*flen+1:j*flen) = toeplitz(s(flen:end,j),s(flen:-1:1,j));
            col = s[flen - 1:, j]
            row = s[flen - 1::-1, j]
            S[:, j * flen: (j + 1) * flen] = scipy.linalg.toeplitz(col, row)

    # % Weighted ...
    # Sw = diag(wa)*S;
    Sw = wa[:, np.newaxis] * S
    # se = diag(wa)*se;
    se_w = wa[:, np.newaxis] * se

    # % ... Least squares
    # gramSw = Sw'*Sw;
    gramSw = Sw.conj().T @ Sw

    # lambda = 10^-15; % regularization parameter
    reg_lambda = 10 ** -15

    # [R testCond] = chol(gramSw+lambda*eye(size(gramSw)));
    try:
        # SciPy cholesky yields upper triangular R matching MATLAB: gramSw = R' * R
        R = scipy.linalg.cholesky(gramSw + reg_lambda * np.eye(gramSw.shape[0]), lower=False)
        testCond = False
    except (scipy.linalg.LinAlgError, ValueError):
        testCond = True

    # if testCond
    if testCond:
        # y = pinv(Sw)*se;
        y = np.linalg.pinv(Sw) @ se_w
    # else
    else:
        # y = R\(R'\Sw'*se);
        b = Sw.conj().T @ se_w
        tmp = scipy.linalg.solve_triangular(R.conj().T, b, lower=True)
        y = scipy.linalg.solve_triangular(R, tmp, lower=False)

    # proj = zeros([size(se), size(s,2)]);
    proj = np.zeros((L, se.shape[1], J), dtype=se.dtype)
    # Wa = diag(wa);
    Wa = wa[:, np.newaxis]
    # for j=1:J
    for j in range(J):
        # proj(:,:,j) = Wa*S(:,(j-1)*flen+(1:flen))*y((j-1)*flen+(1:flen),:);
        proj[:, :, j] = Wa * (S[:, j * flen: (j + 1) * flen] @ y[j * flen: (j + 1) * flen, :])

    # return
    return proj