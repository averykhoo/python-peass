"""Weighted least-squares projection onto delayed versions of the sources.

Transcribed from MATLAB PEASS v2.0.1.
"""
MATLAB_SOURCE = "LSDecompose.m"

import numpy as np
import scipy.linalg

# >>> MATLAB
# function proj = LSDecompose(se,s,flen2,wa)
# % SPROJ Weighted least-squares projection of each channel of se on the subspace
# % spanned by delayed versions of the channels of s, with delays between
# % -flen2 and flen2
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Version 1.0
# % Copyright 2010 Valentin Emiya (INRIA).
# % This software is distributed under the terms of the GNU Public License
# % version 3 (http://www.gnu.org/licenses/gpl.txt).
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#
# <<< MATLAB


def LSDecompose(se, s, flen2, wa):
# >>> MATLAB
# flen = 2*flen2+1;
# J = size(s,2);
#
#
# <<< MATLAB
    flen = 2 * flen2 + 1
    J = s.shape[1]  # MATLAB size(s,2)

    # Everything downstream inherits the input dtype. In the PEASS pipeline
    # `s`/`se` are the complex gammatone subbands, so this is complex128 and
    # MATLAB's `'` below is a CONJUGATE transpose.
    dtype = np.result_type(se.dtype, s.dtype)

# >>> MATLAB
# S = zeros(size(se,1),J*flen);
# for j=1:J
#     try
#         S(:,(j-1)*flen+1:j*flen) =...
#             toeplitzC(s(flen:end,j),s(flen:-1:1,j));
#     catch
#         S(:,(j-1)*flen+1:j*flen) =...
#             toeplitz(s(flen:end,j),s(flen:-1:1,j));
#     end
# end
# <<< MATLAB
    S = np.zeros((se.shape[0], J * flen), dtype=dtype)
    for j in range(1, J + 1):  # MATLAB j=1:J
        # `toeplitzC` is a MEX build of the same Toeplitz construction; the
        # MATLAB falls back to the built-in `toeplitz` when it is not compiled,
        # and that fallback is what is transcribed. Neither MATLAB's two-argument
        # `toeplitz` nor scipy's conjugates, so they agree on complex input.
        #
        # MATLAB s(flen:end,j)  -> first column, length size(s,1)-flen+1
        # MATLAB s(flen:-1:1,j) -> first row, length flen
        column = s[flen - 1:, j - 1]
        row = s[flen - 1::-1, j - 1]
        # MATLAB S(:,(j-1)*flen+1:j*flen)
        S[:, (j - 1) * flen:j * flen] = scipy.linalg.toeplitz(column, row)

# >>> MATLAB
#
# % Weighted ...
# Sw = diag(wa)*S;
# se = diag(wa)*se;
# <<< MATLAB
    # `diag(wa)*X` is a row scaling; kept as an explicit matrix product so the
    # shape contract stays visible, since `wa` is what makes this "weighted".
    Sw = np.diag(wa) @ S
    se = np.diag(wa) @ se

# >>> MATLAB
#
# % ... Least squares
# gramSw = Sw'*Sw;
#
# lambda = 10^-15; % regularization parameter (useful when a sequence of zeroes occurs in sources "s"
# [R testCond] = chol(gramSw+lambda*eye(size(gramSw))); % very fast but raises an error if gramS is not well conditionned
# if testCond
#     % if gramS is not well conditionned, use pseudo-inverse (more
#     % computations)
#     y = pinv(Sw)*se;
# else
#     y = R\(R'\Sw'*se);
# end
# <<< MATLAB
    gramSw = Sw.conj().T @ Sw  # MATLAB `'` is the conjugate transpose

    lambda_ = 10.0 ** -15  # `lambda` is a Python keyword
    # MATLAB's two-output `chol` returns testCond>0 instead of raising when the
    # matrix is not positive definite; scipy raises, so the flag is recovered
    # from the exception.
    try:
        R = scipy.linalg.cholesky(gramSw + lambda_ * np.eye(gramSw.shape[0]), lower=False)
        testCond = 0
    except scipy.linalg.LinAlgError:
        R = None
        testCond = 1
    if testCond:
        y = np.linalg.pinv(Sw) @ se
    else:
        # MATLAB `R\(R'\Sw'*se)`: `\` on a triangular factor is a triangular
        # solve, not a general one. R is upper triangular, so R' is lower.
        y = scipy.linalg.solve_triangular(
            R,
            scipy.linalg.solve_triangular(R.conj().T, Sw.conj().T @ se, lower=True),
            lower=False,
        )

# >>> MATLAB
#
# proj = zeros([size(se), size(s,2)]); % length x channels x sources
# Wa = diag(wa);
# for j=1:J
# %     proj(:,:,j) = S(:,(j-1)*flen+(1:flen))*y((j-1)*flen+(1:flen),:);
#     proj(:,:,j) = Wa*S(:,(j-1)*flen+(1:flen))*y((j-1)*flen+(1:flen),:);
# end
# return
#
#
# <<< MATLAB
    # MATLAB zeros([size(se), size(s,2)]) -> (length, channels, sources)
    proj = np.zeros((se.shape[0], se.shape[1], s.shape[1]), dtype=dtype)
    Wa = np.diag(wa)
    for j in range(1, J + 1):  # MATLAB j=1:J
        # MATLAB (j-1)*flen+(1:flen) -> rows/cols (j-1)*flen .. j*flen-1
        block = slice((j - 1) * flen, (j - 1) * flen + flen)
        proj[:, :, j - 1] = Wa @ S[:, block] @ y[block, :]
    return proj
