"""
PEASS Toolkit - Python Port
Equivalent Rectangular Bandwidth (ERB) calculator.
"""
import numpy as np


def erbBW(fc: np.ndarray) -> np.ndarray:
    """
    % function bw = erbBW(fc)
    % bw = 24.7*(.00437*fc+1);
    % return
    """
    return 24.7 * (0.00437 * fc + 1.0)
