"""
PEASS Auditory Package - Dau 1996/1997 Psychoacoustic Ear Model [1, 2]

This module ports the legacy C/MEX elements (haircell.c, adapt.c) into pure,
performant Python [1, 3]. It simulates the transduction process of the inner hair cells
and the temporal adaptation (forward masking) of the auditory nerve.
"""

from typing import Tuple

import numpy as np
import scipy.signal as signal

from .gammatone import GammatoneAnalyzer


def haircell_transduction(subband_signals: np.ndarray, sampling_frequency: float) -> np.ndarray:
    """
    Models the nonlinear mechanical-to-neural transduction of the inner hair cells.
    Replaces haircell.c MEX script [2, 3].

    Stages:
      1. Half-wave rectification (simulates unidirectional shearing of hair bundle)
      2. 1 kHz first-order lowpass filter (simulates inner hair cell membrane limits)
    """
    # % gain=exp(-pi*2000/fs);
    # % rx=filter(1-gain,[1 -gain],max(rx,0),[],2);
    gain_haircell = np.exp(-np.pi * 2000.0 / sampling_frequency)
    b_hc = np.array([1.0 - gain_haircell])
    a_hc = np.array([1.0, -gain_haircell])

    # Process rectified signals over the sample dimension (axis 1)
    rectified_signals = np.maximum(subband_signals, 0.0)
    return signal.lfilter(b_hc, a_hc, rectified_signals, axis=1)


def adaptation_loops(subband_signals: np.ndarray, sampling_frequency: float) -> np.ndarray:
    """
    Simulates the physiological adaptive properties of the auditory nerve.
    Replaces adapt.c MEX script [2].

    Runs 5 consecutive non-linear feedback loops modeling forward masking,
    vectorized across all bands for optimal execution in Python.
    """
    dbrange = 100.0
    thresh = 10.0 ** (-dbrange / 20.0)
    bw_loop = 1.0 / (np.pi * np.array([0.005, 0.05, 0.129, 0.253, 0.5]))

    # % rx=max(single(rx),thresh);
    rx = np.maximum(subband_signals.astype(np.float32), thresh)
    num_bands, num_samples = rx.shape

    # Process each of the 5 adaptive stages
    sthresh = thresh
    for stage_idx in range(5):
        gain_val = np.exp(-np.pi * bw_loop[stage_idx] / sampling_frequency)
        sthresh = np.sqrt(sthresh)
        factor = np.full(num_bands, sthresh)  # divisor factor for each band

        for sample_idx in range(num_samples):
            # Divide current sample by current divisor factor
            val = rx[:, sample_idx] / factor
            rx[:, sample_idx] = val
            # Update divisor filter state
            factor = np.maximum((1.0 - gain_val) * val + gain_val * factor, sthresh)

    # % rx=double(dbrange/(1-sthresh))*(double(rx)-double(sthresh));
    return (dbrange / (1.0 - sthresh)) * (rx - sthresh)


def generate_internal_representation(
        signal_data: np.ndarray,
        sampling_frequency: float,
        modulation_processing_type: str = 'lp'
) -> Tuple[np.ndarray, float]:
    """
    Generates the 3D internal auditory representation of a signal.
    Equivalent of pemo_internal.m [1].
    """
    if len(signal_data.shape) > 1:
        if signal_data.shape[0] < signal_data.shape[1]:
            signal_data = signal_data.T
        signal_data = signal_data.ravel()

    # Model input scaling (1.0 becomes 100 dB SPL)
    signal_data = 10.0 * signal_data

    # Frequency analysis boundaries
    fmin = 235.0
    fmax = min(0.5 * sampling_frequency, 14500.0)
    if sampling_frequency < 3.0 * fmax:
        new_fs = int(round(1.5 * sampling_frequency))
        signal_data = signal_data.astype(float)
        signal_data = signal.resample(signal_data, int(round(len(signal_data) * new_fs / sampling_frequency)))
        sampling_frequency = float(new_fs)

    analyzer = GammatoneAnalyzer(sampling_frequency, fmin, 1000.0, fmax, 1.0)
    num_bands = len(analyzer.filters)

    # Subband analysis
    subbands = np.real(analyzer.process(signal_data))

    # Transduction and Adaptation stages
    transduced = haircell_transduction(subbands, sampling_frequency)
    adapted = adaptation_loops(transduced, sampling_frequency)

    # Modulation Filtering & Downsampling
    if modulation_processing_type == 'fb':
        adapted = signal.resample(adapted, int(round(adapted.shape[1] * 800.0 / sampling_frequency)), axis=1)
        sampling_frequency = 800.0
        center_frequencies_mod = np.concatenate(([0.0, 5.0], 10.0 * (5.0 / 3.0) ** np.arange(6)))
        bandwidth_mod = np.concatenate(([5.0, 5.0], 5.0 * (5.0 / 3.0) ** np.arange(6)))
    else:
        adapted = signal.resample(adapted, int(round(adapted.shape[1] * 100.0 / sampling_frequency)), axis=1)
        sampling_frequency = 100.0
        center_frequencies_mod = np.array([0.0])
        bandwidth_mod = np.array([15.92])

    num_modulations = len(center_frequencies_mod)
    num_samples = adapted.shape[1]
    internal_representation = np.zeros((num_bands, num_samples, num_modulations), dtype=complex)

    for m in range(num_modulations):
        gain_val = np.exp(-np.pi * bandwidth_mod[m] / sampling_frequency)
        b_mod = np.array([1.0 - gain_val])
        a_mod = np.array([1.0, -gain_val * np.exp(2j * np.pi * center_frequencies_mod[m] / sampling_frequency)])
        internal_representation[:, :, m] = signal.lfilter(b_mod, a_mod, adapted, axis=1)

    # Hilbert envelope extraction above 10 Hz
    above_10_hz = (center_frequencies_mod > 10.0)
    internal_representation[:, :, ~above_10_hz] = np.real(internal_representation[:, :, ~above_10_hz])
    internal_representation[:, :, above_10_hz] = np.abs(internal_representation[:, :, above_10_hz])

    return internal_representation, sampling_frequency
