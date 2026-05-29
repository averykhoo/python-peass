"""
PEASS Toolkit - Python Port
Hohmann 2002 Gammatone Filterbank implementation.
Consolidates Gfb_Filter, Gfb_Analyzer, Gfb_Delay, and Gfb_Mixer.
"""
import math

import numpy as np
import scipy.signal as signal

from erbBW import erbBW


def Gfb_hz2erbscale(Hz: float) -> float:
    # ERBscale = GFB_Q * log(1 + Hz / (GFB_L * GFB_Q));
    return 9.265 * np.log(1.0 + Hz / (24.7 * 9.265))


def Gfb_erbscale2hz(ERBscale: np.ndarray) -> np.ndarray:
    # Hz = (exp(ERBscale / GFB_Q) - 1) * (GFB_L * GFB_Q);
    return (np.exp(ERBscale / 9.265) - 1.0) * (24.7 * 9.265)


def Gfb_center_frequencies(filters_per_ERBaud, lower_cutoff_frequency_hz,
                           specified_center_frequency_hz, upper_cutoff_frequency_hz):
    lower_cutoff_frequency_erb = Gfb_hz2erbscale(lower_cutoff_frequency_hz)
    specified_center_frequency_erb = Gfb_hz2erbscale(specified_center_frequency_hz)
    upper_cutoff_frequency_erb = Gfb_hz2erbscale(upper_cutoff_frequency_hz)

    erbs_below_base_frequency = specified_center_frequency_erb - lower_cutoff_frequency_erb
    num_of_filters_below_base_freq = int(np.floor(erbs_below_base_frequency * filters_per_ERBaud))

    start_frequency_erb = specified_center_frequency_erb - num_of_filters_below_base_freq / filters_per_ERBaud

    # center_frequencies_erb = [start_frequency_erb:(1/filters_per_ERBaud):upper_cutoff_frequency_erb];
    # Adding a small epsilon guarantees the upper bound is included as it is in MATLAB
    center_frequencies_erb = np.arange(start_frequency_erb, upper_cutoff_frequency_erb + 1e-9, 1.0 / filters_per_ERBaud)
    return Gfb_erbscale2hz(center_frequencies_erb)


def Gfb_Filter_new(sampling_frequency_hz, center_frequency_hz, gamma_order=4, bandwidth_factor=1.0):
    audiological_erb = (24.7 + center_frequency_hz / 9.265) * bandwidth_factor
    a_gamma = (np.pi * math.factorial(2 * gamma_order - 2) * (2.0 ** -(2 * gamma_order - 2)) /
               (math.factorial(gamma_order - 1) ** 2))
    b = audiological_erb / a_gamma
    lambda_val = np.exp(-2.0 * np.pi * b / sampling_frequency_hz)
    beta = 2.0 * np.pi * center_frequency_hz / sampling_frequency_hz

    coefficient = lambda_val * np.exp(1j * beta)
    normalization_factor = 2.0 * (1.0 - np.abs(coefficient)) ** gamma_order

    return {
        'coefficient':          coefficient,
        'normalization_factor': normalization_factor,
        'gamma_order':          gamma_order,
        'state':                np.zeros(gamma_order, dtype=complex)
    }


def Gfb_Filter_process(filter_obj, input_data):
    factor = filter_obj['normalization_factor']
    coeff = filter_obj['coefficient']
    gamma = filter_obj['gamma_order']
    filter_state = filter_obj['state'] * coeff

    y = input_data.copy()
    b_stage1 = np.array([factor], dtype=complex)
    a_stage = np.array([1.0, -coeff], dtype=complex)

    new_state = np.zeros(gamma, dtype=complex)
    for i in range(gamma):
        b = b_stage1 if i == 0 else np.array([1.0], dtype=complex)
        zi = np.array([filter_state[i]], dtype=complex)
        y, zf = signal.lfilter(b, a_stage, y, zi=zi)
        new_state[i] = zf[0]

    filter_obj['state'] = new_state / coeff
    return y, filter_obj


def Gfb_Analyzer_new(sampling_frequency_hz, lower_cutoff_frequency_hz, specified_center_frequency_hz,
                     upper_cutoff_frequency_hz, filters_per_ERBaud, gamma_order=4, bandwidth_factor=1.0):
    center_frequencies_hz = Gfb_center_frequencies(filters_per_ERBaud, lower_cutoff_frequency_hz,
                                                   specified_center_frequency_hz, upper_cutoff_frequency_hz)
    filters = [Gfb_Filter_new(sampling_frequency_hz, cf, gamma_order, bandwidth_factor) for cf in center_frequencies_hz]
    return {
        'sampling_frequency_hz': sampling_frequency_hz,
        'center_frequencies_hz': center_frequencies_hz,
        'filters':               filters,
        'bw':                    erbBW(center_frequencies_hz)
    }


def Gfb_Analyzer_process(analyzer, input_data):
    number_of_bands = len(analyzer['filters'])
    output = np.zeros((number_of_bands, input_data.shape[0]), dtype=complex)
    for band in range(number_of_bands):
        output[band, :], analyzer['filters'][band] = Gfb_Filter_process(analyzer['filters'][band], input_data)
    return output, analyzer


def Gfb_Analyzer_zresponse(analyzer, z):
    z = z[:, np.newaxis]
    number_of_bands = analyzer['center_frequencies_hz'].shape[0]
    zresponse = np.ones((z.shape[0], number_of_bands), dtype=complex)
    for band in range(number_of_bands):
        coeff = analyzer['filters'][band]['coefficient']
        norm = analyzer['filters'][band]['normalization_factor']
        gamma = analyzer['filters'][band]['gamma_order']
        zresponse[:, band] = ((1.0 - coeff / z[:, 0]) ** -gamma) * norm
    return zresponse


def Gfb_Delay_new(analyzer, delay_samples):
    impulse = np.zeros(delay_samples + 2)
    impulse[0] = 1.0
    impulse_response, _ = Gfb_Analyzer_process(analyzer, impulse)

    number_of_bands = impulse_response.shape[0]
    ir_slice = np.abs(impulse_response[:, :delay_samples + 1])
    max_indices = np.argmax(ir_slice, axis=1)

    delays_samples = delay_samples - max_indices
    slopes = np.zeros(number_of_bands, dtype=complex)

    for band in range(number_of_bands):
        idx = max_indices[band]
        slopes[band] = impulse_response[band, idx + 1] - impulse_response[band, idx - 1]

    slopes = slopes / np.abs(slopes)
    phase_factors = 1j / slopes

    return {
        'delays_samples': delays_samples,
        'phase_factors':  phase_factors,
        'memory':         np.zeros((number_of_bands, int(np.max(delays_samples))), dtype=complex)
    }


def Gfb_Mixer_new(analyzer, delay, iterations=100):
    center_frequencies = analyzer['center_frequencies_hz']
    number_of_bands = center_frequencies.shape[0]
    sampling_frequency = analyzer['sampling_frequency_hz']

    z_c = np.exp(2j * np.pi * center_frequencies / sampling_frequency)
    gains = np.ones(number_of_bands)

    pos_f_response = Gfb_Analyzer_zresponse(analyzer, z_c)
    neg_f_response = Gfb_Analyzer_zresponse(analyzer, np.conj(z_c))

    for band in range(number_of_bands):
        pos_f_response[:, band] = pos_f_response[:, band] * delay['phase_factors'][band] * (
                z_c ** -delay['delays_samples'][band])
        neg_f_response[:, band] = neg_f_response[:, band] * delay['phase_factors'][band] * (
                np.conj(z_c) ** -delay['delays_samples'][band])

    f_response = (pos_f_response + np.conj(neg_f_response)) / 2.0

    for _ in range(iterations):
        selected_spectrum = f_response @ gains
        gains = gains / np.abs(selected_spectrum)

    return {'gains': gains}


def Gfb_Delay_process(delay, input_data):
    number_of_bands, number_of_samples = input_data.shape
    output = np.zeros((number_of_bands, number_of_samples))
    for band in range(number_of_bands):
        delay_val = int(delay['delays_samples'][band])
        phase_corrected = np.real(input_data[band, :] * delay['phase_factors'][band])
        if delay_val == 0:
            output[band, :] = phase_corrected
        else:
            tmp_out = np.concatenate((delay['memory'][band, :delay_val], phase_corrected))
            delay['memory'][band, :delay_val] = tmp_out[number_of_samples:]
            output[band, :] = tmp_out[:number_of_samples]
    return output, delay


def Gfb_Mixer_process(mixer, input_data):
    return mixer['gains'] @ input_data, mixer


def Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds):
    fs = analyzer['sampling_frequency_hz']
    desired_delay_in_samples = int(round(desired_delay_in_seconds * fs))
    delay = Gfb_Delay_new(analyzer, desired_delay_in_samples)
    mixer = Gfb_Mixer_new(analyzer, delay)
    return {'delay': delay, 'mixer': mixer}


def Gfb_Synthesizer_process(synthesizer, input_data):
    output, delay = Gfb_Delay_process(synthesizer['delay'], input_data)
    synthesizer['delay'] = delay
    output, mixer = Gfb_Mixer_process(synthesizer['mixer'], output)
    synthesizer['mixer'] = mixer
    return output, synthesizer
