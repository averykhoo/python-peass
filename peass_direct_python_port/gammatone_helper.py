# %%% hohmann2002_gammatone_reference.py %%%
import numpy as np
import scipy.signal as signal
from erbBW import erbBW


# % GFB_L = 24.7;
# % GFB_Q = 9.265;
# % GFB_PREFERED_GAMMA_ORDER = 4;
# % GFB_GAINCALC_ITERATIONS  = 100;

def Gfb_hz2erbscale(Hz):
    # ERBscale = GFB_Q * log(1 + Hz / (GFB_L * GFB_Q));
    return 9.265 * np.log(1 + Hz / (24.7 * 9.265))


def Gfb_erbscale2hz(ERBscale):
    # Hz = (exp(ERBscale / GFB_Q) - 1) * (GFB_L * GFB_Q);
    return (np.exp(ERBscale / 9.265) - 1) * (24.7 * 9.265)


# function center_frequencies_hz = Gfb_center_frequencies(filters_per_ERBaud, lower_cutoff_frequency_hz, specified_center_frequency_hz, upper_cutoff_frequency_hz)
def Gfb_center_frequencies(filters_per_ERBaud, lower_cutoff_frequency_hz, specified_center_frequency_hz,
                           upper_cutoff_frequency_hz):
    # lower_cutoff_frequency_erb = Gfb_hz2erbscale(lower_cutoff_frequency_hz);
    lower_cutoff_frequency_erb = Gfb_hz2erbscale(lower_cutoff_frequency_hz)
    # specified_center_frequency_erb = Gfb_hz2erbscale(specified_center_frequency_hz);
    specified_center_frequency_erb = Gfb_hz2erbscale(specified_center_frequency_hz)
    # upper_cutoff_frequency_erb = Gfb_hz2erbscale(upper_cutoff_frequency_hz);
    upper_cutoff_frequency_erb = Gfb_hz2erbscale(upper_cutoff_frequency_hz)

    # erbs_below_base_frequency = specified_center_frequency_erb - lower_cutoff_frequency_erb;
    erbs_below_base_frequency = specified_center_frequency_erb - lower_cutoff_frequency_erb
    # num_of_filters_below_base_freq = floor(erbs_below_base_frequency * filters_per_ERBaud);
    num_of_filters_below_base_freq = int(np.floor(erbs_below_base_frequency * filters_per_ERBaud))

    # start_frequency_erb = specified_center_frequency_erb - num_of_filters_below_base_freq / filters_per_ERBaud;
    start_frequency_erb = specified_center_frequency_erb - num_of_filters_below_base_freq / filters_per_ERBaud
    # center_frequencies_erb = [start_frequency_erb:(1/filters_per_ERBaud):upper_cutoff_frequency_erb];
    center_frequencies_erb = np.arange(start_frequency_erb, upper_cutoff_frequency_erb + 1e-9, 1.0 / filters_per_ERBaud)
    # center_frequencies_hz = Gfb_erbscale2hz(center_frequencies_erb);
    center_frequencies_hz = Gfb_erbscale2hz(center_frequencies_erb)
    return center_frequencies_hz


# function filter = Gfb_Filter_new(sampling_frequency_hz, center_frequency_hz, gamma_order, bandwidth_factor)
def Gfb_Filter_new(sampling_frequency_hz, center_frequency_hz, gamma_order=4, bandwidth_factor=1.0):
    # audiological_erb = (GFB_L + center_frequency_hz / GFB_Q) * bandwidth_factor;
    audiological_erb = (24.7 + center_frequency_hz / 9.265) * bandwidth_factor
    # a_gamma = (pi * factorial(2*filter.gamma_order - 2) * 2 ^ -(2*filter.gamma_order - 2) / factorial(filter.gamma_order - 1) ^ 2);
    import math
    a_gamma = (np.pi * math.factorial(2 * gamma_order - 2) * (2 ** -(2 * gamma_order - 2)) / (
                math.factorial(gamma_order - 1) ** 2))
    # b = audiological_erb / a_gamma;
    b = audiological_erb / a_gamma
    # lambda = exp(-2 * pi * b / sampling_rate_hz);
    lambda_val = np.exp(-2 * np.pi * b / sampling_frequency_hz)
    # beta = 2 * pi * center_frequency_hz / sampling_rate_hz;
    beta = 2 * np.pi * center_frequency_hz / sampling_frequency_hz
    # filter.coefficient = lambda * exp(1i * beta);
    coefficient = lambda_val * np.exp(1j * beta)
    # filter.normalization_factor = 2 * (1 - abs(filter.coefficient)) ^ filter.gamma_order;
    normalization_factor = 2 * (1 - np.abs(coefficient)) ** gamma_order
    return {
        'type':                 'Gfb_Filter',
        'coefficient':          coefficient,
        'normalization_factor': normalization_factor,
        'gamma_order':          gamma_order,
        'state':                np.zeros(gamma_order, dtype=complex)
    }


# function [output, filter_obj] = Gfb_Filter_process(filter_obj, input)
def Gfb_Filter_process(filter_obj, input_data):
    # factor = filter_obj.normalization_factor;
    factor = filter_obj['normalization_factor']
    # coeff = filter_obj.coefficient;
    coeff = filter_obj['coefficient']
    # gamma = filter_obj.gamma_order;
    gamma = filter_obj['gamma_order']
    # filter_state = filter_obj.state * filter_obj.coefficient;
    filter_state = filter_obj['state'] * coeff

    y = input_data.copy()
    b_stage1 = np.array([factor], dtype=complex)
    a_stage = np.array([1.0, -coeff], dtype=complex)

    new_state = np.zeros(gamma, dtype=complex)
    # for i = [1:filter_obj.gamma_order]
    for i in range(gamma):
        # [input, filter_state(i)] = filter(factor, [1, -filter_obj.coefficient], input, filter_state(i));
        b = b_stage1 if i == 0 else np.array([1.0], dtype=complex)
        init_val = filter_state[i]
        zi = np.array([init_val], dtype=complex)
        y, zf = signal.lfilter(b, a_stage, y, zi=zi)
        new_state[i] = zf[0]

    # filter_obj.state = filter_state / filter_obj.coefficient;
    filter_obj['state'] = new_state / coeff
    # output = input;
    return y, filter_obj


# function analyzer = Gfb_Analyzer_new(sampling_frequency_hz, lower_cutoff_frequency_hz, specified_center_frequency_hz, upper_cutoff_frequency_hz, filters_per_ERBaud)
def Gfb_Analyzer_new(sampling_frequency_hz, lower_cutoff_frequency_hz, specified_center_frequency_hz,
                     upper_cutoff_frequency_hz, filters_per_ERBaud, gamma_order=4, bandwidth_factor=1.0):
    # analyzer.center_frequencies_hz = Gfb_center_frequencies(...)
    center_frequencies_hz = Gfb_center_frequencies(filters_per_ERBaud, lower_cutoff_frequency_hz,
                                                   specified_center_frequency_hz, upper_cutoff_frequency_hz)
    filters = []
    # for band = [1:length(analyzer.center_frequencies_hz)]
    for band in range(center_frequencies_hz.shape[0]):
        cf = center_frequencies_hz[band]
        # analyzer.filters(1,band) = Gfb_Filter_new(sampling_frequency_hz, center_frequency_hz, gamma_order, bandwidth_factor)
        filters.append(Gfb_Filter_new(sampling_frequency_hz, cf, gamma_order, bandwidth_factor))
    return {
        'type':                          'Gfb_Analyzer',
        'sampling_frequency_hz':         sampling_frequency_hz,
        'lower_cutoff_frequency_hz':     lower_cutoff_frequency_hz,
        'specified_center_frequency_hz': specified_center_frequency_hz,
        'upper_cutoff_frequency_hz':     upper_cutoff_frequency_hz,
        'filters_per_ERBaud':            filters_per_ERBaud,
        'bandwidth_factor':              bandwidth_factor,
        'center_frequencies_hz':         center_frequencies_hz,
        'filters':                       filters,
        'bw':                            erbBW(center_frequencies_hz)
    }


# function [output, analyzer] = Gfb_Analyzer_process(analyzer, input)
def Gfb_Analyzer_process(analyzer, input_data):
    # number_of_bands = length(analyzer.center_frequencies_hz);
    number_of_bands = len(analyzer['filters'])
    # output = zeros(number_of_bands, length(input));
    output = np.zeros((number_of_bands, input_data.shape[0]), dtype=complex)
    # for band = [1:number_of_bands]
    for band in range(number_of_bands):
        # [output(band,:), analyzer.filters(band)] = Gfb_Filter_process(analyzer.filters(band), input);
        output[band, :], analyzer['filters'][band] = Gfb_Filter_process(analyzer['filters'][band], input_data)
    return output, analyzer


# function zresponse = Gfb_Analyzer_zresponse(analyzer, z)
def Gfb_Analyzer_zresponse(analyzer, z):
    z = z[:, np.newaxis]
    number_of_bands = analyzer['center_frequencies_hz'].shape[0]
    zresponse = np.ones((z.shape[0], number_of_bands), dtype=complex)
    # for band = [1:number_of_bands]
    for band in range(number_of_bands):
        coeff = analyzer['filters'][band]['coefficient']
        norm = analyzer['filters'][band]['normalization_factor']
        gamma = analyzer['filters'][band]['gamma_order']
        # zresponse(:,band) = (1 - filter.coefficient ./ z) .^ -filter.gamma_order * filter.normalization_factor
        zresponse[:, band] = ((1 - coeff / z[:, 0]) ** -gamma) * norm
    return zresponse


# function delay = Gfb_Delay_new(analyzer, delay_samples)
def Gfb_Delay_new(analyzer, delay_samples):
    # impulse = zeros(1, delay_samples + 2);
    impulse = np.zeros(delay_samples + 2)
    # impulse(1) = 1;
    impulse[0] = 1.0
    # impulse_response = Gfb_Analyzer_process(analyzer, impulse);
    impulse_response, _ = Gfb_Analyzer_process(analyzer, impulse)

    number_of_bands = impulse_response.shape[0]
    # [dummy, max_indices] = max(abs(impulse_response(:,1:(delay_samples+1))).');
    ir_slice = np.abs(impulse_response[:, :delay_samples + 1])
    max_indices = np.argmax(ir_slice, axis=1)

    # delay.delays_samples = delay_samples + 1 - max_indices;
    delays_samples = delay_samples - max_indices
    # slopes = zeros(1, number_of_bands);
    slopes = np.zeros(number_of_bands, dtype=complex)
    # for band = [1:number_of_bands]
    for band in range(number_of_bands):
        idx = max_indices[band]
        # slopes(band) = (impulse_response(band, band_max_index+1) - impulse_response(band, band_max_index-1));
        slopes[band] = impulse_response[band, idx + 1] - impulse_response[band, idx - 1]

    # slopes = slopes ./ abs(slopes);
    slopes = slopes / np.abs(slopes)
    # delay.phase_factors = 1i ./ slopes;
    phase_factors = 1j / slopes

    return {
        'type':           'Gfb_Delay',
        'delays_samples': delays_samples,
        'phase_factors':  phase_factors,
        'memory':         np.zeros((number_of_bands, int(np.max(delays_samples))), dtype=complex)
    }


# function mixer = Gfb_Mixer_new(analyzer, delay, iterations)
def Gfb_Mixer_new(analyzer, delay, iterations=100):
    center_frequencies = analyzer['center_frequencies_hz']
    number_of_bands = center_frequencies.shape[0]
    sampling_frequency = analyzer['sampling_frequency_hz']

    # z_c = exp(2i * pi * center_frequencies(:) / sampling_frequency);
    z_c = np.exp(2j * np.pi * center_frequencies / sampling_frequency)
    # mixer.gains = ones(number_of_bands, 1);
    gains = np.ones(number_of_bands)

    # pos_f_response = Gfb_Analyzer_zresponse(analyzer, z_c);
    pos_f_response = Gfb_Analyzer_zresponse(analyzer, z_c)
    # neg_f_response = Gfb_Analyzer_zresponse(analyzer, conj(z_c));
    neg_f_response = Gfb_Analyzer_zresponse(analyzer, np.conj(z_c))

    # for band = [1:number_of_bands]
    for band in range(number_of_bands):
        # pos_f_response(:,band) = pos_f_response(:,band) * delay.phase_factors(band) .* z_c .^ -delay.delays_samples(band);
        pos_f_response[:, band] = pos_f_response[:, band] * \
                                  delay['phase_factors'][band] * \
                                  (z_c ** -delay['delays_samples'][band])
        # neg_f_response(:,band) = neg_f_response(:,band) * delay.phase_factors(band) .* conj(z_c) .^ -delay.delays_samples(band);
        neg_f_response[:, band] = neg_f_response[:, band] * \
                                  delay['phase_factors'][band] * \
                                  (np.conj(z_c) ** -delay['delays_samples'][band])

    # f_response = (pos_f_response + conj(neg_f_response)) / 2;
    f_response = (pos_f_response + np.conj(neg_f_response)) / 2.0

    # for i = [1:iterations]
    for _ in range(iterations):
        # selected_spectrum = f_response * mixer.gains;
        selected_spectrum = f_response @ gains
        # mixer.gains = mixer.gains ./ abs(selected_spectrum);
        gains = gains / np.abs(selected_spectrum)

    return {
        'type':  'Gfb_Mixer',
        'gains': gains
    }


# function [output, delay] = Gfb_Delay_process(delay, input)
def Gfb_Delay_process(delay, input_data):
    number_of_bands, number_of_samples = input_data.shape
    output = np.zeros((number_of_bands, number_of_samples))
    # for band = [1:number_of_bands]
    for band in range(number_of_bands):
        delay_val = int(delay['delays_samples'][band])
        phase_corrected = np.real(input_data[band, :] * delay['phase_factors'][band])
        # if (delay.delays_samples(band) == 0)
        if delay_val == 0:
            # output(band,:) = real(input(band,:) * delay.phase_factors(band));
            output[band, :] = phase_corrected
        # else
        else:
            # tmp_out = [delay.memory(band,1:delay.delays_samples(band)), real(input(band,:) * delay.phase_factors(band))];
            tmp_out = np.concatenate((delay['memory'][band, :delay_val], phase_corrected))
            # delay.memory(band,1:delay.delays_samples(band)) = tmp_out(number_of_samples+1:length(tmp_out));
            delay['memory'][band, :delay_val] = tmp_out[number_of_samples:]
            # output(band,:) = tmp_out(1:number_of_samples);
            output[band, :] = tmp_out[:number_of_samples]
    return output, delay


# function [output, mixer] = Gfb_Mixer_process(mixer, input)
def Gfb_Mixer_process(mixer, input_data):
    # output = mixer.gains * input;
    output = mixer['gains'] @ input_data
    return output, mixer


# function synthesizer = Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds)
def Gfb_Synthesizer_new(analyzer, desired_delay_in_seconds):
    fs = analyzer['sampling_frequency_hz']
    desired_delay_in_samples = int(round(desired_delay_in_seconds * fs))
    # synthesizer.delay = Gfb_Delay_new(analyzer, desired_delay_in_samples);
    delay = Gfb_Delay_new(analyzer, desired_delay_in_samples)
    # synthesizer.mixer = Gfb_Mixer_new(analyzer, synthesizer.delay);
    mixer = Gfb_Mixer_new(analyzer, delay)
    return {
        'type':  'Gfb_Synthesizer',
        'delay': delay,
        'mixer': mixer
    }


# function [output, synthesizer] = Gfb_Synthesizer_process(synthesizer, input)
def Gfb_Synthesizer_process(synthesizer, input_data):
    # [output, synthesizer.delay] = Gfb_Delay_process(synthesizer.delay, input);
    output, delay = Gfb_Delay_process(synthesizer['delay'], input_data)
    synthesizer['delay'] = delay
    # [output, synthesizer.mixer] = Gfb_Mixer_process(synthesizer.mixer, output);
    output, mixer = Gfb_Mixer_process(synthesizer['mixer'], output)
    synthesizer['mixer'] = mixer
    return output, synthesizer