"""
PEASS Test Suite - Subband Least-Squares Decomposition Unit Tests
File path: tests/unit/test_decomposition.py
"""

import pathlib

import numpy as np
import pytest
import scipy.signal as signal

from peass.backend_numpy import decomposition as decomposition_module
from peass.backend_numpy.decomposition import DecompositionConfiguration
from peass.backend_numpy.decomposition import decompose_distortion_components
from peass.backend_numpy.decomposition import get_analysis_modulation_matrix
from peass.backend_numpy.decomposition import get_synthesis_modulation_matrix
from peass.backend_numpy.decomposition import matlab_shade_length
from peass.backend_numpy.decomposition import matlab_shade_window
from peass.backend_numpy.decomposition import perform_least_squares_projection
from peass.backend_numpy.decomposition import perform_time_varying_least_squares_projection
from peass.backend_numpy.decomposition import run_auditory_analysis_filterbank
from peass.backend_numpy.decomposition import run_auditory_synthesis_filterbank

# (fs, shade_ms) pairs exercised by the MATLAB-fidelity shade window tests. The
# 44100/5.0, 44100/25.0 and 22050/10.0 entries land exactly on a .5 sample count,
# which pins MATLAB's round-half-away-from-zero tie rule.
SHADE_WINDOW_CASES = [
    (8000.0, 10.0), (8000.0, 5.0), (8000.0, 25.0),
    (16000.0, 10.0), (16000.0, 5.0), (16000.0, 25.0),
    (22050.0, 10.0),
    (44100.0, 10.0), (44100.0, 5.0), (44100.0, 25.0),
    (48000.0, 10.0), (48000.0, 5.0), (48000.0, 25.0),
    (8000.0, 0.125), (16000.0, 0.2),
]


def matlab_shade_window_reference(fs: float, shade_ms: float) -> np.ndarray:
    """
    Literal transcription of MATLAB `extractDistortionComponents.m` (v2.0.1)::

        wShadeIn = hann(2*round(shadeInMs/1000*fs+1),'periodic');
        wShadeIn = wShadeIn(2:end/2);

    with ``hann(N,'periodic')[k] = 0.5*(1 - cos(2*pi*(k-1)/N))`` for 1-based k=1..N and
    MATLAB's round-half-away-from-zero. Used as an independent oracle for the closed
    form used in production.
    """
    n_total = 2 * int(np.floor(shade_ms / 1000.0 * fs + 1.0 + 0.5))
    k = np.arange(1, n_total + 1)  # 1-based MATLAB index
    full_hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * (k - 1) / n_total))
    # MATLAB w(2:end/2) keeps 1-based k = 2 .. N/2 -> 0-based 1 .. N//2 - 1
    return full_hann[1:n_total // 2]


def apply_window_shading_helper(sig: np.ndarray, fs: float, shade_in: float = 10.0,
                                shade_out: float = 10.0) -> np.ndarray:
    sig_shaded = sig.copy()
    num_samples = sig_shaded.shape[0]

    fade_in_samples = matlab_shade_length(shade_in, fs) if shade_in > 0 else 0
    fade_out_samples = matlab_shade_length(shade_out, fs) if shade_out > 0 else 0

    if 0 < fade_in_samples <= num_samples:
        shade_in_window = matlab_shade_window(fade_in_samples)
        for chan_idx in range(sig_shaded.shape[1]):
            sig_shaded[:fade_in_samples, chan_idx] *= shade_in_window

    if 0 < fade_out_samples <= num_samples:
        shade_out_window = matlab_shade_window(fade_out_samples)[::-1]
        for chan_idx in range(sig_shaded.shape[1]):
            sig_shaded[-fade_out_samples:, chan_idx] *= shade_out_window

    return sig_shaded


@pytest.mark.unit
@pytest.mark.parametrize("fs, shade_ms", SHADE_WINDOW_CASES)
def test_shade_window_matches_matlab_hann_slice(fs, shade_ms):
    """
    The production shade window must reproduce MATLAB's
    `hann(2*round(ms/1000*fs+1),'periodic')` sliced with `(2:end/2)` exactly.
    """
    reference_window = matlab_shade_window_reference(fs, shade_ms)

    fade_samples = matlab_shade_length(shade_ms, fs)
    assert fade_samples == reference_window.size

    produced_window = matlab_shade_window(fade_samples)
    np.testing.assert_allclose(produced_window, reference_window, rtol=0.0, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("fs, shade_ms", SHADE_WINDOW_CASES)
def test_shade_window_is_strict_hann_interior(fs, shade_ms):
    """
    MATLAB drops both the leading zero and the unity midpoint of the Hann rise, so the
    window is strictly inside (0, 1). A plain 0->1 ramp would violate this.
    """
    window = matlab_shade_window(matlab_shade_length(shade_ms, fs))

    assert window.size > 0
    assert np.all(window > 0.0)
    assert np.all(window < 1.0)
    # Strictly increasing rise
    assert np.all(np.diff(window) > 0.0)


@pytest.mark.unit
def test_shade_out_window_is_exact_reverse_of_shade_in():
    """MATLAB derives wShadeOut from wShadeIn via flipud(), so the two must mirror exactly."""
    for fs, shade_ms in SHADE_WINDOW_CASES:
        shade_in_window = matlab_shade_window(matlab_shade_length(shade_ms, fs))
        shade_out_window = matlab_shade_window(matlab_shade_length(shade_ms, fs))[::-1]
        np.testing.assert_array_equal(shade_out_window, shade_in_window[::-1])


@pytest.mark.unit
def test_shade_window_length_uses_matlab_tie_breaking():
    """
    MATLAB rounds halves away from zero; Python's built-in round() rounds half to even.
    At 44.1 kHz a 5 ms shade is exactly 220.5 samples, which discriminates the two.
    """
    assert matlab_shade_length(5.0, 44100.0) == 221
    assert matlab_shade_length(25.0, 44100.0) == 1103
    assert matlab_shade_length(10.0, 22050.0) == 221
    assert matlab_shade_length(10.0, 44100.0) == 441
    # A single-sample window is well defined for the MATLAB formula (the old 0->1 ramp
    # divided by zero here), and evaluates to the Hann midpoint-adjacent value.
    assert matlab_shade_length(0.125, 8000.0) == 1
    np.testing.assert_allclose(matlab_shade_window(1), [0.5], rtol=0.0, atol=1e-15)


@pytest.mark.unit
def test_decomposition_algebraic_reconstruction(synthetic_audio_data):
    """
    Mathematically verifies that the decomposition satisfies the fundamental
    separation error conservation identity:

        estimate_reconstructed = target_distortion + interference + artifacts + true_target

    This verifies that no signal energy is lost or created during the
    subband least-squares and filterbank synthesis stages [1].
    """
    target, interferer, estimate, fs = synthetic_audio_data

    configuration = DecompositionConfiguration()
    decomposition_result = decompose_distortion_components(
        source_files=[target, interferer],
        estimate_file=estimate,
        configuration=configuration,
        sampling_frequency_hz=fs
    )
    waveforms = decomposition_result.waveforms

    assert waveforms.true_target.shape == estimate.shape

    estimate_shaded = apply_window_shading_helper(estimate, fs, 10.0, 10.0)
    number_of_channels = estimate_shaded.shape[1]
    est_reconstructed = np.zeros_like(estimate_shaded)

    def fit_to_length(sig, target_len):
        if len(sig) >= target_len:
            return sig[:target_len]
        return np.pad(sig, (0, target_len - len(sig)), mode='constant')

    for channel_idx in range(number_of_channels):
        subbands, analyzer, modulation_matrix = run_auditory_analysis_filterbank(estimate_shaded[:, channel_idx], fs)
        synth_signal, _ = run_auditory_synthesis_filterbank(subbands, analyzer)
        est_reconstructed[:, channel_idx] = fit_to_length(synth_signal, len(estimate_shaded))

    summed_sub_components = (
            waveforms.true_target + waveforms.target_distortion + waveforms.interference + waveforms.artifacts
    )
    np.testing.assert_allclose(est_reconstructed, summed_sub_components, atol=1e-7, rtol=1e-7)


@pytest.mark.unit
def test_decomposition_file_generation(audio_files_fixture):
    """
    Verifies that the decomposer successfully writes output WAV files to disk
    in file-based execution mode.
    """
    target_path, interferer_path, estimate_path = audio_files_fixture

    # Define the configurations using the new dataclass structure
    configuration = DecompositionConfiguration(
        destination_directory=str(target_path.parent)
    )

    # Execute using the modern, JIT-friendly entry point
    result = decompose_distortion_components(
        source_files=[str(target_path), str(interferer_path)],
        estimate_file=str(estimate_path),
        configuration=configuration
    )

    # Verify that the generated file paths are present
    assert result.file_paths is not None
    output_files = [
        result.file_paths.true_target,
        result.file_paths.target_distortion,
        result.file_paths.interference,
        result.file_paths.artifacts
    ]

    assert len(output_files) == 4
    for file_path in output_files:
        path_obj = pathlib.Path(file_path)
        assert path_obj.is_file()
        assert path_obj.stat().st_size > 0


@pytest.mark.unit
def test_decomposition_input_validation():
    """Checks bounds validation during parameter instantiation."""
    sampling_frequency_hz = 16000.0
    signal_length = 1000
    target = np.random.randn(signal_length, 1)
    interferer = np.random.randn(signal_length, 1)

    estimate_mismatched = np.random.randn(signal_length + 10, 1)
    with pytest.raises(ValueError, match="dimensions|size"):
        decompose_distortion_components(
            source_files=[target, interferer],
            estimate_file=estimate_mismatched,
            sampling_frequency_hz=sampling_frequency_hz
        )

    # Missing sampling rate in in-memory mode
    with pytest.raises(ValueError, match="requires explicit sampling rate"):
        decompose_distortion_components(
            source_files=[target, interferer],
            estimate_file=target,
            sampling_frequency_hz=None
        )


@pytest.mark.unit
def test_decomposition_gain_invariance_parameterized(synthetic_signal):
    """
    Verifies that a constant scaling (gain modification) of the target
    is mapped entirely to Target Distortion, leaving Interference/Artifacts at zero.
    """
    waveform, fs = synthetic_signal
    target = waveform
    silent_interferer = np.zeros_like(target)

    # 30% reduction in level
    estimate = 0.7 * target

    config = DecompositionConfiguration()
    result = decompose_distortion_components(
        source_files=[target, silent_interferer],
        estimate_file=estimate,
        configuration=config,
        sampling_frequency_hz=fs
    )
    waveforms = result.waveforms

    # Compare the synthesized target distortion against the synthesized true target.
    # Both have passed through the identical filterbank, ensuring exact mathematical parity.
    np.testing.assert_allclose(waveforms.target_distortion, -0.3 * waveforms.true_target, atol=1e-4, rtol=1e-4)

    # Interferences and Artifacts remain mathematically negligible
    assert np.max(np.abs(waveforms.interference)) < 1e-4
    assert np.max(np.abs(waveforms.artifacts)) < 1e-4


@pytest.mark.unit
def test_decomposition_in_bounds_delay_parameterized(synthetic_signal):
    """
    Verifies that temporal delays within the window boundaries are absorbed
    exclusively as Target Distortion.
    """
    waveform, fs = synthetic_signal
    target = waveform
    silent_interferer = np.zeros_like(target)

    # Small delay (3 samples, well within the 40ms solver limit)
    shift = 3
    estimate = np.roll(target, shift, axis=0)
    estimate[:shift, :] = 0.0

    config = DecompositionConfiguration()
    result = decompose_distortion_components(
        source_files=[target, silent_interferer],
        estimate_file=estimate,
        configuration=config,
        sampling_frequency_hz=fs
    )
    waveforms = result.waveforms

    # Shift is in-bounds. Subband least-squares modeling allows small numerical leakage
    # on transient edges, which is physically correct.
    assert np.max(np.abs(waveforms.interference)) < 5e-2
    assert np.max(np.abs(waveforms.artifacts)) < 5e-2


def time_varying_projection_frame_by_frame_reference(
        source_estimates: np.ndarray,
        true_sources: np.ndarray,
        filter_length: int,
        window_length: int,
        hop_size: int
) -> np.ndarray:
    """
    Literal frame-at-a-time transcription of the time-varying projection, driving the
    single-frame `perform_least_squares_projection` exactly as the production code did
    before the per-band batching landed. This is the oracle that pins the batched
    implementation to bitwise parity.
    """
    filter_half_length = (filter_length - 1) // 2
    pad_length = filter_length - 1 + window_length - 1
    true_sources = np.pad(true_sources, ((0, pad_length), (0, 0)), mode='constant')
    source_estimates = np.pad(source_estimates, ((0, pad_length), (0, 0)), mode='constant')

    total_samples, num_sources = true_sources.shape
    num_channels = source_estimates.shape[1]

    hann_window = signal.windows.hann(window_length, sym=False)
    analysis_window = np.sqrt(np.flipud(hann_window))
    synthesis_window = np.sqrt(np.flipud(hann_window))

    synthesis_weights = np.zeros((window_length, num_channels, num_sources))
    for channel_idx in range(num_channels):
        for source_idx in range(num_sources):
            synthesis_weights[:, channel_idx, source_idx] = synthesis_window

    projections_accumulation = np.zeros((total_samples, num_channels, num_sources),
                                        dtype=true_sources.dtype)
    window_gain_accumulation = np.zeros((total_samples, 1))

    edge_padding = np.zeros((filter_half_length, num_sources), dtype=true_sources.dtype)
    padded_sources = np.vstack([edge_padding, true_sources, edge_padding])
    frame_source_length = window_length + 2 * filter_half_length

    window_begin = 0
    window_end = window_begin + window_length
    while window_end - window_length / 2.0 <= projections_accumulation.shape[0] - window_length + 1:
        frame_projections = perform_least_squares_projection(
            source_estimates[window_begin:window_end, :],
            padded_sources[window_begin:window_begin + frame_source_length, :],
            filter_half_length,
            analysis_window
        )
        projections_accumulation[window_begin:window_end, :, :] += (
                frame_projections[:window_length, :, :] * synthesis_weights
        )
        window_gain_accumulation[window_begin:window_end, 0] += synthesis_window * analysis_window
        window_begin += hop_size
        window_end += hop_size

    valid_indices = (window_gain_accumulation[:, 0] != 0)
    for source_idx in range(num_sources):
        projections_accumulation[valid_indices, :, source_idx] /= window_gain_accumulation[valid_indices, :]

    return projections_accumulation[:-(window_length - 1), :, :]


def make_subband_projection_case(num_sources: int, num_channels: int, num_samples: int,
                                 silent_prefix: int = 0, seed: int = 7):
    """
    Complex subband-shaped inputs for the least-squares projection, in the layout the
    filterbank hands over.
    """
    rng = np.random.default_rng(seed)
    sources = (rng.standard_normal((num_samples, num_sources))
               + 1j * rng.standard_normal((num_samples, num_sources)))
    if silent_prefix:
        sources[:silent_prefix, :] = 0.0
    estimates = (rng.standard_normal((num_samples, num_channels))
                 + 1j * rng.standard_normal((num_samples, num_channels)))
    return estimates, sources


# (num_sources, num_channels, num_samples, silent_prefix, filter_length, window_length,
#  hop_size). The hop-3 cases produce more frames than one LEAST_SQUARES_FRAME_BATCH,
# so the seam between batches is covered; the silent_prefix cases drive the silence
# bypass, including one where every frame of the leading batch is bypassed.
BATCHED_PROJECTION_CASES = [
    (2, 1, 900, 0, 11, 133, 33),
    (2, 1, 900, 0, 11, 40, 3),
    (2, 1, 900, 600, 11, 40, 3),
    (2, 1, 400, 400, 11, 133, 33),
    (3, 1, 500, 0, 7, 61, 15),
    (2, 2, 500, 0, 11, 61, 15),
    (1, 1, 300, 0, 3, 45, 11),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_sources, num_channels, num_samples, silent_prefix, filter_length, window_length, hop_size",
    BATCHED_PROJECTION_CASES
)
def test_batched_projection_matches_single_frame_bitwise(
        num_sources, num_channels, num_samples, silent_prefix,
        filter_length, window_length, hop_size
):
    """
    The batched per-band Gram/RHS build must be *bitwise* identical to solving one
    frame at a time, not merely close: it issues the same per-frame GEMM shapes, so
    nothing is reassociated. If this ever has to be loosened to `allclose`, the
    batching has started reordering arithmetic and the bit-identity claim in
    `perform_time_varying_least_squares_projection` no longer holds.

    Compared as bytes rather than with `==` so that even the sign of zero has to
    match -- which is what the explicit zeroing of bypassed frames exists to preserve.
    """
    estimates, sources = make_subband_projection_case(
        num_sources, num_channels, num_samples, silent_prefix
    )

    batched = perform_time_varying_least_squares_projection(
        estimates, sources, filter_length, window_length, hop_size
    )
    reference = time_varying_projection_frame_by_frame_reference(
        estimates, sources, filter_length, window_length, hop_size
    )

    assert batched.shape == reference.shape
    assert batched.dtype == reference.dtype
    assert np.array_equal(batched.view(np.uint8), reference.view(np.uint8))


@pytest.mark.unit
def test_batched_projection_cases_span_more_than_one_batch():
    """
    Guards the parametrization above: the seam between two batches is only exercised if
    at least one case really produces more frames than `LEAST_SQUARES_FRAME_BATCH`.
    """
    num_samples, filter_length, window_length, hop_size = 900, 11, 40, 3
    total_samples = num_samples + filter_length - 1 + window_length - 1
    frame_count = len(range(0, total_samples - window_length, hop_size))
    assert frame_count > decomposition_module.LEAST_SQUARES_FRAME_BATCH


@pytest.mark.unit
def test_silent_sources_project_to_exact_zero():
    """
    The silence bypass must yield exact zeros, and the batched energy reduction has to
    agree with the per-frame `np.sum` it replaced on where the threshold falls.
    """
    rng = np.random.default_rng(3)
    sources = np.zeros((400, 2), dtype=complex)
    estimates = rng.standard_normal((400, 1)) + 1j * rng.standard_normal((400, 1))

    projections = perform_time_varying_least_squares_projection(estimates, sources, 11, 61, 15)

    assert np.count_nonzero(projections) == 0


# -----------------------------------------------------------------------------
# Modulation matrix sharing between analysis and synthesis
# -----------------------------------------------------------------------------

MODULATION_CASES = [
    (24000.0, 512, np.array([100.0, 437.5, 1000.0, 4321.0])),
    (24000.0, 4096, np.array([20.0, 1000.0, 11999.0])),
    (12000.0, 1000, np.array([250.0, 2500.0])),
]


@pytest.mark.unit
@pytest.mark.parametrize("sampling_frequency, num_samples, center_frequencies", MODULATION_CASES)
def test_modulation_matrices_are_bitwise_conjugates(sampling_frequency, num_samples, center_frequencies):
    """
    Sharing one np.exp between analysis and synthesis rests on `exp(+ix)` being the
    *bitwise* conjugate of `exp(-ix)`: the two arguments are exact IEEE negations and
    cos/sin are exactly even/odd about zero. Asserted against a freshly evaluated
    np.exp so a NumPy or libm change that broke the symmetry fails here rather than
    silently perturbing the decomposition.

    The identity holds bitwise for every t >= 1 but *not* for t = 0, where the sign of
    the exponent has been multiplied away and both directions land on 1+0j -- so
    conjugating gives 1-0j, equal in value but not in bytes. That is pinned below too,
    because it is exactly why `get_synthesis_modulation_matrix` recomputes column 0
    instead of conjugating it.
    """
    decomposition_module._MODULATION_MATRIX_CACHE.clear()

    time_steps = np.arange(num_samples)
    expected_analysis = np.exp(
        -2j * np.pi / sampling_frequency * center_frequencies[:, np.newaxis] * time_steps
    )
    expected_synthesis = np.exp(
        2j * np.pi / sampling_frequency * center_frequencies[:, np.newaxis] * time_steps
    )

    assert np.array_equal(expected_synthesis[:, 1:].view(np.uint8),
                          np.conjugate(expected_analysis[:, 1:]).view(np.uint8))
    # t = 0: equal in value, opposite in the sign of the zero imaginary part.
    assert np.array_equal(expected_synthesis[:, 0], np.conjugate(expected_analysis[:, 0]))
    assert not np.array_equal(expected_synthesis[:, :1].view(np.uint8),
                              np.conjugate(expected_analysis[:, :1]).view(np.uint8))

    # The public getters must nonetheless reproduce a direct np.exp byte for byte.
    analysis = get_analysis_modulation_matrix(sampling_frequency, num_samples, center_frequencies)
    synthesis = get_synthesis_modulation_matrix(sampling_frequency, num_samples, center_frequencies)
    assert np.array_equal(analysis.view(np.uint8), expected_analysis.view(np.uint8))
    assert np.array_equal(synthesis.view(np.uint8), expected_synthesis.view(np.uint8))


@pytest.mark.unit
@pytest.mark.parametrize("sampling_frequency, num_samples, center_frequencies", MODULATION_CASES)
def test_modulation_matrix_prefix_identity(sampling_frequency, num_samples, center_frequencies):
    """
    Synthesis needs a few hundred more columns than analysis produced, so it reuses the
    analysis matrix as a prefix and only exponentiates the overhang. That is valid only
    if a length-N build is bitwise the first N columns of a longer one.
    """
    decomposition_module._MODULATION_MATRIX_CACHE.clear()
    short_matrix = get_analysis_modulation_matrix(
        sampling_frequency, num_samples, center_frequencies
    ).copy()

    decomposition_module._MODULATION_MATRIX_CACHE.clear()
    long_matrix = get_analysis_modulation_matrix(
        sampling_frequency, num_samples + 337, center_frequencies
    )

    assert np.array_equal(long_matrix[:, :num_samples].view(np.uint8), short_matrix.view(np.uint8))


@pytest.mark.unit
@pytest.mark.parametrize("sampling_frequency, num_samples, center_frequencies", MODULATION_CASES)
def test_synthesis_modulation_matrix_is_independent_of_cache_state(
        sampling_frequency, num_samples, center_frequencies
):
    """
    Synthesis takes the conjugate shortcut when an analysis matrix is cached and falls
    back to a plain np.exp when it is not. The cold path, the fully shared path, and
    the mixed path -- where the cached analysis matrix is shorter than the request, so
    only the overhang is exponentiated -- must agree to the last bit.
    """
    overhang_length = num_samples + 337

    decomposition_module._MODULATION_MATRIX_CACHE.clear()
    cold = get_synthesis_modulation_matrix(
        sampling_frequency, overhang_length, center_frequencies
    ).copy()

    decomposition_module._MODULATION_MATRIX_CACHE.clear()
    get_analysis_modulation_matrix(sampling_frequency, overhang_length, center_frequencies)
    fully_shared = get_synthesis_modulation_matrix(
        sampling_frequency, overhang_length, center_frequencies
    ).copy()

    decomposition_module._MODULATION_MATRIX_CACHE.clear()
    get_analysis_modulation_matrix(sampling_frequency, num_samples, center_frequencies)
    partially_shared = get_synthesis_modulation_matrix(
        sampling_frequency, overhang_length, center_frequencies
    )

    assert np.array_equal(fully_shared.view(np.uint8), cold.view(np.uint8))
    assert np.array_equal(partially_shared.view(np.uint8), cold.view(np.uint8))


@pytest.mark.unit
def test_modulation_matrix_cache_stays_bounded():
    """
    Every entry is a full (num_bands x num_samples) complex matrix, so an unbounded
    cache would be a memory leak in a process scoring many different-length signals.
    """
    decomposition_module._MODULATION_MATRIX_CACHE.clear()
    center_frequencies = np.array([100.0, 1000.0])

    for num_samples in range(64, 64 + 8 * 16, 16):
        get_analysis_modulation_matrix(24000.0, num_samples, center_frequencies)
        get_synthesis_modulation_matrix(24000.0, num_samples, center_frequencies)
        assert (len(decomposition_module._MODULATION_MATRIX_CACHE)
                <= decomposition_module._MODULATION_MATRIX_CACHE_SIZE)
