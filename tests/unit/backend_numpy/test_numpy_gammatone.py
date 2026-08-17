"""
PEASS Test Suite - Gammatone Filterbank Mathematics Unit Tests
File path: tests/unit/test_gammatone.py
"""

import math

import numpy as np
import pytest

from peass.backend_numpy import gammatone as gammatone_module
from peass.backend_numpy.gammatone import GammatoneAnalyzer
from peass.backend_numpy.gammatone import GammatoneFilter
from peass.backend_numpy.gammatone import GammatoneSynthesizer
from peass.backend_numpy.gammatone import calculate_audiological_equivalent_rectangular_bandwidth
from peass.backend_numpy.gammatone import calculate_equivalent_rectangular_bandwidth
from peass.backend_numpy.gammatone import convert_equivalent_rectangular_bandwidth_scale_to_frequency
from peass.backend_numpy.gammatone import convert_frequency_to_equivalent_rectangular_bandwidth_scale
from peass.backend_numpy.gammatone import fast_resample_poly
from peass.backend_numpy.gammatone import get_equivalent_rectangular_bandwidth_center_frequencies
from peass.backend_numpy.gammatone import resample_output_length


@pytest.mark.unit
@pytest.mark.parametrize("frequency_hz", [50.0, 100.0, 440.0, 1000.0, 5000.0, 12000.0])
def test_erb_scale_inversion(frequency_hz):
    """
    Tests that frequency_to_erb_scale and erb_scale_to_frequency are mathematical inverses.
    """
    erb = convert_frequency_to_equivalent_rectangular_bandwidth_scale(frequency_hz)
    f_recon = convert_equivalent_rectangular_bandwidth_scale_to_frequency(erb)
    assert np.isclose(frequency_hz, f_recon)


@pytest.mark.unit
def test_calculate_erb_bandwidth_values():
    """
    Checks calculation of ERB bandwidths against expected values.
    Formula: ERB = 24.7 * (0.00437 * fc + 1.0)
    """
    # At fc = 0: ERB = 24.7
    assert np.isclose(calculate_equivalent_rectangular_bandwidth(0.0), 24.7)

    # At fc = 1000: ERB = 24.7 * (4.37 + 1.0) = 132.639
    assert np.isclose(calculate_equivalent_rectangular_bandwidth(1000.0), 132.639)


def _recover_audiological_bandwidth_from_filter(filter_instance: GammatoneFilter) -> float:
    """
    Inverts the Hohmann 2002 eq. (14) chain to recover the audiological ERB that the
    filter constructor actually used, straight from the constructed pole.

    lambda = exp(-2*pi*b/fs)  ->  b = -ln(lambda) * fs / (2*pi)  ->  erb = b * a_gamma
    """
    order = filter_instance.filter_order
    gamma_constant = (math.pi * math.factorial(2 * order - 2) * (2.0 ** -(2 * order - 2)) /
                      (math.factorial(order - 1) ** 2))
    decay_constant = -math.log(filter_instance.lambda_decay_factor) * filter_instance.sampling_frequency_hz / (
            2.0 * math.pi)
    return decay_constant * gamma_constant


@pytest.mark.unit
@pytest.mark.parametrize("center_frequency_hz", [20.0, 100.0, 440.0, 1000.0, 5000.0, 12000.0])
def test_erb_bandwidth_helper_pins_erbbw_formula(center_frequency_hz):
    """
    Pins `calculate_equivalent_rectangular_bandwidth` to the Glasberg & Moore form used
    by the MATLAB reference's `erbBW.m`: 24.7 * (0.00437 * fc + 1).

    This helper feeds the per-band decimation factor (`myPemoAnalysisFilterBank.m:53`)
    and NOTHING else. It must not be collapsed onto the gammatone filter constructor's
    (GFB_L + fc/GFB_Q) form -- see
    `test_gammatone_filter_bandwidth_pins_hohmann_formula`. The two formulas are the
    same empirical fit with one constant rounded, so a refactor that unifies them looks
    harmless and is not: MATLAB deliberately uses each in a different place, and parity
    requires reproducing that split.
    """
    expected = 24.7 * (0.00437 * center_frequency_hz + 1.0)
    assert calculate_equivalent_rectangular_bandwidth(center_frequency_hz) == pytest.approx(expected, rel=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize("center_frequency_hz", [20.0, 100.0, 440.0, 1000.0, 5000.0, 12000.0])
def test_gammatone_filter_bandwidth_pins_hohmann_formula(center_frequency_hz):
    """
    Pins the gammatone filter constructor's bandwidth to `Gfb_Filter_new.m:61`:
    audiological_erb = (GFB_L + fc / GFB_Q) * bandwidth_factor, with GFB_L = 24.7 and
    GFB_Q = 9.265 (Hohmann 2002 eq. 17, `Gfb_set_constants.m`).

    The constructor must NOT use the `erbBW.m` form 24.7 * (0.00437 * fc + 1), which is
    the same fit with 1/(24.7*0.00437) = 9.264488... rounded to 9.265 and which belongs
    at the decimation call site instead. The gap is only ~5.5e-5 relative on the slope
    term, so this test asserts tightly enough to catch a silent swap back.
    """
    expected = 24.7 + center_frequency_hz / 9.265

    # The standalone helper matches...
    assert calculate_audiological_equivalent_rectangular_bandwidth(center_frequency_hz) == pytest.approx(
        expected, rel=1e-15
    )

    # ...and so does the pole that the constructor actually built.
    filter_instance = GammatoneFilter(sampling_frequency_hz=32000.0, center_frequency_hz=center_frequency_hz)
    assert _recover_audiological_bandwidth_from_filter(filter_instance) == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("bandwidth_factor", [0.5, 1.0, 2.0])
def test_gammatone_filter_bandwidth_scales_with_bandwidth_factor(bandwidth_factor):
    """
    `Gfb_Filter_new.m:61` multiplies the whole audiological ERB by bandwidth_factor.
    """
    center_frequency_hz = 1000.0
    expected = (24.7 + center_frequency_hz / 9.265) * bandwidth_factor

    filter_instance = GammatoneFilter(
        sampling_frequency_hz=32000.0,
        center_frequency_hz=center_frequency_hz,
        bandwidth_factor=bandwidth_factor
    )
    assert _recover_audiological_bandwidth_from_filter(filter_instance) == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_two_erb_formulas_stay_distinct():
    """
    Guards the intent directly: the constructor's ERB and the erbBW helper are close but
    must remain two separate expressions. If a refactor unifies them this test fails,
    which is the point -- the MATLAB reference uses both, in different places.
    """
    center_frequency_hz = 10000.0
    hohmann = calculate_audiological_equivalent_rectangular_bandwidth(center_frequency_hz)
    glasberg_moore = calculate_equivalent_rectangular_bandwidth(center_frequency_hz)

    assert hohmann != glasberg_moore
    # Same fit, one rounded constant: agreement to ~5e-5 relative, but no closer.
    assert hohmann == pytest.approx(glasberg_moore, rel=1e-4)
    assert hohmann != pytest.approx(glasberg_moore, rel=1e-6)


@pytest.mark.unit
def test_analyzer_bandwidths_use_erbbw_for_decimation():
    """
    `GammatoneAnalyzer.bandwidths` is the decimation bandwidth, so it stays on the
    erbBW form even though the analyzer's filters are built with the Hohmann form.
    """
    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=16000.0,
        lower_cutoff_frequency_hz=100.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=4000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    np.testing.assert_allclose(
        analyzer.bandwidths,
        24.7 * (0.00437 * analyzer.center_frequencies + 1.0),
        rtol=1e-15
    )

    for filter_instance in analyzer.filters:
        expected = 24.7 + filter_instance.center_frequency_hz / 9.265
        assert _recover_audiological_bandwidth_from_filter(filter_instance) == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_get_center_frequencies_range():
    """
    Verifies that get_center_frequencies produces frequencies within bounds and contains the base frequency.
    """
    lower = 100.0
    base = 1000.0
    upper = 8000.0
    filters_per_erb = 1.0

    cfs = get_equivalent_rectangular_bandwidth_center_frequencies(filters_per_erb, lower, base, upper)

    assert cfs[0] >= lower
    assert cfs[-1] <= upper
    # There should be an exact match for the base frequency
    assert np.any(np.isclose(cfs, base))


@pytest.mark.unit
def test_gammatone_filter_state_clearing():
    """
    Verifies state setting and clearing inside GammatoneFilter.
    """
    filt = GammatoneFilter(sampling_frequency_hz=16000.0, center_frequency_hz=1000.0)

    # Assert state is initially zeros
    assert np.all(filt.state == 0.0)

    # Process some noise to populate the state
    input_noise = np.random.default_rng(seed=7).standard_normal(100)
    filt.process(input_noise)

    # State should now be populated with non-zero values
    assert np.any(filt.state != 0.0)

    # Clear the state
    filt.clear_state()
    assert np.all(filt.state == 0.0)


@pytest.mark.unit
def test_gammatone_analyzer_state_clearing():
    """
    Verifies state setting and clearing inside GammatoneAnalyzer.
    """
    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=16000.0,
        lower_cutoff_frequency_hz=100.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=4000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    # All filter states should start at zero
    for filt in analyzer.filters:
        assert np.all(filt.state == 0.0)

    # Process signal
    analyzer.process(np.random.default_rng(seed=7).standard_normal(50))

    # States should be non-zero
    for filt in analyzer.filters:
        assert np.any(filt.state != 0.0)

    # Clear states
    analyzer.clear_state()
    for filt in analyzer.filters:
        assert np.all(filt.state == 0.0)


@pytest.mark.unit
def test_gammatone_reconstruction_parameterized(synthetic_signal):
    """
    Verifies that the Gammatone analysis-synthesis filterbank correctly
    reconstructs signals across diverse parameterized waveforms.
    """
    waveform, fs = synthetic_signal
    sig_1d = waveform.ravel()

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=fs,
        lower_cutoff_frequency_hz=80.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=6000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    subbands = analyzer.process(sig_1d)
    desired_delay_sec = 0.004
    synthesizer = GammatoneSynthesizer(analyzer, desired_delay_seconds=desired_delay_sec)
    reconstructed = synthesizer.process(subbands)

    delay_samples = int(round(desired_delay_sec * fs))

    # Slice to discard transient filter-delay periods
    original_sliced = sig_1d[delay_samples:-delay_samples]
    reconstructed_sliced = reconstructed[2 * delay_samples:]

    min_len = min(len(original_sliced), len(reconstructed_sliced))
    original_sliced = original_sliced[:min_len]
    reconstructed_sliced = reconstructed_sliced[:min_len]

    # Verify high phase and shape matching through cross-correlation. The bound
    # is deliberately loose: broadband inputs (chirp, white noise) carry energy
    # outside the [80, 6000] Hz filterbank span that cannot be reconstructed, so
    # ~0.89 is the physical ceiling for those cases (narrowband tones reach ~0.99).
    #
    # FRAGILE -- do not tighten. Measured 2026-08-15 across the parametrization:
    #
    #     signal      correlation    1 - corr
    #     sine        0.9929427      7.06e-03
    #     square      0.9911042      8.90e-03
    #     triangle    0.9956218      4.38e-03
    #     chirp       0.8954640      1.05e-01
    #     noise       0.8896672      1.10e-01   <- worst
    #
    # 0.85 allows a 1.5e-1 deficit, so `noise` runs at 1.36x of the bar -- the
    # narrowest correlation margin in the suite. It is not flaky here (the fixture
    # RNG is seeded and this path has no cross-platform-sensitive arithmetic), but
    # any change that shifts broadband reconstruction even slightly will trip it,
    # and the right fix in that case is to re-measure and lower the bar, not to
    # assume a regression. Do not raise it toward the 0.99 the tonal cases reach.
    corr = np.corrcoef(original_sliced, reconstructed_sliced)[0, 1]
    assert corr > 0.85, f"Reconstruction fidelity too low: {corr:.6f}"


@pytest.mark.unit
def test_gammatone_fallback_vs_jit_equivalence():
    """
    Verifies that disabling Numba JIT acceleration forces the filterbank
    to run on the fallback path and produces identical outputs down to float precision.
    """
    import peass.backend_numpy.gammatone as gammatone
    if not gammatone._HAS_NUMBA:
        pytest.skip("Numba is not installed or enabled in this environment.")
    from peass.backend_numpy.gammatone import GammatoneAnalyzer

    # Generate a random signal
    rng = np.random.default_rng(seed=123)
    signal_input = rng.normal(0.0, 0.5, 1000)
    fs = 16000.0

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=fs,
        lower_cutoff_frequency_hz=80.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=4000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    # 1. Run with JIT active
    original_has_numba = gammatone._HAS_NUMBA
    try:
        gammatone._HAS_NUMBA = True
        analyzer.clear_state()
        jit_output = analyzer.process(signal_input)

        # 2. Force fallback path by mocking _HAS_NUMBA as False
        gammatone._HAS_NUMBA = False
        analyzer.clear_state()
        fallback_output = analyzer.process(signal_input)

        # Assert mathematical parity down to machine precision
        np.testing.assert_allclose(jit_output, fallback_output, rtol=1e-12, atol=1e-12)

    finally:
        # Restore original setting
        gammatone._HAS_NUMBA = original_has_numba


@pytest.mark.unit
def test_gammatone_process_real_is_bit_identical_to_real_of_process():
    """`process_real` must be exactly `np.real(process(...))`, not merely close.

    It exists only to skip the complex128 output the auditory model discards, so
    the recurrence and the mutated filter state have to be untouched. Asserted as
    byte equality on the samples AND on every band's post-call state, from a
    non-zero (warmed) starting state -- a zero state would hide any state bug.
    """
    import peass.backend_numpy.gammatone as gammatone
    if not gammatone._HAS_NUMBA:
        pytest.skip("Numba is not installed or enabled in this environment.")

    rng = np.random.default_rng(seed=20260815)
    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=16000.0,
        lower_cutoff_frequency_hz=235.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=7000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    analyzer.process(rng.normal(0.0, 0.5, 500))  # warm the state so it is non-zero
    warmed_state = [f.state.copy() for f in analyzer.filters]
    assert any(np.any(s != 0.0) for s in warmed_state)

    signal_input = rng.normal(0.0, 0.5, 3000)
    complex_output = analyzer.process(signal_input)
    state_after_complex = [f.state.copy() for f in analyzer.filters]

    for filter_instance, state in zip(analyzer.filters, warmed_state):
        filter_instance.state = state.copy()
    real_output = analyzer.process_real(signal_input)
    state_after_real = [f.state.copy() for f in analyzer.filters]

    expected = np.real(complex_output)
    assert real_output.dtype == np.float64
    assert real_output.flags["C_CONTIGUOUS"]
    assert real_output.tobytes() == np.ascontiguousarray(expected).tobytes()
    assert np.array_equal(real_output, expected)
    for after_complex, after_real in zip(state_after_complex, state_after_real):
        assert np.array_equal(after_complex, after_real)


@pytest.mark.unit
def test_gammatone_process_real_fallback_matches_jit_path():
    """The no-Numba branch of `process_real` must agree with the JIT branch.

    `process_real` is the auditory model's entry point, so its fallback has to
    stay usable when Numba is absent; without this the branch is never executed.
    """
    import peass.backend_numpy.gammatone as gammatone
    if not gammatone._HAS_NUMBA:
        pytest.skip("Numba is not installed or enabled in this environment.")

    rng = np.random.default_rng(seed=99)
    signal_input = rng.normal(0.0, 0.5, 800)
    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=16000.0,
        lower_cutoff_frequency_hz=235.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=4000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    original_has_numba = gammatone._HAS_NUMBA
    try:
        analyzer.clear_state()
        jit_output = analyzer.process_real(signal_input)

        gammatone._HAS_NUMBA = False
        analyzer.clear_state()
        fallback_output = analyzer.process_real(signal_input)
    finally:
        gammatone._HAS_NUMBA = original_has_numba

    assert fallback_output.dtype == np.float64
    assert fallback_output.flags["C_CONTIGUOUS"]
    # Byte equality, not a tolerance: the two branches evaluate the same
    # recurrence in the same order, and the measured deviation is exactly 0.0.
    # A tolerance here would silently accept a real divergence appearing later.
    assert np.array_equal(jit_output, fallback_output)


@pytest.mark.unit
@pytest.mark.parametrize("sampling_frequency_hz", [8000.0, 16000.0, 44100.0])
def test_gammatone_analysis_reconstruction(sampling_frequency_hz):
    """
    Verifies the analysis-synthesis filterbank reconstructs a tone with the
    correct shape AND high fidelity (once the fixed group delay is compensated).
    """
    duration_seconds = 0.5
    num_samples = int(duration_seconds * sampling_frequency_hz)
    signal_input = np.sin(2.0 * np.pi * 500.0 * np.linspace(0, duration_seconds, num_samples))

    # Keep the upper cutoff strictly below Nyquist so the filterbank is valid.
    upper_cutoff = min(8000.0, sampling_frequency_hz / 2.0 * 0.9)
    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=sampling_frequency_hz,
        lower_cutoff_frequency_hz=80.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=upper_cutoff,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    subbands = analyzer.process(signal_input)
    assert subbands.ndim == 2

    desired_delay_seconds = 0.004
    synthesizer = GammatoneSynthesizer(analyzer, desired_delay_seconds=desired_delay_seconds)
    reconstructed = np.real(synthesizer.process(subbands))
    assert reconstructed.ndim == 1
    assert reconstructed.shape[0] == num_samples

    # Compensate the synthesis group delay, then assert high reconstruction fidelity.
    #
    # This is a genuinely lossy round trip, so the bar is set by physics rather than
    # by arithmetic noise and cannot be tightened the way a cross-backend parity bar
    # can. Measured 2026-08-15, worst over the parametrization:
    #
    #     fs        correlation    1 - corr
    #     8000      0.9950593      4.94e-03   <- worst
    #     16000     0.9956465      4.35e-03
    #     44100     0.9960657      3.93e-03
    #
    # 0.98 allows a deficit of 2e-2, i.e. ~4x over the worst case. That is a small
    # margin by the standards of the parity tests, but appropriate here: there is no
    # RNG and no resampling in this path, the two backends' filterbanks agree to
    # 1e-13, and the value moves only 26% across a 5.5x span of sampling rate. The
    # 10x-100x rule of thumb would actually *loosen* this bar to 0.95, so it is left
    # where it is.
    delay_samples = int(round(desired_delay_seconds * sampling_frequency_hz))
    original_aligned = signal_input[:num_samples - delay_samples]
    reconstructed_aligned = reconstructed[delay_samples:delay_samples + len(original_aligned)]
    corr = np.corrcoef(original_aligned, reconstructed_aligned)[0, 1]
    assert corr > 0.98, f"Reconstruction fidelity too low at {sampling_frequency_hz} Hz: {corr:.6f}"


@pytest.mark.unit
def test_filterbank_dc_offset_rejection():
    """
    Verifies that the auditory filterbank correctly blocks constant DC bias offsets.
    """
    fs = 16000.0
    time_steps = np.linspace(0.0, 0.5, int(0.5 * fs), endpoint=False)

    dc_bias = 2.5
    signal_with_dc = np.sin(2.0 * np.pi * 500.0 * time_steps) + dc_bias

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=fs,
        lower_cutoff_frequency_hz=235.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=6000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    subbands = np.real(analyzer.process(signal_with_dc))

    # Every subband must reject the constant input. The residual is *proportional to
    # dc_bias*: measured 2026-08-17, |mean|/dc_bias is 2.018e-03 at the worst band
    # (band 0, cf 236.3 Hz) and constant to four significant figures across dc_bias
    # 0.25, 2.5, 25 and 250. So the bar is a fraction of the bias, not a bare absolute
    # number that would silently re-tune itself the day the bias above is changed.
    #
    # `4e-3` of the bias reproduces the previous literal `1e-2` *exactly* at
    # dc_bias=2.5 (2.5 * 4e-3 == 1e-2 is true in binary floating point, checked), so
    # this is a units fix and nothing else: the bar has not moved in either direction
    # and the margin stays 1.98x against the measured 2.018e-03.
    #
    # That 1.98x is thin by the standards of the cross-backend parity tests, and it is
    # deliberately NOT widened to their ~30x. Those bars bound float64 arithmetic noise
    # whose size varies with the platform's FFT/LAPACK/libm; this one bounds a
    # deterministic property of one filterbank, computed by one backend, with no RNG and
    # no resampling in the path -- there is nothing here for another platform to move by
    # more than a few ULP. Widening it 30x would take the bar to 6e-2 of the bias, i.e.
    # 24 dB of attenuation, which no longer asserts DC rejection at all.
    #
    # NOTE the comment that used to be here claimed "attenuation > 50 dB", which the
    # code never asserted: 4e-3 of the bias is 47.96 dB, and a genuine 50 dB bar would
    # be 3.162e-3 -- *tighter* than what was there. The worst band measures 53.90 dB.
    # Making the 50 dB claim true would leave 1.57x of margin and is a separate,
    # deliberate call, not a drive-by.
    max_leakage = dc_bias * 4e-3
    for band_idx in range(subbands.shape[0]):
        leakage = float(np.abs(np.mean(subbands[band_idx, :])))
        assert leakage < max_leakage, (
            f"Band {band_idx} (cf {analyzer.center_frequencies[band_idx]:.1f} Hz) leaks "
            f"{leakage:.3e} of a {dc_bias} DC bias, above the {max_leakage:.3e} bar "
            f"({20 * np.log10(dc_bias / leakage):.2f} dB of attenuation, "
            f"{20 * np.log10(1.0 / 4e-3):.2f} dB required)"
        )


@pytest.mark.unit
@pytest.mark.parametrize("sampling_frequency_hz", [8000.0, 16000.0, 44100.0])
def test_gammatone_analysis_reconstruction_fidelity(sampling_frequency_hz):
    """
    Verifies that the Gammatone analysis and synthesis filterbank reconstruct
    the input signal with high fidelity (cross-correlation close to 1.0)
    after accounting for group delay.
    """
    duration_seconds = 0.5
    num_samples = int(duration_seconds * sampling_frequency_hz)
    time_steps = np.linspace(0, duration_seconds, num_samples, endpoint=False)
    signal_input = np.sin(2.0 * np.pi * 440.0 * time_steps)

    analyzer = GammatoneAnalyzer(
        sampling_frequency_hz=sampling_frequency_hz,
        lower_cutoff_frequency_hz=80.0,
        specified_center_frequency_hz=1000.0,
        upper_cutoff_frequency_hz=3000.0,
        filters_per_equivalent_rectangular_bandwidth=1.0
    )

    subbands = analyzer.process(signal_input)
    assert subbands.ndim == 2

    desired_delay_seconds = 0.004
    synthesizer = GammatoneSynthesizer(analyzer, desired_delay_seconds=desired_delay_seconds)
    reconstructed = synthesizer.process(subbands)

    # Slice arrays to account for the synthesis delay buffer
    delay_samples = int(round(desired_delay_seconds * sampling_frequency_hz))
    original_slice = signal_input[delay_samples:-delay_samples]
    reconstructed_slice = reconstructed[2 * delay_samples: len(original_slice) + 2 * delay_samples]

    min_len = min(len(original_slice), len(reconstructed_slice))
    original_slice = original_slice[:min_len]
    reconstructed_slice = reconstructed_slice[:min_len]

    # Assert high-fidelity signal reconstruction. Lossy round trip, same as
    # test_gammatone_analysis_reconstruction above -- measured 2026-08-15:
    #
    #     fs        correlation    1 - corr
    #     8000      0.9922756      7.72e-03   <- worst
    #     16000     0.9929189      7.08e-03
    #     44100     0.9930428      6.96e-03
    #
    # Tightened 0.90 -> 0.98 on 2026-08-15. The old bar allowed a 1e-1 deficit
    # against a measured 7.7e-03, i.e. 13x, which let the filterbank degrade by an
    # order of magnitude unnoticed. 0.98 leaves ~2.6x and matches the sibling test's
    # bar for the same physical quantity; a 440 Hz tone against a [80, 3000] Hz bank
    # is the near-ideal case for this filterbank, so the deficit is small and stable
    # (11% spread across a 5.5x span of sampling rate).
    corr = np.corrcoef(original_slice, reconstructed_slice)[0, 1]
    assert corr > 0.98, f"Reconstruction fidelity too low at {sampling_frequency_hz} Hz: {corr:.6f}"


# -----------------------------------------------------------------------------
# fast_resample_poly(out=...) -- in-place scatter destination
# -----------------------------------------------------------------------------
#
# `out=` exists so `run_auditory_synthesis_filterbank` can upsample straight into
# the zero-filled (bands x samples) buffer it owns instead of allocating a block
# and copying it in. The contract these tests pin is twofold: the samples written
# are BITWISE the ones the allocating call returns (the optimization must change
# no arithmetic), and the destination beyond the result length is left completely
# untouched (the caller relies on those zeros surviving).

RESAMPLE_RATIO_CASES = [
    (4, 1),  # pure interpolation -- the synthesis upsampling case
    (1, 4),  # pure decimation -- the analysis case
    (3, 2),  # mixed ratio -- always routed to SciPy's upfirdn
    (5, 5),  # up == down -- the early-out copy
]


# `False` forces the SciPy `upfirdn` fallback, which cannot write in place and so
# takes a different route through `out=`; both routes are covered.
RESAMPLER_BACKENDS = [pytest.param(True, id="numba"), pytest.param(False, id="scipy")]


@pytest.mark.unit
@pytest.mark.parametrize("up, down", RESAMPLE_RATIO_CASES)
@pytest.mark.parametrize("complex_input", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("use_numba", RESAMPLER_BACKENDS)
@pytest.mark.parametrize("slack", [0, 37], ids=["exact_width", "wider_than_result"])
def test_fast_resample_poly_out_matches_allocated(monkeypatch, up, down, complex_input,
                                                  use_numba, slack):
    """`out=` reproduces the allocating result bit for bit and spares the tail."""
    monkeypatch.setattr(gammatone_module, "USE_NUMBA_RESAMPLER", use_numba)

    rng = np.random.default_rng(20260812)
    block = rng.standard_normal((5, 97))
    if complex_input:
        block = block + 1j * rng.standard_normal((5, 97))

    expected = fast_resample_poly(block, up, down, axis=-1)
    result_length = expected.shape[-1]
    assert result_length == resample_output_length(block.shape[-1], up, down)

    sentinel = -12345.5 + (6789.25j if complex_input else 0.0)
    destination = np.zeros((5, result_length + slack), dtype=expected.dtype)
    destination[:, result_length:] = sentinel

    returned = fast_resample_poly(block, up, down, axis=-1, out=destination)

    # Bitwise, not approximately: `out=` must not perturb a single rounding.
    assert returned.dtype == expected.dtype
    assert returned.shape == expected.shape
    assert returned.tobytes() == expected.tobytes()
    assert np.shares_memory(returned, destination)
    assert destination[:, :result_length].tobytes() == expected.tobytes()
    assert np.all(destination[:, result_length:] == sentinel)


@pytest.mark.unit
@pytest.mark.parametrize("use_numba", RESAMPLER_BACKENDS)
def test_fast_resample_poly_out_accepts_one_dimensional_input(monkeypatch, use_numba):
    """A 1D signal is as valid a destination shape as a 2D block."""
    monkeypatch.setattr(gammatone_module, "USE_NUMBA_RESAMPLER", use_numba)

    rng = np.random.default_rng(7)
    signal_input = rng.standard_normal(160)

    expected = fast_resample_poly(signal_input, 3, 1)
    destination = np.zeros(expected.size + 11)
    destination[expected.size:] = 99.0

    returned = fast_resample_poly(signal_input, 3, 1, out=destination)
    assert returned.tobytes() == expected.tobytes()
    assert np.all(destination[expected.size:] == 99.0)


@pytest.mark.unit
def test_resample_output_length_matches_actual_output():
    """The helper the synthesis bounds check relies on must not drift."""
    rng = np.random.default_rng(3)
    for in_len in (1, 2, 37, 97, 160):
        signal_input = rng.standard_normal(in_len)
        for up, down in RESAMPLE_RATIO_CASES + [(2, 3), (7, 1), (1, 7)]:
            predicted = resample_output_length(in_len, up, down)
            assert predicted == fast_resample_poly(signal_input, up, down).shape[-1], (in_len, up, down)


@pytest.mark.unit
def test_fast_resample_poly_out_rejects_short_destination():
    block = np.zeros((3, 40))
    too_short = np.zeros((3, resample_output_length(40, 4, 1) - 1))
    with pytest.raises(ValueError, match="at least"):
        fast_resample_poly(block, 4, 1, out=too_short)


@pytest.mark.unit
def test_fast_resample_poly_out_rejects_wrong_dtype():
    block = np.zeros((3, 40))
    with pytest.raises(ValueError, match="dtype"):
        fast_resample_poly(block, 4, 1, out=np.zeros((3, 400), dtype=complex))

    complex_block = np.zeros((3, 40), dtype=complex)
    with pytest.raises(ValueError, match="dtype"):
        fast_resample_poly(complex_block, 4, 1, out=np.zeros((3, 400), dtype=float))


@pytest.mark.unit
def test_fast_resample_poly_out_rejects_mismatched_leading_shape():
    block = np.zeros((3, 40))
    with pytest.raises(ValueError, match="shape"):
        fast_resample_poly(block, 4, 1, out=np.zeros((4, 400)))


@pytest.mark.unit
def test_fast_resample_poly_out_rejects_non_contiguous_destination():
    """A strided view would still be writable, but not with the layout the kernels
    are handed -- reject it loudly instead of silently dropping the write."""
    block = np.zeros((3, 40))
    strided = np.zeros((3, 800))[:, ::2]
    assert not strided.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        fast_resample_poly(block, 4, 1, out=strided)


@pytest.mark.unit
def test_fast_resample_poly_out_rejects_non_final_axis():
    """`out=` only knows how to reserve a tail on the last axis."""
    block = np.zeros((40, 3))
    with pytest.raises(ValueError, match="axis"):
        fast_resample_poly(block, 4, 1, axis=0, out=np.zeros((400, 3)))
