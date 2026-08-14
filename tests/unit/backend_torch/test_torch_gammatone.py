import math

import numpy as np
import pytest
import torch

from peass.backend_torch.gammatone import GammatoneAnalyzerTorch
from peass.backend_torch.gammatone import GammatoneSynthesizerTorch
from peass.backend_torch.gammatone import calculate_audiological_erb
from peass.backend_torch.gammatone import calculate_erb
from tests.conftest import to_numpy_format


@pytest.mark.parametrize("device_str", ["cpu", "cuda", "mps"])
def test_torch_gammatone_filterbank(device_str):
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if device_str == "mps":
        pytest.skip("MPS does not support float64; the PEASS torch backend is float64-only.")

    device = torch.device(device_str)

    # Use a predictable sine wave instead of random noise for strict reconstruction checks
    fs = 16000.0
    t = torch.linspace(0.0, 1.0, 16000, device=device, dtype=torch.float64)
    x = torch.sin(2.0 * math.pi * 440.0 * t)

    analyzer = GammatoneAnalyzerTorch(fs, 80.0, 1000.0, 6000.0, 1.0, device, torch.float64)
    subbands = analyzer.process(x)

    assert subbands.device == device
    assert subbands.shape[1] == 16000

    synth = GammatoneSynthesizerTorch(analyzer, 0.004)
    reconstructed = synth.process(subbands)

    assert reconstructed.device == device
    assert reconstructed.shape[0] == 16000

    x_np = to_numpy_format(x)
    recon_np = to_numpy_format(reconstructed)

    # Account for synthesizer delay offset (0.004 seconds * 16000 Hz = 64 samples)
    delay = int(round(0.004 * fs))
    orig_slice = x_np[delay: -delay]
    recon_slice = recon_np[2 * delay: len(orig_slice) + 2 * delay]

    # Lossy analysis -> synthesis round trip, so this bar is set by physics rather
    # than arithmetic noise. Measured on CPU 2026-08-15: correlation 0.9929697,
    # i.e. a deficit of 7.03e-03. The NumPy backend on the identical setup measures
    # 0.9929697 too -- the two reconstructions agree to 1.6e-13 -- so this is the
    # filterbank's own reconstruction floor, not a torch approximation.
    #
    # Tightened 0.90 -> 0.98 on CPU 2026-08-15; the old bar allowed a 1e-1 deficit
    # against a measured 7.0e-03 (14x) and let the filterbank degrade an order of
    # magnitude unnoticed. 0.98 leaves ~2.8x and matches the NumPy siblings in
    # tests/unit/backend_numpy/test_numpy_gammatone.py.
    #
    # CUDA has never been measured for this quantity and is not exercised in CI, so
    # it keeps the old documented-loose bound rather than silently inheriting the
    # CPU floor. Tighten it once there is a measurement to justify a number.
    minimum_correlation = 0.90 if device_str == "cuda" else 0.98

    corr = np.corrcoef(orig_slice, recon_slice)[0, 1]
    assert corr > minimum_correlation, (
        f"PyTorch reconstruction fidelity failed on {device_str}. Correlation is "
        f"{corr:.6f}, below the {minimum_correlation} floor"
    )


def gather_synthesis_reference(synth: GammatoneSynthesizerTorch, subbands: torch.Tensor,
                               alignment: torch.Tensor = None) -> torch.Tensor:
    """
    Independent oracle for `GammatoneSynthesizerTorch.process`, kept in the pre-fusion
    form: build the whole phase-aligned block, shift it with `gather` against an index
    tensor, mask the pre-onset samples with `where`, then contract with the mixer gains.

    The production version fuses all four steps into one per-band accumulate, which is
    ~2.7x faster but re-associates the arithmetic (`(x*mod)*phase` -> `x*(mod*phase)`,
    and a different summation order over the bands). Everything below asserts the two
    agree to a few ULP -- they compute the same quantity, so a real divergence here is a
    bug, not rounding.
    """
    aligned = (subbands * (synth.phase_factors.view(-1, 1) if alignment is None else alignment)).real
    time_steps = aligned.shape[-1]

    idx = torch.arange(time_steps, device=aligned.device).unsqueeze(0) - synth.delays.unsqueeze(1)
    valid = idx >= 0
    idx_clamped = torch.clamp(idx, min=0)

    shape_prefix = [1] * (aligned.dim() - 2)
    idx_clamped = idx_clamped.view(*shape_prefix, synth.delays.shape[0], time_steps).expand_as(aligned)
    valid = valid.view(*shape_prefix, synth.delays.shape[0], time_steps).expand_as(aligned)

    shifted = torch.gather(aligned, -1, idx_clamped)
    out = torch.where(valid, shifted, 0.0)
    return torch.einsum('b, ...bt -> ...t', synth.gains.to(out.dtype), out)


def _random_subbands(num_bands: int, shape_prefix: tuple, time_steps: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    size = shape_prefix + (num_bands, time_steps)
    return (torch.randn(*size, generator=generator, dtype=torch.float64)
            + 1j * torch.randn(*size, generator=generator, dtype=torch.float64))


@pytest.mark.unit
@pytest.mark.parametrize("shape_prefix", [(), (1,), (2,), (3, 2)])
def test_torch_synthesizer_matches_gather_formulation(shape_prefix):
    """
    The fused per-band delay-and-sum must reproduce the gather/where/einsum formulation
    it replaced, batched and unbatched. Stereo reaches this through the batched path.
    """
    analyzer = GammatoneAnalyzerTorch(16000.0, 80.0, 1000.0, 4000.0, 1.0, torch.device("cpu"), torch.float64)
    synth = GammatoneSynthesizerTorch(analyzer, 0.004)
    num_bands = len(analyzer.center_frequencies)

    subbands = _random_subbands(num_bands, shape_prefix, 2000, seed=4242)
    expected = gather_synthesis_reference(synth, subbands)
    actual = synth.process(subbands)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # Re-association only: a few ULP of the output's own scale.
    assert float((actual - expected).abs().max()) < 1e-13 * float(expected.abs().max())


@pytest.mark.unit
def test_torch_synthesizer_alignment_argument_matches_premultiplying():
    """
    Passing a fused `(Bands, Time)` alignment matrix must equal pre-multiplying the
    subbands by it and letting `process` apply the phase factors itself. This is exactly
    what the decomposition does with the cached modulation-times-phase matrix, so a
    broadcasting or indexing slip here would silently corrupt every reconstruction.
    """
    fs = 16000.0
    time_steps = 1500
    analyzer = GammatoneAnalyzerTorch(fs, 80.0, 1000.0, 4000.0, 1.0, torch.device("cpu"), torch.float64)
    synth = GammatoneSynthesizerTorch(analyzer, 0.004)
    num_bands = len(analyzer.center_frequencies)

    steps = torch.arange(time_steps, dtype=torch.float64)
    modulation = torch.exp(2j * math.pi / fs * analyzer.center_frequencies.unsqueeze(1) * steps)
    alignment = modulation * synth.phase_factors.unsqueeze(-1)

    for shape_prefix in [(), (2,)]:
        subbands = _random_subbands(num_bands, shape_prefix, time_steps, seed=99)
        expected = synth.process(subbands * modulation)
        actual = synth.process(subbands, alignment=alignment)
        assert float((actual - expected).abs().max()) < 1e-13 * float(expected.abs().max())


@pytest.mark.unit
def test_torch_synthesizer_zeroes_samples_before_each_band_onset():
    """
    `where(valid, shifted, 0)` used to zero every sample before a band's delay onset.
    The fused loop expresses that as the region of the output buffer the accumulate never
    touches, so it has to be checked directly: feeding a single band must leave exactly
    its first `delay` samples at zero, and the rest equal to the gain-scaled band.
    """
    analyzer = GammatoneAnalyzerTorch(16000.0, 80.0, 1000.0, 4000.0, 1.0, torch.device("cpu"), torch.float64)
    synth = GammatoneSynthesizerTorch(analyzer, 0.004)
    num_bands = len(analyzer.center_frequencies)
    time_steps = 400

    band = int(torch.argmax(synth.delays))
    delay = int(synth.delays[band])
    assert delay > 0, "test needs a band with a non-zero delay"

    subbands = torch.zeros(num_bands, time_steps, dtype=torch.complex128)
    subbands[band] = _random_subbands(1, (), time_steps, seed=7)[0]
    out = synth.process(subbands)

    assert torch.all(out[:delay] == 0.0)
    expected_tail = (synth.gains[band] * (subbands[band] * synth.phase_factors[band]).real)[:time_steps - delay]
    assert float((out[delay:] - expected_tail).abs().max()) < 1e-14 * float(expected_tail.abs().max())


@pytest.mark.unit
def test_torch_synthesizer_drops_bands_delayed_past_the_signal():
    """
    A signal shorter than a band's delay contributes nothing from that band. The gather
    formulation got this from an all-False `valid` mask; the fused loop gets it from
    skipping the band. Both must agree, and a signal shorter than every delay must give
    exact zeros.
    """
    analyzer = GammatoneAnalyzerTorch(16000.0, 80.0, 1000.0, 4000.0, 1.0, torch.device("cpu"), torch.float64)
    synth = GammatoneSynthesizerTorch(analyzer, 0.004)
    num_bands = len(analyzer.center_frequencies)
    longest = int(synth.delays.max())

    for time_steps in (1, longest // 2, longest, longest + 1):
        subbands = _random_subbands(num_bands, (), time_steps, seed=time_steps)
        expected = gather_synthesis_reference(synth, subbands)
        actual = synth.process(subbands)
        assert actual.shape == expected.shape
        scale = float(expected.abs().max())
        assert float((actual - expected).abs().max()) <= 1e-13 * max(scale, 1e-300)


@pytest.mark.unit
@pytest.mark.parametrize("shape_prefix", [(), (2,)])
def test_torch_synthesizer_gradients_match_gather_formulation(shape_prefix):
    """
    The synthesizer sits on the training path, and the fused version accumulates in place
    into a fresh buffer. In-place writes are the classic way to silently break autograd,
    so the gradient is compared against the out-of-place formulation directly.
    """
    analyzer = GammatoneAnalyzerTorch(16000.0, 80.0, 1000.0, 4000.0, 1.0, torch.device("cpu"), torch.float64)
    synth = GammatoneSynthesizerTorch(analyzer, 0.004)
    num_bands = len(analyzer.center_frequencies)

    subbands = _random_subbands(num_bands, shape_prefix, 800, seed=1234)
    weights = torch.randn(*(shape_prefix + (800,)), generator=torch.Generator().manual_seed(5), dtype=torch.float64)

    reference_input = subbands.clone().requires_grad_(True)
    (gather_synthesis_reference(synth, reference_input) * weights).sum().backward()

    fused_input = subbands.clone().requires_grad_(True)
    (synth.process(fused_input) * weights).sum().backward()

    assert fused_input.grad is not None
    assert torch.isfinite(fused_input.grad).all()
    assert float((fused_input.grad - reference_input.grad).abs().max()) < 1e-13 * float(
        reference_input.grad.abs().max())


@pytest.mark.unit
def test_torch_synthesizer_passes_gradcheck():
    """Numerical gradient check of the fused accumulate on a small configuration."""
    analyzer = GammatoneAnalyzerTorch(8000.0, 500.0, 1000.0, 2000.0, 1.0, torch.device("cpu"), torch.float64)
    synth = GammatoneSynthesizerTorch(analyzer, 0.002)
    num_bands = len(analyzer.center_frequencies)

    subbands = _random_subbands(num_bands, (), 40, seed=31).requires_grad_(True)
    assert torch.autograd.gradcheck(synth.process, (subbands,), eps=1e-6, atol=1e-9)


def _recover_audiological_bandwidths(analyzer: GammatoneAnalyzerTorch) -> np.ndarray:
    """
    Inverts the Hohmann 2002 eq. (14) chain to recover the audiological ERBs that the
    analyzer's filter construction actually used, straight from the constructed poles.

    lambda = |coefficient| = exp(-2*pi*b/fs)  ->  b = -ln(lambda)*fs/(2*pi)
    erb = b * a_gamma, with a_gamma the 4th-order gamma constant.
    """
    gamma_const = (math.pi * math.factorial(6) * (2.0 ** -6) / (math.factorial(3) ** 2))
    lambda_decay = to_numpy_format(torch.abs(analyzer.coefs))
    decay_const = -np.log(lambda_decay) * analyzer.fs / (2.0 * math.pi)
    return decay_const * gamma_const


@pytest.mark.unit
@pytest.mark.parametrize("center_frequency_hz", [20.0, 100.0, 440.0, 1000.0, 5000.0, 12000.0])
def test_torch_erb_helper_pins_erbbw_formula(center_frequency_hz):
    """
    Pins `calculate_erb` to the Glasberg & Moore form used by the MATLAB reference's
    `erbBW.m`: 24.7 * (0.00437 * fc + 1).

    This helper feeds the per-band decimation factor (`myPemoAnalysisFilterBank.m:53`)
    and NOTHING else. It must not be collapsed onto the gammatone filter constructor's
    (GFB_L + fc/GFB_Q) form -- see `test_torch_gammatone_bandwidth_pins_hohmann_formula`.
    The two formulas are the same empirical fit with one constant rounded, so a refactor
    that unifies them looks harmless and is not: MATLAB deliberately uses each in a
    different place, and parity requires reproducing that split.
    """
    fc = torch.tensor([center_frequency_hz], dtype=torch.float64)
    expected = 24.7 * (0.00437 * center_frequency_hz + 1.0)
    assert float(calculate_erb(fc)[0]) == pytest.approx(expected, rel=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize("center_frequency_hz", [20.0, 100.0, 440.0, 1000.0, 5000.0, 12000.0])
def test_torch_gammatone_bandwidth_pins_hohmann_formula(center_frequency_hz):
    """
    Pins the gammatone filter construction's bandwidth to `Gfb_Filter_new.m:61`:
    audiological_erb = (GFB_L + fc / GFB_Q) * bandwidth_factor, with GFB_L = 24.7 and
    GFB_Q = 9.265 (Hohmann 2002 eq. 17, `Gfb_set_constants.m`).

    The coefficients must NOT be built from the `erbBW.m` form 24.7*(0.00437*fc + 1),
    which is the same fit with 1/(24.7*0.00437) = 9.264488... rounded to 9.265 and which
    belongs at the decimation call site instead. The gap is only ~5.5e-5 relative on the
    slope term, so this test asserts tightly enough to catch a silent swap back.
    """
    fc = torch.tensor([center_frequency_hz], dtype=torch.float64)
    expected = 24.7 + center_frequency_hz / 9.265
    assert float(calculate_audiological_erb(fc)[0]) == pytest.approx(expected, rel=1e-15)


@pytest.mark.unit
def test_torch_analyzer_splits_the_two_erb_formulas():
    """
    The analyzer must keep both formulas alive at once: erbBW for `bandwidths` (which
    drives `decimations`), Hohmann for the poles in `coefs`. Recovering the bandwidth
    back out of the constructed poles is what makes a silent collapse onto one formula
    fail here.
    """
    analyzer = GammatoneAnalyzerTorch(16000.0, 100.0, 1000.0, 4000.0, 1.0, torch.device("cpu"), torch.float64)
    center_frequencies = to_numpy_format(analyzer.center_frequencies)

    # Decimation bandwidths stay on the erbBW form.
    np.testing.assert_allclose(
        to_numpy_format(analyzer.bandwidths),
        24.7 * (0.00437 * center_frequencies + 1.0),
        rtol=1e-15
    )

    # The filter poles come from the Hohmann form.
    np.testing.assert_allclose(
        _recover_audiological_bandwidths(analyzer),
        24.7 + center_frequencies / 9.265,
        rtol=1e-9
    )


@pytest.mark.unit
def test_torch_two_erb_formulas_stay_distinct():
    """
    Guards the intent directly: the constructor's ERB and the erbBW helper are close but
    must remain two separate expressions. If a refactor unifies them this test fails,
    which is the point -- the MATLAB reference uses both, in different places.
    """
    fc = torch.tensor([10000.0], dtype=torch.float64)
    hohmann = float(calculate_audiological_erb(fc)[0])
    glasberg_moore = float(calculate_erb(fc)[0])

    assert hohmann != glasberg_moore
    # Same fit, one rounded constant: agreement to ~5e-5 relative, but no closer.
    assert hohmann == pytest.approx(glasberg_moore, rel=1e-4)
    assert hohmann != pytest.approx(glasberg_moore, rel=1e-6)


@pytest.mark.unit
def test_torch_and_numpy_gammatone_bandwidths_agree():
    """
    Both backends must build their poles from the same Hohmann formula; drift between
    them is exactly the class of bug this pins down.
    """
    from peass.backend_numpy.gammatone import calculate_audiological_equivalent_rectangular_bandwidth

    center_frequencies = [20.0, 100.0, 440.0, 1000.0, 5000.0, 12000.0]
    torch_values = to_numpy_format(calculate_audiological_erb(torch.tensor(center_frequencies, dtype=torch.float64)))
    numpy_values = np.array([calculate_audiological_equivalent_rectangular_bandwidth(f) for f in center_frequencies])

    np.testing.assert_allclose(torch_values, numpy_values, rtol=1e-15)
