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
    subbands by it and letting `process` apply the phase factors itself.

    The decomposition no longer drives this: P5 folded the synthesis modulation into the
    interpolation taps, so `run_auditory_synthesis_filterbank_torch` takes the
    `alignment=None` default and the fused modulation-times-phase matrix this test was
    written against is gone. The hook and the test stay because the `(Bands, Time)`
    pre-multiply is a general argument on a public-ish method, and because the equivalence
    it pins -- that `alignment` multiplies the *subband* time index, the same index the
    interpolator produces -- is precisely the property that lets the fold be index-aligned
    with no shift. A broadcasting or indexing slip here would invalidate that reasoning.
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


# ---------------------------------------------------------------------------------------
# The half-spectrum (`process_real`) filter identity. One shared comparison, exercised
# once positively against `peass` and once per trap against a deliberately wrong build.
# ---------------------------------------------------------------------------------------

# Of the filter's own peak. A gross-error net, NOT the thing that separates the two
# known traps from the correct construction -- see the positive test's docstring.
_HALF_SPECTRUM_REFLECTION_BAR = 1e-13

_HALF_SPECTRUM_SIZES = (4096, 8192, 32768)


def _half_spectrum_filter_args():
    """The one filterbank config these tests use, shaped as `_get_gammatone_H_*` takes it."""
    fs = 24000.0
    analyzer = GammatoneAnalyzerTorch(fs, 235.0, 1000.0, 11000.0, 1.0, torch.device("cpu"), torch.float64)
    return (
        fs,
        tuple(analyzer.center_frequencies.tolist()),
        tuple(analyzer.norms.tolist()),
        tuple(analyzer.coefs.tolist()),
        "cpu",
    )


def _rebuild_gammatone_H(n_fft, args, *, half, grid_dtype=torch.float64,
                         nyquist_convention="fftfreq", nyquist_fixup=True):
    """`_get_gammatone_H_torch` / `_get_gammatone_H_real_torch`, re-derived here.

    Spelled out rather than imported so that the negative tests below can build the
    *wrong* variants without mutating anything under `peass/`. Every knob corresponds to
    one implementation decision the real code makes and must keep making:

    * `grid_dtype`   -- the explicit `dtype=torch.float64` on the `fftfreq` call.
    * `nyquist_convention` -- `fftfreq`'s first half (-0.5 at Nyquist) vs `rfftfreq` (+0.5).
    * `nyquist_fixup` -- the `combined[:, -1] = forward[:, -1].real` self-reflection line.

    With the defaults this reproduces `_get_gammatone_H_real_torch` bit-for-bit; the
    positive test asserts exactly that, so the knobs cannot silently rot.
    """
    _, _, norms_tuple, coefs_tuple, _ = args
    coefs = torch.tensor(coefs_tuple, dtype=torch.complex128).view(-1, 1)
    norms = torch.tensor(norms_tuple, dtype=torch.float64).view(-1, 1)

    if not half:
        freqs_norm = torch.fft.fftfreq(n_fft, dtype=grid_dtype)
    elif nyquist_convention == "rfftfreq":
        freqs_norm = torch.fft.rfftfreq(n_fft, dtype=grid_dtype)
    else:
        freqs_norm = torch.fft.fftfreq(n_fft, dtype=grid_dtype)[:n_fft // 2 + 1]

    z_inv = torch.exp(-2j * math.pi * freqs_norm)
    denom = 1.0 - coefs * z_inv.unsqueeze(0)
    denom_sq = denom * denom
    forward = norms / (denom_sq * denom_sq)
    if not half:
        return forward

    denom_r = 1.0 - coefs * torch.conj(z_inv).unsqueeze(0)
    denom_r_sq = denom_r * denom_r
    reflected = torch.conj(norms / (denom_r_sq * denom_r_sq))

    combined = (forward + reflected) * 0.5
    if nyquist_fixup and n_fft % 2 == 0:
        combined[:, -1] = forward[:, -1].real
    return combined


def _assert_half_spectrum_is_the_reflection(half_H, full_H, n_fft):
    """The comparison. Three separable criteria, only one of which is a tolerance.

    Factored out so the negative tests can assert it *rejects* a wrong build, rather
    than each of them re-stating a bar that would then drift out of step with this one.
    Each `assert` carries a distinguishable message so `pytest.raises(match=...)` can
    pin down which criterion did the rejecting.
    """
    assert half_H.shape == (full_H.shape[0], n_fft // 2 + 1)

    k = torch.arange(n_fft // 2 + 1)
    expected = (full_H[:, k] + torch.conj(full_H[:, (-k) % n_fft])) * 0.5

    peak = half_H.abs().max().item()
    deviation = (half_H - expected).abs().max().item()
    assert deviation < _HALF_SPECTRUM_REFLECTION_BAR * peak, (
        f"n_fft={n_fft}: half-grid filter deviates {deviation:.3e} from the "
        f"reflection of the full-grid one, {deviation / peak:.3e} of peak "
        f"({deviation / (peak * torch.finfo(torch.float64).eps):.1f} ULP)."
    )

    # These two are structural rather than numerical, and stay exact on every platform:
    # a Hermitian spectrum needs both self-reflecting bins purely real. DC cancels
    # algebraically (z_inv[0] is exactly 1+0j, so the reflected term is the conjugate of
    # the forward one) and Nyquist is assigned `forward[:, -1].real` outright.
    assert torch.equal(half_H[:, 0].imag, torch.zeros_like(half_H[:, 0].imag)), (
        f"n_fft={n_fft}: DC bin must be purely real, got max |imag| "
        f"{half_H[:, 0].imag.abs().max().item():.3e}"
    )
    assert torch.equal(half_H[:, -1].imag, torch.zeros_like(half_H[:, -1].imag)), (
        f"n_fft={n_fft}: Nyquist bin must be purely real, got max |imag| "
        f"{half_H[:, -1].imag.abs().max().item():.3e}"
    )


@pytest.mark.unit
def test_torch_gammatone_half_spectrum_filter_is_the_exact_reflection():
    """
    `_get_gammatone_H_real_torch` must be `(H[k] + conj(H[-k])) / 2` taken from the
    full-grid filter -- that identity is the whole basis of `process_real`.

    The tolerance is a few hundred ULP of the filter's own peak rather than bit-identity,
    and the distinction is deliberate. This assertion was `torch.equal` when it landed --
    exact on Windows across 32 config x size combinations -- and it failed on
    ubuntu-latest at `n_fft=4096` the first time CI saw it. The two constructions call
    `torch.exp` on grids of different length (the full `N_fft` vs this half grid), and a
    complex `exp` is free to vectorize differently per length and per platform libm; the
    last-ULP difference then runs through a 4th power with a resonant denominator. Same
    family as the batched-FFT non-invariance recorded in ARCHIVE.md, and the second time
    this project has written a Windows-only exactness down as a universal invariant. Do
    not restore `torch.equal` against the full-grid reflection.

    WHY THE TRAP COVERAGE LIVES IN THE THREE TESTS BELOW AND NOT IN THAT TOLERANCE.
    This test used to claim the two known ways of getting the identity wrong -- the
    `rfftfreq` (+0.5) Nyquist convention, and omitting the `combined[:, -1]` fixup --
    "fail far louder than 1e-13". That was true only while the frequency grid inherited
    torch's default float32. Since the grid carries an explicit `dtype=torch.float64`,
    measured on this config at n_fft 4096/8192/32768 (identical at all three):

        variant                  float32 grid    float64 grid    bar = 1.000e-13 * peak
        correct                     0.000e+00       0.000e+00     passes
        rfftfreq convention         2.446e-07       1.110e-16     was 2.4e6x over, now 900x under
        no Nyquist fixup            1.486e-07       1.304e-16     was 1.5e6x over, now 770x under

    So the accuracy fix pushed BOTH traps ~800x *under* the bar: the correct path became
    ~9 orders more accurate and the tolerance stopped separating right from wrong. Do not
    respond to that by re-tuning the number -- a scalar bar blinded once by an accuracy
    improvement will be blinded again by the next one, and here it cannot work in
    principle, because on this platform the correct path deviates by exactly 0 while a
    *wrong* one deviates by 1.1e-16, which is below the cross-platform libm noise the
    tolerance has to leave room for.

    What replaces it, per trap:

    * Omitted Nyquist fixup is caught *exactly*, by the Nyquist-imag criterion in
      `_assert_half_spectrum_is_the_reflection`, on every platform and at every
      precision -- `exp(+i*pi)` is `-1 + 1.2246e-16j`, never `-1 + 0j`, so
      `(H + conj(H)) / 2` computed rather than assigned leaves a residue (measured
      1.180e-16) where the contract demands a hard zero. See
      `test_torch_gammatone_half_spectrum_check_rejects_a_missing_nyquist_fixup`.
    * The `rfftfreq` convention cannot be caught by any value-level bar at float64,
      and that is a fact about the mathematics, not a weak test: under the fixup the
      Nyquist bin is `Re(H(f_nyq))`, and `exp(-2j*pi*(-0.5))` and `exp(-2j*pi*(+0.5))`
      are both exactly -1, so the two conventions are the *same number* in exact
      arithmetic and differ only by float64 rounding asymmetry (1.110e-16). It is
      pinned two other ways instead: by grid provenance, in
      `test_torch_gammatone_half_spectrum_filter_is_built_on_the_float64_fftfreq_grid`,
      and by showing the tolerance does reject it once the grid loses precision, in
      `test_torch_gammatone_half_spectrum_check_rejects_the_rfftfreq_convention_on_a_float32_grid`
      -- which is also what keeps the `dtype=torch.float64` honest, since that argument
      is the only reason the convention is currently harmless.

    The tolerance is kept anyway, and still does real work: it is the only check here
    that looks at all `N_fft/2 + 1` bins rather than just the two self-reflecting ones.
    Dropping the `dtype=torch.float64` from this function's `fftfreq` call was measured
    at 7.496e-06 of peak, which it catches by 7.5e7x.
    """
    from peass.backend_torch.gammatone import _get_gammatone_H_real_torch
    from peass.backend_torch.gammatone import _get_gammatone_H_torch

    args = _half_spectrum_filter_args()

    for n_fft in _HALF_SPECTRUM_SIZES:
        full = _get_gammatone_H_torch(n_fft, *args)
        half = _get_gammatone_H_real_torch(n_fft, *args)

        _assert_half_spectrum_is_the_reflection(half, full, n_fft)

        # `_rebuild_gammatone_H` is the oracle the negative tests below perturb, so it
        # has to be the same construction. Same grid length and same op sequence as the
        # implementation, hence bit-exact on every platform -- unlike the full-vs-half
        # comparison above, which is not.
        assert torch.equal(half, _rebuild_gammatone_H(n_fft, args, half=True))
        assert torch.equal(full, _rebuild_gammatone_H(n_fft, args, half=False))


@pytest.mark.unit
def test_torch_gammatone_half_spectrum_check_rejects_a_missing_nyquist_fixup():
    """
    Negative counterpart: `_assert_half_spectrum_is_the_reflection` must REJECT a build
    that drops `combined[:, -1] = forward[:, -1].real`.

    The Nyquist bin reflects onto itself, so `conj(H[-N/2])` is `conj(H[N/2])` and the
    conjugate-grid formula does not apply there; the fixup is what makes that bin
    `Re(H)`. Computing `(forward + reflected) / 2` there instead is wrong by only
    1.304e-16 of peak at float64 -- 770x under the tolerance, which is precisely why
    this test asserts on the exact-zero imaginary part rather than on a magnitude. The
    separation is not a ratio: the correct build's Nyquist imaginary part is a hard 0.0
    (a real tensor assigned into a complex slice), and this one's is 1.180e-16.
    """
    args = _half_spectrum_filter_args()

    for n_fft in _HALF_SPECTRUM_SIZES:
        full = _rebuild_gammatone_H(n_fft, args, half=False)
        no_fixup = _rebuild_gammatone_H(n_fft, args, half=True, nyquist_fixup=False)

        # The tolerance is blind to this (1.304e-16 of peak against a 1.000e-13 bar);
        # that is stated in the positive test's docstring rather than asserted here,
        # because "the bar must not catch it" is not a property worth pinning. What is
        # worth pinning is that it gets caught anyway, exactly, in the bin it corrupts.
        residue = no_fixup[:, -1].imag.abs().max().item()
        assert residue > 0.0, (
            "the no-fixup build must leave a nonzero Nyquist imaginary residue, or this "
            "negative test proves nothing"
        )
        with pytest.raises(AssertionError, match="Nyquist bin must be purely real"):
            _assert_half_spectrum_is_the_reflection(no_fixup, full, n_fft)

        # Nothing else moves: the fixup touches exactly one bin.
        correct = _rebuild_gammatone_H(n_fft, args, half=True)
        assert torch.equal(no_fixup[:, :-1], correct[:, :-1])


@pytest.mark.unit
def test_torch_gammatone_half_spectrum_check_rejects_the_rfftfreq_convention_on_a_float32_grid():
    """
    Negative counterpart for the `rfftfreq` trap, and the test that keeps the explicit
    `dtype=torch.float64` on the `fftfreq` call load-bearing.

    `fftfreq(n)[n // 2]` is -0.5 and `rfftfreq(n)[-1]` is +0.5 -- exactly, at any dtype.
    Mathematically `exp(-2j*pi*f)` is -1 for both, so at float64 the two conventions
    agree to 1.110e-16 and no tolerance can separate them (see the positive test). At
    float32 `exp(-+i*pi)` carries an ~8.742e-08 imaginary residue instead of ~1.225e-16,
    the two signs differ by 1.749e-07 on `z_inv`, and a 4th power with a resonant
    denominator turns that into 2.446e-07 of peak -- 2.4e6x over the tolerance.

    So this test asserts the pair: on a single-precision grid the correct convention
    still passes and the `rfftfreq` one is rejected. If the `dtype=torch.float64` ever
    goes missing from `peass`, the positive test's tolerance starts doing real work
    again; while it is there, grid provenance is what guards the convention.
    """
    args = _half_spectrum_filter_args()

    for n_fft in _HALF_SPECTRUM_SIZES:
        assert torch.fft.fftfreq(n_fft, dtype=torch.float64)[n_fft // 2].item() == -0.5
        assert torch.fft.rfftfreq(n_fft, dtype=torch.float64)[-1].item() == +0.5

        full32 = _rebuild_gammatone_H(n_fft, args, half=False, grid_dtype=torch.float32)
        correct32 = _rebuild_gammatone_H(n_fft, args, half=True, grid_dtype=torch.float32)
        rfft32 = _rebuild_gammatone_H(n_fft, args, half=True, grid_dtype=torch.float32,
                                      nyquist_convention="rfftfreq")

        # Isolates the convention, not the precision: float32 alone passes every
        # criterion (measured deviation 0.000e+00 -- the grid *values* are exact in
        # both dtypes, only `exp` loses precision, and both sides lose it equally).
        _assert_half_spectrum_is_the_reflection(correct32, full32, n_fft)

        peak = rfft32.abs().max().item()
        deviation = (rfft32 - correct32).abs().max().item()
        assert deviation > 1000.0 * _HALF_SPECTRUM_REFLECTION_BAR * peak, (
            f"n_fft={n_fft}: the rfftfreq trap must be loud on a float32 grid "
            f"(measured 2.446e-07 of peak), got {deviation:.3e}"
        )
        with pytest.raises(AssertionError, match="deviates"):
            _assert_half_spectrum_is_the_reflection(rfft32, full32, n_fft)


@pytest.mark.unit
def test_torch_gammatone_half_spectrum_filter_is_built_on_the_float64_fftfreq_grid(monkeypatch):
    """
    The only check that catches the `rfftfreq` trap at the precision `peass` actually
    runs at, and it does so structurally: `_get_gammatone_H_real_torch` must take
    `fftfreq`'s first half, in float64, and must never reach for `rfftfreq`.

    Value-level detection is impossible here -- under the Nyquist fixup the two
    conventions are the same number in exact arithmetic and differ by 1.110e-16 of
    peak in float64, below the cross-platform noise the tolerance must tolerate. Grid
    provenance has no such problem: -0.5 vs +0.5 is exact everywhere, so a spy on the
    two constructors separates right from wrong deterministically and cannot be blinded
    by any future accuracy improvement. That is the whole reason this test exists in
    this form.
    """
    from peass.backend_torch.gammatone import _get_gammatone_H_real_torch

    args = _half_spectrum_filter_args()  # built before the patch; it uses no FFT grids
    seen_dtypes = []
    real_fftfreq = torch.fft.fftfreq

    def spy_fftfreq(n, *rest, **kwargs):
        seen_dtypes.append(kwargs.get("dtype"))
        return real_fftfreq(n, *rest, **kwargs)

    def forbidden_rfftfreq(*_args, **_kwargs):
        raise AssertionError(
            "_get_gammatone_H_real_torch must take fftfreq's first half, not rfftfreq: "
            "the two disagree at the Nyquist bin (-0.5 vs +0.5), and that bin is the "
            "one the self-reflection fixup writes."
        )

    monkeypatch.setattr(torch.fft, "fftfreq", spy_fftfreq)
    monkeypatch.setattr(torch.fft, "rfftfreq", forbidden_rfftfreq)

    # `lru_cache`d, so a warm entry would hide the call entirely. Cleared afterwards too,
    # so no value computed under the patch outlives this test.
    _get_gammatone_H_real_torch.cache_clear()
    try:
        for n_fft in _HALF_SPECTRUM_SIZES:
            seen_dtypes.clear()
            _get_gammatone_H_real_torch.cache_clear()
            _get_gammatone_H_real_torch(n_fft, *args)
            assert seen_dtypes == [torch.float64], (
                f"n_fft={n_fft}: expected exactly one fftfreq call at dtype=torch.float64 "
                f"(the explicit dtype is what keeps `exp(-+i*pi)`'s imaginary residue at "
                f"~1.225e-16 rather than ~8.742e-08), saw {seen_dtypes}"
            )
    finally:
        _get_gammatone_H_real_torch.cache_clear()


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(4000,), (2, 4000), (2, 3, 1500)])
def test_torch_gammatone_process_real_matches_the_complex_path(shape):
    """
    `process_real` computes the same quantity as `process(x).real`, via a half-length
    inverse transform. Not bit-identical -- rfft/irfft sum the same terms in a
    different order -- so the bar is a couple of ULP of the signal's own peak.
    """
    fs = 16000.0
    analyzer = GammatoneAnalyzerTorch(fs, 235.0, 1000.0, 7000.0, 1.0, torch.device("cpu"), torch.float64)
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)

    expected = analyzer.process(x).real
    actual = analyzer.process_real(x)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float64
    # A contiguous real block, which is the point of the path. Only the new value's own
    # layout is asserted: whether `.real` on a complex tensor is a strided view is
    # torch's business, not this function's contract.
    assert actual.is_contiguous()

    peak = expected.abs().max().item()
    assert (actual - expected).abs().max().item() < 8.0 * np.finfo(np.float64).eps * peak


@pytest.mark.unit
def test_torch_gammatone_process_real_rejects_complex_input():
    """The identity only holds for real input; complex signals must keep `process`."""
    analyzer = GammatoneAnalyzerTorch(16000.0, 235.0, 1000.0, 7000.0, 1.0, torch.device("cpu"), torch.float64)

    with pytest.raises(ValueError, match="real input"):
        analyzer.process_real(torch.randn(2, 512, dtype=torch.complex128))


@pytest.mark.unit
def test_torch_gammatone_process_real_is_differentiable():
    """
    The metrics are trained through this path, so the half-spectrum route has to carry
    the same gradient the complex one did.
    """
    analyzer = GammatoneAnalyzerTorch(16000.0, 235.0, 1000.0, 7000.0, 1.0, torch.device("cpu"), torch.float64)
    generator = torch.Generator().manual_seed(5)
    num_bands = len(analyzer.center_frequencies)
    weights = torch.randn(1, num_bands, 800, generator=generator, dtype=torch.float64)

    x = torch.randn(1, 800, generator=generator, dtype=torch.float64, requires_grad=True)

    (analyzer.process_real(x) * weights).sum().backward()
    actual = x.grad.clone()

    x.grad = None
    (analyzer.process(x).real * weights).sum().backward()
    expected = x.grad.clone()

    assert actual.shape == expected.shape
    np.testing.assert_allclose(to_numpy_format(actual), to_numpy_format(expected), rtol=0.0,
                               atol=8.0 * np.finfo(np.float64).eps * expected.abs().max().item())
