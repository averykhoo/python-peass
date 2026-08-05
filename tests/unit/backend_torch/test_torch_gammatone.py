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

    # Convert to numpy and check that reconstruction is highly correlated (>0.90)
    x_np = to_numpy_format(x)
    recon_np = to_numpy_format(reconstructed)

    # Account for synthesizer delay offset (0.004 seconds * 16000 Hz = 64 samples)
    delay = int(round(0.004 * fs))
    orig_slice = x_np[delay: -delay]
    recon_slice = recon_np[2 * delay: len(orig_slice) + 2 * delay]

    corr = np.corrcoef(orig_slice, recon_slice)[0, 1]
    assert corr > 0.90, f"PyTorch reconstruction fidelity failed. Correlation is {corr:.4f}"


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
