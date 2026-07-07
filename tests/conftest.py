"""
PEASS Test Suite - Pytest Configuration and Fixtures
File path: tests/conftest.py
"""

import pathlib
import tempfile
from typing import Generator

import numpy as np
import pytest
import scipy.signal as signal
import soundfile as sf

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# PyTorch is an optional extra. When it is not installed, skip collection of the
# torch-only test modules entirely so a bare `pytest` run does not abort with
# import errors (paths are relative to this conftest's directory).
if not _HAS_TORCH:
    collect_ignore_glob = [
        "unit/backend_torch/*.py",
        "regression/test_differential_numpy_vs_torch.py",
    ]


@pytest.fixture(scope="session")
def project_root():
    return pathlib.Path(__file__).parent.parent


@pytest.fixture(scope="session")
def resources_dir(project_root):
    return project_root / "tests" / "resources"


@pytest.fixture(scope="session")
def db_resources(resources_dir):
    """Path to the consolidated database audio clips."""
    return resources_dir / "database"


@pytest.fixture(scope="session")
def matlab_ref_resources(resources_dir):
    """Path to the MATLAB v2.0.1 gold standard reference files."""
    return resources_dir / "matlab_reference"


@pytest.fixture(scope="session")
def training_dataset_dir(project_root):
    """Path to the full training dataset."""
    return project_root / "dataset"


@pytest.fixture(scope="module", params=[(16000.0, 2.0), (32000.0, 1.0)])
def synthetic_audio_data(request) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generates synthetic target, interferer, and estimate signals for DSP testing.
    Parameterized over (sampling_frequency, duration_seconds) to automatically
    test all downstream consumer tests at different sample rates and lengths.
    """
    sampling_frequency, duration_seconds = request.param
    num_samples = int(duration_seconds * sampling_frequency)
    time_steps = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

    # 1. Generate clean target (440 Hz)
    target_waveform = np.sin(2.0 * np.pi * 440.0 * time_steps)[:, np.newaxis]

    # 2. Generate interferer (1200 Hz)
    interferer_waveform = np.sin(2.0 * np.pi * 1200.0 * time_steps)[:, np.newaxis]

    # 3. Simulate Target/Spatial Distortion by lowpassing the target
    butterworth_b, butterworth_a = signal.butter(4, 800.0 / (sampling_frequency / 2.0), btype='low')
    target_distorted = signal.lfilter(butterworth_b, butterworth_a, target_waveform, axis=0)

    # 4. Generate Artifacts (low level white Gaussian noise)
    artifact_noise = 0.01 * np.random.default_rng(seed=1234).standard_normal((num_samples, 1))

    # 5. Compile simulated Separation Estimate
    estimate_waveform = target_distorted + 0.15 * interferer_waveform + artifact_noise

    return target_waveform, interferer_waveform, estimate_waveform, sampling_frequency


@pytest.fixture(scope="module")
def audio_files_fixture(
        synthetic_audio_data: tuple[np.ndarray, np.ndarray, np.ndarray, float]
) -> Generator[tuple[pathlib.Path, pathlib.Path, pathlib.Path], None, None]:
    """
    Writes synthetic waveforms to temporary WAV files on disk to test file-based modes.
    """
    target, interferer, estimate, fs = synthetic_audio_data

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        target_path = temp_path / "target_source.wav"
        interferer_path = temp_path / "interferer.wav"
        estimate_path = temp_path / "estimate.wav"

        sf.write(target_path, target, int(fs))
        sf.write(interferer_path, interferer, int(fs))
        sf.write(estimate_path, estimate, int(fs))

        yield target_path, interferer_path, estimate_path


@pytest.fixture(params=["sine", "square", "triangle", "chirp", "noise"])
def synthetic_signal(request) -> tuple[np.ndarray, float]:
    """
    Generates parameterized waveforms to verify mathematical robustness across
    periodic, discontinuous, and stochastic signal profiles.
    """
    signal_type = request.param
    sampling_frequency = 16000.0
    duration = 1.5
    num_samples = int(duration * sampling_frequency)
    time_steps = np.linspace(0.0, duration, num_samples, endpoint=False)

    rng = np.random.default_rng(seed=42)

    if signal_type == "sine":
        data = np.sin(2.0 * np.pi * 440.0 * time_steps)
    elif signal_type == "square":
        data = signal.square(2.0 * np.pi * 200.0 * time_steps)
    elif signal_type == "triangle":
        data = signal.sawtooth(2.0 * np.pi * 200.0 * time_steps, width=0.5)
    elif signal_type == "chirp":
        data = signal.chirp(time_steps, f0=50.0, t1=duration, f1=8000.0, method="linear")
    elif signal_type == "noise":
        data = rng.normal(0.0, 0.2, num_samples)
    else:
        raise ValueError("Invalid signal type requested.")

    # Normalize to prevent digital clipping
    data = data / (np.max(np.abs(data)) + 1e-9) * 0.9
    return data[:, np.newaxis], sampling_frequency


@pytest.fixture(params=[
    ("exp01_target.wav", "exp01_InterfSrc1.wav", "mono_16k"),  # Mono, 16kHz
    ("exp03_target.wav", "exp03_InterfSrc1.wav", "stereo_16k"),  # Stereo, 16kHz
])
def database_audio_pair(request, db_resources) -> tuple[np.ndarray, np.ndarray, float, str]:
    """
    Yields real audio pairs from the database covering different channel layouts.
    Returns: (target, interferer, sampling_rate, identifier_string)
    """
    target_filename, interferer_filename, identifier = request.param
    target_path = db_resources / target_filename
    interferer_path = db_resources / interferer_filename

    target, fs = sf.read(target_path)
    interferer, _ = sf.read(interferer_path)

    return target, interferer, float(fs), identifier


@pytest.fixture(params=["numpy", "torch_cpu", "torch_gpu"])
def peass_backend(request):
    """
    Parametrized fixture yielding (backend_type, device) configurations.
    Automatically skips unavailable hardware environments (CUDA / MPS).
    """
    mode = request.param
    if mode.startswith("torch"):
        if not _HAS_TORCH:
            pytest.skip("PyTorch is not installed in this environment.")

        if mode == "torch_cpu":
            return "torch", torch.device("cpu")
        elif mode == "torch_gpu":
            if torch.cuda.is_available():
                return "torch", torch.device("cuda")
            elif torch.backends.mps.is_available():
                return "torch", torch.device("mps")
            else:
                pytest.skip("GPU hardware (CUDA or MPS) is not available.")

    return "numpy", None


def to_backend_format(data, backend_type: str, device=None):
    """Converts signals and arrays to the target backend type and device."""
    if backend_type == "torch":
        import torch
        if isinstance(data, list):
            return [to_backend_format(x, "torch", device) for x in data]
        if isinstance(data, np.ndarray):
            # Bypass torch.from_numpy C-binding issues in specific environments
            return torch.tensor(data, device=device, dtype=torch.float64)
    return data


def to_numpy_format(data):
    """Casts outputs back to standard NumPy arrays for validation assertions."""
    if _HAS_TORCH and isinstance(data, torch.Tensor):
        # Convert via nested list to completely bypass local C-binding limitations
        return np.array(data.detach().cpu().tolist())
    if hasattr(data, "__dict__"):  # Handle dataclass structures
        return data
    return data
