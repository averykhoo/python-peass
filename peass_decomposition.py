"""
PEASS Decomposition Step - Complete Python Port

Command-line script demonstrating execution of the fully optimized PEASS package.
"""

import numpy as np

# Direct domain-import from the internal module
from peass import (
    DecompositionConfiguration,
    decompose_distortion_components,
    calculate_bss_eval_energy_ratios
)

if __name__ == '__main__':
    sampling_frequency_hz = 16000.0
    duration_seconds = 2.0
    time_steps = np.linspace(0, duration_seconds, int(duration_seconds * sampling_frequency_hz), endpoint=False)

    speech_waveform = np.sin(2 * np.pi * 400 * time_steps)
    noise_waveform = 0.5 * np.random.randn(len(time_steps))

    estimate_waveform = 0.8 * speech_waveform + 0.05 * noise_waveform + 0.01 * np.random.randn(len(time_steps))

    speech_waveform = speech_waveform[:, np.newaxis]
    noise_waveform = noise_waveform[:, np.newaxis]
    estimate_waveform = estimate_waveform[:, np.newaxis]

    print("Beginning JIT-optimized PEASS error decomposition...")

    configuration = DecompositionConfiguration()
    decomposition_result = decompose_distortion_components(
        source_files=[speech_waveform, noise_waveform],
        estimate_file=estimate_waveform,
        configuration=configuration,
        sampling_frequency_hz=sampling_frequency_hz
    )
    waveforms = decomposition_result.waveforms

    (
        source_to_spatial_distortion_ratio,
        source_to_interference_ratio,
        source_to_artifacts_ratio,
        source_to_distortion_ratio
    ) = calculate_bss_eval_energy_ratios(
        waveforms.true_target,
        waveforms.target_distortion,
        waveforms.interference,
        waveforms.artifacts
    )

    print("\n--- Python Port Metrics Result ---")
    print(f"Overall Quality (SDR):                     {source_to_distortion_ratio:.2f} dB")
    print(f"Target Preservation (ISR):                 {source_to_spatial_distortion_ratio:.2f} dB")
    print(f"Interference Rejection (SIR):              {source_to_interference_ratio:.2f} dB")
    print(f"Artifact Level (SAR):                      {source_to_artifacts_ratio:.2f} dB")