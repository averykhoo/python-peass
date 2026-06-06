import soundfile as sf

from peass.predictor import predict_perceptual_evaluation_scores


def test_stereo_database_file_execution(db_resources):
    """Tests the full pipeline using the Stereo (exp03) assets from the DB folder."""
    target_path = db_resources / "exp03_target.wav"
    interf_path = db_resources / "exp03_InterfSrc1.wav"

    # Create a dummy estimate (Target + low level Interferer)
    t, fs = sf.read(target_path)
    i, _ = sf.read(interf_path)
    estimate = t + 0.05 * i

    # Run full scoring
    results = predict_perceptual_evaluation_scores(
        original_files=[t, i],
        estimate_file=estimate,
        sampling_frequency_hz=float(fs)
    )

    # Verify scores generated for stereo don't crash and remain in a reasonable perceptual band
    assert 0.0 <= results.overall_perceptual_score <= 100.0
    assert results.overall_perceptual_score > 70.0  # Adjusted threshold for 5% active leakage
    assert 0.0 <= results.target_perceptual_score <= 100.0
    assert 0.0 <= results.interference_perceptual_score <= 100.0
    assert 0.0 <= results.artifact_perceptual_score <= 100.0
