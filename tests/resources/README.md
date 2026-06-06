# Test Resources

This directory contains static WAV files used by `pytest` for validation and regression checks.

## Contents
*   `database/`: Mono and stereo source audio used for baseline correctness validations.
*   `matlab_reference/`: Sourced from the MATLAB PEASS v2.0.1 codebase. Used to guarantee that the Python implementation matches legacy outputs within acceptable floating-point tolerances.
