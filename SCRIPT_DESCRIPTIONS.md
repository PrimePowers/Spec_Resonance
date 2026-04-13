# Script Descriptions

## Generation scripts

### `Gen_10_9_fixed.py`
Generates log-grid prime-counting signals up to 1e9 and exports normalized channels for resonance analysis.

### `Gen_10_10_partA.py`
Builds the 1e10 prime-side dataset on a logarithmic grid for downstream spectral experiments.

### `Gen_10_11_partA.py`
Creates the 1e11 prime-side dataset and serves as the main high-scale generation entry point.

## Analysis scripts

### `Gen_10_10_partB.py`
Analyzes the 1e10 dataset with global spectral summaries, null comparisons, and stability diagnostics.

### `Gen_10_11_partB.py`
Runs the main 1e11 resonance analysis and exports high-scale summary tables.

### `Gen_10_11_partC_first20.py`
Scores the first 20 zeta ordinates using local significance and consistency tests.

### `analyze_first30_lhs_phase.py`
Initial FFT-style exploration of the first 30 candidate ordinates in LHS phase data.

### `analyze_first30_lhs_phase_fixed.py`
Refined version of the first-30 LHS phase analysis with corrected matching behavior.

### `lppl_first30_analysis.py`
Compares blind and matched peak recovery against the first 30 zeta ordinates.

### `advanced_null_loc_1e11_colab.py`
Provides advanced robustness diagnostics, blind-cluster tracking, and localization error analysis at 1e11 scale.

## Angle scripts

### `build_angles_upto_1e10.py`
Builds the angle-domain dataset used for transformed resonance experiments.

### `evaluate_angles_resonance_v2.py`
Evaluates angle-domain resonance alignment and exports summary result tables.

### `high_precision_angles_pipeline.py`
Runs a more detailed angle-based validation pipeline with residual and spectral diagnostics.

## Figure scripts

### `paper_figures_1e11_colab.py`
Generates publication-style figures from the 1e11 analysis outputs.
