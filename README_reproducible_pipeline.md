# Reproducible Config + Runner

This file documents the new reproducibility layer added on top of the compact repository release.

## Files added

- `config.yaml` — canonical configuration file collecting script parameters
- `run_fd_pipeline.py` — unified runner for the repository scripts

The runner supports two execution modes:

1. **Native CLI mode** for scripts that already expose command-line arguments.
2. **Patched-script mode** for legacy scripts that still hard-code constants at the top of the file.

In patched-script mode, the runner creates a temporary copy of the script, replaces the configured top-level constants, and executes the patched copy. The original repository files remain unchanged.

---

## Quick start

Run the full pipeline from the repository root:

```bash
python run_fd_pipeline.py --config config.yaml
```

Run only selected stages:

```bash
python run_fd_pipeline.py --config config.yaml --tasks generation analysis
python run_fd_pipeline.py --config config.yaml --tasks angles
python run_fd_pipeline.py --config config.yaml --tasks figures
```

---

## What `config.yaml` controls

### 1. Generation scripts

These settings patch the top-level constants in:

- `scripts/generation/Gen_10_9_fixed.py`
- `scripts/generation/Gen_10_10_partA.py`
- `scripts/generation/Gen_10_11_partA.py`

For each scale, the config exposes:

- `x_min`
- `x_max`
- `n_log_grid`
- `segment_size`
- `out_prefix`

### Default values reconstructed from the repository

#### 1e9 generation
- `x_min = 10`
- `x_max = 1_000_000_000`
- `n_log_grid = 20000`
- `segment_size = 5_000_000`
- `out_prefix = lhs_phase_1e9`

**Outputs**
- `lhs_phase_1e9_loggrid.csv`
- `lhs_phase_1e9_loggrid.npz`

#### 1e10 generation
- `x_min = 10`
- `x_max = 10_000_000_000`
- `n_log_grid = 16000`
- `segment_size = 10_000_000`
- `out_prefix = lhs_phase_1e10`

**Outputs**
- `lhs_phase_1e10_loggrid.csv`
- `lhs_phase_1e10_loggrid.npz`

#### 1e11 generation
- `x_min = 10`
- `x_max = 100_000_000_000`
- `n_log_grid = 12000`
- `segment_size = 20_000_000`
- `out_prefix = lhs_phase_1e11`

**Outputs**
- `lhs_phase_1e11_loggrid.csv`
- `lhs_phase_1e11_loggrid.npz`

---

### 2. Main analysis scripts

These are legacy scripts with hard-coded constants. The runner patches the relevant input filenames and selected numerical parameters.

#### `scripts/analysis/Gen_10_10_partB.py`
**Input**: `lhs_phase_1e10_loggrid.npz`

**Fixed analysis parameters found in the script**
- known zeros: first 10 zeta ordinates
- Hann window: enabled
- detrending: enabled
- band half-width: `0.35`
- random band test trials: `400`
- random frequency range: `[10, 55]`
- shift grid: `np.linspace(-3, 3, 121)`
- jitter sizes: `0.2, 0.4, 0.6`
- jitter trials: `300`
- window-level random trials: `250`
- seeds:
  - random band: `2026`
  - jitter 0.2: `2027`
  - jitter 0.4: `2028`
  - jitter 0.6: `2029`
  - window random: `2030`

**Outputs**
- `analysis_global_summary_1e10.csv`
- `analysis_window_stability_1e10.csv`
- `analysis_peaks_theta_norm_1e10.csv`
- `analysis_peaks_psi_norm_1e10.csv`

#### `scripts/analysis/Gen_10_11_partB.py`
Same structure as the 1e10 script, but for `lhs_phase_1e11_loggrid.npz`.

**Outputs**
- `analysis_global_summary_1e11.csv`
- `analysis_window_stability_1e11.csv`
- `analysis_peaks_theta_norm_1e11.csv`
- `analysis_peaks_psi_norm_1e11.csv`

#### `scripts/analysis/Gen_10_11_partC_first20.py`
**Input**: `lhs_phase_1e11_loggrid.npz`

**Fixed parameters reconstructed from the script**
- known zeros: first 20 zeta ordinates
- local random trials: `400`
- Hann window: enabled
- detrending: enabled
- default window caps sweep:
  - `1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9, 2e9, 5e9, 1e10, 2e10, 5e10, 1e11`

**Outputs**
- `analysis_first20_local_scores_1e11.csv`
- `analysis_first20_window_scores_1e11.csv`
- `analysis_first20_consistency_1e11.csv`
- `analysis_first20_scoreboard_1e11.csv`

#### `scripts/analysis/advanced_null_loc_1e11_colab.py`
**Input**: `lhs_phase_1e11_loggrid.npz`

**Fixed parameters reconstructed from the script**
- first 10 zeros
- `WINDOW_CAPS` default sweep:
  - `1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9, 2e9, 5e9, 1e10, 2e10, 5e10, 1e11`
- `TOP_N_BLIND = 20`
- `MATCH_HALF_WINDOW = 0.8`
- `TOP_N_NEAR_ZERO = 3`
- Hann window: enabled

**Outputs**
- `advanced_tracking_first10_1e11.csv`
- `advanced_blind_peaks_1e11.csv`
- `advanced_error_vs_scale_1e11.csv`
- `advanced_blind_clusters_1e11.csv`
- `advanced_blind_clusters_vs_first10_1e11.csv`
- figure files `fig_adv*.png`

#### `scripts/analysis/analyze_first30_lhs_phase.py`
#### `scripts/analysis/analyze_first30_lhs_phase_fixed.py`
**Input**: `lhs_phase_1e11_loggrid.npz`

**Fixed parameters**
- first 30 zeros
- FFT / spectral comparison in log space
- Hann window enabled in the fixed version

**Outputs**
- `top30_peaks_lhs_phase.csv`
- `matched_first30_lhs_phase.csv` (fixed version)
- `fft_first30_lhs_phase.png`

#### `scripts/analysis/lppl_first30_analysis.py`
**Input expectation**
- uses first-30 matching outputs already present in the working directory

**Outputs**
- `results_first30.csv`
- `blind_clusters_first30.csv`

---

### 3. Angle scripts

#### `scripts/angles/build_angles_upto_1e10.py`
Native CLI script.

```bash
python scripts/angles/build_angles_upto_1e10.py   --input lhs_phase_1e10_loggrid.csv   --output angles_rebuilt_upto_1e10.csv   --xmax 1e10
```

**Outputs**
- `angles_rebuilt_upto_1e10.csv`

#### `scripts/angles/evaluate_angles_resonance_v2.py`
Native CLI script.

Parameters exposed in `config.yaml`:
- `input`
- `summary_out`
- `peaks_out`
- `matches_out`
- `low_omega`
- `high_omega`
- `top_k`
- `peak_distance`
- `tol`
- `match_series`

Defaults reconstructed from the script:
- `low_omega = 10.0`
- `high_omega = 55.0`
- `top_k = 15`
- `peak_distance = 20`
- `tol = 0.75`
- `match_series = theta_norm`

**Outputs**
- `angles_eval_summary_v2.csv`
- `angles_eval_peaks_v2.csv`
- `angles_eval_matches_v2.csv`

#### `scripts/angles/high_precision_angles_pipeline.py`
Legacy script patched by the runner.

**Reconstructed parameters**
- input angles file
- frequency range `[10, 55]`
- output spectrum CSVs:
  - `theta_norm.csv`
  - `psi_norm.csv`
- top peak export:
  - `top20_peaks_from_angles.csv`

**Outputs**
- `theta_norm.csv`
- `psi_norm.csv`
- `top20_peaks_from_angles.csv`
- angle-domain diagnostic figure

---

### 4. Figure script

#### `scripts/figures/paper_figures_1e11_colab.py`
Patched by the runner.

**Inputs**
- `lhs_phase_1e11_loggrid.npz`
- `analysis_first20_scoreboard_1e11.csv`
- `analysis_first20_window_scores_1e11.csv`

**Parameters**
- `OMEGA_MIN = 10`
- `OMEGA_MAX = 55`
- first 10 zeros

**Outputs**
- `fig1_global_resonance_1e11.png`
- `fig2_zoom_first10_1e11.png`
- `fig3_gap_first10_1e11.png`
- `fig4_local_z_first10_1e11.png`
- `fig5_window_heatmap_first10_1e11.png`

---

## Operator-level parameters captured for documentation

These are not consumed by `run_fd_pipeline.py`, but are included in `config.yaml` because they were canonically identified in the research workflow and should be preserved in a reproducible release.

### Canonical theta core
- window: `[18.2, 20.45]`
- `edge_width = 0.32`
- `resid_dim = 3`

### Canonical psi core
- window: `[18.2, 20.45]`
- `edge_width = 0.32`
- `resid_dim = 4`

### 27-setup theta sweep
- `left_edges = [18.15, 18.20, 18.25]`
- `right_edges = [20.40, 20.45, 20.50]`
- `edge_widths = [0.26, 0.32, 0.38]`
- `resid_dim = 3`

These settings describe the stable compressed-core regime discussed in the paper, even though the corresponding scripts are not yet fully unified into the compact repository release.


