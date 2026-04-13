# Prime-Zeta Resonance Experiments

## Overview

This repository contains a research-oriented collection of Python scripts for studying resonance-like structure in prime-derived signals and comparing those structures to low-lying Riemann zeta ordinates.

The project is organized as a reproducible computational archive rather than a packaged Python library. The scripts generate logarithmic-grid datasets, evaluate spectral concentration and localization, test robustness under null models, and produce publication-style figures.

## Core idea

The workflow starts from prime-counting objects such as theta and psi, builds normalized residual signals on logarithmic grids, and then examines whether strong spectral features align with known zeta ordinates. Several scripts extend this with local significance tests, blind clustering, angle-domain transforms, and scale-dependent robustness checks.

## Repository layout

```text
prime_zeta_repo_pro/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── CITATION.cff
├── REPOSITORY_GUIDE.md
├── SCRIPT_DESCRIPTIONS.md
└── scripts/
    ├── generation/
    ├── analysis/
    ├── angles/
    └── figures/
```

## Workflows

### 1. Generate datasets

- `scripts/generation/Gen_10_9_fixed.py`
- `scripts/generation/Gen_10_10_partA.py`
- `scripts/generation/Gen_10_11_partA.py`

### 2. Run resonance analysis

- `scripts/analysis/Gen_10_10_partB.py`
- `scripts/analysis/Gen_10_11_partB.py`
- `scripts/analysis/Gen_10_11_partC_first20.py`
- `scripts/analysis/analyze_first30_lhs_phase.py`
- `scripts/analysis/analyze_first30_lhs_phase_fixed.py`
- `scripts/analysis/lppl_first30_analysis.py`
- `scripts/analysis/advanced_null_loc_1e11_colab.py`

### 3. Run angle-domain validation

- `scripts/angles/build_angles_upto_1e10.py`
- `scripts/angles/evaluate_angles_resonance_v2.py`
- `scripts/angles/high_precision_angles_pipeline.py`

### 4. Produce figures

- `scripts/figures/paper_figures_1e11_colab.py`

## Reproducibility notes

This repository reflects an experimental research workflow. Some scripts assume the presence of precomputed CSV, NPZ, or PNG outputs in the working directory, so the cleanest reproduction path is:

generation -> main analysis -> focused diagnostics -> figures

## Environment

Install the main dependencies with:

```bash
pip install -r requirements.txt
```

## Citation

If you use this repository in academic or technical work, include a citation to the repository and, if applicable, to the related paper or manuscript. A starter `CITATION.cff` file is included.


## Compact release layout

This v4 release keeps the repository compact for GitHub. Generated CSV outputs are bundled into `data/outputs_csv_bundle.tar.gz`, while PNG figures remain visible in `figures/` for quick browsing on GitHub.

### Included result artifacts

- Archived CSV files: 70
- Visible PNG figures: 30

To inspect all tabular outputs locally, extract the archive in `data/`.
