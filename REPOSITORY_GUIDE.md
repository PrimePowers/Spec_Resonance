# Repository Guide

## Purpose

This repository is a reproducible research archive for prime-side resonance experiments.

## Directories

### `scripts/generation`
Creates the underlying logarithmic-grid datasets from prime-counting functions.

### `scripts/analysis`
Runs the main statistical and spectral evaluations, including targeted tests for low-lying zeta ordinates.

### `scripts/angles`
Implements an alternate angle-based pipeline for transformed resonance validation.

### `scripts/figures`
Builds final presentation and paper-style figures from precomputed outputs.

## Usage style

Run the repository as a sequence of standalone scripts rather than importing it as a package. Most scripts write CSV, NPZ, or PNG artifacts into the current working directory.


## Output packaging

For a cleaner public repository, CSV outputs are stored in the compressed archive `data/outputs_csv_bundle.tar.gz`. Figures are kept individually in `figures/` so they remain directly viewable on GitHub.
