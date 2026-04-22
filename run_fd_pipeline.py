
#!/usr/bin/env python3
"""
Unified runner for the Prime–Zeta FD repository.

This script does two things:
1. Runs scripts that already expose a CLI directly.
2. Runs hard-coded scripts by creating a temporary patched copy with constants
   replaced from config.yaml.

It is intended as a reproducibility wrapper for the compact repository release.
"""
from __future__ import annotations
import argparse, copy, os, re, shlex, subprocess, sys, tempfile
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_subprocess(cmd, cwd: Path):
    print('[RUN]', ' '.join(shlex.quote(str(c)) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def patch_constants(src: str, mapping: dict) -> str:
    out = src
    for name, value in mapping.items():
        if isinstance(value, str):
            replacement = repr(value)
        else:
            replacement = repr(value)
        # replace simple top-level assignments NAME = ...
        pattern = rf'(?m)^({re.escape(name)}\s*=\s*).*$'
        new_out, n = re.subn(pattern, rf'\1{replacement}', out)
        if n == 0:
            print(f'[WARN] constant {name} not found for patching')
        out = new_out
    return out


def run_patched_script(script_path: Path, mapping: dict, cwd: Path):
    src = script_path.read_text(encoding='utf-8')
    patched = patch_constants(src, mapping)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / script_path.name
        tmp.write_text(patched, encoding='utf-8')
        run_subprocess([sys.executable, str(tmp)], cwd=cwd)


def task_generation(cfg: dict, root: Path):
    gen = cfg['generation']
    mapping_by_file = {
        'Gen_10_9_fixed.py': {'X_MAX': gen['scale_1e9']['x_max'], 'X_MIN': gen['scale_1e9']['x_min'], 'N_LOG_GRID': gen['scale_1e9']['n_log_grid'], 'SEGMENT_SIZE': gen['scale_1e9']['segment_size'], 'OUT_PREFIX': gen['scale_1e9']['out_prefix']},
        'Gen_10_10_partA.py': {'X_MAX': gen['scale_1e10']['x_max'], 'X_MIN': gen['scale_1e10']['x_min'], 'N_LOG_GRID': gen['scale_1e10']['n_log_grid'], 'SEGMENT_SIZE': gen['scale_1e10']['segment_size'], 'OUT_PREFIX': gen['scale_1e10']['out_prefix']},
        'Gen_10_11_partA.py': {'X_MAX': gen['scale_1e11']['x_max'], 'X_MIN': gen['scale_1e11']['x_min'], 'N_LOG_GRID': gen['scale_1e11']['n_log_grid'], 'SEGMENT_SIZE': gen['scale_1e11']['segment_size'], 'OUT_PREFIX': gen['scale_1e11']['out_prefix']},
    }
    for fname, mapping in mapping_by_file.items():
        scale_key = 'scale_1e9' if '10_9' in fname else 'scale_1e10' if '10_10' in fname else 'scale_1e11'
        if not gen[scale_key]['enabled']:
            continue
        run_patched_script(root/'scripts'/'generation'/fname, mapping, root)


def task_analysis(cfg: dict, root: Path):
    # scripts are hard-coded; patch input file names and key analysis parameters
    analysis = cfg['analysis']
    mapping_b = {
        'Gen_10_10_partB.py': {'NPZ_FILE': analysis['global_1e10']['input_npz']},
        'Gen_10_11_partB.py': {'NPZ_FILE': analysis['global_1e11']['input_npz']},
        'Gen_10_11_partC_first20.py': {'NPZ_FILE': analysis['first20_1e11']['input_npz'], 'LOCAL_RANDOM_TRIALS': analysis['first20_1e11']['local_random_trials'], 'WINDOW_CAPS': analysis['first20_1e11']['window_caps']},
        'advanced_null_loc_1e11_colab.py': {'NPZ_FILE': analysis['advanced_1e11']['input_npz'], 'WINDOW_CAPS': analysis['advanced_1e11']['window_caps'], 'MATCH_HALF_WINDOW': analysis['advanced_1e11']['first10_half_window'], 'TOP_N_BLIND': analysis['advanced_1e11']['top_n_blind'], 'TOP_N_NEAR_ZERO': analysis['advanced_1e11']['top_n_near_zero']},
        'analyze_first30_lhs_phase_fixed.py': {'NPZ_FILE': analysis['first30_fixed']['input_npz']},
        'analyze_first30_lhs_phase.py': {'NPZ_FILE': analysis['first30_fixed']['input_npz']},
    }
    for fname, mapping in mapping_b.items():
        key = None
        if fname=='Gen_10_10_partB.py': key='global_1e10'
        elif fname=='Gen_10_11_partB.py': key='global_1e11'
        elif fname=='Gen_10_11_partC_first20.py': key='first20_1e11'
        elif fname=='advanced_null_loc_1e11_colab.py': key='advanced_1e11'
        elif 'first30' in fname: key='first30_fixed'
        if key and analysis.get(key,{}).get('enabled', False):
            run_patched_script(root/'scripts'/'analysis'/fname, mapping, root)
    if analysis['lppl_first30']['enabled']:
        # lppl script seems input-driven via expected file in cwd
        run_patched_script(root/'scripts'/'analysis'/'lppl_first30_analysis.py', {}, root)


def task_angles(cfg: dict, root: Path):
    ang = cfg['angles']
    if ang['build_upto_1e10']['enabled']:
        c = ang['build_upto_1e10']
        run_subprocess([sys.executable, 'scripts/angles/build_angles_upto_1e10.py', '--input', c['input'], '--output', c['output'], '--xmax', str(c['xmax'])], root)
    if ang['evaluate_resonance_v2']['enabled']:
        c = ang['evaluate_resonance_v2']
        run_subprocess([sys.executable, 'scripts/angles/evaluate_angles_resonance_v2.py', '--input', c['input'], '--summary_out', c['summary_out'], '--peaks_out', c['peaks_out'], '--matches_out', c['matches_out'], '--low_omega', str(c['low_omega']), '--high_omega', str(c['high_omega']), '--top_k', str(c['top_k']), '--peak_distance', str(c['peak_distance']), '--tol', str(c['tol']), '--match_series', c['match_series']], root)
    if ang['high_precision_pipeline']['enabled']:
        c = ang['high_precision_pipeline']
        mapping = {'ANGLES_FILE': c['input'], 'OUT_THETA': c['output_theta'], 'OUT_PSI': c['output_psi'], 'PLOT2': c['output_plot'], 'OMEGA_MIN': c['omega_min'], 'OMEGA_MAX': c['omega_max'], 'N_FREQS': c['num_omega']}
        run_patched_script(root/'scripts'/'angles'/'high_precision_angles_pipeline.py', mapping, root)


def task_figures(cfg: dict, root: Path):
    fig = cfg['figures']['paper_figures_1e11']
    if not fig['enabled']:
        return
    mapping = {'NPZ_FILE': fig['npz_file'], 'SCORE_FILE': fig['score_file'], 'WINDOW_FILE': fig['window_file'], 'OMEGA_MIN': fig['omega_min'], 'OMEGA_MAX': fig['omega_max']}
    run_patched_script(root/'scripts'/'figures'/'paper_figures_1e11_colab.py', mapping, root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--tasks', nargs='+', default=['generation','analysis','angles','figures'], choices=['generation','analysis','angles','figures'])
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config))
    if 'generation' in args.tasks:
        task_generation(cfg, ROOT)
    if 'analysis' in args.tasks:
        task_analysis(cfg, ROOT)
    if 'angles' in args.tasks:
        task_angles(cfg, ROOT)
    if 'figures' in args.tasks:
        task_figures(cfg, ROOT)

if __name__ == '__main__':
    main()
