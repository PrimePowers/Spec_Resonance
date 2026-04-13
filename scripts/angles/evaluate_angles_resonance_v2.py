"""
Prime-Zeta Resonance Experiments
Evaluates resonance alignment in the angle-domain data and exports summary result tables.
"""

# Evaluate an angles file with angular-frequency matching and detrending for unwrapped angle series
import argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram, detrend
from tqdm import tqdm

ZETA_GAMMA = np.array([
    14.1347251417, 21.0220396388, 25.0108575801, 30.4248761259, 32.9350615877,
    37.5861781588, 40.9187190121, 43.3270732813, 48.0051508812, 49.7738324777
])


def load_frame(input_path):
    if input_path.endswith('.csv'):
        return pd.read_csv(input_path)
    if input_path.endswith('.parquet'):
        return pd.read_parquet(input_path)
    raise ValueError('Unsupported input format')


def pick_series(data_df):
    name_list = [
        'theta_norm', 'psi_norm',
        'angle_theta_norm', 'angle_psi_norm',
        'angle_theta_norm_unwrapped', 'angle_psi_norm_unwrapped'
    ]
    return {name: data_df[name].to_numpy() for name in name_list if name in data_df.columns}


def preprocess_series(series_name, series_vals):
    centered_vals = series_vals - np.mean(series_vals)
    if 'unwrapped' in series_name:
        return detrend(centered_vals, type='linear')
    return centered_vals


def evaluate_series(u_vals, series_name, series_vals, low_omega, high_omega, top_k, peak_distance, tol):
    fs_val = 1.0 / np.median(np.diff(u_vals))
    clean_vals = preprocess_series(series_name, series_vals)
    freq_vals, spec_vals = periodogram(clean_vals, fs=fs_val, scaling='spectrum')
    omega_vals = 2.0 * np.pi * freq_vals
    band_mask = (omega_vals >= low_omega) & (omega_vals <= high_omega)
    band_omega = omega_vals[band_mask]
    band_spec = spec_vals[band_mask]
    peak_idx, _ = find_peaks(band_spec, distance=peak_distance)
    peak_omega = band_omega[peak_idx]
    peak_pows = band_spec[peak_idx]
    if len(peak_omega) == 0:
        return {
            'series': series_name,
            'mean_gap_to_first10': np.nan,
            'captured_first10_within_tol': 0,
            'resonance_score': 0.0,
            'top_omega': ''
        }, []
    top_order = np.argsort(peak_pows)[-top_k:]
    top_omega = np.sort(peak_omega[top_order])
    top_pows = np.sort(peak_pows[top_order])
    mean_gap = np.mean([np.min(np.abs(top_omega - g_val)) for g_val in ZETA_GAMMA])
    captured = int(np.sum([np.min(np.abs(top_omega - g_val)) <= tol for g_val in ZETA_GAMMA]))
    resonance_score = float(np.sum([band_spec[np.argmin(np.abs(band_omega - g_val))] for g_val in ZETA_GAMMA]))
    peak_rows = []
    for omega_val, pow_val in zip(top_omega, top_pows):
        peak_rows.append({'series': series_name, 'peak_omega': omega_val, 'peak_power': pow_val})
    summary_row = {
        'series': series_name,
        'mean_gap_to_first10': mean_gap,
        'captured_first10_within_tol': captured,
        'resonance_score': resonance_score,
        'top_omega': ', '.join([str(round(x_val, 6)) for x_val in top_omega])
    }
    return summary_row, peak_rows


def build_match_table(peaks_df, match_series):
    base_peaks = peaks_df[peaks_df['series'] == match_series]['peak_omega'].to_numpy()
    match_rows = []
    if len(base_peaks) == 0:
        return pd.DataFrame(match_rows)
    for gamma_val in ZETA_GAMMA:
        nearest_peak = base_peaks[np.argmin(np.abs(base_peaks - gamma_val))]
        match_rows.append({
            'series': match_series,
            'zeta_gamma': gamma_val,
            'nearest_detected_peak': nearest_peak,
            'abs_gap': abs(nearest_peak - gamma_val)
        })
    return pd.DataFrame(match_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--summary_out', default='angles_eval_summary_v2.csv')
    parser.add_argument('--peaks_out', default='angles_eval_peaks_v2.csv')
    parser.add_argument('--matches_out', default='angles_eval_matches_v2.csv')
    parser.add_argument('--low_omega', type=float, default=10.0)
    parser.add_argument('--high_omega', type=float, default=55.0)
    parser.add_argument('--top_k', type=int, default=15)
    parser.add_argument('--peak_distance', type=int, default=20)
    parser.add_argument('--tol', type=float, default=0.75)
    parser.add_argument('--match_series', default='theta_norm')
    args = parser.parse_args()

    print('loading')
    data_df = load_frame(args.input)
    data_df = data_df.sort_values('x').reset_index(drop=True)
    u_vals = data_df['u_logx'].to_numpy()
    series_map = pick_series(data_df)

    summary_rows = []
    peak_rows = []
    for series_name in tqdm(series_map.keys()):
        summary_row, local_peaks = evaluate_series(
            u_vals,
            series_name,
            series_map[series_name],
            args.low_omega,
            args.high_omega,
            args.top_k,
            args.peak_distance,
            args.tol
        )
        summary_rows.append(summary_row)
        peak_rows.extend(local_peaks)

    summary_df = pd.DataFrame(summary_rows).sort_values(['captured_first10_within_tol', 'mean_gap_to_first10'], ascending=[False, True])
    peaks_df = pd.DataFrame(peak_rows)
    match_df = build_match_table(peaks_df, args.match_series)

    summary_df.to_csv(args.summary_out, index=False)
    peaks_df.to_csv(args.peaks_out, index=False)
    match_df.to_csv(args.matches_out, index=False)

    print(summary_df.head())
    print(peaks_df.head())
    print(match_df.head())
    print(args.summary_out)
    print(args.peaks_out)
    print(args.matches_out)


if __name__ == '__main__':
    main()
