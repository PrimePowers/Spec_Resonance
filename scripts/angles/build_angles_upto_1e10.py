"""
Prime-Zeta Resonance Experiments
Constructs the angle-domain dataset up to 1e10 for transformed resonance experiments.
"""

# Build angles data up to 1e10 from a log-grid phase source file
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_phase_frame(input_path):
    if input_path.endswith('.csv'):
        return pd.read_csv(input_path)
    if input_path.endswith('.parquet'):
        return pd.read_parquet(input_path)
    if input_path.endswith('.npz'):
        npz_obj = np.load(input_path)
        data_map = {key: npz_obj[key] for key in npz_obj.files}
        return pd.DataFrame(data_map)
    raise ValueError('Unsupported input format')


def ensure_columns(phase_df):
    needed_cols = ['x', 'u_logx', 'theta_norm', 'psi_norm', 'theta_minus_x', 'psi_minus_x']
    missing_cols = [col for col in needed_cols if col not in phase_df.columns]
    if missing_cols:
        raise ValueError('Missing columns: ' + ', '.join(missing_cols))


def add_angle_columns(phase_df):
    out_df = phase_df.copy()
    out_df['angle_theta_norm'] = np.angle(np.exp(1j * out_df['theta_norm'].to_numpy()))
    out_df['angle_psi_norm'] = np.angle(np.exp(1j * out_df['psi_norm'].to_numpy()))
    out_df['angle_theta_minus_x'] = np.angle(np.exp(1j * out_df['theta_minus_x'].to_numpy()))
    out_df['angle_psi_minus_x'] = np.angle(np.exp(1j * out_df['psi_minus_x'].to_numpy()))
    out_df['angle_theta_norm_unwrapped'] = np.unwrap(out_df['angle_theta_norm'].to_numpy())
    out_df['angle_psi_norm_unwrapped'] = np.unwrap(out_df['angle_psi_norm'].to_numpy())
    return out_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='angles_rebuilt_upto_1e10.csv')
    parser.add_argument('--xmax', type=float, default=1e10)
    args = parser.parse_args()

    print('loading')
    phase_df = load_phase_frame(args.input)
    ensure_columns(phase_df)

    print('filtering')
    phase_df = phase_df[phase_df['x'] <= args.xmax].copy()
    phase_df = phase_df.sort_values('x').reset_index(drop=True)

    print('building angles')
    tqdm.pandas()
    out_df = add_angle_columns(phase_df)

    print('saving')
    if args.output.endswith('.parquet'):
        out_df.to_parquet(args.output, index=False)
    else:
        out_df.to_csv(args.output, index=False)
    print(args.output)
    print(len(out_df))


if __name__ == '__main__':
    main()
