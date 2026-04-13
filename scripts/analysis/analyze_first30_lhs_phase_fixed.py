"""
Prime-Zeta Resonance Experiments
Corrected and refined first-30 LHS phase analysis with improved matching outputs.
"""

# ============================================================
# CLEAN PIPELINE: FIRST 30 ZETA ORDINATES FROM lhs_phase_1e11_loggrid.npz
# FIXED VERSION
# ------------------------------------------------------------
# Uses the actual keys stored by the generation pipeline:
#   - u_logx
#   - theta_norm
#   - psi_norm
#
# Outputs:
#   - top30_peaks_lhs_phase.csv
#   - fft_first30_lhs_phase.png
#   - matched_first30_lhs_phase.csv
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

FILE = "lhs_phase_1e11_loggrid.npz"

FIRST30 = np.array([
    14.134725141, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
    52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
    67.079810529, 69.546401711, 72.067157674, 75.704690699, 77.144840069,
    79.337375020, 82.910380854, 84.735492981, 87.425274613, 88.809111208,
    92.491899271, 94.651344041, 95.870634228, 98.831194218, 101.317851006
], dtype=np.float64)

OMEGA_MIN = 10.0
OMEGA_MAX = 110.0

print("Loading NPZ file...")
data = np.load(FILE)

print("Available keys:", list(data.keys()))

# Robust key selection
if "u_logx" in data:
    u = data["u_logx"].astype(np.float64)
elif "u" in data:
    u = data["u"].astype(np.float64)
else:
    raise KeyError("Neither 'u_logx' nor 'u' found in NPZ archive.")

if "theta_norm" not in data or "psi_norm" not in data:
    raise KeyError("NPZ must contain both 'theta_norm' and 'psi_norm'.")

theta = data["theta_norm"].astype(np.float64)
psi = data["psi_norm"].astype(np.float64)

print(f"Loaded {len(theta):,} points")

# FFT in log-space
def compute_fft(signal, u_vals, apply_hann=True):
    y = signal - np.mean(signal)
    if apply_hann:
        y = y * np.hanning(len(y))

    du = float(np.median(np.diff(u_vals)))
    fft_vals = np.fft.rfft(y)
    omega = 2 * np.pi * np.fft.rfftfreq(len(y), d=du)
    power = np.abs(fft_vals) ** 2
    power = power / power.max() if power.max() > 0 else power
    return omega, power

omega, power_theta = compute_fft(theta, u)
_, power_psi = compute_fft(psi, u)

mask = (omega > OMEGA_MIN) & (omega < OMEGA_MAX)
omega = omega[mask]
power_theta = power_theta[mask]
power_psi = power_psi[mask]

# Peaks for theta spectrum
peaks, props = find_peaks(power_theta, height=0.03, distance=8)

peak_omega = omega[peaks]
peak_val = power_theta[peaks]

peak_df = pd.DataFrame({
    "omega": peak_omega,
    "power": peak_val
}).sort_values("power", ascending=False).head(30).sort_values("omega").reset_index(drop=True)

peak_df.to_csv("top30_peaks_lhs_phase.csv", index=False)

print("\nTop 30 peaks from theta_norm spectrum:")
print(peak_df.to_string(index=False))

# Local matching of first 30 ordinates
rows = []
for k, gamma in enumerate(FIRST30, start=1):
    local = (omega >= gamma - 0.8) & (omega <= gamma + 0.8)
    if not np.any(local):
        rows.append({
            "k": k,
            "gamma": gamma,
            "theta_peak": np.nan,
            "theta_gap": np.nan,
            "psi_peak": np.nan,
            "psi_gap": np.nan,
        })
        continue

    om = omega[local]
    pt = power_theta[local]
    pp = power_psi[local]

    theta_peak = om[np.argmax(pt)]
    psi_peak = om[np.argmax(pp)]

    rows.append({
        "k": k,
        "gamma": gamma,
        "theta_peak": theta_peak,
        "theta_gap": abs(theta_peak - gamma),
        "psi_peak": psi_peak,
        "psi_gap": abs(psi_peak - gamma),
    })

match_df = pd.DataFrame(rows)
match_df.to_csv("matched_first30_lhs_phase.csv", index=False)

print("\nMatched first 30 ordinates:")
print(match_df.to_string(index=False))

# Plot spectrum vs ordinates
plt.figure(figsize=(13, 6))
plt.plot(omega, power_theta, label="theta_norm", lw=2)
plt.plot(omega, power_psi, label="psi_norm", lw=1.7, alpha=0.8)

for g in FIRST30:
    plt.axvline(g, color="green", alpha=0.18, lw=1)

plt.title("FFT spectrum vs first 30 zeta ordinates")
plt.xlabel("omega")
plt.ylabel("normalized power")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("fft_first30_lhs_phase.png", dpi=250)
plt.show()

print("\nSaved:")
print("- top30_peaks_lhs_phase.csv")
print("- matched_first30_lhs_phase.csv")
print("- fft_first30_lhs_phase.png")
