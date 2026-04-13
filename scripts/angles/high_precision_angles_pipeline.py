"""
Prime-Zeta Resonance Experiments
Runs a higher-precision angle-based validation workflow with spectral diagnostics and residual analysis.
"""

# ============================================================
# HIGH-PRECISION PIPELINE FOR angles_*.csv.gz
# ------------------------------------------------------------
# This Colab-ready script:
#   1) loads an angles_*.csv.gz file
#   2) reconstructs a log-space time axis
#   3) removes the global linear trend
#   4) computes a Lomb-Scargle spectrum
#   5) saves:
#        - theta_norm.csv
#        - psi_norm.csv
#   6) also saves diagnostic plots
#
# IMPORTANT:
# This script assumes the uploaded file contains a single column
# of angle / phase values, one per line.
#
# By default, it uses a simple reconstructed x-grid from X_MIN..X_MAX.
# For your current file angles_100000000.csv.gz, the default X_MAX=1e8
# is correct.
#
# If later you use a file from a different range, change X_MAX below.
# ============================================================

import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import lombscargle

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
ANGLE_FILE = "angles_100000000.csv.gz"   # change if needed
X_MIN = 1.0
X_MAX = 100_000_000.0                    # set to the range of your file
OMEGA_MIN = 10.0
OMEGA_MAX = 80.0
N_OMEGA = 4000

OUT_THETA = "theta_norm.csv"
OUT_PSI = "psi_norm.csv"

PLOT1 = "angles_trend_residual.png"
PLOT2 = "lomb_scargle_spectrum.png"

# First 30 zeta ordinates for reference in the spectrum plot
FIRST30 = np.array([
    14.134725141, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
    52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
    67.079810529, 69.546401711, 72.067157674, 75.704690699, 77.144840069,
    79.337375020, 82.910380854, 84.735492981, 87.425274613, 88.809111208,
    92.491899271, 94.651344041, 95.870634228, 98.831194218, 101.317851006
], dtype=np.float64)

plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 10

# ------------------------------------------------------------
# LOAD ANGLES
# ------------------------------------------------------------
print(f"Loading file: {ANGLE_FILE}")
with gzip.open(ANGLE_FILE, "rt") as f:
    angles = np.loadtxt(f, dtype=np.float64)

n = len(angles)
print(f"Loaded {n:,} angle values")

if n < 100:
    raise ValueError("Too few points loaded. Please check the file format.")

# ------------------------------------------------------------
# RECONSTRUCT x AND u = log(x)
# ------------------------------------------------------------
# We use a monotone x-grid on [X_MIN, X_MAX], then u=log(x).
# This is appropriate if the file corresponds to a full sweep up to X_MAX.
x = np.linspace(X_MIN, X_MAX, n, dtype=np.float64)
u = np.log(x)

# ------------------------------------------------------------
# REMOVE GLOBAL LINEAR TREND IN LOG-SPACE
# ------------------------------------------------------------
coef = np.polyfit(u, angles, 1)
trend = np.polyval(coef, u)
residual = angles - trend

print("Linear fit in log-space:")
print(f"  slope     = {coef[0]:.12f}")
print(f"  intercept = {coef[1]:.12f}")

# ------------------------------------------------------------
# DIAGNOSTIC PLOT: ANGLES, TREND, RESIDUAL
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(u, angles, lw=1.0, label="angles")
axes[0].plot(u, trend, lw=2.0, label="linear trend")
axes[0].set_ylabel("angle")
axes[0].set_title("Angle signal and global linear trend in log-space")
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].plot(u, residual, lw=1.0, label="residual")
axes[1].set_xlabel(r"$u = \log x$")
axes[1].set_ylabel("residual")
axes[1].set_title("Detrended residual")
axes[1].grid(alpha=0.25)
axes[1].legend()

fig.tight_layout()
fig.savefig(PLOT1, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# LOMB-SCARGLE SPECTRUM
# ------------------------------------------------------------
omega = np.linspace(OMEGA_MIN, OMEGA_MAX, N_OMEGA, dtype=np.float64)

# lombscargle expects angular frequencies and zero-mean signal
resid0 = residual - residual.mean()
power = lombscargle(u, resid0, omega, normalize=False)

# normalize to max=1 for easier downstream use
if power.max() > 0:
    power_norm = power / power.max()
else:
    power_norm = power.copy()

# ------------------------------------------------------------
# SAVE theta_norm.csv and psi_norm.csv
# ------------------------------------------------------------
# For this pipeline, both are written from the same spectrum object.
# This is a practical bridge into your later scripts that expect both files.
# If later you build a true psi-based angles file, replace psi_norm.csv accordingly.
df_spec = pd.DataFrame({
    "omega": omega,
    "value": power_norm
})

df_spec.to_csv(OUT_THETA, index=False)
df_spec.to_csv(OUT_PSI, index=False)

print(f"Saved: {OUT_THETA}")
print(f"Saved: {OUT_PSI}")

# ------------------------------------------------------------
# DIAGNOSTIC PLOT: SPECTRUM WITH FIRST 30 ZETA ORDINATES
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(omega, power_norm, lw=2, label="Lomb-Scargle spectrum")

for g in FIRST30:
    if OMEGA_MIN <= g <= OMEGA_MAX:
        ax.axvline(g, color="green", alpha=0.18, lw=1)

ax.set_xlabel(r"frequency $\omega$")
ax.set_ylabel("normalized power")
ax.set_title("Lomb-Scargle spectrum of detrended angle signal")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(PLOT2, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# PRINT TOP PEAKS
# ------------------------------------------------------------
top_idx = np.argsort(power_norm)[-20:][::-1]
top_df = pd.DataFrame({
    "omega": omega[top_idx],
    "value": power_norm[top_idx]
}).sort_values("omega").reset_index(drop=True)

print("\nTop 20 spectral peaks:")
print(top_df.to_string(index=False))

# ------------------------------------------------------------
# OPTIONAL: SAVE TOP PEAKS
# ------------------------------------------------------------
top_df.to_csv("top20_peaks_from_angles.csv", index=False)
print("Saved: top20_peaks_from_angles.csv")

print("\nDone.")
