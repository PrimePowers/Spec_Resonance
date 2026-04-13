"""
Prime-Zeta Resonance Experiments
Initial exploratory FFT-style analysis of the first 30 candidate ordinates in LHS phase data.
"""

# ============================================================
# CLEAN PIPELINE: FIRST 30 ZETA ORDINATES FROM lhs_phase_1e11_loggrid.npz
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
])

print("Loading NPZ file...")
data = np.load(FILE)

theta = data["theta_norm"]
psi = data["psi_norm"]
u = data["u"]

print(f"Loaded {len(theta)} points")

# FFT (fast & robust)
def compute_fft(signal, u):
    du = u[1] - u[0]
    fft_vals = np.fft.rfft(signal - signal.mean())
    omega = 2*np.pi*np.fft.rfftfreq(len(signal), d=du)
    power = np.abs(fft_vals)**2
    power /= power.max()
    return omega, power

omega, power_theta = compute_fft(theta, u)
_, power_psi = compute_fft(psi, u)

# restrict range
mask = (omega > 10) & (omega < 110)
omega = omega[mask]
power_theta = power_theta[mask]
power_psi = power_psi[mask]

# peaks
peaks, _ = find_peaks(power_theta, height=0.05, distance=10)

peak_omega = omega[peaks]
peak_val = power_theta[peaks]

df = pd.DataFrame({
    "omega": peak_omega,
    "power": peak_val
})

df = df.sort_values("power", ascending=False).head(30).sort_values("omega")
df.to_csv("top30_peaks_lhs_phase.csv", index=False)

print("\nTop peaks:")
print(df)

# plot
plt.figure(figsize=(12,6))
plt.plot(omega, power_theta, label="theta_norm")
plt.plot(omega, power_psi, label="psi_norm", alpha=0.7)

for g in FIRST30:
    plt.axvline(g, color="green", alpha=0.2)

plt.title("FFT Spectrum vs first 30 zeta ordinates")
plt.xlabel("omega")
plt.ylabel("power")
plt.legend()
plt.grid()
plt.savefig("fft_first30_lhs_phase.png", dpi=200)
plt.show()

print("\nSaved:")
print("- top30_peaks_lhs_phase.csv")
print("- fft_first30_lhs_phase.png")
