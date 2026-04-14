"""
Prime-Zeta Resonance Experiments
Generates publication-style figures from the 1e11 analysis outputs.
"""

# ============================================================
# PAPER FIGURES FOR 1e11 DATA (Colab-ready)
#
# This script generates 5 publication-style figures from:
#   - lhs_phase_1e11_loggrid.npz
#   - analysis_first20_scoreboard_1e11.csv
#   - analysis_first20_window_scores_1e11.csv
#
# Figures:
#   1) Global resonance spectrum with first 10 zeta ordinates
#   2) 10-panel zoom around the first 10 ordinates
#   3) Peak-gap plot for first 10 ordinates
#   4) Local random z-score plot for first 10 ordinates
#   5) Window-stability heatmaps (theta_norm / psi_norm)
#
# Output files:
#   - fig1_global_resonance_1e11.png
#   - fig2_zoom_first10_1e11.png
#   - fig3_gap_first10_1e11.png
#   - fig4_local_z_first10_1e11.png
#   - fig5_window_heatmap_first10_1e11.png
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
NPZ_FILE = "lhs_phase_1e11_loggrid.npz"
SCORE_FILE = "analysis_first20_scoreboard_1e11.csv"
WINDOW_FILE = "analysis_first20_window_scores_1e11.csv"

OUT_DIR = Path(".")
OUT_DIR.mkdir(exist_ok=True)

OMEGA_MIN = 10
OMEGA_MAX = 55

known_zeros_10 = np.array([
    14.134725141, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478
], dtype=np.float64)

# ------------------------------------------------------------
# PLOT STYLE
# ------------------------------------------------------------
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def linear_detrend(u_vals, y_vals):
    coeff = np.polyfit(u_vals, y_vals, 1)
    trend = np.polyval(coeff, u_vals)
    return y_vals - trend

def spectrum_in_log_space(u_vals, y_vals, apply_hann=True):
    y = np.asarray(y_vals, dtype=np.float64).copy()
    y = linear_detrend(u_vals, y)
    y = y - y.mean()

    if apply_hann:
        y = y * np.hanning(len(y))

    du = np.median(np.diff(u_vals))
    fft_vals = np.fft.rfft(y)
    omega = 2 * np.pi * np.fft.rfftfreq(len(y), d=du)
    amp = np.abs(fft_vals)
    amp_norm = amp / amp.max() if amp.max() > 0 else amp
    return omega, amp_norm

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
data = np.load(NPZ_FILE)
u = data["u_logx"].astype(np.float64)
theta_norm = data["theta_norm"].astype(np.float64)
psi_norm = data["psi_norm"].astype(np.float64)

score = pd.read_csv(SCORE_FILE)
win = pd.read_csv(WINDOW_FILE)

score10 = score.sort_values("k").head(10).copy()
win10 = win[win["k"] <= 10].copy()

omega_t, amp_t = spectrum_in_log_space(u, theta_norm)
omega_p, amp_p = spectrum_in_log_space(u, psi_norm)

# ------------------------------------------------------------
# FIGURE 1: GLOBAL RESONANCE SPECTRUM
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
mask_t = (omega_t >= OMEGA_MIN) & (omega_t <= OMEGA_MAX)
mask_p = (omega_p >= OMEGA_MIN) & (omega_p <= OMEGA_MAX)

ax.plot(omega_t[mask_t], amp_t[mask_t], lw=2, label="theta_norm")
ax.plot(omega_p[mask_p], amp_p[mask_p], lw=2, label="psi_norm")

for z in known_zeros_10:
    ax.axvline(z, color="green", alpha=0.25, lw=1)

ax.set_xlabel(r"Frequency $\omega$")
ax.set_ylabel("Normalized amplitude")
ax.set_title("Global resonance spectrum near the first 10 zeta ordinates")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_global_resonance_1e11.png", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# FIGURE 2: 10-PANEL ZOOM
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(16, 6), dpi=160)
axes = axes.ravel()

for i, z in enumerate(known_zeros_10):
    ax = axes[i]
    mt = (omega_t >= z - 0.8) & (omega_t <= z + 0.8)
    mp = (omega_p >= z - 0.8) & (omega_p <= z + 0.8)

    ax.plot(omega_t[mt], amp_t[mt], lw=2, label="theta")
    ax.plot(omega_p[mp], amp_p[mp], lw=2, label="psi")
    ax.axvline(z, color="green", alpha=0.6, lw=1.2)

    ax.set_title(rf"$\gamma_{{{i+1}}}$")
    ax.grid(alpha=0.25)

fig.suptitle("Local zoom around the first 10 zeta ordinates", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_zoom_first10_1e11.png", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# FIGURE 3: PEAK GAP PLOT
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(score10["k"], score10["theta_peak_gap"], marker="o", lw=2, label="theta gap")
ax.plot(score10["k"], score10["psi_peak_gap"], marker="o", lw=2, label="psi gap")
ax.set_xlabel(r"Zero index $k$")
ax.set_ylabel("Peak gap")
ax.set_title("Peak misalignment for the first 10 zeta ordinates")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_gap_first10_1e11.png", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# FIGURE 4: LOCAL RANDOM Z-SCORES
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(score10["k"] - 0.15, score10["theta_rand_z"], width=0.3, label="theta z")
ax.bar(score10["k"] + 0.15, score10["psi_rand_z"], width=0.3, label="psi z")
ax.set_xlabel(r"Zero index $k$")
ax.set_ylabel("Local random z-score")
ax.set_title("Local significance for the first 10 zeta ordinates")
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_local_z_first10_1e11.png", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# FIGURE 5: WINDOW-STABILITY HEATMAPS
# ------------------------------------------------------------
pivot_theta = win10[win10["signal"] == "theta_norm"].pivot(index="k", columns="x_max", values="rand_z")
pivot_psi = win10[win10["signal"] == "psi_norm"].pivot(index="k", columns="x_max", values="rand_z")

fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=160)

im1 = axes[0].imshow(pivot_theta.values, aspect="auto", origin="lower")
axes[0].set_title("theta_norm: local z-score heatmap")
axes[0].set_xlabel(r"$x_{\max}$ window")
axes[0].set_ylabel(r"Zero index $k$")
axes[0].set_yticks(range(len(pivot_theta.index)))
axes[0].set_yticklabels(pivot_theta.index)
axes[0].set_xticks(range(len(pivot_theta.columns)))
axes[0].set_xticklabels([f"{c:.0e}" for c in pivot_theta.columns], rotation=45, ha="right")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(pivot_psi.values, aspect="auto", origin="lower")
axes[1].set_title("psi_norm: local z-score heatmap")
axes[1].set_xlabel(r"$x_{\max}$ window")
axes[1].set_ylabel(r"Zero index $k$")
axes[1].set_yticks(range(len(pivot_psi.index)))
axes[1].set_yticklabels(pivot_psi.index)
axes[1].set_xticks(range(len(pivot_psi.columns)))
axes[1].set_xticklabels([f"{c:.0e}" for c in pivot_psi.columns], rotation=45, ha="right")
fig.colorbar(im2, ax=axes[1])

fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_window_heatmap_first10_1e11.png", bbox_inches="tight")
plt.show()

print("Saved:")
for fn in [
    "fig1_global_resonance_1e11.png",
    "fig2_zoom_first10_1e11.png",
    "fig3_gap_first10_1e11.png",
    "fig4_local_z_first10_1e11.png",
    "fig5_window_heatmap_first10_1e11.png",
]:
    print("-", fn)
