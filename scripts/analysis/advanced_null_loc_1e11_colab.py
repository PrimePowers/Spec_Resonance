"""
Prime-Zeta Resonance Experiments
Advanced robustness, localization, and blind-cluster diagnostics for the 1e11 study.
"""

# ============================================================
# ADVANCED 1e11 ANALYSIS SCRIPT
# ------------------------------------------------------------
# This Colab-ready script adds the next-level analyses:
#
# (1) Error vs. scale:
#     - mean gap(x_max)
#     - median gap(x_max)
#
# (2) Frequency regression / convergence:
#     - omega_peak_k(x_max) for the first 10 zeta ordinates
#     - absolute and relative error vs scale
#
# (3) Blind detection without using the zeta list for peak finding:
#     - top peaks per window
#     - clustering across windows
#     - comparison to the first 10 known ordinates only afterwards
#
# (4) Normalized error analysis:
#     - |omega_peak - gamma| / gamma
#
# INPUT:
#   - lhs_phase_1e11_loggrid.npz
#
# OPTIONAL INPUT:
#   - analysis_first20_window_scores_1e11.csv (not required)
#
# OUTPUT:
#   - CSV files
#   - PNG figures
#
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
NPZ_FILE = "lhs_phase_1e11_loggrid.npz"
OUT_DIR = Path(".")
OUT_DIR.mkdir(exist_ok=True)

SIGNALS_TO_ANALYZE = ["theta_norm", "psi_norm"]

WINDOW_CAPS = [
    500_000,
    1_000_000,
    1_500_000,
    2_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    100_000_000,
    300_000_000,
    500_000_000,
    1_000_000_000,
    3_000_000_000,
    5_000_000_000,
    10_000_000_000,
    30_000_000_000,
    50_000_000_000,
    100_000_000_000,
]

FIRST10 = np.array([
    14.134725141, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478
], dtype=np.float64)

OMEGA_MIN = 10.0
OMEGA_MAX = 80.0
TOP_N_BLIND = 20
MATCH_HALF_WINDOW = 0.8
TOP_N_NEAR_ZERO = 3  # use strongest local peak within local window

# For blind clustering:
CLUSTER_TOL = 0.20

# Plot style
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
    return omega, amp, amp_norm


def top_peaks_global(omega, amp_norm, n_peaks=20, omega_min=10, omega_max=80):
    mask = (omega >= omega_min) & (omega <= omega_max)
    om = omega[mask]
    am = amp_norm[mask]
    if len(om) == 0:
        return pd.DataFrame(columns=["peak_omega", "amp_norm"])

    idx = np.argsort(am)[-n_peaks:][::-1]
    out = pd.DataFrame({
        "peak_omega": om[idx],
        "amp_norm": am[idx]
    }).sort_values("peak_omega").reset_index(drop=True)
    return out


def strongest_peak_near_zero(omega, amp_norm, gamma, local_half_window=0.8):
    mask = (omega >= gamma - local_half_window) & (omega <= gamma + local_half_window)
    om = omega[mask]
    am = amp_norm[mask]
    if len(om) == 0:
        return np.nan, np.nan
    idx = np.argmax(am)
    peak = om[idx]
    gap = abs(peak - gamma)
    return peak, gap


def blind_cluster_peaks(all_peak_rows, tol=0.20):
    """
    Greedy clustering of peak frequencies across windows/signals.
    Returns DataFrame with cluster center, support count, mean amplitude.
    """
    rows = sorted(all_peak_rows, key=lambda r: r["peak_omega"])
    clusters = []

    for row in rows:
        w = row["peak_omega"]
        placed = False
        for cl in clusters:
            if abs(w - cl["center"]) <= tol:
                cl["members"].append(row)
                vals = [m["peak_omega"] for m in cl["members"]]
                cl["center"] = float(np.mean(vals))
                placed = True
                break
        if not placed:
            clusters.append({
                "center": float(w),
                "members": [row]
            })

    out_rows = []
    for i, cl in enumerate(clusters, start=1):
        amps = [m["amp_norm"] for m in cl["members"]]
        wins = [m["x_max"] for m in cl["members"]]
        sigs = [m["signal"] for m in cl["members"]]
        out_rows.append({
            "cluster_id": i,
            "cluster_center": cl["center"],
            "support_count": len(cl["members"]),
            "mean_amp": float(np.mean(amps)),
            "max_amp": float(np.max(amps)),
            "unique_windows": len(set(wins)),
            "unique_signals": len(set(sigs)),
        })

    out = pd.DataFrame(out_rows).sort_values(
        ["support_count", "mean_amp", "max_amp"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    return out


def nearest_known_zero(value, zeros):
    idx = int(np.argmin(np.abs(zeros - value)))
    return zeros[idx], abs(value - zeros[idx]), idx + 1


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
data = np.load(NPZ_FILE)
x = data["x"].astype(np.int64)
u = data["u_logx"].astype(np.float64)

signals = {
    "theta_norm": data["theta_norm"].astype(np.float64),
    "psi_norm": data["psi_norm"].astype(np.float64),
}

# ------------------------------------------------------------
# MAIN ANALYSIS OVER WINDOWS
# ------------------------------------------------------------
tracking_rows = []
blind_peak_rows = []

for cap in WINDOW_CAPS:
    mask = x <= cap
    if mask.sum() < 512:
        continue

    u_w = u[mask]

    for sig_name in SIGNALS_TO_ANALYZE:
        sig_w = signals[sig_name][mask]
        omega, amp, amp_norm = spectrum_in_log_space(u_w, sig_w, apply_hann=True)

        # Track first 10 zeros using local window around each gamma
        for k, gamma in enumerate(FIRST10, start=1):
            peak, gap = strongest_peak_near_zero(
                omega, amp_norm, gamma, local_half_window=MATCH_HALF_WINDOW
            )
            rel_gap = gap / gamma if np.isfinite(gap) else np.nan
            tracking_rows.append({
                "x_max": cap,
                "signal": sig_name,
                "k": k,
                "gamma": gamma,
                "peak_omega": peak,
                "abs_gap": gap,
                "rel_gap": rel_gap,
            })

        # Blind peaks: top peaks without using zeta list
        peaks = top_peaks_global(
            omega, amp_norm,
            n_peaks=TOP_N_BLIND,
            omega_min=OMEGA_MIN,
            omega_max=OMEGA_MAX
        )
        for _, row in peaks.iterrows():
            blind_peak_rows.append({
                "x_max": cap,
                "signal": sig_name,
                "peak_omega": float(row["peak_omega"]),
                "amp_norm": float(row["amp_norm"]),
            })

tracking_df = pd.DataFrame(tracking_rows)
blind_df = pd.DataFrame(blind_peak_rows)

tracking_df.to_csv(OUT_DIR / "advanced_tracking_first10_1e11.csv", index=False)
blind_df.to_csv(OUT_DIR / "advanced_blind_peaks_1e11.csv", index=False)

# ------------------------------------------------------------
# (1) ERROR VS SCALE
# ------------------------------------------------------------
err_rows = []
for sig_name in SIGNALS_TO_ANALYZE:
    sub = tracking_df[tracking_df["signal"] == sig_name]
    for cap, group in sub.groupby("x_max"):
        err_rows.append({
            "x_max": cap,
            "signal": sig_name,
            "mean_abs_gap": group["abs_gap"].mean(),
            "median_abs_gap": group["abs_gap"].median(),
            "mean_rel_gap": group["rel_gap"].mean(),
            "median_rel_gap": group["rel_gap"].median(),
        })

err_df = pd.DataFrame(err_rows).sort_values(["signal", "x_max"])
err_df.to_csv(OUT_DIR / "advanced_error_vs_scale_1e11.csv", index=False)

# Plot mean + median abs gap
fig, ax = plt.subplots(figsize=(11, 6))
for sig_name, marker in [("theta_norm", "o"), ("psi_norm", "s")]:
    sub = err_df[err_df["signal"] == sig_name].sort_values("x_max")
    ax.plot(sub["x_max"], sub["median_abs_gap"], marker=marker, lw=2, label=rf"{sig_name}: median gap")
    ax.plot(sub["x_max"], sub["mean_abs_gap"], marker=marker, lw=1.5, linestyle="--", label=rf"{sig_name}: mean gap")

ax.set_xscale("log")
ax.set_xlabel(r"$x_{\max}$")
ax.set_ylabel("Gap")
ax.set_title("Mean and median gap vs scale (first 10 ordinates)")
ax.grid(alpha=0.3, which="both")
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_adv1_error_vs_scale_1e11.png", bbox_inches="tight")
plt.show()

# Plot relative error
fig, ax = plt.subplots(figsize=(11, 6))
for sig_name, marker in [("theta_norm", "o"), ("psi_norm", "s")]:
    sub = err_df[err_df["signal"] == sig_name].sort_values("x_max")
    ax.plot(sub["x_max"], sub["median_rel_gap"], marker=marker, lw=2, label=rf"{sig_name}: median rel gap")
    ax.plot(sub["x_max"], sub["mean_rel_gap"], marker=marker, lw=1.5, linestyle="--", label=rf"{sig_name}: mean rel gap")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$x_{\max}$")
ax.set_ylabel(r"Relative gap $|\omega-\gamma|/\gamma$")
ax.set_title("Relative error vs scale (first 10 ordinates)")
ax.grid(alpha=0.3, which="both")
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_adv2_relative_error_vs_scale_1e11.png", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# (2) FREQUENCY REGRESSION / CONVERGENCE PLOTS
# ------------------------------------------------------------
# For each of the first 10 zeros, plot peak_omega(x_max)
for sig_name in SIGNALS_TO_ANALYZE:
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), dpi=160)
    axes = axes.ravel()

    sub_all = tracking_df[tracking_df["signal"] == sig_name]
    for k in range(1, 11):
        ax = axes[k - 1]
        sub = sub_all[sub_all["k"] == k].sort_values("x_max")
        gamma = sub["gamma"].iloc[0]

        ax.plot(sub["x_max"], sub["peak_omega"], marker="o", lw=1.8)
        ax.axhline(gamma, color="green", alpha=0.7, lw=1.2)
        ax.set_xscale("log")
        ax.set_title(rf"$\gamma_{{{k}}}$")
        ax.grid(alpha=0.25)

    fig.suptitle(f"Peak frequency convergence vs scale ({sig_name})", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_adv3_peak_convergence_{sig_name}_1e11.png", bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------
# (3) BLIND DETECTION WITHOUT USING ZETA LIST
# ------------------------------------------------------------
clusters = blind_cluster_peaks(blind_peak_rows, tol=CLUSTER_TOL)

# Compare top blind clusters to known zeros only afterwards
cluster_rows = []
for _, row in clusters.head(30).iterrows():
    z, gap, k = nearest_known_zero(row["cluster_center"], FIRST10)
    cluster_rows.append({
        "cluster_id": row["cluster_id"],
        "cluster_center": row["cluster_center"],
        "support_count": row["support_count"],
        "mean_amp": row["mean_amp"],
        "max_amp": row["max_amp"],
        "unique_windows": row["unique_windows"],
        "unique_signals": row["unique_signals"],
        "nearest_gamma": z,
        "nearest_k": k,
        "gap_to_nearest_first10": gap,
    })

cluster_compare_df = pd.DataFrame(cluster_rows)
clusters.to_csv(OUT_DIR / "advanced_blind_clusters_1e11.csv", index=False)
cluster_compare_df.to_csv(OUT_DIR / "advanced_blind_clusters_vs_first10_1e11.csv", index=False)

print("\n=== TOP BLIND CLUSTERS (compared afterwards to first 10 zeros) ===")
print(cluster_compare_df.to_string(index=False))

# Plot top blind clusters
top_blind = cluster_compare_df.head(15)
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(top_blind["cluster_center"], top_blind["support_count"], s=80)
for _, row in top_blind.iterrows():
    ax.annotate(
        f"c{int(row['cluster_id'])}\n→γ{int(row['nearest_k'])}",
        (row["cluster_center"], row["support_count"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=9
    )

for z in FIRST10:
    ax.axvline(z, color="green", alpha=0.2, lw=1)

ax.set_xlabel(r"Blind cluster center $\omega$")
ax.set_ylabel("Support count across windows/signals")
ax.set_title("Blind peak clustering across windows")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_adv4_blind_clusters_1e11.png", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# (4) NORMALIZED ERROR BY ZERO INDEX
# ------------------------------------------------------------
# Use largest window only for a clean summary
max_cap = tracking_df["x_max"].max()
largest = tracking_df[tracking_df["x_max"] == max_cap].copy()

fig, ax = plt.subplots(figsize=(10, 5))
for sig_name, marker in [("theta_norm", "o"), ("psi_norm", "s")]:
    sub = largest[largest["signal"] == sig_name].sort_values("k")
    ax.plot(sub["k"], sub["rel_gap"], marker=marker, lw=2, label=sig_name)

ax.set_yscale("log")
ax.set_xlabel("Zero index k")
ax.set_ylabel(r"Relative gap $|\omega-\gamma|/\gamma$")
ax.set_title(rf"Relative localization error at largest scale ($x_{{\max}}={int(max_cap):.0e}$)")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_adv5_relative_error_by_index_1e11.png", bbox_inches="tight")
plt.show()

# Also absolute gap by index at largest scale
fig, ax = plt.subplots(figsize=(10, 5))
for sig_name, marker in [("theta_norm", "o"), ("psi_norm", "s")]:
    sub = largest[largest["signal"] == sig_name].sort_values("k")
    ax.plot(sub["k"], sub["abs_gap"], marker=marker, lw=2, label=sig_name)

ax.set_yscale("log")
ax.set_xlabel("Zero index k")
ax.set_ylabel(r"Absolute gap $|\omega-\gamma|$")
ax.set_title(rf"Absolute localization error at largest scale ($x_{{\max}}={int(max_cap):.0e}$)")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_adv6_absolute_error_by_index_1e11.png", bbox_inches="tight")
plt.show()

print("\nSaved files:")
for fn in [
    "advanced_tracking_first10_1e11.csv",
    "advanced_blind_peaks_1e11.csv",
    "advanced_error_vs_scale_1e11.csv",
    "advanced_blind_clusters_1e11.csv",
    "advanced_blind_clusters_vs_first10_1e11.csv",
    "fig_adv1_error_vs_scale_1e11.png",
    "fig_adv2_relative_error_vs_scale_1e11.png",
    "fig_adv3_peak_convergence_theta_norm_1e11.png",
    "fig_adv3_peak_convergence_psi_norm_1e11.png",
    "fig_adv4_blind_clusters_1e11.png",
    "fig_adv5_relative_error_by_index_1e11.png",
    "fig_adv6_absolute_error_by_index_1e11.png",
]:
    print("-", fn)
