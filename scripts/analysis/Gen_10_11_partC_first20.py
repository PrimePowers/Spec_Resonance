"""
Prime-Zeta Resonance Experiments
Scores the first 20 zeta ordinates using local significance tests, jitter baselines, and consistency metrics.
"""

# ============================================================
# TEIL C: SCOREBOARD FÜR DIE ERSTEN 20 ZETA-ORDINATEN
# Für 1e11-Daten (oder allgemein NPZ-Datei mit theta_norm/psi_norm)
#
# Ziel:
#   - die ersten 20 Zeta-Ordinaten als Resonanzzentren testen
#   - lokaler Random-Test
#   - lokaler Jitter-Test
#   - Peak-Gap
#   - Fensterkonsistenz
#   - Meta-Score
#
# Standardmäßig auf:
#   lhs_phase_1e11_loggrid.npz
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# EINSTELLUNGEN
# ------------------------------------------------------------
DATA_FILE = "lhs_phase_1e11_loggrid.npz"

# erste 20 bekannte Zeta-Ordinaten
known_zeros_20 = np.array([
    14.134725141,
    21.022039639,
    25.010857580,
    30.424876126,
    32.935061588,
    37.586178159,
    40.918719012,
    43.327073281,
    48.005150881,
    49.773832478,
    52.970321478,
    56.446247697,
    59.347044003,
    60.831778525,
    65.112544048,
    67.079810529,
    69.546401711,
    72.067157674,
    75.704690699,
    77.144840069,
], dtype=np.float64)

HALF_WIDTH = 0.30
LOCAL_RANDOM_TRIALS = 400
LOCAL_JITTER_TRIALS = 300
LOCAL_JITTER = 0.25
PEAK_SEARCH_MIN = 10
PEAK_SEARCH_MAX = 80

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
    100_000_000_000
]

# ------------------------------------------------------------
# DATEI LADEN
# ------------------------------------------------------------
data = np.load(DATA_FILE)
x = data["x"].astype(np.int64)
u = data["u_logx"].astype(np.float64)
theta_norm = data["theta_norm"].astype(np.float64)
psi_norm = data["psi_norm"].astype(np.float64)

signals = {
    "theta_norm": theta_norm,
    "psi_norm": psi_norm,
}

# ------------------------------------------------------------
# HILFSFUNKTIONEN
# ------------------------------------------------------------
def linear_detrend(u_vals, y_vals):
    coeff = np.polyfit(u_vals, y_vals, 1)
    trend = np.polyval(coeff, u_vals)
    resid = y_vals - trend
    return coeff, trend, resid

def spectrum_in_log_space(u_vals, y_vals, detrend=True, apply_hann=True):
    y = np.asarray(y_vals, dtype=np.float64).copy()

    if detrend:
        _, _, y = linear_detrend(u_vals, y)

    y = y - y.mean()

    if apply_hann:
        window = np.hanning(len(y))
        y = y * window

    du = np.median(np.diff(u_vals))
    fft_vals = np.fft.rfft(y)
    omega = 2 * np.pi * np.fft.rfftfreq(len(y), d=du)
    amp = np.abs(fft_vals)
    power = amp ** 2
    amp_norm = amp / amp.max() if amp.max() > 0 else amp
    return omega, amp, amp_norm, power

def local_band_energy(omega, power, center, half_width):
    mask = (omega >= center - half_width) & (omega <= center + half_width)
    return np.sum(power[mask])

def local_random_test(omega, power, center, half_width, low=10, high=80, n_trials=400, seed=1234):
    rng = np.random.default_rng(seed)
    true_e = local_band_energy(omega, power, center, half_width)
    rand_centers = rng.uniform(low, high, size=n_trials)
    rand_e = np.array([local_band_energy(omega, power, rc, half_width) for rc in rand_centers])

    mu = rand_e.mean()
    sd = rand_e.std(ddof=1) if len(rand_e) > 1 else 0.0
    z = (true_e - mu) / sd if sd > 0 else np.nan
    p_emp = np.mean(rand_e >= true_e)

    return {
        "true_energy": true_e,
        "rand_mean": mu,
        "rand_std": sd,
        "rand_z": z,
        "rand_p": p_emp,
    }

def local_jitter_test(omega, power, center, half_width, jitter=0.25, n_trials=300, seed=1234):
    rng = np.random.default_rng(seed)
    true_e = local_band_energy(omega, power, center, half_width)
    jit_centers = center + rng.uniform(-jitter, jitter, size=n_trials)
    jit_e = np.array([local_band_energy(omega, power, jc, half_width) for jc in jit_centers])

    mu = jit_e.mean()
    sd = jit_e.std(ddof=1) if len(jit_e) > 1 else 0.0
    z = (true_e - mu) / sd if sd > 0 else np.nan
    p_emp = np.mean(jit_e >= true_e)

    return {
        "true_energy": true_e,
        "jitter_mean": mu,
        "jitter_std": sd,
        "jitter_z": z,
        "jitter_p": p_emp,
    }

def strongest_local_peak_gap(omega, amp_norm, center, local_half_window=0.8):
    mask = (omega >= center - local_half_window) & (omega <= center + local_half_window)
    om = omega[mask]
    am = amp_norm[mask]
    if len(om) == 0:
        return np.nan, np.nan
    idx = np.argmax(am)
    peak = om[idx]
    gap = abs(peak - center)
    return peak, gap

def score_from_components(rand_z, jitter_z, gap, p005_fraction, theta_psi_agreement_term):
    gap_term = 1.0 / (gap + 0.02) if np.isfinite(gap) else 0.0
    rand_term = max(rand_z, 0.0) if np.isfinite(rand_z) else 0.0
    jitter_term = max(jitter_z, 0.0) if np.isfinite(jitter_z) else 0.0
    p_term = 4.0 * p005_fraction
    agreement_term = theta_psi_agreement_term
    return rand_term + 0.75 * jitter_term + 0.15 * gap_term + p_term + agreement_term

# ------------------------------------------------------------
# GLOBALE SPEKTREN
# ------------------------------------------------------------
spectra = {}
for name, sig in signals.items():
    spectra[name] = spectrum_in_log_space(u, sig, detrend=True, apply_hann=True)

# ------------------------------------------------------------
# 1) LOKALE ANALYSE PRO NULLSTELLE UND SIGNAL
# ------------------------------------------------------------
all_rows = []

for sig_name, sig in signals.items():
    omega, amp, amp_norm, power = spectra[sig_name]

    for k, gamma in enumerate(known_zeros_20, start=1):
        rr = local_random_test(
            omega, power, gamma, HALF_WIDTH,
            low=PEAK_SEARCH_MIN, high=PEAK_SEARCH_MAX,
            n_trials=LOCAL_RANDOM_TRIALS,
            seed=1000 + k
        )
        jr = local_jitter_test(
            omega, power, gamma, HALF_WIDTH,
            jitter=LOCAL_JITTER,
            n_trials=LOCAL_JITTER_TRIALS,
            seed=2000 + k
        )
        peak_loc, peak_gap = strongest_local_peak_gap(
            omega, amp_norm, gamma, local_half_window=0.8
        )

        all_rows.append({
            "signal": sig_name,
            "k": k,
            "gamma": gamma,
            "local_energy": rr["true_energy"],
            "rand_z": rr["rand_z"],
            "rand_p": rr["rand_p"],
            "jitter_z": jr["jitter_z"],
            "jitter_p": jr["jitter_p"],
            "local_peak": peak_loc,
            "peak_gap": peak_gap,
        })

local_df = pd.DataFrame(all_rows)

# ------------------------------------------------------------
# 2) FENSTERKONSISTENZ PRO NULLSTELLE
# ------------------------------------------------------------
window_rows = []

for cap in WINDOW_CAPS:
    mask = x <= cap
    if mask.sum() < 512:
        continue

    u_w = u[mask]

    for sig_name, sig in signals.items():
        sig_w = sig[mask]
        omega, amp, amp_norm, power = spectrum_in_log_space(u_w, sig_w, detrend=True, apply_hann=True)

        for k, gamma in enumerate(known_zeros_20, start=1):
            rr = local_random_test(
                omega, power, gamma, HALF_WIDTH,
                low=PEAK_SEARCH_MIN, high=PEAK_SEARCH_MAX,
                n_trials=200,
                seed=3000 + k + int(cap % 1000)
            )
            peak_loc, peak_gap = strongest_local_peak_gap(
                omega, amp_norm, gamma, local_half_window=0.8
            )

            window_rows.append({
                "x_max": cap,
                "signal": sig_name,
                "k": k,
                "gamma": gamma,
                "rand_z": rr["rand_z"],
                "rand_p": rr["rand_p"],
                "peak_gap": peak_gap,
            })

window_df = pd.DataFrame(window_rows)

cons_rows = []
for sig_name in signals.keys():
    for k, gamma in enumerate(known_zeros_20, start=1):
        sub = window_df[(window_df["signal"] == sig_name) & (window_df["k"] == k)]
        if len(sub) == 0:
            continue

        p005_frac = np.mean(sub["rand_p"] <= 0.05)
        mean_gap = sub["peak_gap"].mean()
        median_gap = sub["peak_gap"].median()
        mean_z = sub["rand_z"].mean()

        cons_rows.append({
            "signal": sig_name,
            "k": k,
            "gamma": gamma,
            "window_count": len(sub),
            "p005_fraction": p005_frac,
            "mean_window_rand_z": mean_z,
            "mean_window_gap": mean_gap,
            "median_window_gap": median_gap,
        })

cons_df = pd.DataFrame(cons_rows)

# ------------------------------------------------------------
# 3) THETA/PSI-ZUSAMMENFÜHRUNG UND META-SCORE
# ------------------------------------------------------------
theta_local = local_df[local_df["signal"] == "theta_norm"].copy()
psi_local = local_df[local_df["signal"] == "psi_norm"].copy()

theta_cons = cons_df[cons_df["signal"] == "theta_norm"].copy()
psi_cons = cons_df[cons_df["signal"] == "psi_norm"].copy()

theta_local = theta_local.rename(columns={
    "local_energy": "theta_local_energy",
    "rand_z": "theta_rand_z",
    "rand_p": "theta_rand_p",
    "jitter_z": "theta_jitter_z",
    "jitter_p": "theta_jitter_p",
    "local_peak": "theta_local_peak",
    "peak_gap": "theta_peak_gap",
})
psi_local = psi_local.rename(columns={
    "local_energy": "psi_local_energy",
    "rand_z": "psi_rand_z",
    "rand_p": "psi_rand_p",
    "jitter_z": "psi_jitter_z",
    "jitter_p": "psi_jitter_p",
    "local_peak": "psi_local_peak",
    "peak_gap": "psi_peak_gap",
})

theta_cons = theta_cons.rename(columns={
    "window_count": "theta_window_count",
    "p005_fraction": "theta_p005_fraction",
    "mean_window_rand_z": "theta_mean_window_rand_z",
    "mean_window_gap": "theta_mean_window_gap",
    "median_window_gap": "theta_median_window_gap",
})
psi_cons = psi_cons.rename(columns={
    "window_count": "psi_window_count",
    "p005_fraction": "psi_p005_fraction",
    "mean_window_rand_z": "psi_mean_window_rand_z",
    "mean_window_gap": "psi_mean_window_gap",
    "median_window_gap": "psi_median_window_gap",
})

merged = theta_local.merge(
    psi_local[["k","gamma","psi_local_energy","psi_rand_z","psi_rand_p","psi_jitter_z","psi_jitter_p","psi_local_peak","psi_peak_gap"]],
    on=["k","gamma"],
    how="inner"
)

merged = merged.merge(
    theta_cons[["k","gamma","theta_window_count","theta_p005_fraction","theta_mean_window_rand_z","theta_mean_window_gap","theta_median_window_gap"]],
    on=["k","gamma"],
    how="left"
)

merged = merged.merge(
    psi_cons[["k","gamma","psi_window_count","psi_p005_fraction","psi_mean_window_rand_z","psi_mean_window_gap","psi_median_window_gap"]],
    on=["k","gamma"],
    how="left"
)

agreement_rows = []
for _, row in merged.iterrows():
    z_diff = abs(row["theta_rand_z"] - row["psi_rand_z"])
    gap_diff = abs(row["theta_peak_gap"] - row["psi_peak_gap"])

    agreement_term = 1.0 / (1.0 + z_diff + 3.0 * gap_diff)

    avg_rand_z = np.nanmean([row["theta_rand_z"], row["psi_rand_z"]])
    avg_jitter_z = np.nanmean([row["theta_jitter_z"], row["psi_jitter_z"]])
    avg_gap = np.nanmean([row["theta_peak_gap"], row["psi_peak_gap"]])
    avg_p005 = np.nanmean([row["theta_p005_fraction"], row["psi_p005_fraction"]])

    score = score_from_components(
        rand_z=avg_rand_z,
        jitter_z=avg_jitter_z,
        gap=avg_gap,
        p005_fraction=avg_p005,
        theta_psi_agreement_term=agreement_term
    )

    agreement_rows.append({
        "k": row["k"],
        "gamma": row["gamma"],
        "theta_rand_z": row["theta_rand_z"],
        "psi_rand_z": row["psi_rand_z"],
        "theta_jitter_z": row["theta_jitter_z"],
        "psi_jitter_z": row["psi_jitter_z"],
        "theta_peak_gap": row["theta_peak_gap"],
        "psi_peak_gap": row["psi_peak_gap"],
        "theta_p005_fraction": row["theta_p005_fraction"],
        "psi_p005_fraction": row["psi_p005_fraction"],
        "agreement_term": agreement_term,
        "composite_score": score,
    })

scoreboard_df = pd.DataFrame(agreement_rows).sort_values("composite_score", ascending=False).reset_index(drop=True)

print("\n===== LOKALE ANALYSE (erste 20) =====")
print(local_df.to_string(index=False))

print("\n===== FENSTERKONSISTENZ (erste 20) =====")
print(cons_df.to_string(index=False))

print("\n===== FINALES SCOREBOARD (erste 20) =====")
print(scoreboard_df.to_string(index=False))

# ------------------------------------------------------------
# 4) VISUALISIERUNGEN
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.bar(scoreboard_df["k"].astype(str), scoreboard_df["composite_score"])
plt.title("Composite localization score for the first 20 zeta ordinates")
plt.xlabel("Zero index k")
plt.ylabel("Composite score")
plt.grid(axis="y", alpha=0.3)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(scoreboard_df["k"], scoreboard_df["theta_peak_gap"], marker="o", label="theta peak gap")
plt.plot(scoreboard_df["k"], scoreboard_df["psi_peak_gap"], marker="o", label="psi peak gap")
plt.title("Peak gap vs zero index")
plt.xlabel("Zero index k")
plt.ylabel("Gap")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(scoreboard_df["k"], scoreboard_df["theta_rand_z"], marker="o", label="theta local rand z")
plt.plot(scoreboard_df["k"], scoreboard_df["psi_rand_z"], marker="o", label="psi local rand z")
plt.title("Local random z-score vs zero index")
plt.xlabel("Zero index k")
plt.ylabel("Local random z-score")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# ------------------------------------------------------------
# 5) CSV EXPORT
# ------------------------------------------------------------
local_df.to_csv("analysis_first20_local_scores_1e11.csv", index=False)
window_df.to_csv("analysis_first20_window_scores_1e11.csv", index=False)
cons_df.to_csv("analysis_first20_consistency_1e11.csv", index=False)
scoreboard_df.to_csv("analysis_first20_scoreboard_1e11.csv", index=False)

print("\nGespeichert:")
print("- analysis_first20_local_scores_1e11.csv")
print("- analysis_first20_window_scores_1e11.csv")
print("- analysis_first20_consistency_1e11.csv")
print("- analysis_first20_scoreboard_1e11.csv")
