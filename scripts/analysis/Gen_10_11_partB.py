"""
Prime-Zeta Resonance Experiments
Analyzes the 1e11 dataset with global resonance scoring, peak extraction, and stability diagnostics.
"""

# ============================================================
# TEIL B: RESONANZANALYSE FÜR 1e11
# Benötigt die von Teil A erzeugte Datei:
#    lhs_phase_1e11_loggrid.npz
#
# Lean version:
#   Schwerpunkt auf theta_norm und psi_norm
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# DATEI LADEN
# ------------------------------------------------------------
DATA_FILE = "lhs_phase_1e11_loggrid.npz"
data = np.load(DATA_FILE)

x = data["x"].astype(np.int64)
u = data["u_logx"].astype(np.float64)

theta_norm = data["theta_norm"].astype(np.float64)
psi_norm = data["psi_norm"].astype(np.float64)

# ------------------------------------------------------------
# BEKANNTE ZETA-ORDINATEN (erste 10)
# ------------------------------------------------------------
known_zeros = np.array([
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
], dtype=np.float64)

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
    power = amp**2

    amp_norm = amp / amp.max() if amp.max() > 0 else amp
    return omega, amp, amp_norm, power

def top_peaks(omega, amp_norm, n_peaks=20, omega_min=10, omega_max=55):
    mask = (omega >= omega_min) & (omega <= omega_max)
    om = omega[mask]
    am = amp_norm[mask]

    if len(om) == 0:
        return pd.DataFrame(columns=["peak_omega", "amp_norm"])

    idx = np.argsort(am)[-n_peaks:][::-1]
    peaks = pd.DataFrame({
        "peak_omega": om[idx],
        "amp_norm": am[idx]
    }).sort_values("peak_omega").reset_index(drop=True)
    return peaks

def attach_nearest_zero(peaks_df, zeros):
    if len(peaks_df) == 0:
        return peaks_df

    nearest = []
    gaps = []
    for w in peaks_df["peak_omega"].values:
        z = zeros[np.argmin(np.abs(zeros - w))]
        nearest.append(z)
        gaps.append(abs(w - z))

    out = peaks_df.copy()
    out["nearest_zero"] = nearest
    out["abs_gap"] = gaps
    return out

def band_energy(omega, power, centers, half_width=0.35):
    total = 0.0
    for c in centers:
        mask = (omega >= c - half_width) & (omega <= c + half_width)
        total += np.sum(power[mask])
    return total

def random_frequency_sets(low, high, n_freq, n_trials=400, seed=1234):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_trials, n_freq))

def random_band_test(omega, power, zeros, half_width=0.35,
                     n_trials=400, low=10, high=55, seed=1234):
    true_e = band_energy(omega, power, zeros, half_width=half_width)

    rand_sets = random_frequency_sets(low, high, len(zeros), n_trials=n_trials, seed=seed)
    rand_e = np.array([band_energy(omega, power, rs, half_width=half_width) for rs in rand_sets])

    mu = rand_e.mean()
    sd = rand_e.std(ddof=1) if len(rand_e) > 1 else 0.0
    z = (true_e - mu) / sd if sd > 0 else np.nan
    p_emp = np.mean(rand_e >= true_e)

    return {
        "true_energy": true_e,
        "rand_mean": mu,
        "rand_std": sd,
        "rand_z": z,
        "emp_p": p_emp,
    }

def shift_test(omega, power, zeros, shifts=np.linspace(-3, 3, 121), half_width=0.35):
    zero_e = band_energy(omega, power, zeros, half_width=half_width)
    energies = []
    for s in shifts:
        e = band_energy(omega, power, zeros + s, half_width=half_width)
        energies.append(e)
    energies = np.array(energies)

    best_idx = np.argmax(energies)
    rank_zero = 1 + np.sum(energies > zero_e)

    return {
        "best_shift": shifts[best_idx],
        "best_energy": energies[best_idx],
        "zero_shift_energy": zero_e,
        "zero_shift_rank": int(rank_zero),
        "shifts": shifts,
        "energies": energies
    }

def jittered_zero_sets(zeros, jitter=0.2, n_trials=400, seed=1234):
    rng = np.random.default_rng(seed)
    jitter_sets = []
    for _ in range(n_trials):
        eps = rng.uniform(-jitter, jitter, size=len(zeros))
        jitter_sets.append(zeros + eps)
    return np.array(jitter_sets)

def jitter_test(omega, power, zeros, jitter=0.2, half_width=0.35, n_trials=400, seed=1234):
    true_e = band_energy(omega, power, zeros, half_width=half_width)
    jit_sets = jittered_zero_sets(zeros, jitter=jitter, n_trials=n_trials, seed=seed)
    jit_e = np.array([band_energy(omega, power, js, half_width=half_width) for js in jit_sets])

    mu = jit_e.mean()
    sd = jit_e.std(ddof=1) if len(jit_e) > 1 else 0.0
    z = (true_e - mu) / sd if sd > 0 else np.nan
    p_emp = np.mean(jit_e >= true_e)

    return {
        "true_energy": true_e,
        "jitter_mean": mu,
        "jitter_std": sd,
        "jitter_z": z,
        "emp_p": p_emp,
    }

def moving_average_complex(z, window=151):
    if window < 3:
        return z
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window, dtype=np.float64) / window
    zr = np.convolve(np.real(z), kernel, mode="same")
    zi = np.convolve(np.imag(z), kernel, mode="same")
    return zr + 1j * zi

def demodulated_phase(u_vals, y_vals, gamma, smooth_window=151):
    y = np.asarray(y_vals, dtype=np.float64)
    y = y - np.mean(y)
    z = y * np.exp(-1j * gamma * u_vals)
    z_s = moving_average_complex(z, window=smooth_window)
    phase = np.unwrap(np.angle(z_s))
    amp = np.abs(z_s)
    return phase, amp

# ------------------------------------------------------------
# SIGNALE
# ------------------------------------------------------------
signals = {
    "theta_norm": theta_norm,
    "psi_norm": psi_norm,
}

# ------------------------------------------------------------
# 1) GLOBALE AUSWERTUNG
# ------------------------------------------------------------
summary_rows = []
peak_tables = {}
spectra = {}

for name, sig in signals.items():
    omega, amp, amp_norm, power = spectrum_in_log_space(u, sig, detrend=True, apply_hann=True)
    spectra[name] = (omega, amp, amp_norm, power)

    peaks = top_peaks(omega, amp_norm, n_peaks=20, omega_min=10, omega_max=55)
    peaks = attach_nearest_zero(peaks, known_zeros)
    peak_tables[name] = peaks

    rb = random_band_test(omega, power, known_zeros, half_width=0.35, n_trials=400, low=10, high=55, seed=2026)
    st = shift_test(omega, power, known_zeros, shifts=np.linspace(-3, 3, 121), half_width=0.35)
    jt02 = jitter_test(omega, power, known_zeros, jitter=0.2, half_width=0.35, n_trials=300, seed=2027)
    jt04 = jitter_test(omega, power, known_zeros, jitter=0.4, half_width=0.35, n_trials=300, seed=2028)
    jt06 = jitter_test(omega, power, known_zeros, jitter=0.6, half_width=0.35, n_trials=300, seed=2029)

    summary_rows.append({
        "signal": name,
        "true_band_energy": rb["true_energy"],
        "rand_mean": rb["rand_mean"],
        "rand_std": rb["rand_std"],
        "rand_z": rb["rand_z"],
        "rand_emp_p": rb["emp_p"],
        "best_shift": st["best_shift"],
        "zero_shift_rank": st["zero_shift_rank"],
        "jitter02_z": jt02["jitter_z"],
        "jitter02_p": jt02["emp_p"],
        "jitter04_z": jt04["jitter_z"],
        "jitter04_p": jt04["emp_p"],
        "jitter06_z": jt06["jitter_z"],
        "jitter06_p": jt06["emp_p"],
    })

summary_df = pd.DataFrame(summary_rows).sort_values("rand_z", ascending=False)
print("\n===== GLOBALE AUSWERTUNG 1e11 =====")
print(summary_df.to_string(index=False))

# ------------------------------------------------------------
# 2) PEAK-TABELLEN
# ------------------------------------------------------------
for name, peaks in peak_tables.items():
    print(f"\n===== TOP-20 PEAKS: {name} =====")
    print(peaks.to_string(index=False))

# ------------------------------------------------------------
# 3) FENSTERSTABILITÄT BIS 1e11
# ------------------------------------------------------------
window_caps = [
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

window_rows = []

for cap in window_caps:
    mask = x <= cap
    if mask.sum() < 1024:
        continue

    u_w = u[mask]
    for name in ["theta_norm", "psi_norm"]:
        sig = signals[name][mask]
        omega, amp, amp_norm, power = spectrum_in_log_space(u_w, sig, detrend=True, apply_hann=True)
        peaks = top_peaks(omega, amp_norm, n_peaks=20, omega_min=10, omega_max=55)
        peaks = attach_nearest_zero(peaks, known_zeros)

        median_gap = peaks["abs_gap"].median() if len(peaks) else np.nan
        mean_gap = peaks["abs_gap"].mean() if len(peaks) else np.nan
        best_gap = peaks["abs_gap"].min() if len(peaks) else np.nan

        rb = random_band_test(omega, power, known_zeros, half_width=0.35, n_trials=250, low=10, high=55, seed=2030)
        st = shift_test(omega, power, known_zeros, shifts=np.linspace(-3, 3, 121), half_width=0.35)

        window_rows.append({
            "x_max": cap,
            "signal": name,
            "median_gap": median_gap,
            "mean_gap": mean_gap,
            "best_gap": best_gap,
            "rand_z": rb["rand_z"],
            "emp_p": rb["emp_p"],
            "best_shift": st["best_shift"],
            "zero_shift_rank": st["zero_shift_rank"],
        })

window_df = pd.DataFrame(window_rows)
print("\n===== FENSTERSTABILITÄT =====")
print(window_df.to_string(index=False))

# ------------------------------------------------------------
# 4) DEMODULIERTE PHASE FÜR gamma_1, gamma_2, gamma_3
# ------------------------------------------------------------
gammas_to_check = known_zeros[:3]

for name in ["theta_norm", "psi_norm"]:
    sig = signals[name]

    plt.figure(figsize=(12, 8))
    for i, gamma in enumerate(gammas_to_check, start=1):
        phase, amp_demod = demodulated_phase(u, sig, gamma, smooth_window=301)

        plt.subplot(3, 1, i)
        plt.plot(u, phase, lw=1.2, label=f"{name}: demodulated phase @ gamma={gamma:.6f}")
        plt.ylabel("phase")
        plt.grid(alpha=0.3)
        plt.legend()

    plt.suptitle(f"Demodulated local phase for {name}")
    plt.xlabel("u = log x")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    for i, gamma in enumerate(gammas_to_check, start=1):
        phase, amp_demod = demodulated_phase(u, sig, gamma, smooth_window=301)

        plt.subplot(3, 1, i)
        plt.plot(u, amp_demod, lw=1.2, label=f"{name}: demodulated amplitude @ gamma={gamma:.6f}")
        plt.ylabel("amplitude")
        plt.grid(alpha=0.3)
        plt.legend()

    plt.suptitle(f"Demodulated local amplitude for {name}")
    plt.xlabel("u = log x")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 5) SPEKTRALPLOTS MIT ZETA-LINIEN
# ------------------------------------------------------------
plt.figure(figsize=(14, 7))
for name, color in [("theta_norm", "tab:blue"), ("psi_norm", "tab:red")]:
    omega, amp, amp_norm, power = spectra[name]
    mask = (omega >= 10) & (omega <= 55)
    plt.plot(omega[mask], amp_norm[mask], lw=2, color=color, label=name)

for z in known_zeros:
    plt.axvline(z, color="green", alpha=0.25, lw=1)

plt.title("Normalized spectral amplitude vs known zeta ordinates (1e11)")
plt.xlabel("omega")
plt.ylabel("normalized amplitude")
plt.grid(alpha=0.25)
plt.legend()
plt.show()

# ------------------------------------------------------------
# 6) SHIFT-PLOTS
# ------------------------------------------------------------
for name in ["theta_norm", "psi_norm"]:
    omega, amp, amp_norm, power = spectra[name]
    st = shift_test(omega, power, known_zeros, shifts=np.linspace(-3, 3, 121), half_width=0.35)

    plt.figure(figsize=(10, 5))
    plt.plot(st["shifts"], st["energies"], lw=2)
    plt.axvline(0.0, color="red", linestyle="--", alpha=0.7, label="zero shift")
    plt.axvline(st["best_shift"], color="green", linestyle="--", alpha=0.7,
                label=f"best shift = {st['best_shift']:.3f}")
    plt.title(f"Shift test: {name}")
    plt.xlabel("shift")
    plt.ylabel("band energy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

# ------------------------------------------------------------
# 7) CSV-EXPORT
# ------------------------------------------------------------
summary_df.to_csv("analysis_global_summary_1e11.csv", index=False)
window_df.to_csv("analysis_window_stability_1e11.csv", index=False)

for name, peaks in peak_tables.items():
    peaks.to_csv(f"analysis_peaks_{name}_1e11.csv", index=False)

print("\nGespeichert:")
print("- analysis_global_summary_1e11.csv")
print("- analysis_window_stability_1e11.csv")
for name in peak_tables:
    print(f"- analysis_peaks_{name}_1e11.csv")
