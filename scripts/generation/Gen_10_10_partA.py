"""
Prime-Zeta Resonance Experiments
Builds the 1e10 logarithmic-grid prime-side dataset used in the resonance workflow.
"""

# ============================================================
# TEIL A: DATENGENERIERUNG BIS 1e10 (segmentiert)
# Erzeugt theta(x), psi(x), theta_norm, psi_norm
# auf einem log-uniformen Gitter.
#
# Ausgabe:
#   - lhs_phase_1e10_loggrid.csv
#   - lhs_phase_1e10_loggrid.npz
#
# Für Google Colab / lokale Python-Umgebung.
# ============================================================

import numpy as np
import pandas as pd
import time
from math import isqrt

# ------------------------------------------------------------
# EINSTELLUNGEN
# ------------------------------------------------------------
X_MAX = 10_000_000_000      # 1e10
X_MIN = 10
N_LOG_GRID = 16000          # für ersten Lauf bewusst etwas konservativer
SEGMENT_SIZE = 10_000_000   # kann ggf. auf 20_000_000 erhöht werden
OUT_PREFIX = "lhs_phase_1e10"

# ------------------------------------------------------------
# BASIS-SIEB BIS sqrt(X_MAX)
# ------------------------------------------------------------
def simple_sieve(limit: int) -> np.ndarray:
    """Alle Primzahlen <= limit."""
    if limit < 2:
        return np.array([], dtype=np.int64)

    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p * p : limit + 1 : p] = False
    return np.flatnonzero(sieve).astype(np.int64)

# ------------------------------------------------------------
# PRIME-POWERS FÜR psi(x)
# Nur p <= sqrt(X_MAX) nötig, da p^2 <= X_MAX
# ------------------------------------------------------------
def build_prime_power_events(base_primes: np.ndarray, x_max: int):
    """
    Liefert sortierte Events (pp_x, pp_w) mit:
      pp_x = p^k <= x_max, k>=2
      pp_w = log(p)
    """
    pp_x = []
    pp_w = []

    cutoff = isqrt(x_max)
    base = base_primes[base_primes <= cutoff]

    for p in base:
        lp = np.log(p)
        val = p * p
        while val <= x_max:
            pp_x.append(val)
            pp_w.append(lp)

            if val > x_max // p:
                break
            val *= p

    if len(pp_x) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    pp_x = np.array(pp_x, dtype=np.int64)
    pp_w = np.array(pp_w, dtype=np.float64)

    order = np.argsort(pp_x, kind="mergesort")
    return pp_x[order], pp_w[order]

# ------------------------------------------------------------
# SEGMENTIERTES PRIMZAHLSIEB
# ------------------------------------------------------------
def segmented_primes(low: int, high: int, base_primes: np.ndarray) -> np.ndarray:
    """
    Erzeugt Primzahlen im Intervall [low, high].
    """
    if high < 2 or high < low:
        return np.array([], dtype=np.int64)

    low = max(low, 2)
    size = high - low + 1
    sieve = np.ones(size, dtype=bool)

    for p in base_primes:
        p2 = p * p
        if p2 > high:
            break
        start = max(p2, ((low + p - 1) // p) * p)
        sieve[start - low : size : p] = False

    return (np.flatnonzero(sieve) + low).astype(np.int64)

# ------------------------------------------------------------
# LOG-GITTER
# ------------------------------------------------------------
def build_log_grid(x_min: int, x_max: int, n_grid: int):
    u_grid = np.linspace(np.log(x_min), np.log(x_max), n_grid)
    x_grid = np.unique(np.clip(np.round(np.exp(u_grid)).astype(np.int64), x_min, x_max))
    u_grid = np.log(x_grid).astype(np.float64)
    return x_grid, u_grid

# ------------------------------------------------------------
# THETA / PSI AUF LOG-GITTER BERECHNEN
# ------------------------------------------------------------
def evaluate_theta_psi_on_grid(
    x_grid: np.ndarray,
    x_max: int,
    segment_size: int,
    base_primes: np.ndarray,
    pp_x: np.ndarray,
    pp_w: np.ndarray,
):
    """
    Berechnet theta(x), psi(x) auf den Punkten des x_grid.
    """
    theta_vals = np.zeros(len(x_grid), dtype=np.float64)

    # Prime-power-Korrektur für psi separat
    if len(pp_x) > 0:
        pp_cum = np.cumsum(pp_w)
        pp_idx = np.searchsorted(pp_x, x_grid, side="right")
        pp_corr = np.zeros(len(x_grid), dtype=np.float64)
        mask_pp = pp_idx > 0
        pp_corr[mask_pp] = pp_cum[pp_idx[mask_pp] - 1]
    else:
        pp_corr = np.zeros(len(x_grid), dtype=np.float64)

    running_theta = 0.0
    grid_ptr = 0

    for seg_low in range(2, x_max + 1, segment_size):
        seg_high = min(seg_low + segment_size - 1, x_max)
        print(f"Segment [{seg_low:,}, {seg_high:,}]")

        primes_seg = segmented_primes(seg_low, seg_high, base_primes)
        if len(primes_seg) == 0:
            while grid_ptr < len(x_grid) and x_grid[grid_ptr] <= seg_high:
                theta_vals[grid_ptr] = running_theta
                grid_ptr += 1
            continue

        prime_logs = np.log(primes_seg).astype(np.float64)
        theta_seg_cum = np.cumsum(prime_logs)

        while grid_ptr < len(x_grid) and x_grid[grid_ptr] <= seg_high:
            xq = x_grid[grid_ptr]
            idx = np.searchsorted(primes_seg, xq, side="right")
            if idx == 0:
                theta_vals[grid_ptr] = running_theta
            else:
                theta_vals[grid_ptr] = running_theta + theta_seg_cum[idx - 1]
            grid_ptr += 1

        running_theta += theta_seg_cum[-1]

    while grid_ptr < len(x_grid):
        theta_vals[grid_ptr] = running_theta
        grid_ptr += 1

    psi_vals = theta_vals + pp_corr
    return theta_vals, psi_vals, pp_corr

# ------------------------------------------------------------
# HAUPTLAUF
# ------------------------------------------------------------
def main():
    t0 = time.time()

    print("==> Erzeuge Basis-Primzahlen bis sqrt(X_MAX)")
    base_limit = isqrt(X_MAX)
    base_primes = simple_sieve(base_limit)
    print(f"Basis-Primzahlen <= {base_limit}: {len(base_primes):,}")

    t1 = time.time()

    print("==> Erzeuge Prime-Power-Ereignisse")
    pp_x, pp_w = build_prime_power_events(base_primes, X_MAX)
    print(f"Anzahl Prime-Power-Ereignisse: {len(pp_x):,}")

    t2 = time.time()

    print("==> Baue log-uniformes Gitter")
    x_grid, u_grid = build_log_grid(X_MIN, X_MAX, N_LOG_GRID)
    print(f"Anzahl eindeutiger Gitterpunkte: {len(x_grid):,}")

    t3 = time.time()

    print("==> Werte theta(x) und psi(x) auf dem Gitter aus")
    theta_vals, psi_vals, pp_corr = evaluate_theta_psi_on_grid(
        x_grid=x_grid,
        x_max=X_MAX,
        segment_size=SEGMENT_SIZE,
        base_primes=base_primes,
        pp_x=pp_x,
        pp_w=pp_w,
    )

    t4 = time.time()

    print("==> Baue normierte Kandidaten")
    x_float = x_grid.astype(np.float64)
    sqrt_x = np.sqrt(x_float)

    theta_minus_x = theta_vals - x_float
    psi_minus_x = psi_vals - x_float
    theta_norm = theta_minus_x / sqrt_x
    psi_norm = psi_minus_x / sqrt_x

    df = pd.DataFrame({
        "x": x_grid,
        "u_logx": u_grid,
        "theta": theta_vals,
        "psi": psi_vals,
        "prime_power_corr": pp_corr,
        "theta_minus_x": theta_minus_x,
        "psi_minus_x": psi_minus_x,
        "theta_norm": theta_norm,
        "psi_norm": psi_norm,
    })

    csv_name = f"{OUT_PREFIX}_loggrid.csv"
    npz_name = f"{OUT_PREFIX}_loggrid.npz"

    print("==> Speichere CSV und NPZ")
    df.to_csv(csv_name, index=False)
    np.savez_compressed(
        npz_name,
        x=df["x"].values,
        u_logx=df["u_logx"].values,
        theta=df["theta"].values,
        psi=df["psi"].values,
        prime_power_corr=df["prime_power_corr"].values,
        theta_minus_x=df["theta_minus_x"].values,
        psi_minus_x=df["psi_minus_x"].values,
        theta_norm=df["theta_norm"].values,
        psi_norm=df["psi_norm"].values,
    )

    t5 = time.time()

    print("\n==> FERTIG")
    print(f"CSV gespeichert: {csv_name}")
    print(f"NPZ gespeichert: {npz_name}")
    print("\nZeit Basis-Sieb:      %.2f s" % (t1 - t0))
    print("Zeit Prime-Powers:   %.2f s" % (t2 - t1))
    print("Zeit Log-Gitter:     %.2f s" % (t3 - t2))
    print("Zeit theta/psi:      %.2f s" % (t4 - t3))
    print("Zeit Speichern:      %.2f s" % (t5 - t4))
    print("Gesamtzeit:          %.2f s" % (t5 - t0))

if __name__ == "__main__":
    main()
