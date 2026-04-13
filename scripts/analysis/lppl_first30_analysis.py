"""
Prime-Zeta Resonance Experiments
Blind and matched peak-recovery analysis for the first 30 zeta ordinates.
"""

# ============================================================
# ADVANCED LPPL ANALYSIS – FIRST 30 ZETA ORDINATES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# ============================================================
# 1. ZETA ORDINATES (FIRST 30)
# ============================================================

gamma = np.array([
14.134725,21.022040,25.010858,30.424876,32.935062,
37.586178,40.918719,43.327073,48.005151,49.773832,
52.970321,56.446248,59.347044,60.831779,65.112544,
67.079811,69.546402,72.067158,75.704691,77.144840,
79.337375,82.910381,84.735492,87.425274,88.809112,
92.491899,94.651344,95.870634,98.831194,101.317851
])

K = len(gamma)

# ============================================================
# 2. LOAD DATA
# ============================================================

df_theta = pd.read_csv("theta_norm.csv")
df_psi   = pd.read_csv("psi_norm.csv")

freq = df_theta["omega"].values
theta = df_theta["value"].values
psi   = df_psi["value"].values

theta = theta / np.max(theta)
psi   = psi   / np.max(psi)

# ============================================================
# 3. PEAK DETECTION
# ============================================================

def detect_peaks(signal):
    peaks, _ = find_peaks(signal, height=0.1, distance=5)
    return freq[peaks], signal[peaks]

theta_peaks_x, theta_peaks_y = detect_peaks(theta)
psi_peaks_x, psi_peaks_y     = detect_peaks(psi)

# ============================================================
# 4. MATCH TO TRUE ZEROS
# ============================================================

def match_peaks(peaks_x, gamma):
    matches = []
    for g in gamma:
        idx = np.argmin(np.abs(peaks_x - g))
        matches.append(peaks_x[idx])
    return np.array(matches)

theta_match = match_peaks(theta_peaks_x, gamma)
psi_match   = match_peaks(psi_peaks_x, gamma)

# ============================================================
# 5. ERROR ANALYSIS
# ============================================================

theta_error = np.abs(theta_match - gamma)
psi_error   = np.abs(psi_match   - gamma)

theta_rel = theta_error / gamma
psi_rel   = psi_error   / gamma

# ============================================================
# 6. BLIND CLUSTERING
# ============================================================

all_peaks = np.concatenate([theta_peaks_x, psi_peaks_x]).reshape(-1,1)

clustering = DBSCAN(eps=0.3, min_samples=3).fit(all_peaks)

labels = clustering.labels_

clusters = []

for lab in set(labels):
    if lab == -1:
        continue
    pts = all_peaks[labels==lab].flatten()
    center = np.mean(pts)
    clusters.append((lab, center, len(pts)))

clusters_df = pd.DataFrame(clusters, columns=["cluster_id","center","count"])

# ============================================================
# 7. MATCH CLUSTERS TO ZEROS
# ============================================================

def nearest_gamma(val):
    idx = np.argmin(np.abs(gamma - val))
    return gamma[idx], idx+1

clusters_df["nearest_gamma"] = clusters_df["center"].apply(lambda x: nearest_gamma(x)[0])
clusters_df["k"] = clusters_df["center"].apply(lambda x: nearest_gamma(x)[1])
clusters_df["gap"] = np.abs(clusters_df["center"] - clusters_df["nearest_gamma"])

# ============================================================
# 8. PLOTS
# ============================================================

plt.figure(figsize=(14,6))
plt.plot(freq, theta, label="theta_norm")
plt.plot(freq, psi, label="psi_norm")
for g in gamma:
    plt.axvline(g, color='green', alpha=0.2)
plt.title("Global spectrum (first 30 zeros)")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(range(1,K+1), theta_rel, marker='o', label="theta_rel")
plt.plot(range(1,K+1), psi_rel, marker='s', label="psi_rel")
plt.yscale("log")
plt.xlabel("k")
plt.ylabel("relative error")
plt.title("Relative localization error (first 30 zeros)")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(clusters_df["center"], clusters_df["count"])
for _, row in clusters_df.iterrows():
    plt.text(row["center"], row["count"], f"k={int(row['k'])}")
for g in gamma:
    plt.axvline(g, color='green', alpha=0.1)
plt.xlabel("frequency")
plt.ylabel("cluster support")
plt.title("Blind clusters vs true zeros")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(range(1,K+1), theta_match, label="theta peaks")
plt.plot(range(1,K+1), psi_match, label="psi peaks")
plt.plot(range(1,K+1), gamma, linestyle="dashed", label="true gamma")
plt.xlabel("k")
plt.ylabel("frequency")
plt.title("Recovered vs true zeros")
plt.legend()
plt.show()

# ============================================================
# 9. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "k": np.arange(1,K+1),
    "gamma": gamma,
    "theta_match": theta_match,
    "psi_match": psi_match,
    "theta_rel_error": theta_rel,
    "psi_rel_error": psi_rel
})

results.to_csv("results_first30.csv", index=False)
clusters_df.to_csv("blind_clusters_first30.csv", index=False)

print("Saved:")
print("- results_first30.csv")
print("- blind_clusters_first30.csv")
