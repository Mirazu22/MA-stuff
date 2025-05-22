# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:51:44 2025

@author: maxuh
"""

import numpy as np
import matplotlib.pyplot as plt

# ── your data ── 
# Suppose you have arrays (or pandas Series) of length 11:
#   steps       = [1, 2, …, 10, 11]
#   means       = [mean_corr at step 1 … mean_corr at step 10,   limit_corr]
#   ci_lower    = [… lower bounds for steps 1–10,                NaN]
#   ci_upper    = [… upper bounds for steps 1–10,                NaN]
# The last entry has NaN CIs because we don’t plot error bars there.

steps    = np.arange(1, 12)
means    = np.array([0.52, 0.65, 0.74, 0.80, 0.84, 0.87, 0.89, 0.90, 0.91, 0.915, 0.917])
ci_lower = np.array([0.48, 0.60, 0.70, 0.77, 0.82, 0.85, 0.88, 0.89, 0.90, 0.912,   np.nan])
ci_upper = np.array([0.56, 0.70, 0.78, 0.83, 0.86, 0.89, 0.90, 0.91, 0.92, 0.918,   np.nan])

# compute asymmetric errors for steps 1–10
yerr_lower = means - ci_lower
yerr_upper = ci_upper - means
yerr = np.vstack([yerr_lower, yerr_upper])

# ── plotting ──
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(figsize=(7, 4.5))

# 1) draw line + points for all steps
ax.plot(steps, means, marker='o', linestyle='-', lw=1.5, label='Mean corr.')

# 2) add error bars only where they exist (i.e. steps 1–10)
ax.errorbar(
    steps[:-1],            # omit the last index
    means[:-1],
    yerr=yerr[:, :-1],
    fmt='none',
    ecolor='gray',
    alpha=0.7,
    capsize=4,
    label='95% CI'
)

# 3) highlight the infinite-step point in a different color/marker
ax.scatter(
    steps[-1],
    means[-1],
    s=100,
    facecolor='white',
    edgecolor='C0',
    linewidth=2,
    zorder=5,
    label='Limit (∞ step)'
)

# 4) customize ticks: show 1–10 normally, then “∞”
ax.set_xticks(steps)
ax.set_xticklabels([str(i) for i in range(1, 11)] + ['∞'])
ax.set_xlabel('Step number')
ax.set_ylabel('Correlation')
ax.set_title('Convergence of Correlation over Iterated Matrix Multiplication')

# 5) polish legend & layout
ax.legend(frameon=True)
plt.tight_layout()
plt.show()
