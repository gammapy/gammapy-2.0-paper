# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ULs per realization with side histograms
========================================

F. Acero, October 2026. Editted by K. Feijen, October 2026.
One panel per exposure (rows: 0.03 h and 100 h) and constraint (columns: amplitude >= 0, unconstrained),
profile likelihood vs Bayesian ULs on the same realizations. Reads the main run of
`UL_simulation_run.py` (uls_main.ecsv), so run that first.

Script accompanying the gammapy v2 paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import astropy.units as u
from astropy.table import QTable

exposures = ["0.03 h", "100 h"]  # Poisson and Gaussian regimes (0.3 h skipped)
PRIORS = {"zero": "amplitude $\\geq 0$", "negative": "unconstrained"}
COLORS = {"profile": "tab:orange", "sampler": "tab:blue"}
MARKERS = {"profile": "v", "sampler": "o"}
LABELS = {"profile": "UL frequentist", "sampler": "UL Bayesian"}
METHODS = ("profile", "sampler")
UL_METHOD = {"profile-zero": "exact", "profile-negative": "exact", "sampler-zero": "grid", "sampler-negative": "grid"}
YMAX = {"0.03 h": 5e-12, "100 h": 3e-14}

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 15


def select(table, sort="seed", **filters):
    mask = np.ones(len(table), dtype=bool)
    for column, value in filters.items():
        if column not in table.colnames:
            raise KeyError(f"Unknown column {column!r}, available: {table.colnames}")
        if isinstance(value, (list, tuple, set, np.ndarray)):
            mask &= np.isin(table[column], list(value))
        else:
            mask &= table[column] == value
    result = table[mask]
    if sort is not None and len(result):
        result.sort(sort)
    return result

def get(table, exposure, case, dataset_type="on", column="ul", **filters):
    rows = select(table, exposure=exposure, case=case, ul_method=UL_METHOD[case], dataset_type=dataset_type, **filters)
    values = rows[column]
    return values.to_value("cm-2 s-1 TeV-1") if isinstance(values, u.Quantity) else np.asarray(values)
    
    
def plot_lc_triangle(ax, axh, uls, ylim=None):
    """Scatter of the ULs vs realization (``ax``) and their histograms (``axh``), log scale. """
    n = len(uls["profile"])
    x = np.arange(n)
    for method in METHODS:
        values = uls[method]
        color = COLORS[method]
        ax.plot(x, values[:n], markersize=3, alpha=0.6, marker=MARKERS[method], color=color, 
                linestyle='None', label=LABELS[method], markeredgewidth=0)
        ax.axhline(np.nanmedian(values), ls="--", color=color, label="median")

    positive = np.concatenate([uls["profile"], uls["sampler"]])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 25)
    # Use reverse METHODS because we want the profile on top??
    for method in reversed(METHODS):
        v = uls[method]
        axh.hist(v[np.isfinite(v) & (v > 0)], bins=bins, orientation="horizontal", color=COLORS[method], alpha=0.6)
        axh.axhline(np.nanmedian(v), ls="--", color=COLORS[method])

    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlim(-0.01 * n, 1.01 * n)
    axh.tick_params(labelleft=False)
    
# Plot the figure
table = QTable.read("uls_main_dev.ecsv", format="ascii.ecsv")

fig = plt.figure(figsize=(14, 4 * len(exposures)))
outer = fig.add_gridspec(len(exposures), len(PRIORS), wspace=0.12, hspace=0.35)
for i, exposure in enumerate(exposures):
    values = (np.concatenate([get(table, exposure, f"{m}-negative") for m in METHODS]))
    ymin = 0.8*(values[np.isfinite(values) & (values > 0)]).min()
    ymax = YMAX[exposure]
    for j, (prior, prior_label) in enumerate(PRIORS.items()):        
        inner = outer[i, j].subgridspec(1, 2, width_ratios=[3, 1], wspace=0.04)
        ax = fig.add_subplot(inner[0])
        axh = fig.add_subplot(inner[1], sharey=ax)
        uls = {m: get(table, exposure, f"{m}-{prior}") for m in METHODS}
        plot_lc_triangle(ax, axh, uls, ylim=(ymin, ymax))
        ax.set_title(f"{exposure}, {prior_label}")
        if j == 0:
            ax.set_ylabel("95% UL [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$]")
        if i == len(exposures) - 1:
            ax.set_xlabel("Realization")
            axh.set_xlabel("Occurrence")
        if (i, j) == (0, 0):
            ax.legend(fontsize=12, ncol=2, loc="lower left")
plt.savefig(f"../figures/UL_per_realization.pdf", dpi=500, bbox_inches="tight")
