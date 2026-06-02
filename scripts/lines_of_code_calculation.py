# Imports
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from io import StringIO
import numpy as np
import json
from pathlib import Path

# Create function to obtain the lines of code
def run_cloc_all(path):
    """
    Run cloc on Gammapy and return a DataFrame with:
    - Python API
    - Python Tests
    - DocStrings
    - reStructuredText, C, and Others
    """
    # API code
    result_api = subprocess.run(
        ["cloc", "--not-match-d=test", "--json", path],
        capture_output=True, text=True, check=True
    )
    # Tests
    result_test = subprocess.run(
        ["cloc", "--match-d=test", "--json", path],
        capture_output=True, text=True, check=True
    )

    api_data = pd.read_json(StringIO(result_api.stdout)).T
    test_data = pd.read_json(StringIO(result_test.stdout)).T

    # Initialize dictionary
    stats = defaultdict(int)

    # Python API / Tests / DocStrings
    if "Python" in api_data.index:
        stats["Python API"] = api_data.loc["Python", "code"]
        stats["DocStrings"] = api_data.loc["Python", "comment"]
    if "Python" in test_data.index:
        stats["Python Tests"] = test_data.loc["Python", "code"]

    # Other languages from API cloc output
    for lang in api_data.index:
        if lang not in ["Python", "SUM"]:
            stats[lang] = api_data.loc[lang, "code"]

    # Compute Others (sum of small languages if >5)
    others = defaultdict(int)
    i = 0
    for lang, row in api_data.iterrows():
        if lang not in ["Python", "SUM", "reStructuredText", "C"]:
            if i >= 5:  # group everything after top 5
                others[lang] = row["code"]
            i += 1
    if others:
        stats["Others"] = sum(others.values())

    # Convert to DataFrame
    df = pd.DataFrame(stats, index=[0])
    return df
    
# Plotting function
def plot_version_comparison(labels, v1, v2, ylabel="Lines of code", figsize=(6, 4),
                        width=0.25, ylim=(0, 53000), rotation=0, fontsize=11):

    x = np.arange(len(labels))
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width/2, v1, width, label="v1.0")
    bars2 = ax.bar(x + width/2, v2, width, label="v2.0")

    pct_change = 100 * (v2 - v1) / np.where(v1 == 0, np.nan, v1)
    for bar, pct in zip(bars2, pct_change):
        if not np.isnan(pct):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=fontsize - 1)

    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotation, ha="right" if rotation else "center")
    ax.legend()
    ax.set_ylim(ylim)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.5)
    ax.get_yaxis().set_major_formatter(lambda x, p: f"{int(x):,}" )
    plt.tight_layout()
    
### First download the zips from: 
# https://github.com/gammapy/gammapy/releases/tag/v1.0
# https://github.com/gammapy/gammapy/releases/tag/v2.0

path_v1 = "gammapy-1.0"
path_v2 = "gammapy-2.0"

# Obtain the lines of code
df_cloc_all_v1 = run_cloc_all(path_v1)
df_cloc_all_v2 = run_cloc_all(path_v2)


# Create the labels
labels = ["Python API", "Python Tests", "DocStrings", "reStructuredText", "Others"]
sizes_v1 = np.array([df_cloc_all_v1[label].values[0] for label in labels])
sizes_v2 = np.array([df_cloc_all_v2[label].values[0] for label in labels])


# Plot
plot_version_comparison(labels=labels, v1=sizes_v1, v2=sizes_v2, figsize=(7, 5), width=0.25, rotation=30)
plt.savefig('LOC_pie_comparison.pdf', bbox_inches='tight')        
