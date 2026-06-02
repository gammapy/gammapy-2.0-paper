# This code produces the Gammapy visits map for v2.0 paper
# >>> python visitors_map.py

# Imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn
import geopandas
import difflib

# Load the Matomo file
df = pd.read_csv("GammapyDoc_Country_15Sept2024-6Feb2026.csv", encoding="utf-16")

# Add the world map
url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world = geopandas.read_file(url)

# Rename the incorrectly named places
unmatched = set(df["Label"]) - set(world["SOVEREIGNT"])
for name in unmatched:
    matches = difflib.get_close_matches(name, world["SOVEREIGNT"], n=3, cutoff=0.5)
name_corrections = {
    "United States": "United States of America",
    "Côte d’Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Myanmar (Burma)": "Myanmar",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Bahamas": "The Bahamas"
}
df["Label"] = df["Label"].replace(name_corrections)    
world = world.merge(df, left_on="SOVEREIGNT", right_on="Label", how="left")

# Remove Antarctica for clearer plotting
world = world[world["CONTINENT"] != "Antarctica"]

# Set the projection to equal Earth 
world_proj = world.to_crs("+proj=eqearth")

# Plot the figure
fig, ax = plt.subplots(figsize=(18, 4))
world_proj.plot(column="Visits", cmap="Blues", legend=True, linewidth=0.4, edgecolor='black', ax=ax,
                norm=LogNorm(vmin=world_proj.Visits.min(), vmax=world_proj.Visits.max()), 
              missing_kwds={"color": "lightgrey", "edgecolor": "0.2",  "hatch": "///", "label": "No data"},
              legend_kwds={"label": "Number of Visits", "orientation": "vertical",  "shrink": 0.6,  "pad": 0})
# Add custom legend entry for missing data
import matplotlib.patches as mpatches
missing_patch = mpatches.Patch(
    facecolor="lightgrey",
    edgecolor="0.2",
    hatch="///",
    label="No data"
)
ax.legend(handles=[missing_patch], loc="lower left")
ax.set_axis_off()
ax.set_title("Matamo analytics for Gammapy Documentation", fontsize=14, fontweight="bold", pad=2)
plt.tight_layout()
plt.savefig('../figures/number_of_visitors.pdf', bbox_inches='tight')
