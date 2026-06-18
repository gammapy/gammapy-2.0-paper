# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Sampling analysis on a multi-modal example
==========================================
F. Acero, June 2026.
Simulation of a fake 3D dataset with two sources simulated and only one fitted.
A nested sampling analysis is compared to an iMinuit gradient descent fit.
Script accompanying the gammapy v2 paper.

In this script we :  
- simulate a 3D dataset with two sources well separated
- define a `Skymodel` where only one source is defined
- fit the dataset with a) gradient descent via `Fit` and b) via the `Sampler` and the UltraNest backend
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from gammapy.irf import load_irf_dict_from_file
from gammapy.maps import WcsGeom, MapAxis
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    PointSpatialModel,
    SkyModel,
    Models,
    FoVBackgroundModel,
    UniformPrior,
    LogUniformPrior
)
from gammapy.datasets import Datasets, MapDataset
from gammapy.makers import MapDatasetMaker
from gammapy.data import Observation
from gammapy.modeling import Fit
from gammapy.modeling.sampler import Sampler

from itertools import combinations
from matplotlib.collections import LineCollection
from chainconsumer import Chain, ChainConsumer, PlotConfig
from pandas import DataFrame

# Create the observations
irfs = load_irf_dict_from_file(
    "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
)

observation = Observation.create(
    pointing=SkyCoord(0 * u.deg, 0 * u.deg, frame="galactic"),
    livetime=5 * u.h,
    irfs=irfs,
)

# Define map geometry
axis = MapAxis.from_edges(
    np.logspace(-1, 1, 10), unit="TeV", name="energy", interp="log"
)

geom = WcsGeom.create(
    skydir=(0, 0), binsz=0.02, width=(0.8, 0.8), frame="galactic", axes=[axis]
)

empty_dataset = MapDataset.create(geom=geom, name="dataset-fake")
maker = MapDatasetMaker(selection=["background", "edisp", "psf", "exposure"])
dataset = maker.run(empty_dataset, observation)


# Define the models
spatial_model = PointSpatialModel(
lon_0="-0.1 deg", lat_0="-0.1 deg", frame="galactic"
)

spectral_model = PowerLawSpectralModel(
    index=2.,
    amplitude="1e-12 cm-2 s-1 TeV-1",
    reference="1 TeV",
)

sky_model = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model, name="source"
)


bkg_model = FoVBackgroundModel(dataset_name=dataset.name)


sky_model.spectral_model.amplitude.prior = UniformPrior(min=1e-13, max=50e-13)
sky_model.spectral_model.index.prior = UniformPrior(min=1.5, max=3.)
sky_model.spatial_model.lon_0.prior = UniformPrior(min=-0.15, max=0.15)
sky_model.spatial_model.lat_0.prior = UniformPrior(min=-0.15, max=0.15)
sky_model.spatial_model.lon_0.min = -0.2
sky_model.spatial_model.lon_0.max = 0.2
sky_model.spatial_model.lat_0.min = -0.2
sky_model.spatial_model.lat_0.max = 0.2
sky_model.spatial_model.lon_0.frozen = False
sky_model.spatial_model.lat_0.frozen = False

# A 2nd fainter source
sky_model2 = sky_model.copy()
sky_model2.spatial_model.lon_0.value = 0.1
sky_model2.spatial_model.lat_0.value = 0.1

bkg_model.spectral_model.norm.prior = UniformPrior(min=0.9, max=1.1)
bkg_model.spectral_model.norm.frozen = False

sky_model_true= sky_model.copy()

models = Models([sky_model, sky_model2, bkg_model])
models_true = models.copy()  # comparison later between true and fitted values

# Fake the dataset
dataset.models = models
dataset.fake(random_state=1)

# Making an Asimov dataset 
dataset.counts.data = dataset.npred().data

sky_model.spatial_model.lon_0.value = 0.01
sky_model.spatial_model.lat_0.value = 0.0
sky_model.spectral_model.amplitude.value = 0.8e-12
sky_model.spectral_model.index.value =2.1

bkg_model.spectral_model.norm.frozen=True
dataset.models = Models([sky_model, bkg_model]) #assuming just one src instead of two

# Fit with the normal routine
fit = Fit(store_trace=True, backend='minuit')
result_fit = fit.run(datasets=[dataset])

# Using the Sampler
bkg_model.spectral_model.norm.value = 1.0
bkg_model.spectral_model.norm.frozen = True
dataset.models = Models([sky_model, bkg_model]) #assuming just one src instead of two
sampler_opts = {
    "live_points": 1000,
    "frac_remain": 0.01,
    "log_dir": "multi-modal",
    "resume":"overwrite", 
}
sampler = Sampler(backend="ultranest", sampler_opts=sampler_opts)
result = sampler.run(dataset)

# Record the trace
trace = result_fit.trace 

# Plotting
#---------------------
c = ChainConsumer()

# Create chain
names = result.models[0].parameters.free_parameters.names
c.add_chain(
    Chain(
        samples=DataFrame(result.samples, columns=names),
        name="joint",
        color="b",
        smooth=10,
        shade=True,
        linewidth=2,
        kde=False,
        plot_point=False,
        plot_cloud=True,
        multimodal=True,
        shade_alpha=0.2,
        zorder=-10,
    )
)

# Plot ranges
c.set_plot_config(
    PlotConfig(
        extents={
            "lat_0": (-0.15, 0.15),
            "lon_0": (-0.15, 0.15),
            "index": (1.4, 2.6),
            "amplitude": (0.5e-12, 1.5e-12),
        }
    )
)
# labels are for the x and y

fig = c.plotter.plot()
axes = np.array(fig.axes).reshape(4, 4)

params = [
    "source.spectral.index",
    "source.spectral.amplitude",
    "source.spatial.lon_0",
    "source.spatial.lat_0",
]

for (i, p1), (j, p2) in combinations(enumerate(params), 2):
    x, y = trace[p1][::2], trace[p2][::2]
    ax = axes.T[i, j]

    segs = np.stack(
        [np.column_stack([x[:-1], y[:-1]]),
         np.column_stack([x[1:],  y[1:]])],
        axis=1,
    )

    ax.add_collection(
        LineCollection(segs, colors="grey", linewidths=2)
    )

    ax.scatter(
        x[0], y[0],
        marker="s",
        facecolors="none",
        edgecolors="k",
        s=150,
        label="start",
    )

plt.savefig("../figures/nested_sampling_mulitmodal.pdf", dpi=200, bbox_inches='tight')









