# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Joint TS maps and combined significance maps 
============================================
Fermi-LAT and H.E.S.S. analysis example toward RXJ 1713.7-3946
"""

import astropy.units as u
import matplotlib.pyplot as plt 
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import numpy as np 

from gammapy.datasets import Datasets
from gammapy.modeling.models import Models, PointSpatialModel, PowerLawSpectralModel, SkyModel
from gammapy.modeling import Fit
from gammapy.estimators import TSMapEstimator, ExcessMapEstimator
from gammapy.estimators.utils import get_combined_significance_maps

plt.rcParams["font.size"] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 14


# Datasets setup
datasets_fermi = Datasets.read(
    filename="./datasets/fermi_lat_psf23_rxj_datasets.yaml", filename_models="./models/fermi_lat_psf23_rxj_models.yaml"
)

datasets_hess = Datasets.read(
    filename="./datasets/hess_rxj_joint_datasets.yaml", filename_models="./models/hess_rxj_joint_models.yaml"
)

# Create a joint dataset for Fermi and H.E.S.S.
datasets = Datasets(list(datasets_fermi) + list(datasets_hess))

# We select are going to select only the fermi-lat models that are not seen by H.E.S.S.
models_hgps_geom = Models.read("./models/hgps_rxj_models.yaml")

fermi_sources = Models(datasets_fermi.models.select(name_substring="4FGL"))
selection = np.array([np.all(m.position.separation(models_hgps_geom.positions) > 0.1*u.deg) for m in fermi_sources])
fermi_sources_selection = fermi_sources[selection]

model_iem = Models(datasets_fermi.models[ "IEM_varmin_rescaled"])
models_fermi_iso = Models(datasets_fermi.models.select(tag="const", model_type="spatial"))
models_hess_bkg = Models(datasets_hess.models.select(tag="fov-bkg"))

# You can check the `datasets_names` attribute on a given model to check on which dataset it is applied.
# By default it is `None` and apply to everything.
# Now we can assign the models to the datasets (and this will take into account `datasets_names` assignment)
models = fermi_sources_selection + model_iem + models_fermi_iso + models_hess_bkg
datasets.models = models

# Joint TS map
#---------------------
# This use forward folding to convolve a given nev model by the IRFs and 
# move it in every pixel to evalaute the significance of the excess above 
# the existing models.
spatial_model = PointSpatialModel()
spectral_model = PowerLawSpectralModel(index=2)
model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

ts_estimator = TSMapEstimator(
    model,
    kernel_width="1 deg",  # set this close to the 95-99% containment radius of the PSF
    selection_optional=[],
    sum_over_energy_groups=True,
    energy_edges=[10, 1000] * u.GeV,
    n_jobs=4, #this will run in parallel
)

ts_results_fermi = ts_estimator.run(datasets_fermi)
ts_results_hess = ts_estimator.run(datasets_hess)
ts_results_joint = ts_estimator.run(datasets)

# Plotting
def plot_cutout(image, ax, plot_kwargs=None, margin=2*u.deg):
    image = image.cutout(
        image.geom.center_skydir, width=np.max(image.geom.width) - 2 * margin
    )
    kwargs = dict(ax=ax)
    if plot_kwargs : 
        kwargs.update(plot_kwargs)
    image.plot(**kwargs)

margin=2.42*u.deg
geom = datasets[0].counts.geom
fig_geom = geom.cutout(geom.center_skydir, width=np.max(geom.width) - 2 * margin)
resi_kwargs =dict(
        clim=[-9, 9],
        cmap=plt.cm.RdBu_r,
        add_cbar=False,
    )

def plot_cutout(image, ax, plot_kwargs=None, margin=2*u.deg):
    image = image.cutout(
        image.geom.center_skydir, width=np.max(image.geom.width) - 2 * margin
    )
    kwargs = dict(ax=ax)
    if plot_kwargs : 
        kwargs.update(plot_kwargs)
    image.plot(**kwargs)

margin=2.42*u.deg
geom = datasets[0].counts.geom
fig_geom = geom.cutout(geom.center_skydir, width=np.max(geom.width) - 2 * margin)
resi_kwargs =dict(
        clim=[-9, 9],
        cmap=plt.cm.RdBu_r,
        add_cbar=False,
    )

# Plotting
fig = plt.figure(figsize=(12, 5))
ax1 = plt.subplot(131, projection=fig_geom.wcs)
plot_cutout(ts_results_fermi["sqrt_ts"], ax1, resi_kwargs, margin)
ax1.set_title("$Fermi$-LAT")
ax1.text(0.5, 0.92, '10 GeV - 1 TeV', transform=ax1.transAxes,
         ha='center', va='bottom', fontsize=13)

ax2 = plt.subplot(132, projection=fig_geom.wcs)
plot_cutout(ts_results_hess["sqrt_ts"], ax2, resi_kwargs, margin)
ax2.set_title("H.E.S.S.")
ax2.tick_params(axis="y", labelbottom=False)
ax2.text(0.5, 0.92, '500 GeV - 50 TeV', transform=ax2.transAxes,
         ha='center', va='bottom', fontsize=13)

ax3 = plt.subplot(133, projection=fig_geom.wcs)
plot_cutout(ts_results_joint["sqrt_ts"], ax3, resi_kwargs, margin)
ax3.set_title("$Fermi$-LAT + H.E.S.S.")
ax3.tick_params(axis="y", labelbottom=False)
ax3.text(0.5, 0.92, '10 GeV - 50 TeV', transform=ax3.transAxes,
         ha='center', va='bottom', fontsize=13)

norm = colors.Normalize(vmin=resi_kwargs["clim"][0], vmax=resi_kwargs["clim"][1])
sm = ScalarMappable(norm=norm, cmap=resi_kwargs["cmap"])
sm.set_array([])

cb_ax = fig.add_axes([ .999, .21, 0.023 , .691])
cb = fig.colorbar(sm, cax=cb_ax, orientation='vertical')  
cb.ax.xaxis.set_ticks_position("top")
cb.ax.xaxis.set_label_position("top")
cb.ax.tick_params(axis="both", which="major")
cb.ax.set_ylabel(r"$\sqrt{TS}$ [$\sigma$]")

plt.tight_layout()
plt.savefig('../../figures/joint_ts_fermilat_hess_rxj.pdf', bbox_inches='tight')

# Excess Maps
#----------------------------
# This use backward folding to estimate the TS and the flux in a given correlation radius.
# In that case the predicted counts are the excess counts. 
# The flux is the excess counts divided by the reconstructed exposure. 
# So only the flux depends on the spectral model assumption not the TS.

estimator_rcorr0p1 = ExcessMapEstimator(correlation_radius="0.1 deg", spectral_model=spectral_model)

# This will give use one estimation per dataset that we can combine together 
# using `get_combined_significance_maps`.The significance computation assumes
# that the model contains one degree of freedom per valid energy bin in each 
# dataset. This method implemented here is valid under the assumption that the 
# TS in each independent bin follows a Chi2 distribution, then the sum of the 
# TS also follows a Chi2 distribution (with the sum of degree of freedom).

results_combined_fermi_rcorr0p1 = get_combined_significance_maps(estimator_rcorr0p1, datasets_fermi)
results_combined_hess_rcorr0p1 = get_combined_significance_maps(estimator_rcorr0p1, datasets_hess)
results_combined_rcorr0p1 = get_combined_significance_maps(estimator_rcorr0p1, datasets)


# Plotting
fig = plt.figure(figsize=(12, 5))
ax1 = plt.subplot(131, projection=fig_geom.wcs)
plot_cutout(results_combined_fermi_rcorr0p1["significance"], ax1, resi_kwargs, margin)
ax1.set_title("$Fermi$-LAT")
ax1.text(0.5, 0.92, '10 GeV - 1 TeV', transform=ax1.transAxes,
         ha='center', va='bottom', fontsize=13)

ax2 = plt.subplot(132, projection=fig_geom.wcs)
plot_cutout(results_combined_hess_rcorr0p1["significance"], ax2, resi_kwargs, margin)
ax2.set_title("H.E.S.S.")
ax2.tick_params(axis="y", labelbottom=False)
ax2.text(0.5, 0.92, '500 GeV - 50 TeV', transform=ax2.transAxes,
         ha='center', va='bottom', fontsize=13)

ax3 = plt.subplot(133, projection=fig_geom.wcs)
plot_cutout(results_combined_rcorr0p1["significance"], ax3, resi_kwargs, margin)
ax3.set_title("$Fermi$-LAT + H.E.S.S.")
ax3.tick_params(axis="y", labelbottom=False)
ax3.text(0.5, 0.92, '10 GeV - 50 TeV', transform=ax3.transAxes,
         ha='center', va='bottom', fontsize=13)

norm = colors.Normalize(vmin=resi_kwargs["clim"][0], vmax=resi_kwargs["clim"][1])
sm = ScalarMappable(norm=norm, cmap=resi_kwargs["cmap"])
sm.set_array([])

cb_ax = fig.add_axes([ .999, .21, 0.023 , .691])
cb = fig.colorbar(sm, cax=cb_ax, orientation='vertical')  
cb.ax.xaxis.set_ticks_position("top")
cb.ax.xaxis.set_label_position("top")
cb.ax.tick_params(axis="both", which="major", labelsize=14)
cb.ax.set_ylabel(r"Significance [$\sigma$]", fontsize=14)

plt.tight_layout()
plt.savefig('../../figures/combined_significance_fermilat_hess_rxj.pdf', bbox_inches='tight')

















