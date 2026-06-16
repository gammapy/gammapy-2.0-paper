# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Fake dataset generation and upper-limits
========================================

F. Acero, June 2026.
Simulation of fake datasets to probe the behavior of upper-limit computation gammapy v2.0.1.  
Script accompanying the gammapy v2 paper.

In this script we :  

- Build a fake `SpectrumDatasetOnOff` with CTAO IRFs and pure background for a given exposure time
- Use `FluxPointsEstimator` to derive upper-limits+sensitivity using the profile likelihood (frequentist approach)
- Use the `Sampler` class to perform a nested sampling analysis and derive upper-limits via the samples
"""
  
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from scipy.stats import norm
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
from matplotlib.ticker import MaxNLocator

from gammapy.data import FixedPointingInfo, Observation, observatory_locations
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.estimators import FluxPointsEstimator
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom
from gammapy.modeling import Sampler
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel, UniformPrior
from gammapy.utils.parallel import run_multiprocessing

import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 13

def build_observation(livetime="1 h"):
    # Define simulation parameters parameters
    livetime = u.Quantity(livetime)

    pointing_position = SkyCoord(0, 0, unit="deg", frame="galactic")
    # We want to simulate an observation pointing at a fixed position in the sky.
    # For this, we use the `FixedPointingInfo` class
    pointing = FixedPointingInfo(
        fixed_icrs=pointing_position.icrs,
    )
    
    irfs = load_irf_dict_from_file(
        "$GAMMAPY_DATA/cta-caldb/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz"
    )

    location = observatory_locations["ctao_south"]
    return Observation.create(
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=location,
    )


def build_dataset(obs, offset="0.5 deg"):
    offset = u.Quantity(offset)

    # Reconstructed and true energy axis
    energy_axis = MapAxis.from_energy_bounds(0.05, 50, 6, per_decade=True, unit="TeV")
    energy_axis_true = MapAxis.from_energy_bounds(0.02, 100, 12, per_decade=True, unit="TeV", name="energy_true")

    on_region_radius = 0.11*u.deg

    pointing_position = obs.get_pointing_icrs(obs.tmid)
    center = pointing_position.directional_offset_by(
        position_angle=0 * u.deg, separation=offset
    )
    
    on_region = CircleSkyRegion(center=center, radius=on_region_radius)

    # Make the SpectrumDataset
    geom = RegionGeom.create(region=on_region, axes=[energy_axis])

    dataset_empty = SpectrumDataset.create(
        geom=geom, energy_axis_true=energy_axis_true, name="obs-0"
    )
    maker = SpectrumDatasetMaker(selection=["exposure", "edisp", "background"])

    return maker.run(dataset_empty, obs)

def fake_dataset(dataset, model, seed):
    dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
        dataset=dataset, acceptance=1, acceptance_off=10
    )
    dataset_on_off.models = model.copy()

    dataset_on_off.fake(npred_background=dataset.npred_background(),random_state=seed)
    return dataset_on_off

def build_model_pow(amplitude=1e-12):
    model_simu = PowerLawSpectralModel(index=2, amplitude=f'{amplitude} cm-2 s-1 TeV-1')
    return SkyModel(spectral_model=model_simu, name="source")    

def fake_analyse(dataset, simu_model, fit_model, fpe_config, seed):
    results = {}

    sampler_model = fit_model.copy()

    min_amplitudes = [-3e-14,0]
    max_amplitude  = 1e-12

    for min_amplitude, name in zip(min_amplitudes,["negative", "zero"]):
    
        faked_dataset = fake_dataset(dataset, simu_model, seed)
        faked_dataset.models =  fit_model #reset model to a reasonable model for fitting
    
        faked_dataset.models[0].spectral_model.amplitude.min=min_amplitude
        faked_dataset.models[0].spectral_model.amplitude.max=max_amplitude
        faked_dataset.models[0].spectral_model.amplitude.scan_min=min_amplitude
        faked_dataset.models[0].spectral_model.amplitude.scan_max=max_amplitude

        print(faked_dataset.models[0].spectral_model.amplitude)    
        
        estimator = FluxPointsEstimator(
            source="source",
            energy_edges=[E1,E2], **fpe_config,
        )        
        results[f'profile-{name}'] = estimator.run([faked_dataset])


        sampler_model.spectral_model.amplitude.prior = UniformPrior(min=min_amplitude, max=max_amplitude)
        faked_dataset.models = sampler_model
        
        sampler_opts = {"live_points": 300, "frac_remain": 0.3,"log_dir": None, "show_status": False}
        sampler = Sampler(backend="ultranest", sampler_opts=sampler_opts)
    
        results[f'uniform-{name}'] = sampler.run([faked_dataset])

        
    return results

def extract_ul_from_samples(samples, n_sigma_ul=2):
    return np.quantile(samples[:,0], norm.cdf(n_sigma_ul))

def perform_simulation(nsim, dataset, simu_model, fit_model, fpe_config, seed="random-seed"):
    indices = np.arange(nsim)

    if seed=="random-seed":
        inputs = [(dataset, simu_model, fit_model, fpe_config, "random-seed")  for seed in indices]
    if seed=="range":
        inputs = [(dataset, simu_model, fit_model, fpe_config, seed)  for seed in indices]
    
    fps = run_multiprocessing(fake_analyse, inputs, task_name="simulation")
    
    return fps

exposure="1 h"
obs = build_observation(livetime=exposure)
dataset = build_dataset(obs)
E1,E2= 0.05*u.TeV, 50*u.TeV
dataset.mask_fit = dataset.counts.geom.energy_mask(E1,E2)

 #no signal
skymodel_simu=build_model_pow(amplitude = 1e-20) # almost no signal for simu
skymodel_fit=build_model_pow(amplitude = 1e-13)  # reasonable starting point for fitting UL
skymodel_fit.spectral_model.index.frozen = True

bkg_counts=dataset.npred_background().data[dataset.mask].sum()
print(f"Expected signal in ON region: {dataset.npred_signal().data[dataset.mask].sum():.2f}")
print(f"Expected background in ON region: {bkg_counts:.2f}")

fpe_config = {"selection_optional":["all"], "n_sigma_ul":2, "n_sigma_sensitivity":2}

# multiple runs 
Nrun=20
seed="range" #the seed in range(Nrun). the same everytime you run the script.
results = perform_simulation(Nrun, dataset, skymodel_simu, skymodel_fit, fpe_config,seed=seed)

ref_amplitude = skymodel_fit.spectral_model.amplitude.quantity

uls_profile_zero = np.array([res['profile-zero']['norm_ul'].data.flatten()[0] for res in results])*ref_amplitude
sensitivity_zero = np.array([res['profile-zero']['norm_sensitivity'].data.flatten()[0] for res in results])*ref_amplitude
uls_uniform_zero = np.array([extract_ul_from_samples(res["uniform-zero"].samples) for res in results])

uls_uniform_negative = np.array([extract_ul_from_samples(res["uniform-negative"].samples) for res in results])
uls_profile_negative = np.array([res['profile-negative']['norm_ul'].data.flatten()[0] for res in results])*ref_amplitude


# Plot the figure
arrowfraction = 0.1
x = np.arange(Nrun)
xerr = np.ones(Nrun)*0.5

fig, ax = plt.subplots()
with quantity_support():
    ax.errorbar(x, uls_uniform_zero, 
                xerr=xerr, 
                yerr=arrowfraction*uls_uniform_zero, 
                label="UL sampling", uplims=True, fmt='None')
    ax.errorbar(x, uls_profile_zero, 
                xerr=xerr, 
                yerr=arrowfraction*uls_profile_zero, 
                marker='o', color='tab:orange', 
                label="UL likelihood profile", uplims=True, fmt='None')
    ax.plot(x, sensitivity_zero, color='k', label="Sensitivity")

ax.xaxis.set_major_locator(MaxNLocator(nbins=11))
ax.set_xlabel('Realisation')
ax.set_ylabel(r'Differential Flux 95% UL [TeV$^{-1}$ cm$^{-2}$ s$^{-1}$]')
ax.set_ylim(1e-14, 2.5e-13)
plt.legend()
plt.savefig(f"../figures/UL_comparison.pdf", dpi=500, bbox_inches="tight")
















