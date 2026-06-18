# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
HAWC Joint Sensitivity Estimation
==================================
Reduces HAWC data across multiple event types and computes joint flux points
and sensitivity using an Asimov dataset via the FluxPointsEstimator.
"""

import os

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion

from gammapy.data import DataStore, HDUIndexTable, ObservationTable
from gammapy.datasets import Datasets, MapDataset
from gammapy.estimators import FluxPointsEstimator
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 13

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WHICH = "NN"  # energy estimator
DATA_PATH = os.path.join(os.environ["GAMMAPY_DATA"], "hawc/crab_events_pass4/")
HDU_FILENAME = f"hdu-index-table-{WHICH}-Crab.fits.gz"
OBS_FILENAME = f"obs-index-table-{WHICH}-Crab.fits.gz"
DATASETS_FILE = "HAWC_365_transit.yaml"
OUTPUT_FIGURE = "../figures/HAWC_sensitivity.pdf"

CRAB_CENTER = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
N_TRANSITS = 365  # ~ 1yr of HAWC sensitivity

EVENT_TYPES = np.arange(5, 10)  # nHit bins 5–9

# Simulated source parameters
SIM_SPECTRAL_INDEX = 3.5
SIM_AMPLITUDE = "3e-13 cm-2 s-1 TeV-1"
SIM_REFERENCE = "5 TeV"
SIM_SIGMA = 0.2 * u.deg



def build_datasets():
    # build 1 dataset per event type
    """Load HAWC IRFs for each event type and return a joint Datasets object."""
    obs_table = ObservationTable.read(DATA_PATH + OBS_FILENAME)

    energy_axis = MapAxis.from_edges(
        [1.00, 1.78, 3.16, 5.62, 10.0, 17.8, 31.6, 56.2, 100, 177, 316] * u.TeV,
        name="energy",
        interp="log",
    )
    energy_axis_true = MapAxis.from_energy_bounds(
        1e-2, 1e3, nbin=50, unit="TeV", name="energy_true"
    )

    geom = WcsGeom.create(
        skydir=CRAB_CENTER,
        width=6 * u.deg,
        axes=[energy_axis],
        binsz=0.05,
    )

    maker = MapDatasetMaker(selection=["background", "exposure", "edisp", "psf"])
    safemask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

    datasets = Datasets()

    for bin_id in EVENT_TYPES:
        hdu_table = HDUIndexTable.read(DATA_PATH + HDU_FILENAME, hdu=bin_id)
        hdu_table[-1]["HDU_CLASS"] = "psf_map_reco"

        data_store = DataStore(hdu_table=hdu_table, obs_table=obs_table)
        observation = data_store.get_observations()[0]

        dataset_empty = MapDataset.create(
            geom=geom,
            name=f"nHit-{bin_id}",
            energy_axis_true=energy_axis_true,
            reco_psf=True,
            binsz_irf=1 * u.deg,
            rad_axis=MapAxis.from_bounds(0, 3, 200, unit="deg", name="rad"),
        )

        dataset = maker.run(dataset_empty, observation)
        dataset.exposure.meta["livetime"] = 6.0*u.h #one transit
        dataset = safemask_maker.run(dataset)

        dataset.background.data *= N_TRANSITS
        dataset.exposure.data *= N_TRANSITS

        datasets.append(dataset)

    datasets.write(DATASETS_FILE, overwrite=True)
    return datasets


def simulate_datasets(datasets):
    """Inject a simulated source and draw Poisson-fluctuated counts."""
    spatial_model = GaussianSpatialModel(frame="icrs", sigma=SIM_SIGMA)
    spatial_model.position = CRAB_CENTER

    spectral_model = PowerLawSpectralModel(
        index=SIM_SPECTRAL_INDEX,
        amplitude=SIM_AMPLITUDE,
        reference=SIM_REFERENCE,
    )

    model_simu = SkyModel(
        spatial_model=spatial_model,
        spectral_model=spectral_model,
        name="model-simu",
    )
    datasets.models = model_simu

    print(
        "Simulated integrated energy flux (1–100 TeV): "
        f"{spectral_model.energy_flux(1 * u.TeV, 100 * u.TeV).to(u.erg / u.cm**2 / u.s):.3e}"
    )

    simulated = Datasets()
    for dataset in datasets:
        dataset.fake()
        print(f"  {dataset.name}: {dataset.counts.data.sum():.0f} counts")
        simulated.append(dataset)

    return simulated



def fit_model(datasets: Datasets) -> SkyModel:
    """Fit a power-law point source to the simulated datasets."""
    model_fit = SkyModel(
        spatial_model=GaussianSpatialModel(frame="icrs"),
        spectral_model=PowerLawSpectralModel(reference="5 TeV"),
        name="model-fit",
    )
    model_fit.spatial_model.position = CRAB_CENTER
    

    # Initial parameter guesses
    model_fit.spectral_model.index.value = 2.2
    model_fit.spectral_model.amplitude.value = 1e-14
    model_fit.spectral_model.amplitude.min = 1e-16
    model_fit.spatial_model.sigma.value = 0.1
    model_fit.spatial_model.sigma.max = 0.3


    # Constrain sky position to a sensible range
    model_fit.spatial_model.lon_0.min = 82.0
    model_fit.spatial_model.lon_0.max = 85.0
    model_fit.spatial_model.lat_0.min = 20.0
    model_fit.spatial_model.lat_0.max = 24.0

    datasets.models = [model_fit]

    fit = Fit()
    result = fit.run(datasets)
    print(model_fit)

    return model_fit




def compute_flux_points(datasets: Datasets, model_fit: SkyModel):
    """Run the FluxPointsEstimator and return the flux points result."""
    # Use the energy axis from the first dataset
    energy_axis = datasets[0].geoms["geom"].axes["energy"]

    fpe = FluxPointsEstimator(
        selection_optional="all",
        energy_edges=energy_axis.edges,
        n_sigma_ul=5,
        source="model-fit",
        n_jobs=6,
    )
    fp = fpe.run(datasets)
    fp.sqrt_ts_threshold_ul = 5
    return fp, energy_axis



def plot_results(fp, model_fit, energy_axis):
    """Plot flux points, sensitivity, and the fitted model."""
    fig, ax = plt.subplots()

    fp.plot(ax=ax, sed_type="e2dnde", label="Reconstructed flux points")
    fp.e2dnde_sensitivity.plot(ax=ax, label="Sensitivity", ls="dashed")

    e_min = fp.geom.axes["energy"].edges[2]
    e_max = fp.geom.axes["energy"].edges[-2]
    model_fit.spectral_model.plot_error(
        ax=ax,
        energy_bounds=[e_min, e_max],
        sed_type="e2dnde",
        label="Fitted model",
    )

    ax.legend()
    ax.set_ylabel(r"$\mathrm{E}^2 \times \mathrm{flux\ sensitivity\ [erg\ cm^{-2}\ s^{-1}]}$")
    ax.set_xlabel("Energy [TeV]")

    fig.savefig(OUTPUT_FIGURE, dpi=500, bbox_inches="tight")
    print(f"Figure saved to {OUTPUT_FIGURE}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Step 1: Data reduction ===")
    datasets = build_datasets()

    print("\n=== Step 2: Simulate source ===")
    sim_datasets = simulate_datasets(datasets)

    print("\n=== Step 3: Fit model ===")
    model_fit = fit_model(sim_datasets)

    print("\n=== Step 4: Flux points & sensitivity ===")
    fp, energy_axis = compute_flux_points(sim_datasets, model_fit)

    print("\n=== Step 5: Plot ===")
    plot_results(fp, model_fit, energy_axis)
