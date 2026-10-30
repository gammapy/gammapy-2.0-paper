# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Upper-limit simulation main
===========================

F. Acero, October 2026. Editted by K. Feijen, October 2026.
Background-only CTAO spectral datasets to probe the
behaviour of upper-limit (UL) estimators in gammapy v2.0.2:

- profile likelihood (``FluxPointsEstimator`` stat scan, frequentist),
- nested sampling (``Sampler``, Bayesian) with uniform priors.

The different test cases (profile / sampler, constrained / unconstrained
amplitude, ...) are described by `Case` objects. `run_simulation` fakes one
dataset per (exposure, dataset type, seed), runs *all* cases on that same
faked dataset, in parallel over seeds, and returns a long-format astropy
`~astropy.table.Table` (one row per exposure x dataset_type x seed x case x
ul_method).

This needs to be run on the dev version - but will be
updated to v2.0.2 when released.
"""

import contextlib
import datetime
import io
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, Table
from regions import CircleSkyRegion
from scipy import interpolate

__all__ = [
    "Case",
    "default_cases",
    "build_observation",
    "build_dataset",
    "build_model_pow",
    "build_simulation",
    "fake_dataset",
    "fake_dataset_on_off",
    "asimov_dataset",
    "fake",
    "run_realization",
    "run_simulation",
    "PROFILE_UL_METHODS",
    "SAMPLER_UL_METHODS",
    "DATASET_TYPES",
]

PROFILE_UL_METHODS = ("scipy", "fpe", "interp", "exact")
SAMPLER_UL_METHODS = ("samples", "grid")
DATASET_TYPES = ("on", "onoff", "asimov")
UL_UNIT = u.Unit("cm-2 s-1 TeV-1")
EXTRA_COLUMNS = ("sensitivity", "best_fit", "ts")
REF_AMPLITUDE = 1e-13  # amplitude of the fit model, reference of the FluxPointsEstimator norm
ASIMOV_SEED = -1  # seed value recorded for the (single, noiseless) Asimov realization

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

DEFAULT_SAMPLER_OPTIONS = {
    "min_num_live_points": 300,
    "log_dir": None,
    "viz_callback": False,
    "show_status": False,
}


@dataclass
class Case:
    """One upper-limit test case, run on every faked dataset.

    Parameters
    ----------
    name : str
        Unique case name, e.g. "profile-zero".
    method : {"profile", "sampler"}
        Profile likelihood (FluxPointsEstimator) or nested sampling (Sampler).
    prior : str
        Label used to group cases in the plots, e.g. "zero" (constrained,
        amplitude >= 0) or "negative" (unconstrained, amplitude can go negative).
    amplitude_min, amplitude_max : float
        Bounds of the amplitude in cm-2 s-1 TeV-1. Set as ``min``/``max`` (and
        scan range) of the fitted amplitude for the profile, and as the bounds
        of the `UniformPrior` for the sampler.
    norm_min, norm_scan_min, norm_scan_max, norm_scan_n_values, norm_interp :
        Settings of the ``norm`` parameter of the `FluxPointsEstimator`
        (profile only). Use `Case.norm_scan_zero` / `Case.norm_scan_negative`
        for the notebook settings.
    norm_scan_extend_factor, norm_scan_max_extensions : float, int
        If the stat profile does not reach ``delta_ts`` within the scan range
        (low statistics), ``norm_scan_max`` is multiplied by
        ``norm_scan_extend_factor`` and the profile is recomputed, at most
        ``norm_scan_max_extensions`` times (0 disables it). The scan is never
        extended beyond ``amplitude_max``.
    profile_ul_methods : tuple of str
        Profile UL methods to compute, among:

        - "scipy": `stat_profile_ul_scipy` on the stat scan (default),
        - "fpe": ``norm_ul`` computed by the `FluxPointsEstimator`,
        - "interp": ``interp1d`` of the stat scan on a fine grid (notebook method),
        - "exact": direct minimisation of the likelihood within the amplitude
          bounds and root finding of stat - stat_min = delta_ts, without the
          `FluxPointsEstimator` scan (only amplitude free). Use it for wide
          bounds, where a fixed norm scan would be too coarse.
    sampler_ul_methods : tuple of str
        Bayesian UL methods to compute (sampler only), among:

        - "samples": quantile of the nested sampling posterior samples (default),
        - "grid": same posterior quantile, computed by direct numerical
          integration of exp(-stat/2) x prior on a fine amplitude grid. Exact and
          deterministic as long as amplitude is the only free parameter (no
          sampler run when "samples" is not selected).
    sampler_backend : str
        Sampler backend (sampler only).
    sampler_options : dict
        Options passed to `Sampler` (sampler only).
    confidence_level_samples : float
        Quantile of the amplitude posterior samples used as UL (sampler only).
    """

    name: str
    method: str
    prior: str = "zero"
    amplitude_min: float = 0.0
    amplitude_max: float = 1e-11
    norm_min: float = 0.0
    norm_scan_min: float = 1e-6
    norm_scan_max: float = 10.0
    norm_scan_n_values: int = 50
    norm_interp: str = "log"
    norm_scan_extend_factor: float = 3.0
    norm_scan_max_extensions: int = 3
    profile_ul_methods: tuple = ("scipy",)
    sampler_ul_methods: tuple = ("samples",)
    sampler_backend: str = "ultranest"
    sampler_options: dict = field(default_factory=lambda: dict(DEFAULT_SAMPLER_OPTIONS))
    confidence_level_samples: float = 0.954

    # norm scan settings of the notebook
    norm_scan_zero = dict(
        norm_min=0.0, norm_scan_min=1e-6, norm_scan_max=10.0, norm_scan_n_values=50, norm_interp="log"
    )
    norm_scan_negative = dict(
        norm_min=-0.1, norm_scan_min=-0.1, norm_scan_max=5.0, norm_scan_n_values=100, norm_interp="lin"
    )

    def __post_init__(self):
        if self.method not in ("profile", "sampler"):
            raise ValueError(f"Case {self.name}: method must be 'profile' or 'sampler', got {self.method!r}")
        self.profile_ul_methods = tuple(self.profile_ul_methods)
        unknown = set(self.profile_ul_methods) - set(PROFILE_UL_METHODS)
        if unknown:
            raise ValueError(f"Case {self.name}: unknown profile_ul_methods {unknown}, choices are {PROFILE_UL_METHODS}")
        self.sampler_ul_methods = tuple(self.sampler_ul_methods)
        unknown = set(self.sampler_ul_methods) - set(SAMPLER_UL_METHODS)
        if unknown:
            raise ValueError(f"Case {self.name}: unknown sampler_ul_methods {unknown}, choices are {SAMPLER_UL_METHODS}")
        if self.norm_interp == "linear":
            self.norm_interp = "lin"

    @property
    def ul_methods(self):
        """UL methods (one output row each) for this case."""
        return self.profile_ul_methods if self.method == "profile" else self.sampler_ul_methods

    def make_norm(self, scan_max=None):
        """Norm parameter for the `FluxPointsEstimator`."""
        from gammapy.modeling import Parameter

        return Parameter(
            name="norm",
            value=1,
            min=self.norm_min,
            scan_min=self.norm_scan_min,
            scan_max=self.norm_scan_max if scan_max is None else scan_max,
            scan_n_values=self.norm_scan_n_values,
            interp=self.norm_interp,
        )

    def to_dict(self):
        d = asdict(self)
        d["profile_ul_methods"] = list(self.profile_ul_methods)
        d["sampler_ul_methods"] = list(self.sampler_ul_methods)
        return d


def default_cases(
    profile_ul_methods=("scipy",), sampler_ul_methods=("samples",), amplitude_negative_min=-3e-14, amplitude_max=1e-11, **kwargs
):
    """The 4 cases of the notebook: profile/sampler x constrained (zero)/unconstrained (negative).

    Extra ``kwargs`` (e.g. ``confidence_level_samples``, ``sampler_options``) are
    passed to every `Case`.
    """
    zero = dict(prior="zero", amplitude_min=0.0, amplitude_max=amplitude_max, **Case.norm_scan_zero)
    # same lower bound for the profile (FluxPointsEstimator norm, reference amplitude 1e-13) and the sampler
    norm_negative = dict(Case.norm_scan_negative)
    norm_negative["norm_min"] = norm_negative["norm_scan_min"] = amplitude_negative_min / REF_AMPLITUDE
    negative = dict(
        prior="negative", amplitude_min=amplitude_negative_min, amplitude_max=amplitude_max, **norm_negative
    )
    return [
        Case("profile-zero", "profile", profile_ul_methods=profile_ul_methods, **zero, **kwargs),
        Case("profile-negative", "profile", profile_ul_methods=profile_ul_methods, **negative, **kwargs),
        Case("sampler-zero", "sampler", sampler_ul_methods=sampler_ul_methods, **zero, **kwargs),
        Case("sampler-negative", "sampler", sampler_ul_methods=sampler_ul_methods, **negative, **kwargs),
    ]


# ---------------------------------------------------------------------------
# Dataset / model builders
# ---------------------------------------------------------------------------

def build_observation(livetime="1 h"):
    from gammapy.data import FixedPointingInfo, Observation, observatory_locations
    from gammapy.irf import load_irf_dict_from_file

    livetime = u.Quantity(livetime)

    pointing_position = SkyCoord(0, 0, unit="deg", frame="galactic")
    # We want to simulate an observation pointing at a fixed position in the sky.
    # For this, we use the `FixedPointingInfo` class
    pointing = FixedPointingInfo(fixed_icrs=pointing_position.icrs)

    irfs = load_irf_dict_from_file(
        "$GAMMAPY_DATA/cta-caldb/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz"
    )

    location = observatory_locations["ctao_south"]
    return Observation.create(pointing=pointing, livetime=livetime, irfs=irfs, location=location)


def build_dataset(obs, offset="0.5 deg"):
    from gammapy.datasets import SpectrumDataset
    from gammapy.makers import SpectrumDatasetMaker
    from gammapy.maps import MapAxis, RegionGeom

    offset = u.Quantity(offset)

    # Reconstructed and true energy axis
    energy_axis = MapAxis.from_energy_bounds(0.05, 50, 6, per_decade=True, unit="TeV")
    energy_axis_true = MapAxis.from_energy_bounds(0.02, 100, 12, per_decade=True, unit="TeV", name="energy_true")

    on_region_radius = 0.11 *u.deg

    pointing_position = obs.get_pointing_icrs(obs.tmid)
    center = pointing_position.directional_offset_by(position_angle=0 * u.deg, separation=offset)

    on_region = CircleSkyRegion(center=center, radius=on_region_radius)

    # Make the SpectrumDataset
    geom = RegionGeom.create(region=on_region, axes=[energy_axis])

    dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true, name="obs-0")
    maker = SpectrumDatasetMaker(selection=["exposure", "edisp", "background"])

    return maker.run(dataset_empty, obs)


def build_model_pow(amplitude=1e-12):
    from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

    model_simu = PowerLawSpectralModel(index=2, amplitude=f"{amplitude} cm-2 s-1 TeV-1")
    return SkyModel(spectral_model=model_simu, name="source")


def build_simulation(exposure, E1, E2, simu_amplitude=1e-20):
    """Build the dataset (fit range E1-E2) and the fit / simulation models.

    ``simu_amplitude`` (cm-2 s-1 TeV-1) is the injected signal, 1e-20 = no signal.

    Returns
    -------
    dataset, skymodel_fit, skymodel_simu
    """
    obs = build_observation(livetime=exposure)
    dataset = build_dataset(obs)
    dataset.mask_fit = dataset.counts.geom.energy_mask(E1, E2)

    # no signal
    skymodel_simu = build_model_pow(amplitude=simu_amplitude)  # default: almost no signal for simu
    skymodel_fit = build_model_pow(amplitude=REF_AMPLITUDE)  # reasonable starting point for fitting UL
    skymodel_fit.spectral_model.index.frozen = True
    skymodel_simu.spectral_model.index.frozen = True

    return dataset, skymodel_fit, skymodel_simu


def background_counts(dataset):
    """Expected background counts in the ON region, within the fit/safe mask."""
    return float(dataset.npred_background().data[dataset.mask].sum())


# ---------------------------------------------------------------------------
# Fake datasets. They all work on copies: the input dataset and model are never modified.
# ---------------------------------------------------------------------------

def fake_dataset(dataset, model, seed):
    """`SpectrumDataset` with Poisson counts from ``model`` + known background."""
    faked = dataset.copy(name=dataset.name)
    faked.models = model.copy()
    faked.fake(random_state=seed)
    return faked


def fake_dataset_on_off(dataset, model, seed):
    """`SpectrumDatasetOnOff` (acceptance_off=10) with Poisson ON and OFF counts."""
    from gammapy.datasets import SpectrumDatasetOnOff

    dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
        dataset=dataset.copy(name=dataset.name), acceptance=1, acceptance_off=10
    )
    dataset_on_off.models = model.copy()
    dataset_on_off.fake(npred_background=dataset.npred_background(), random_state=seed)
    return dataset_on_off


def asimov_dataset(dataset, model):
    """Asimov dataset: counts = npred (no Poisson noise)."""
    faked = dataset.copy(name=dataset.name)
    faked.models = model.copy()
    faked.counts = faked.npred()
    return faked


def fake(dataset, model, seed, dataset_type="on"):
    """Dispatch to the fake function corresponding to ``dataset_type``."""
    if dataset_type == "on":
        return fake_dataset(dataset, model, seed)
    elif dataset_type == "onoff":
        return fake_dataset_on_off(dataset, model, seed)
    elif dataset_type == "asimov":
        return asimov_dataset(dataset, model)
    raise ValueError(f"Dataset type {dataset_type!r} not supported. Choices are {DATASET_TYPES}")


# ---------------------------------------------------------------------------
# Upper limits for one case
# ---------------------------------------------------------------------------

def _amplitude(model):
    return model.spectral_model.amplitude


def _ul_from_scan_interp(norm_scan, dstat, delta_ts, n_values, interp):
    """Notebook "interp1d" method: closest point to ``delta_ts`` on a 50x finer grid.

    Only the part of the profile above its minimum is searched, otherwise for an
    unconstrained profile the lower crossing could be returned.
    """
    f = interpolate.interp1d(norm_scan, dstat, fill_value="extrapolate")
    if interp == "log":
        grid = np.logspace(np.log10(norm_scan.min()), np.log10(norm_scan.max()), n_values * 50)
    else:
        grid = np.linspace(norm_scan.min(), norm_scan.max(), n_values * 50)
    grid = grid[grid >= norm_scan[np.nanargmin(dstat)]]
    if np.nanmax(dstat[norm_scan >= grid[0]]) < delta_ts:
        # otherwise the scan edge would be returned as UL
        raise RuntimeError("Stat profile does not reach delta_ts within the scan range.")
    return grid[np.argmin(np.abs(f(grid) - delta_ts))]


def _scan_reaches(res, delta_ts):
    """Whether the stat profile rises by ``delta_ts`` above its minimum, on its upper side."""
    stat = res.stat_scan.data.flatten()
    if not np.any(np.isfinite(stat)):
        return False
    i_min = np.nanargmin(stat)
    return bool(np.nanmax(stat[i_min:]) - stat[i_min] >= delta_ts)


def _coarse_grid(a, b, n_coarse=200):
    """Symmetric log-spaced grid on [a, b] (b > 0), dense around 0, used to bracket the likelihood minimum."""
    tiny = 1e-6 * min(abs(b), abs(a) if a != 0 else abs(b))
    coarse = np.geomspace(tiny, b, n_coarse)
    if a < 0:
        return np.concatenate([-np.geomspace(-a, tiny, n_coarse // 2), [0.0], coarse])
    return np.concatenate([[a], coarse[coarse > a]])


def _profile_ul_exact(dataset, amplitude, a, b, delta_ts):
    """Profile UL by direct minimisation in [a, b] and root finding. Returns (ul, best_fit, ts).

    Only valid if ``amplitude`` is the only free parameter. ``ts`` is stat(0) - stat_min.
    """
    from scipy.optimize import brentq, minimize_scalar

    def stat(x):
        amplitude.value = x
        value = dataset.stat_sum()
        return value if np.isfinite(value) else np.inf

    coarse = _coarse_grid(a, b)
    s = np.array([stat(x) for x in coarse])
    i = int(np.argmin(s))
    if not np.isfinite(s[i]):
        raise RuntimeError("No finite fit statistic within the bounds.")
    lo, hi = coarse[max(i - 1, 0)], coarse[min(i + 1, len(coarse) - 1)]
    res = minimize_scalar(stat, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-4 * (hi - lo)))
    x_min, s_min = (res.x, res.fun) if res.fun < s[i] else (coarse[i], s[i])
    above = np.nonzero((s - s_min > delta_ts) & (coarse > x_min))[0]
    if len(above) == 0:
        raise RuntimeError("Stat profile does not reach delta_ts within the bounds.")
    x_hi = coarse[above[0]]
    ul = brentq(lambda x: stat(x) - s_min - delta_ts, x_min, x_hi, xtol=1e-6 * (x_hi - x_min))
    return ul, x_min, stat(0.0) - s_min


def _run_profile(case, dataset, fit_model, fpe_config, E1, E2):
    """Profile likelihood ULs. Returns {ul_method: (ul, success, message)}, extra."""
    from gammapy.estimators import FluxPointsEstimator
    from gammapy.modeling import stat_profile_ul_scipy

    model = fit_model.copy(name=fit_model.name)
    amplitude = _amplitude(model)
    amplitude.min = case.amplitude_min
    amplitude.max = case.amplitude_max
    amplitude.scan_min = case.amplitude_min
    amplitude.scan_max = case.amplitude_max
    dataset.models = model
    delta_ts = fpe_config.get("n_sigma_ul", 2) ** 2

    uls, extra = {}, {}
    if "exact" in case.profile_ul_methods:
        try:
            if len(dataset.models.parameters.free_unique_parameters) != 1:
                raise RuntimeError("exact method requires amplitude to be the only free parameter")
            ul, best_fit, ts = _profile_ul_exact(dataset, amplitude, case.amplitude_min, case.amplitude_max, delta_ts)
            uls["exact"] = (float(ul), bool(np.isfinite(ul)), "")
            extra = {"best_fit": float(best_fit), "ts": float(ts)}
        except Exception as exc:
            uls["exact"] = (np.nan, False, f"{type(exc).__name__}: {exc}")
    if set(case.profile_ul_methods) == {"exact"}:
        return uls, extra
    amplitude.value = REF_AMPLITUDE

    config = dict(fpe_config)
    config["n_jobs"] = 1  # no nested multiprocessing
    if "selection_optional" not in config:
        # compute only what is needed
        selection = ["sensitivity"]
        if {"scipy", "interp"} & set(case.profile_ul_methods):
            selection.append("scan")
        if "fpe" in case.profile_ul_methods:
            selection.append("ul")
        config["selection_optional"] = selection

    refamp = amplitude.quantity.to_value(UL_UNIT)
    uses_scan = bool({"scipy", "interp"} & set(case.profile_ul_methods))
    scan_max = case.norm_scan_max
    # largest norm allowed by the amplitude bound
    scan_max_limit = max(case.amplitude_max / refamp, case.norm_scan_max)

    for n_extension in range(case.norm_scan_max_extensions + 1):
        # FluxPointsEstimator so that we have the amplitude param, not the flux
        estimator = FluxPointsEstimator(
            source=model.name, energy_edges=[E1, E2], norm=case.make_norm(scan_max), **config
        )
        res = estimator.run([dataset])
        if not uses_scan or _scan_reaches(res, delta_ts) or scan_max >= scan_max_limit:
            break
        # the profile does not reach delta_ts: the UL is beyond the scan range
        scan_max = min(scan_max * case.norm_scan_extend_factor, scan_max_limit)
    scan_note = f"norm scan extended to {scan_max:g}" if scan_max != case.norm_scan_max else ""

    refamp = res.reference_spectral_model.amplitude.quantity.to_value(UL_UNIT)
    for key, column, scale in [("sensitivity", "norm_sensitivity", refamp), ("best_fit", "norm", refamp), ("ts", "ts", 1)]:
        try:
            extra.setdefault(key, float(res[column].data.flatten()[0]) * scale)
        except Exception:
            extra.setdefault(key, np.nan)

    if {"scipy", "interp"} & set(case.profile_ul_methods):
        norm_values = res.stat_scan.geom.axes["norm"].center
        stat = res.stat_scan.data.flatten()
        dstat = stat - np.nanmin(stat)

    for method in case.profile_ul_methods:
        if method == "exact":
            continue
        try:
            if method == "scipy":
                ul = stat_profile_ul_scipy(norm_values, dstat, delta_ts=delta_ts) * refamp
            elif method == "fpe":
                ul = float(res["norm_ul"].data.flatten()[0]) * refamp
            elif method == "interp":
                ul = _ul_from_scan_interp(
                    np.asarray(norm_values) * refamp, dstat, delta_ts, case.norm_scan_n_values, case.norm_interp
                )
            ul = float(ul)
            uls[method] = (ul, bool(np.isfinite(ul)), scan_note if method != "fpe" else "")
        except Exception as exc:
            uls[method] = (np.nan, False, f"{type(exc).__name__}: {exc}")
    return uls, extra


def _bayes_ul_grid(dataset, amplitude, a, b, confidence_level, n_coarse=200, n_fine=1000, max_dstat=50):
    """Posterior quantile of the amplitude, uniform prior on [a, b], by numerical integration.

    The posterior is exp(-(stat - stat_min) / 2) on [a, b] (stat = -2 log L).
    A coarse log-spaced grid locates the region where stat - stat_min < max_dstat,
    which is then integrated on a fine linear grid. Only valid if ``amplitude``
    is the only free parameter. Returns (ul, posterior median).
    """
    def stat(x):
        amplitude.value = x
        value = dataset.stat_sum_likelihood()
        return value if np.isfinite(value) else np.inf

    coarse = _coarse_grid(a, b, n_coarse)
    s = np.array([stat(x) for x in coarse])
    i_min = int(np.argmin(s))
    if not np.isfinite(s[i_min]):
        raise RuntimeError("No finite fit statistic in the prior range.")
    above = np.nonzero(s - s[i_min] > max_dstat)[0]
    hi = coarse[above[above > i_min][0]] if np.any(above > i_min) else b
    lo = coarse[above[above < i_min][-1]] if np.any(above < i_min) else coarse[0]

    fine = np.linspace(lo, hi, n_fine)
    s = np.array([stat(x) for x in fine])
    post = np.exp(-0.5 * (s - np.min(s)))
    cdf = np.concatenate([[0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(fine))])
    cdf /= cdf[-1]
    return float(np.interp(confidence_level, cdf, fine)), float(np.interp(0.5, cdf, fine))


def _run_sampler(case, dataset, fit_model):
    """Bayesian UL: quantile of the amplitude posterior (uniform prior)."""
    from gammapy.modeling import Sampler
    from gammapy.modeling.models import UniformPrior

    model = fit_model.copy(name=fit_model.name)
    amplitude = _amplitude(model)
    amplitude.prior = UniformPrior(min=case.amplitude_min, max=case.amplitude_max)
    dataset.models = model

    uls, extra = {}, {}
    if "samples" in case.sampler_ul_methods:
        sampler = Sampler(backend=case.sampler_backend, options=dict(case.sampler_options))
        result = sampler.run([dataset])

        # select the amplitude column by identity among the free parameters (= sampler columns)
        free = dataset.models.parameters.free_unique_parameters
        idx = [i for i, par in enumerate(free) if par is amplitude]
        if len(idx) != 1:
            raise RuntimeError(f"Amplitude not found among free parameters {free.names}")
        samples = np.atleast_2d(np.asarray(result.samples).T).T
        ul = float(np.quantile(samples[:, idx[0]], case.confidence_level_samples))
        message = "" if result.success else "sampler not converged"
        uls["samples"] = (ul, bool(np.isfinite(ul)), message)
        extra["best_fit"] = float(np.median(samples[:, idx[0]]))

    if "grid" in case.sampler_ul_methods:
        try:
            if len(dataset.models.parameters.free_unique_parameters) != 1:
                raise RuntimeError("grid method requires amplitude to be the only free parameter")
            ul, median = _bayes_ul_grid(
                dataset, amplitude, case.amplitude_min, case.amplitude_max, case.confidence_level_samples
            )
            uls["grid"] = (ul, bool(np.isfinite(ul)), "")
            extra.setdefault("best_fit", median)
        except Exception as exc:
            uls["grid"] = (np.nan, False, f"{type(exc).__name__}: {exc}")
    return uls, extra


def run_case(case, dataset, fit_model, fpe_config, E1, E2):
    """Run one case on an already faked dataset (dataset is copied, not modified).

    Returns
    -------
    uls : dict
        {ul_method: (ul in cm-2 s-1 TeV-1, success, message)}
    extra : dict
        Scalars stored with each row (see `EXTRA_COLUMNS`), missing ones are NaN:
        sensitivity (profile), best_fit (profile: maximum likelihood amplitude,
        sampler: posterior median), ts (profile).
    """
    dataset = dataset.copy(name=dataset.name)
    if case.method == "profile":
        uls, extra = _run_profile(case, dataset, fit_model, fpe_config, E1, E2)
    else:
        uls, extra = _run_sampler(case, dataset, fit_model)
    return uls, {key: extra.get(key, np.nan) for key in EXTRA_COLUMNS}


def run_realization(dataset, simu_model, fit_model, cases, fpe_config, E1, E2, seed, dataset_type="on"):
    """Fake one dataset for ``seed`` and run all ``cases`` on that same faked dataset.

    Exceptions are caught per case: the corresponding rows get ``ul=NaN`` and
    ``success=False`` (the error is stored in ``message``).

    Returns
    -------
    rows : list of dict
        One plain dict (scalars only) per case x ul_method.
    """
    try:
        faked = fake(dataset, simu_model, seed, dataset_type)
        fake_error = None
    except Exception as exc:
        faked, fake_error = None, f"fake failed: {type(exc).__name__}: {exc}"

    rows = []
    for case in cases:
        if fake_error is None:
            try:
                uls, extra = run_case(case, faked, fit_model, fpe_config, E1, E2)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                uls, extra = {m: (np.nan, False, msg) for m in case.ul_methods}, {}
        else:
            uls, extra = {m: (np.nan, False, fake_error) for m in case.ul_methods}, {}

        for ul_method, (ul, success, message) in uls.items():
            rows.append(
                dict(
                    seed=int(seed) if dataset_type != "asimov" else ASIMOV_SEED,
                    dataset_type=dataset_type,
                    case=case.name,
                    method=case.method,
                    prior=case.prior,
                    ul_method=ul_method,
                    ul=ul,
                    **{key: extra.get(key, np.nan) for key in EXTRA_COLUMNS},
                    success=success,
                    message=message,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------

def _init_worker(quiet):
    """Worker initializer (spawn-safe, module level)."""
    for var in _THREAD_ENV_VARS:
        os.environ[var] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass
    if quiet:
        warnings.simplefilter("ignore")
        logging.getLogger("gammapy").setLevel(logging.ERROR)
        logging.getLogger("ultranest").setLevel(logging.ERROR)


def _task(exposure_index, dataset, simu_model, fit_model, cases, fpe_config, E1, E2, seed, dataset_type, quiet=True):
    args = (dataset, simu_model, fit_model, cases, fpe_config, E1, E2, seed, dataset_type)
    if not quiet:
        return exposure_index, run_realization(*args)
    # the sampler prints a summary on stdout even with show_status=False
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return exposure_index, run_realization(*args)


def _make_seeds(seed, n_runs):
    if isinstance(seed, str):
        if seed == "range":  # the seed in range(n_runs): the same every time you run the script
            return np.arange(n_runs)
        if seed in ("random", "random-seed", "random-state"):  # random seeds, recorded in the table
            return np.random.default_rng().integers(0, 2**31 - 1, n_runs)
        raise ValueError(f"Unknown seed option {seed!r}, use 'range', 'random' or a list of int")
    if np.isscalar(seed):  # offset
        return int(seed) + np.arange(n_runs)
    seeds = np.asarray(seed, dtype=int)
    if len(seeds) != n_runs:
        raise ValueError("Explicit seed list must have n_runs elements")
    return seeds


def run_simulation(
    exposures,
    cases,
    n_runs,
    fpe_config,
    E1,
    E2,
    dataset_types=("on", "asimov"),
    seed="range",
    n_jobs=None,
    quiet=True,
    progress=True,
    simu_amplitude=1e-20,
):
    """Run all ``cases`` on ``n_runs`` fake datasets for each exposure.

    For each exposure the dataset is built once. Then, for each dataset type
    and seed, one dataset is faked and all cases are run on it. Realizations
    are distributed over ``n_jobs`` processes (``ProcessPoolExecutor``, spawn).
    The "asimov" dataset type has a single realization.

    Parameters
    ----------
    exposures : list of str or Quantity
        Livetimes, e.g. ["0.03 h", "0.3 h", "100 h"].
    cases : list of `Case`
    n_runs : int
        Number of Poisson realizations per exposure (for non-Asimov types).
    fpe_config : dict
        Extra arguments for `FluxPointsEstimator` (e.g. n_sigma_ul,
        n_sigma_sensitivity). If ``selection_optional`` is not given, only the
        quantities needed by the cases are computed. The profile UL threshold
        is ``delta_ts = n_sigma_ul**2``.
    E1, E2 : Quantity
        Fit energy range.
    dataset_types : tuple of str
        Among "on", "onoff", "asimov".
    seed : "range", "random", int or list of int
        "range": seeds 0..n_runs-1 (reproducible), "random": random seeds
        (recorded in the table), int: offset, list: explicit seeds.
    n_jobs : int, optional
        Number of processes. None: ``os.cpu_count()``. 1: run serially in this process.
    quiet : bool
        Silence warnings and gammapy/ultranest logging in the workers.
    progress : bool
        Show a tqdm progress bar.
    simu_amplitude : float or dict
        Injected amplitude in cm-2 s-1 TeV-1 (1e-20 = background only), or
        {exposure: amplitude} to inject a different signal per exposure (e.g.
        for coverage tests).

    Returns
    -------
    table : `~astropy.table.QTable`
        Long format, one row per exposure x dataset_type x seed x case x ul_method.
    """
    names = [c.name for c in cases]
    if len(set(names)) != len(names):
        raise ValueError(f"Case names must be unique: {names}")
    for dataset_type in dataset_types:
        if dataset_type not in DATASET_TYPES:
            raise ValueError(f"Dataset type {dataset_type!r} not supported. Choices are {DATASET_TYPES}")

    exposures = [str(e) for e in exposures]
    seeds = _make_seeds(seed, n_runs)

    # build datasets once per exposure, in the main process
    built = []
    for exposure in exposures:
        amp = simu_amplitude[exposure] if isinstance(simu_amplitude, dict) else simu_amplitude
        dataset, fit_model, simu_model = build_simulation(exposure, E1, E2, simu_amplitude=amp)
        bkg = background_counts(dataset)
        log.info(f"exposure={exposure}: expected background in ON region: {bkg:.2f}")
        built.append((exposure, dataset, fit_model, simu_model, bkg))

    tasks = []
    for i, (exposure, dataset, fit_model, simu_model, bkg) in enumerate(built):
        for dataset_type in dataset_types:
            type_seeds = [ASIMOV_SEED] if dataset_type == "asimov" else seeds
            for s in type_seeds:
                tasks.append((i, dataset, simu_model, fit_model, cases, fpe_config, E1, E2, int(s), dataset_type, quiet))

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(int(n_jobs), len(tasks)))

    from tqdm.auto import tqdm

    pbar = tqdm(total=len(tasks), desc="realizations", disable=not progress)
    results = []
    if n_jobs == 1:
        for task in tasks:
            results.append(_task(*task))
            pbar.update()
    else:
        # children inherit the environment at spawn time: limit BLAS/OpenMP threads before numpy is imported there
        saved_env = {var: os.environ.get(var) for var in _THREAD_ENV_VARS}
        for var in _THREAD_ENV_VARS:
            os.environ[var] = "1"
        try:
            with ProcessPoolExecutor(
                max_workers=n_jobs, mp_context=get_context("spawn"), initializer=_init_worker, initargs=(quiet,)
            ) as executor:
                futures = [executor.submit(_task, *task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update()
        finally:
            for var, value in saved_env.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
    pbar.close()

    rows = []
    for i, task_rows in results:
        exposure, _, _, _, bkg = built[i]
        livetime = u.Quantity(exposure).to_value("h")
        for row in task_rows:
            simu_amp = _amplitude(built[i][3]).quantity.to_value(UL_UNIT)
            rows.append(dict(exposure=exposure, livetime=livetime, bkg_counts=bkg, simu_amplitude=simu_amp, **row))

    table = _rows_to_table(rows)
    table.meta.update(
        dict(
            E1=str(E1),
            E2=str(E2),
            fpe_config=_yaml_safe(fpe_config),
            delta_ts=float(fpe_config.get("n_sigma_ul", 2) ** 2),
            n_runs=int(n_runs),
            seed=seed if isinstance(seed, str) else "explicit",
            exposures=exposures,
            bkg_counts={exposure: bkg for exposure, _, _, _, bkg in built},
            simu_amplitude={b[0]: float(_amplitude(b[3]).quantity.to_value(UL_UNIT)) for b in built},
            dataset_types=list(dataset_types),
            cases=[c.to_dict() for c in cases],
            gammapy_version=_gammapy_version(),
            date=datetime.datetime.now().isoformat(timespec="seconds"),
        )
    )
    return table


_COLUMNS = (
    "exposure", "livetime", "bkg_counts", "simu_amplitude", "dataset_type", "seed",
    "case", "method", "prior", "ul_method", "ul", "sensitivity", "best_fit", "ts", "success", "message",
)

def _rows_to_table(rows):
    order = {"on": 0, "onoff": 1, "asimov": 2}
    rows = sorted(rows, key=lambda r: (r["livetime"], order[r["dataset_type"]], r["seed"], r["case"], r["ul_method"]))
    table = Table(rows=rows, names=_COLUMNS) if rows else Table(names=_COLUMNS)
    table["livetime"].unit = u.h
    table["ul"].unit = UL_UNIT
    for column in ("sensitivity", "best_fit", "simu_amplitude"):
        table[column].unit = UL_UNIT
    return QTable(table)


def _gammapy_version():
    import gammapy

    return gammapy.__version__


def _yaml_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


if __name__ == "__main__":
    exposures = ["0.03 h", "0.3 h", "100 h"]
    AMPLITUDE_MAX = 1e-11
    AMPLITUDE_NEGATIVE_MIN = -1e-11
    CL_SAMPLES = 0.95
    case_options = dict(
        confidence_level_samples=CL_SAMPLES,
        amplitude_negative_min=AMPLITUDE_NEGATIVE_MIN,
        amplitude_max=AMPLITUDE_MAX,
    )
    cases = default_cases(profile_ul_methods=("exact",), sampler_ul_methods=("grid",), **case_options)
    n_runs = 1000
    fpe_config = {
        "n_sigma_ul": 1.645,
        "n_sigma_sensitivity": 2,
    }
    E1, E2 = 0.05 * u.TeV, 50 * u.TeV
    N_JOBS = 8

    table = run_simulation(
        exposures,
        cases,
        n_runs,
        fpe_config,
        E1,
        E2,
        dataset_types=("on", "asimov"),
        n_jobs=N_JOBS,
    )

    table.write("uls_main_dev.ecsv", format="ascii.ecsv", overwrite=True)
