#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.units import Quantity
from astropy.visualization import quantity_support
from gammapy.maps import MapAxis, RegionNDMap
from gammapy.modeling.models import PowerLawSpectralModel, ExpCutoffPowerLawSpectralModel
from gammapy.modeling.models.spectral import scale_plot_flux


class SpectralErrorPropagation:
    """
    Compute error bands using analytical error propagation.

    Parameters
    ----------
    model : `~gammapy.modeling.models.SpectralModel`
        The model on wich to apply error propagation
    epsilon : float
        Step size of the gradient evaluation. Given as a fraction of the parameter error.
 
    
    """
    
    def __init__(self, model, epsilon=1e-4):
        self.model = model
        self.epsilon = epsilon

    def _propagate_error(self, fct, **kwargs):
        """Evaluate error for a given function with uncertainty propagation.

        Parameters
        ----------
        fct : `~astropy.units.Quantity`
            Function to estimate the error.
        epsilon : float
            Step size of the gradient evaluation. Given as a
            fraction of the parameter error.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        f_cov : `~astropy.units.Quantity`
            Error of the given function.
        """
        eps = np.sqrt(np.diag(self.model.covariance)) * self.epsilon

        n, f_0 = len(self.model.parameters), fct(**kwargs)
        shape = (n, len(np.atleast_1d(f_0)))
        df_dp = np.zeros(shape)

        for idx, parameter in enumerate(self.model.parameters):
            if parameter.frozen or eps[idx] == 0:
                continue

            parameter.value += eps[idx]
            df = fct(**kwargs) - f_0

            df_dp[idx] = df.value / eps[idx]
            parameter.value -= eps[idx]

        f_cov = df_dp.T @ self.model.covariance @ df_dp
        f_err = np.sqrt(np.diagonal(f_cov))
        return u.Quantity([np.atleast_1d(f_0.value), f_err], unit=f_0.unit).squeeze()

    def evaluate_error(self, energy):
        """Evaluate spectral model with error propagation.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy at which to evaluate.

        Returns
        -------
        dnde, dnde_error : tuple of `~astropy.units.Quantity`
            Tuple of flux and flux error.
        """
        return self._propagate_error(fct=self.model, energy=energy)

    def integral(self, energy_min, energy_max, **kwargs):
        r"""Integrate spectral model numerically if no analytical solution defined.

        .. math::
            F(E_{min}, E_{max}) = \int_{E_{min}}^{E_{max}} \phi(E) dE

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Lower and upper bound of integration range.
        **kwargs : dict
            Keyword arguments passed to :func:`~gammapy.modeling.models.spectral.integrate_spectrum`.
        """
        if hasattr(self, "evaluate_integral"):
            kwargs = {par.name: par.quantity for par in self.parameters}
            kwargs = self._convert_evaluate_unit(kwargs, energy_min)
            return self.evaluate_integral(energy_min, energy_max, **kwargs)
        else:
            return integrate_spectrum(self, energy_min, energy_max, **kwargs)


    def integral_error(self, energy_min, energy_max, **kwargs):
        """Evaluate the error of the integral flux of a given spectrum in a given energy range.

        Parameters
        ----------
        energy_min, energy_max :  `~astropy.units.Quantity`
            Lower and upper bound of integration range.

        Returns
        -------
        flux, flux_err : tuple of `~astropy.units.Quantity`
            Integral flux and flux error between energy_min and energy_max.
        """
        return self._propagate_error(
            fct=self.integral,
            energy_min=energy_min,
            energy_max=energy_max,
            **kwargs,
        )

    
    def energy_flux_error(self, energy_min, energy_max, **kwargs):
        """Evaluate the error of the energy flux of a given spectrum in a given energy range.

        Parameters
        ----------
        energy_min, energy_max :  `~astropy.units.Quantity`
            Lower and upper bound of integration range.

        Returns
        -------
        energy_flux, energy_flux_err : tuple of `~astropy.units.Quantity`
            Energy flux and energy flux error between energy_min and energy_max.
        """
        return self._propagate_error(
            fct=self.energy_flux,
            energy_min=energy_min,
            energy_max=energy_max,
            **kwargs,
        )

def _get_plot_flux(model, energy, sed_type):
    flux = RegionNDMap.create(region=None, axes=[energy])
    flux_err = RegionNDMap.create(region=None, axes=[energy])

    error_model = SpectralErrorPropagation(model)
    
    if sed_type in ["dnde", "norm"]:
        flux.quantity, flux_err.quantity = error_model.evaluate_error(energy.center)

    elif sed_type == "e2dnde":
        flux.quantity, flux_err.quantity = energy.center**2 * error_model.evaluate_error(
            energy.center
        )

    elif sed_type == "flux":
        flux.quantity, flux_err.quantity = error_model.integral_error(
            energy.edges_min, energy.edges_max
        )

    elif sed_type == "eflux":
        flux.quantity, flux_err.quantity = error_model.energy_flux_error(
            energy.edges_min, energy.edges_max
        )
    else:
        raise ValueError(f"Not a valid SED type: '{sed_type}'")

    return flux, flux_err


def plot_error_model(
        model,
        energy_bounds,
        ax=None,
        sed_type="dnde",
        energy_power=0,
        n_points=100,
        **kwargs,
    ):
        """Plot spectral model error band.

        .. note::

            This method calls ``ax.set_yscale("log", nonpositive='clip')`` and
            ``ax.set_xscale("log", nonposx='clip')`` to create a log-log representation.
            The additional argument ``nonposx='clip'`` avoids artefacts in the plot,
            when the error band extends to negative values (see also
            https://github.com/matplotlib/matplotlib/issues/8623).

            When you call ``plt.loglog()`` or ``plt.semilogy()`` explicitly in your
            plotting code and the error band extends to negative values, it is not
            shown correctly. To circumvent this issue also use
            ``plt.loglog(nonposx='clip', nonpositive='clip')``
            or ``plt.semilogy(nonpositive='clip')``.

        Parameters
        ----------
        energy_bounds : `~astropy.units.Quantity`, list of `~astropy.units.Quantity` or `~gammapy.maps.MapAxis`
            Energy bounds between which the model is to be plotted. Or an
            axis defining the energy bounds between which the model is to be plotted.
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        sed_type : {"dnde", "flux", "eflux", "e2dnde"}
            Evaluation methods of the model. Default is "dnde".
        energy_power : int, optional
            Power of energy to multiply flux axis with. Default is 0.
        n_points : int, optional
            Number of evaluation nodes. Default is 100.
        **kwargs : dict
            Keyword arguments forwarded to `matplotlib.pyplot.fill_between`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes.

        Notes
        -----
        If ``energy_bounds`` is supplied as a list, tuple, or Quantity, an ``energy_axis`` is created internally with
        ``n_points`` bins between the given bounds.
        """
        from gammapy.estimators.map.core import DEFAULT_UNIT

        if model.is_norm_spectral_model:
            sed_type = "norm"

        if isinstance(energy_bounds, (tuple, list, Quantity)):
            energy_min, energy_max = energy_bounds
            energy = MapAxis.from_energy_bounds(
                energy_min,
                energy_max,
                n_points,
            )
        elif isinstance(energy_bounds, MapAxis):
            energy = energy_bounds

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("facecolor", "black")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)

        if ax.yaxis.units is None:
            ax.yaxis.set_units(DEFAULT_UNIT[sed_type] * energy.unit**energy_power)

        flux, flux_err = _get_plot_flux(model, sed_type=sed_type, energy=energy)
        y_lo = scale_plot_flux(flux - flux_err, energy_power).quantity[:, 0, 0]
        y_hi = scale_plot_flux(flux + flux_err, energy_power).quantity[:, 0, 0]

        with quantity_support():
            ax.fill_between(energy.center, y_lo, y_hi, **kwargs)

        model._plot_format_ax(ax, energy_power, sed_type)
        return ax


def make_figure(model, e_range=[0.2,20]*u.TeV):
    fig = plt.figure(figsize=(6, 5))
    ax = model.plot(e_range, sed_type="e2dnde")
    ax = model.plot_error(e_range, sed_type="e2dnde", label='Sampling', ax=ax)
    ax = plot_error_model(model, e_range, sed_type="e2dnde", facecolor='r', label='Error propagation', ax=ax)
    ax.set_ylabel(r"$\rm E^2\frac{dN}{dE}, [erg\,cm^{-2}\,s^{-1}]$")
    ax.set_xlim(e_range)
    plt.legend()
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Figure with PL only
    pl = PowerLawSpectralModel(index=2.3)
    pl.index.error = 0.3
    pl.amplitude.error = 0.3 * pl.amplitude.value
    fig1 = make_figure(pl)
    fig1.savefig("figures/sampling_error_propagation_comparison_pl.pdf", dpi=200)

    # Figure with ECPL
    ecpl = ExpCutoffPowerLawSpectralModel(index=2.3, lambda_='0.1 TeV-1')
    ecpl.index.error=0.2
    ecpl.amplitude.error = 0.2*pl.amplitude.value
    ecpl.lambda_.error = 0.03
    fig2 = make_figure(ecpl, e_range=[0.2,40]*u.TeV)
    fig2.savefig("figures/sampling_error_propagation_comparison_ecpl.pdf", dpi=200)

    plt.show()


