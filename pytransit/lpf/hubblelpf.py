#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import Optional, Union, List
from math import ceil, floor

# import astropy.units as u

from pathlib import Path

# from astropy.stats import sigma_clipped_stats
# from astropy.table import Table
# from astropy.time import Time
# from astropy.timeseries import TimeSeries
from corner import corner
from matplotlib.pyplot import setp
from matplotlib.pyplot import subplots
from numba import njit
from numpy import (
    zeros,
    squeeze,
    arange,
    digitize,
    full,
    nan,
    sqrt,
    # percentile,
    isfinite,
    argsort,
    ones_like,
    atleast_2d,
    # median,
    ndarray,
    # unique,
    # nanmedian,
    concatenate,
    array,
    linspace,
    pi,
    inf,
    where,
    roll,
    tile,
    exp,
    min as npmin,
)
# from numpy.random import permutation
# from pytransit.utils.tess import read_tess_spoc
import batman

from .loglikelihood import CeleriteLogLikelihood
from .lpf import BaseLPF
from .. import TransitModel
# from ..orbits import epoch
from ..utils.keplerlc import KeplerLC
from ..utils.misc import fold
from ..orbits.orbits_py import (
    # duration_eccentric,
    as_from_rhop,
    i_from_ba,
    # i_from_baew,
    # d_from_pkaiews,
    # epoch,
)
from ..utils.hubble import read_hubble_spoc
from ..param.parameter import (
    # ParameterSet,
    # PParameter,
    GParameter,
    # LParameter
    )
from ..param.parameter import (
    UniformPrior as U,
    # NormalPrior as N,
    # GammaPrior as GM
    )

try:
    from ldtk import tess

    with_ldtk = True
except ImportError:
    with_ldtk = False


@njit
def downsample_time(time, vals, inttime=1.0):
    duration = time.max() - time.min()
    nbins = int(ceil(duration / inttime))
    bins = arange(nbins)
    edges = time[0] + bins * inttime
    bids = digitize(time, edges) - 1
    bt, bv, be = full(nbins, nan), zeros(nbins), zeros(nbins)
    for i, bid in enumerate(bins):
        bmask = bid == bids
        if bmask.sum() > 0:
            bt[i] = time[bmask].mean()
            bv[i] = vals[bmask].mean()
            if bmask.sum() > 2:
                be[i] = vals[bmask].std() / sqrt(bmask.sum())
            else:
                be[i] = nan
    m = isfinite(be)
    return bt[m], bv[m], be[m]


class HubbleLPF(BaseLPF):
    bjdrefi = 2457000

    def __init__(
        self,
        name: str,
        datadir: Path = None,
        # tic: int = None,
        times: ndarray = None,
        fluxes: ndarray = None,
        errors: ndarray = None,
        zero_epoch: float = None,
        period: float = None,
        nsamples: int = 2,
        trdur: float = 0.125,
        bldur: float = 0.5,
        # use_pdc=True,
        sectors: Optional[Union[List[int], str]] = "all",
        split_transits=False,
        separate_noise=False,
        tm: TransitModel = None,
        data_scan=None,
        # minpt=10,
    ):
        df_lc, df_par = read_hubble_spoc(datadir)
        if times is None:
            times = df_lc["time"].values
        if fluxes is None:
            fluxes = df_lc["flux"].values
        if errors is None:
            errors = df_lc["flux_err"].values
        period = df_par.P.values[0]
        zero_epoch = df_par.t_mid_bjd_tdb.values[0]
        sectors = array([1 for time in df_lc.time.values])
        data_scan = df_lc.data_scan.values

        self.lc = KeplerLC(
            times, fluxes, sectors, zero_epoch, period, trdur, bldur
            )

        if split_transits:
            times = self.lc.time_per_transit
            # fluxes = self.lc.normalized_flux_per_transit
            fluxes = self.lc.flux_per_transit
        else:
            times = concatenate(self.lc.time_per_transit)
            # fluxes = concatenate(self.lc.normalized_flux_per_transit)
            fluxes = concatenate(self.lc.flux_per_transit)

        tref = floor(concatenate(array([times])).min())

        self.zero_epoch = zero_epoch
        self.period = period
        self.transit_duration = trdur
        self.baseline_duration = bldur

        wnids = arange(len(times)) if separate_noise else None
        BaseLPF.__init__(
            self,
            name,
            ["Hubble"],
            times=times,
            fluxes=fluxes,
            nsamples=nsamples,
            exptimes=0.00139,
            wnids=wnids,
            tref=tref,
            tm=tm,
        )
        self.tm.interpolate = False
        self.data_scan = data_scan

    def _init_lnlikelihood(self):
        self._add_lnlikelihood_model(CeleriteLogLikelihood(self))

    def add_ldtk_prior(teff, logg, z):
        if with_ldtk:
            super().add_ldtk_prior(teff, logg, z, passbands=(tess,))
        else:
            raise ImportError(
                "Could not import LDTk, cannot add an LDTk prior."
                )

    def batman_model(self, time):
        pv = self.posterior_samples()
        transit_par = batman.TransitParams()

        par = pv.median().values
        transit_par.t0 = par[0]
        transit_par.per = par[1]
        transit_par.rp = sqrt(par[4:5])
        transit_par.a = as_from_rhop(par[2], transit_par.per)
        transit_par.inc = i_from_ba(par[3], transit_par.a) / pi * 180
        # transit_par.inc = 90
        transit_par.ecc = 0
        transit_par.w = 180
        transit_par.limb_dark = "quadratic"
        transit_par.u = par[5:7]
        # transit_par.fp = model_fp_over_fs

        m = batman.TransitModel(transit_par, array(time))
        signal = m.light_curve(transit_par)

        return signal

    def _init_p_instrument(self):
        # pass
        pins = [
            # initial.append(0.001)
            # limits1.append(-10.0)
            # limits2.append(10.0)
            GParameter(
                "r_a1",
                "long term ramp - 1st order",
                "ppt",
                U(-10, 10),
                bounds=(-inf, inf),
            ),
            GParameter(
                "r_a2",
                "long term ramp - 2nd order",
                "ppt",
                U(-2, 2),
                bounds=(-inf, inf),
            ),
            GParameter(
                "r_b1",
                "short term ramp - amplitude",
                "ppt",
                U(-2, 2),
                bounds=(-inf, inf),
            ),
            GParameter(
                "mor_b1",
                "short term mid-orbit ramp - amplitude",
                "ppt",
                U(-2, 2),
                bounds=(-inf, inf),
            ),
            GParameter(
                "for_b1",
                "short term first-orbit ramp - amplitude",
                "ppt",
                U(-2, 2),
                bounds=(-inf, inf),
            ),
            GParameter(
                "r_b2",
                "short term ramp - decay",
                "d",
                U(0, 3.5),
                bounds=(0, inf),
            ),
            GParameter(
                "mor_b2",
                "short term mid-orbit ramp - decay",
                "d",
                U(0, 3.5),
                bounds=(0, inf),
            ),
            GParameter(
                "for_b2",
                "short term first-orbit ramp - decay",
                "d",
                U(0, 3.5),
                bounds=(0, inf),
            ),
        ]
        self.ps.add_global_block("instrument", pins)

    def _init_norm_p(self):
        p_norm = [
            GParameter(
                "norm_f",
                "normalisation factor - flux",
                "ppt", U(-10, 10)
                ),
            GParameter(
                "norm_r",
                "normalisation factor - ramp",
                "ppt", U(-10, 10)
                ),
        ]
        self.ps.add_global_block("norm", p_norm)

    def hubble_phase(self):
        htime = self.timea
        orbits = where(abs(htime - roll(htime, 1)) > 20.0 / 60.0 / 24.0)[0]
        dumps = where(abs(htime - roll(htime, 1)) > 5.0 / 60.0 / 24.0)[0]
        dphase = zeros(len(htime))
        for i in range(1, len(dumps)):
            if dumps[i] not in orbits:
                if i != len(dumps) - 1:
                    for j in range(dumps[i], dumps[i + 1]):
                        dphase[j] = 1
                else:
                    for j in range(dumps[i], len(dphase)):
                        dphase[j] = 1
        htime = self.timea
        orbits = where(abs(htime - roll(htime, 1)) > 5.0 / 60.0 / 24.0)[0]
        orbits = htime[orbits]
        fphase = where(htime < orbits[1], 1, 0)
        htime = self.timea
        orbits = where(abs(htime - roll(htime, 1)) > 5.0 / 60.0 / 24.0)[0]
        t0s = htime[orbits]
        ophase = []
        for pp in t0s:
            ppp = htime - pp
            ppp = where(ppp < 0, 1000, ppp)
            ophase.append(ppp)

        ophase = npmin(ophase, 0)

        return dphase, fphase, ophase

    def hubble_trend(self, pv):
        pv = atleast_2d(pv)
        model_r_a1 = pv[:, 7]
        model_r_a2 = pv[:, 8]
        model_r_b1 = pv[:, 9]
        model_mor_b1 = pv[:, 10]
        model_for_b1 = pv[:, 11]
        model_r_b2 = pv[:, 12]
        model_mor_b2 = pv[:, 13]
        model_for_b2 = pv[:, 14]
        model_mid_time = pv[:, 0] - self._tref
        model_norm_f = pv[:, 15]
        model_norm_r = pv[:, 16]

        # model_time = self.timea - self._tref
        model_time = tile(self.timea, (pv.shape[0], 1)).T - self._tref
        data_scan = self.data_scan

        data_scan = tile(data_scan, (pv.shape[0], 1)).T

        normalization = where(
            data_scan > 0, 10**model_norm_f, 10**model_norm_r
            )

        detrend1 = (
            1.0
            - model_r_a1 * (model_time - model_mid_time)
            + model_r_a2 * ((model_time - model_mid_time) ** 2)
        )

        data_dphase, data_fphase, data_ophase = self.hubble_phase()

        data_dphase = tile(data_dphase, (pv.shape[0], 1)).T
        data_fphase = tile(data_fphase, (pv.shape[0], 1)).T
        data_ophase = tile(data_ophase, (pv.shape[0], 1)).T

        ramp_ampl = where(data_dphase == 0, model_r_b1, model_mor_b1)
        ramp_ampl = where(data_fphase == 0, ramp_ampl, model_for_b1)
        ramp_decay = where(data_dphase == 0, model_r_b2, model_mor_b2)
        ramp_decay = where(data_fphase == 0, ramp_decay, model_for_b2)
        detrend2 = 1.0 - ramp_ampl * exp(-(10**ramp_decay) * data_ophase)

        return (normalization * detrend1 * detrend2).T

    def plot_results(self):
        pv = self.posterior_samples()
        time = self.timea
        flux = self.ofluxa
        tc_model = pv.median().values[0]
        trend = self.hubble_trend(pv.median().values)[0]

        delta_t = max(abs(time - tc_model))
        time_model = linspace(tc_model - delta_t, tc_model + delta_t, 1000)
        # time_model = np.linspace(time.max(), time.max(), 1000)
        flux_model = self.batman_model(time_model)

        fig, axes = subplots(3, 1, figsize=(10, 10), sharex=True)

        axes[0].plot(
            time,
            self.flux_model(pv.median().values)[0],
            "o",
            alpha=0.5,
            label="model"
        )
        axes[0].plot(time, flux, "k.", alpha=0.5, label="observation")
        axes[0].set_ylabel("Flux")
        axes[0].legend()
        # axes[0].set_xlim(time_model.min(), time_model.max())

        axes[1].plot(time_model, flux_model, "k-", alpha=0.5, label="model")
        axes[1].plot(time, flux / trend, "o", alpha=0.5, label="detrended")
        axes[1].set_ylabel("Normalized Flux")
        axes[1].legend()

        axes[2].plot(
            time,
            (flux / trend - self.transit_model(pv.median().values)) * 1e6,
            "o",
            alpha=0.5,
            label="model",
        )
        axes[2].set_xlabel("Time [BJD - 2457000]")
        axes[2].set_ylabel("Residuals [ppm]")

        return fig, axes

    def plot_folded_transit(
        self,
        method="de",
        figsize=(13, 6),
        ylim=(0.9975, 1.002),
        xlim=None,
        binwidth=8,
        remove_baseline: bool = False,
    ):
        if method == "de":
            pv = self.de.minimum_location
            tc, p = pv[[0, 1]]
        else:
            raise NotImplementedError

        phase = p * fold(self.timea, p, tc, 0.5)
        binwidth = binwidth / 24 / 60
        sids = argsort(phase)

        tm = self.transit_model(pv)

        if remove_baseline:
            gp = self._lnlikelihood_models[0]
            bl = squeeze(gp.predict_baseline(pv))
        else:
            bl = ones_like(self.ofluxa)

        bp, bfo, beo = downsample_time(
            phase[sids], (self.ofluxa / bl)[sids], binwidth
            )

        fig, ax = subplots(figsize=figsize)
        ax.plot(phase - 0.5 * p, self.ofluxa / bl, ".", alpha=0.15)
        ax.errorbar(bp - 0.5 * p, bfo, beo, fmt="ko")
        ax.plot(phase[sids] - 0.5 * p, tm[sids], "k")
        xlim = (
            xlim if xlim is not None else 1.01 * (
                bp[isfinite(bp)][[0, -1]] - 0.5 * p
                )
        )
        setp(
            ax,
            ylim=ylim,
            xlim=xlim,
            xlabel="Time - Tc [d]",
            ylabel="Normalised flux"
            )
        fig.tight_layout()
        return fig

    def plot_basic_posteriors(self):
        samples = self.posterior_samples()

        selected_samples = samples[
            ["tc",
             "p",
             "rho",
             "k2",
             "r_a1",
             "r_a2",
             "r_b1",
             "r_b2",
             "norm_f",
             "norm_r"
             ]
        ]

        # Plot the corner plot
        fig = corner(selected_samples)
        return fig
