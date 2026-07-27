import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import astropy.units as u
from astropy.time import Time
from astropy.table import Table
import astromet

from utils import astrometry_sim, rotation

from cmap import Colormap
cm = Colormap('tol:muted')

color_event, color_ref, color_true, color_sample, color_median = cm(1), cm(2), cm(3), cm(6), cm(7)

def compute_u_NE(time, t0, tE, u0, piE, phi, sc_a, fw, fz):
    # compute linear part of motion:
    tau = (time-t0)/tE
    beta = u0*np.ones_like(time)
    
    # linear motion in NE:
    u_linear_NE = rotation(np.pi - phi) @ np.array([tau, beta]) # source wrt. lens -> accounted for with reflection by pi
    
    # compute parallax part of motion:
    f_NE = np.column_stack((
        np.cos(sc_a) * fw + np.sin(sc_a) * fz,
        np.sin(sc_a) * fw - np.cos(sc_a) * fz
    ))
    
    u_parallax_NE = piE * f_NE.T * (-1) # source wrt. lens -> negative sign
    
    u_NE = u_linear_NE + u_parallax_NE
    
    return u_NE
    
    
def shift_from_u(u_NE):
    shift_NE = u_NE/(np.linalg.norm(u_NE, axis=0)**2 + 2)
    return shift_NE
    

def getFactors(ts_continuous, refEpoch, ra, dec):

    params = astromet.params()
    params.ra, params.dec = ra, dec
    params.parallax, params.pmrac, params.pmdec = 1, 0, 0
    params.epoch = refEpoch
    fE, fN = astromet.track(ts_continuous, params)

    return fE, fN


def plotContinuous(posterior=None, obsdata=None, num_samples=100, output_dir='', source_id='', residuals=True, errorbars=True, true_params=None):
    
    
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 18
    
    min_time, max_time = np.min(obsdata['time'])-0.1, np.max(obsdata['time'])+0.1
    ts_continuous = np.linspace(min_time, max_time, num=1000)

    refEpoch, ra, dec = obsdata['refEpoch'], obsdata['ra'], obsdata['dec']
    fE, fN = getFactors(ts_continuous, refEpoch, ra, dec)

    fig, ax = plt.subplots(figsize=(9,10))

    ax.plot([-2000, -2000], [0, 1], color=color_sample, zorder=2, lw=1, alpha=0.8, label='centroid (samples)')

    for sample in range(num_samples):
        plotSample(ax, ts_continuous, refEpoch, ra, dec, fE, fN,
        posterior['tE'][sample], posterior['u0'][sample], posterior['piE'][sample], posterior['pmrac_S'][sample], posterior['pmdec_S'][sample], posterior['thetaE'][sample], posterior['varpi_S'][sample], posterior['t0'][sample], posterior['m0'][sample], posterior['alpha0_S'][sample], posterior['delta0_S'][sample], posterior['phi'][sample])

    pmrac_S_med, pmdec_S_med, varpi_S_med, alpha0_S_med, delta0_S_med = np.median(posterior['pmrac_S']), np.median(posterior['pmdec_S']), np.median(posterior['varpi_S']), np.median(posterior['alpha0_S']), np.median(posterior['delta0_S'])
    tE_med, u0_med, piE_med, thetaE_med, t0_med, m0_med, phi_med = np.median(posterior['tE']), np.median(posterior['u0']), np.median(posterior['piE']), np.median(posterior['thetaE']), np.median(posterior['t0']), np.median(posterior['m0']), np.median(posterior['phi'])

    plotSample(ax, ts_continuous, refEpoch, ra, dec, fE, fN,
    tE_med, u0_med, piE_med, pmrac_S_med, pmdec_S_med, thetaE_med, varpi_S_med, t0_med, m0_med, alpha0_S_med, delta0_S_med, phi_med, median=True)

    source_params = astromet.params()
    source_params.ra, source_params.dec = ra, dec
    source_params.drac, source_params.ddec, source_params.parallax, source_params.pmrac, source_params.pmdec = alpha0_S_med, delta0_S_med, varpi_S_med, pmrac_S_med, pmdec_S_med
    source_params.epoch = refEpoch


    time, sc_a, fw, fz = obsdata['time'], obsdata['sc_a'], obsdata['fw'], obsdata['fz']

    u_NE = compute_u_NE(time, t0_med, tE_med, u0_med, piE_med, phi_med, sc_a, fw, fz)
    shift_u_NE = shift_from_u(u_NE)

    racs, decs = astromet.track(time, source_params)
    track_NE = [decs, racs]
    track_NE += shift_u_NE*thetaE_med

    if residuals:
        projection_dir = np.array( [np.cos(sc_a),
                                    np.sin(sc_a)] )
        ws_med, mags_med = astrometry_sim(
                time, time-refEpoch, sc_a, fw, fz,
                tE_med, u0_med, piE_med, pmrac_S_med, pmdec_S_med, thetaE_med, varpi_S_med, t0_med, m0_med, alpha0_S_med, delta0_S_med, phi_med)
        residuals_w = ws_med - obsdata['w']
        residuals_NE = residuals_w * projection_dir
        ax.scatter(track_NE[1]+residuals_NE[1], track_NE[0]+residuals_NE[0], color='black', alpha=0.8, s=10, zorder=2, marker='s')
    if errorbars:
        projection_dir = np.array( [np.cos(sc_a),
                                    np.sin(sc_a)] )
        err = obsdata['wError'] * projection_dir
        ax.plot(track_NE[1]+residuals_NE[1]+[err[1],-err[1]], track_NE[0]+residuals_NE[0]+[err[0],-err[0]], color='black', alpha=0.8, zorder=4, lw=1)

    ax.errorbar([-2000, -2000], [0, 1], [1, 1], color='black', zorder=2, lw=1, fmt='s', ms=10, alpha=0.8, label='Gaia measurements')
    ax.set_xlim(np.max(racs)+1, np.min(racs)-1)
    ax.set_ylim(np.min(decs)-0.5, np.max(decs)+0.5)
    ax.grid()
    ax.set_ylabel(r'$\Delta\delta$ [mas]')
    ax.set_xlabel(r'$\Delta\alpha*$ [mas]')
    ax.legend(loc='best')

    plt.savefig(f'{output_dir}/{source_id}_posterior_over_data_2D.png',dpi=200)

    fig, ax = plt.subplots(figsize=(9,10))
    
    ax.plot([-2000, -2000], [0, 1], color=color_sample, zorder=2, lw=1, alpha=0.8, label='centroid (samples)')
    
    for sample in range(num_samples):
        plotSample(ax, ts_continuous, refEpoch, ra, dec, fE, fN,
        posterior['tE'][sample], posterior['u0'][sample], posterior['piE'][sample], posterior['pmrac_S'][sample], posterior['pmdec_S'][sample], posterior['thetaE'][sample], posterior['varpi_S'][sample], posterior['t0'][sample], posterior['m0'][sample], posterior['alpha0_S'][sample], posterior['delta0_S'][sample], posterior['phi'][sample], shift_only=True)
    
    plotSample(ax, ts_continuous, refEpoch, ra, dec, fE, fN,
    tE_med, u0_med, piE_med, pmrac_S_med, pmdec_S_med, thetaE_med, varpi_S_med, t0_med, m0_med, alpha0_S_med, delta0_S_med, phi_med, median=True, shift_only=True)

    if residuals:
        projection_dir = np.array( [np.cos(sc_a),
                                    np.sin(sc_a)] )
        ws_med, mags_med = astrometry_sim(
                time, time-refEpoch, sc_a, fw, fz,
                tE_med, u0_med, piE_med, pmrac_S_med, pmdec_S_med, thetaE_med, varpi_S_med, t0_med, m0_med, alpha0_S_med, delta0_S_med, phi_med)
        residuals_w = ws_med - obsdata['w']
        residuals_NE = residuals_w * projection_dir
        ax.scatter(shift_u_NE[1]*thetaE_med+residuals_NE[1], shift_u_NE[0]*thetaE_med+residuals_NE[0], color='black', alpha=0.8, s=10, zorder=2, marker='s')
    if errorbars:
        projection_dir = np.array( [np.cos(sc_a),
                                    np.sin(sc_a)] )
        err = obsdata['wError'] * projection_dir
        ax.plot(shift_u_NE[1]*thetaE_med+residuals_NE[1]+[err[1],-err[1]], shift_u_NE[0]*thetaE_med+residuals_NE[0]+[err[0],-err[0]], color='black', alpha=0.8, zorder=4, lw=1)

    ax.errorbar([-2000, -2000], [0, 1], [1, 1], color='black', zorder=2, lw=1, fmt='s', ms=10, alpha=0.8, label='Gaia measurements')

    ax.set_xlim(np.max(shift_u_NE[1]*thetaE_med)+1, np.min(shift_u_NE[1]*thetaE_med)-1)
    ax.set_ylim(np.min(shift_u_NE[0]*thetaE_med)-0.5, np.max(shift_u_NE[0]*thetaE_med)+0.5)
    ax.grid()
    ax.set_ylabel(r'$\Delta\delta$ [mas]')
    ax.set_xlabel(r'$\Delta\alpha*$ [mas]')
    ax.legend(loc='best')

    plt.savefig(f'{output_dir}/{source_id}_shift_over_data_2D.png',dpi=200)

    
    med_continuous = plotSamplePhot(ax, ts_continuous, refEpoch, ra, dec, fE, fN,
    tE_med, u0_med, piE_med, pmrac_S_med, pmdec_S_med, thetaE_med, varpi_S_med, t0_med, m0_med, alpha0_S_med, delta0_S_med, phi_med, median=True, getmag=True)
    
    ts_ideal = np.linspace(t0_med - 5.5, t0_med + 5.5, 300)
    refEpoch, ra, dec = obsdata['refEpoch'], obsdata['ra'], obsdata['dec']
    fEid, fNid = getFactors(ts_ideal, refEpoch, ra, dec)
    
    fig, axes = plt.subplots(2, 1, figsize=(9,10), height_ratios=[3,1], sharex=True)
    plt.subplots_adjust(hspace=0)
    ax, resx = axes[0], axes[1]
    for sample in range(num_samples):
        tmp_continuous = plotSamplePhot(ax, ts_continuous, refEpoch, ra, dec, fE, fN,
        posterior['tE'][sample], posterior['u0'][sample], posterior['piE'][sample], posterior['pmrac_S'][sample], posterior['pmdec_S'][sample], posterior['thetaE'][sample], posterior['varpi_S'][sample], posterior['t0'][sample], posterior['m0'][sample], posterior['alpha0_S'][sample], posterior['delta0_S'][sample], posterior['phi'][sample], getmag=True)
        resx.plot(ts_continuous, tmp_continuous - med_continuous, color=color_sample, zorder=2, lw=1, alpha=0.05)

    ax.errorbar(obsdata['time'], obsdata['mags'], yerr=obsdata['emags'], fmt='o', zorder=4, color='black', ms=4, alpha=0.8, label='Gaia photometry')
    ax.plot([-2000, -2000], [0, 1], color=color_sample, zorder=2, lw=1, alpha=0.8, label='posterior samples')
    ax.grid()
    resx.grid()
    ax.invert_yaxis()
    ax.set_ylabel('G [mag]')
    resx.set_ylabel('residuals [mag]', fontsize=14)
    ax.set_xlabel('time [year]')
    resx.set_xlim(min_time, max_time)
    ax.set_ylim(np.max(obsdata['mags']) + 0.15, np.min(obsdata['mags']) - 0.3)
    ax.legend(loc='best')
    
        
    fEobs, fNobs = getFactors(obsdata['time'], refEpoch, ra, dec)
    
    median_model_mags = medianModelMags(obsdata['time'], refEpoch, ra, dec, fEobs, fNobs, tE_med, u0_med, piE_med, pmrac_S_med, pmdec_S_med, thetaE_med, varpi_S_med, t0_med, m0_med, alpha0_S_med, delta0_S_med, phi_med)
    
    resx.errorbar(obsdata['time'], obsdata['mags']-median_model_mags, yerr=obsdata['emags'], fmt='o', zorder=4, color='black', ms=4, alpha=0.8, label='Gaia photometry')
    resx.axhline(y=0, color='black', zorder=3, lw=1.5, ls=':', alpha=0.8)
    plt.savefig(f'{output_dir}/{source_id}_posterior_over_data_LC.png',dpi=200)



def plotSample(ax, ts_continuous, refEpoch, ra, dec, fE, fN,
tE, u0, piE, pmrac_S, pmdec_S, thetaE, varpi_S, t0, m0, alpha0_S, delta0_S, phi, median=False, shift_only=False):

    source_params = astromet.params()
    source_params.ra, source_params.dec = ra, dec
    source_params.drac, source_params.ddec, source_params.parallax, source_params.pmrac, source_params.pmdec = alpha0_S, delta0_S, varpi_S, pmrac_S, pmdec_S
    source_params.epoch = refEpoch

    u_NE = compute_u_NE(ts_continuous, t0, tE, u0, piE, phi, np.zeros_like(ts_continuous), fN, -fE)
    shift_u_NE = shift_from_u(u_NE)

    racs, decs = astromet.track(ts_continuous, source_params)
    if shift_only:
        track_NE = shift_u_NE*thetaE
    else:
        track_NE = [decs, racs]
        track_NE += shift_u_NE*thetaE

    if median:
        ax.plot(track_NE[1], track_NE[0], color=color_median, zorder=500, lw=2, ls='--', alpha=0.1, label='model fit (median)')
        if not shift_only: ax.plot(racs, decs, color='black', zorder=1, lw=1.5, ls=':', alpha=0.8, label='source (median)')
    else:
        ax.plot(track_NE[1], track_NE[0], color=color_sample, zorder=4, lw=1, alpha=0.05)



def plotSamplePhot(ax, ts_continuous, refEpoch, ra, dec, fE, fN,
tE, u0, piE, pmrac_S, pmdec_S, thetaE, varpi_S, t0, m0, alpha0_S, delta0_S, phi, median=False, getmag=False):

    source_params = astromet.params()
    source_params.ra, source_params.dec = ra, dec
    source_params.drac, source_params.ddec, source_params.parallax, source_params.pmrac, source_params.pmdec = alpha0_S, delta0_S, varpi_S, pmrac_S, pmdec_S
    source_params.epoch = refEpoch

    u_NE = compute_u_NE(ts_continuous, t0, tE, u0, piE, phi, np.zeros_like(ts_continuous), fN, -fE)
    _u = np.sqrt(u_NE[0]**2 + u_NE[1]**2)
    ampl = (_u**2 + 2)/(_u*np.sqrt(_u**2+4))
    mags = m0 - 2.5 * np.log10(ampl)

    if median:
        ax.plot(ts_continuous, mags+0.1, color=color_median, zorder=500, lw=2, ls='--', alpha=1)

    if not median:
        ax.plot(ts_continuous, mags, color=color_sample, zorder=2, lw=1, alpha=0.05)

    if getmag:
        return mags


def medianModelMags(t_obs, refEpoch, ra, dec, fE, fN, tE, u0, piE, pmrac_S, pmdec_S, thetaE, varpi_S, t0, m0, alpha0_S, delta0_S, phi):
    source_params = astromet.params()
    source_params.ra, source_params.dec = ra, dec
    source_params.drac, source_params.ddec, source_params.parallax, source_params.pmrac, source_params.pmdec = alpha0_S, delta0_S, varpi_S, pmrac_S, pmdec_S
    source_params.epoch = refEpoch

    u_NE = compute_u_NE(t_obs, t0, tE, u0, piE, phi, np.zeros_like(t_obs), fN, -fE)
    _u = np.sqrt(u_NE[0]**2 + u_NE[1]**2)
    ampl = (_u**2 + 2)/(_u*np.sqrt(_u**2+4))
    mags = m0 - 2.5 * np.log10(ampl)
    
    return mags

def plot_times(ax, ref=True, refEpoch=2017.5, labels=True):

    if(labels):
        label_ref, label_t0, label_tE, label_max = 'reference epoch', r'lensing event $t_0$', r'$t_0 \pm t_{\rm E}$', r'max astrom. shift'
    else:
        label_dr, label_ref, label_dec, label_t0, label_tE, label_max = '', '', '', '', '', ''

    if ref: ax.axvline(refEpoch, color=color_ref, label=label_ref)

    if phot_params is not None:
        t0, tE, u0 = phot_params.t0, phot_params.tE, phot_params.u0
        tau_shift_max = np.sqrt(2 - u0**2)
        ax.axvline(t0, color=color_event, ls='-', label=label_t0)


def plotAL(posterior, obsdata, output_dir, source_id, true_params=None):

        pmrac_S_med, pmdec_S_med, varpi_S_med, alpha0_S_med, delta0_S_med = np.median(posterior['pmrac_S']), np.median(posterior['pmdec_S']), np.median(posterior['varpi_S']), np.median(posterior['alpha0_S']), np.median(posterior['delta0_S'])
        tE_med, u0_med, piE_med, thetaE_med, t0_med, m0_med, phi_med = np.median(posterior['tE']), np.median(posterior['u0']), np.median(posterior['piE']), np.median(posterior['thetaE']), np.median(posterior['t0']), np.median(posterior['m0']), np.median(posterior['phi'])

        t, wObs, wErr, refEpoch, sc_a, fw, fz = obsdata['time'], obsdata['w'], obsdata['wError'], obsdata['refEpoch'], obsdata['sc_a'], obsdata['fw'], obsdata['fz']

        ws_med, mags_med = astrometry_sim(
                t, t-refEpoch, sc_a, fw, fz,
                tE_med, u0_med, piE_med, pmrac_S_med, pmdec_S_med, thetaE_med, varpi_S_med, t0_med, m0_med, alpha0_S_med, delta0_S_med, phi_med)

        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[2,1], figsize=(16, 8), sharex=True)
        
        plot_times(ax=ax1, refEpoch = refEpoch)
        plot_times(ax=ax2, refEpoch = refEpoch)

        ax1.errorbar(t, wObs, yerr=wErr, color='black',
                     label='observations', zorder=2, fmt='.', ms=12, elinewidth=1)
        ax2.axhline(0, color='black')
        ax1.errorbar(t, ws_med, yerr=wErr, color=color_median,
                         zorder=2, fmt='.', ms=8, elinewidth=0.5, alpha=0.8, label='model fit')

        ax2.errorbar(t, wObs - ws_med, yerr=wErr, color=color_median,
                         zorder=2, fmt='.', ms=8, elinewidth=0.5, alpha=0.8)

        if true_params is not None:
            ws_true, mags_true = astrometry_sim(
            t, t-refEpoch, sc_a, fw, fz,
            true_params['tE'], true_params['u0'], true_params['piE'], true_params['pmrac_S'], true_params['pmdec_S'], true_params['thetaE'], true_params['varpi_S'], true_params['t0'], true_params['m0'], true_params['alpha0_S'], true_params['delta0_S'], true_params['phi'])
            ax1.scatter(t, ws_true, edgecolor=color_true, facecolor='None',
                         label='ground truth', s=100, alpha=0.5, zorder=5)
        
        # Axes
        ax1.grid()
        ax2.grid()
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.1))
        plt.subplots_adjust(hspace=0.)
        ax1.set_ylabel(r'$w$ [mas]')
        ax2.set_ylabel('Residuals [mas]')
        ax2.set_xlabel('Time [years]')
        ax2.set_xlim(2014.25, 2020.25)
        
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{source_id}_median_vs_obs_AL.png',dpi=200)
