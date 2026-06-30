import numpy as np

import dynesty
from dynesty import utils as dyfunc

from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from astropy import units as u

import pickle
import corner
import matplotlib.pyplot as plt

from priors import *

y = (1.0*u.year).to(u.day).value
deg_to_mas = (1.0*u.degree).to(u.mas).value

# === Priors ===

def get_prior(param_dict):
    """Coverts prior config files to scipy.stats prior distributions"""
    if param_dict['distribution']=='uniform':
        return Uniform(**param_dict['parameters'])
    if param_dict['distribution']=='normal':
        return Normal(**param_dict['parameters'])
    if param_dict['distribution']=='loguniform':
        return LogUniform(**param_dict['parameters'])
    if param_dict['distribution']=='signedloguniform':
        return SignedLogUniform(**param_dict['parameters'])
    if param_dict['distribution']=='log10normal':
        return Log10Normal(**param_dict['parameters'])
    return None

def get_all_priors(obsdata=None, config=None):
    prior_dict = config['priors']
    order = ['tE', 'u0', 'piE', 'pmrac_S', 'pmdec_S', 'thetaE', 'varpi_S']
    priors = [get_prior(prior_dict[param]) for param in order]
    return priors + [
        t0_prior(obsdata['time']),
        m0_prior(obsdata['mags']),
        alpha0_prior(obsdata['w']),
        delta0_prior(obsdata['w']),
        phi_prior()
    ]

def t0_prior(times):
    """Sets the prior on the closest approach time to be uniform between the
    min and max times in the light curve"""
    return Uniform(lower=np.min(times), upper=np.max(times) - np.min(times))

def m0_prior(mags):
    """Sets the prior on the baseline magnitude to be uniform within one
    magnitude of the median magnitude of the light curve"""
    return Normal(mean=np.median(mags), std=np.std(mags), lower=np.median(mags)-0.5, upper=np.median(mags)+0.5)


def alpha0_prior(ws):
    """Sets the prior on the reference RA position in mas to be uniform
    within extents of +- maximum 1D AL deviation values"""
    wmax = np.max(np.abs(ws))
    return Normal(mean=0, std=np.std(ws), lower=-wmax, upper=wmax)


def delta0_prior(ws):
    """Sets the prior on the reference DEC position in mas to be uniform
    within extents of +- maximum 1D AL deviation values"""
    wmax = np.max(np.abs(ws))
    return Normal(mean=0, std=np.std(ws), lower=-wmax, upper=wmax)

def phi_prior():
    """Sets the prior on the relative motion angle to be uniform between 0 and 2pi"""
    return Uniform(lower=0, upper=2*np.pi)


def prior_transform(params, priors):
    """Returns the unit cube transform required for nested sampling"""
    return np.array([prior.ppf(param) for prior, param in zip(priors, params)])

# === Likelihood ===

def log_likelihood(pars, obsdata=None):
    """Log likelihood of point source point lens model with parallax"""
    # Convenient pointers to obs times,
    time, timeRel = obsdata['time'], obsdata['timeRel']
    # scanning angles/parallax factors,
    sc_a, fw, fz = obsdata['sc_a'], obsdata['fw'], obsdata['fz']
    # photometric data,
    mags, emags = obsdata['mags'], obsdata['emags']
    # astrometric data
    ws, ews = obsdata['w'], obsdata['wError']

    # Model astrometry and photometry computed at observed times given params
    ws_predicted, mags_predicted = astrometry_sim(
        time, timeRel, sc_a, fw, fz, *pars
    )

    # Calculate likelihood of parameters
    loglike = gaussian_log_pdf(
        mags_predicted, mags, emags,
        ws_predicted, ws, ews
    )

    return loglike


def gaussian_log_pdf(
        predicted_mags,
        mags,
        emags,
        predicted_ws,
        ws,
        ews
):
    """Log probability density function of a Gaussian with diagonal covariance
    matrix"""
    pdf_mags = -np.sum(  0.5*((predicted_mags-mags)/emags)**2
                   + 0.5*np.log(2*np.pi)+np.log(emags))
    pdf_ws = -np.sum(    0.5*((predicted_ws-ws)/ews)**2
                   + 0.5*np.log(2*np.pi)+np.log(ews))
    return pdf_mags + pdf_ws


# === Creating astrometric and photometric time series  ===

def astrometry_sim(
        time, timeRel, sc_a, fw, fz,
        tE, u0, piE, pmrac_S, pmdec_S, thetaE, varpi_S, t0, m0, alpha0_S, delta0_S, phi
):

    """
    Gets the positions (as seen on the sky) and magnitudes of an event for given observation times.

    Args:

     fixed by observations:
        - time     ndarray - observation times, decimal year
        - sc_a      ndarray - scanning angles, -
        - fw, fz    parallax factors pre-computed for the observation times, -

     model parameters - photometric:
        - t0        float - reference time, decimal year
        - tE        float - Einstein time, days
        - u0        float - impact parameter, units of thetaE
        - piE       float - microlensing parallax, units of thetaE
        - phi       float - angle determining direction of the piE vector, [0 - 2pi]
        - m0        float - event magnitude at baseline, mag

     model parameters - additional for astrometry:
        - alpha0_S    float - reference position on the E axis of the source at t0, deg
        - delta0_S    float - reference position on the N axis of the source at t0, deg
        - pmrac_S     float - proper motion of the source in right ascension, mas/yr
        - pmdec_S     float - proper motion of the source in declination, mas/yr
        - varpi_S     float - distance of the source, kpc
        - thetaE      float - the angular Einstein radius, mas

    Returns:
        - ws     ndarray - observed position along the scanning angle, mas
        - mags   ndarray - observed magnitude, mag
    """

    
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

    _u = np.sqrt(u_NE[0]**2 + u_NE[1]**2)
    ampl = (_u**2 + 2)/(_u*np.sqrt(_u**2+4))
    mags = m0 - 2.5 * np.log10(ampl)

    # # get w:
    # get the last column of the array from u_NE
    shift_NE = u_NE/(np.linalg.norm(u_NE, axis=0)**2 + 2)
    shift_projected = np.cos(sc_a)*shift_NE[0,:] + np.sin(sc_a)*shift_NE[1,:]

    # make the array: standard 5p. solution with an extra dimension for the astrom. shift scaled by thetaE
    R = np.array([np.sin(sc_a), np.cos(sc_a), fw, np.sin(sc_a)*timeRel, np.cos(sc_a)*timeRel, shift_projected]).T
    A = [alpha0_S, delta0_S, varpi_S, pmrac_S, pmdec_S, thetaE]
    
    ws = R @ A

    return ws, mags

# Rotation matrix: defined clockwise
def rotation(phi):
    R = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
    return R

# === Nested sampling  ===

def run_dynesty_sampling(
        log_likelihood, prior_unit_transform, config, num_params=12, extended=False):
    config = config['nested_sampling']
    sampler = dynesty.DynamicNestedSampler(log_likelihood,
                                           prior_unit_transform,
                                           num_params,
                                           sample="rwalk")
    if extended:
        sampler.run_nested(wt_kwargs={"pfrac": 1.0}, print_progress=True, nlive_init=5000, dlogz_init=0.001)
    else:
        sampler.run_nested(wt_kwargs={"pfrac": 1.0}, print_progress=True,
                       nlive_init=config['nlive'])
    return sampler.results


def save_dynesty_results(results, source_id, config):
    """Save dynesty inference output to pkl file"""
    path = config['paths']['output_directory']

    with open(f'{path}{source_id}_dynesty.pkl', 'wb') as handle:
        pickle.dump(results, handle)


def get_posterior_samples(source_id=None, path=None, size=10000):
    """Turns dynesty inference pkl file into dictionary of posterior samples,
    size is the number of samples that you want"""
    #path = config['paths']['output_directory']
    # Order is important!
    parameter_names = ['tE', 'u0', 'piE', 'pmrac_S', 'pmdec_S', 'thetaE', 'varpi_S', 't0', 'm0', 'alpha0_S', 'delta0_S', 'phi']

    with open(f'{path}{source_id}_dynesty.pkl', 'rb') as handle:
        inference = pickle.load(handle)

    samples, weights = inference.samples, np.exp(inference.logwt - inference.logz[-1])
    posterior = dyfunc.resample_equal(samples, weights)

    if size is not None:
        num_samples = posterior.shape[0]
        random_sample_indicies = np.random.choice(num_samples,
                                                  size=size,
                                                  replace=False)
        posterior = posterior[random_sample_indicies, :]

    posterior_samples = {parameter_name: posterior[:, index]
                         for index, parameter_name in enumerate(parameter_names)}

    return posterior_samples

# === Sample distributions  ===

def plot_corner(samples=None, output_dir='', source_id='', true_params=None, verbose=False):
    """Plots 2D corner plot on samples"""
    param_names = list(samples.keys())
    samples_array = np.array([samples[param] for param in param_names]).T
    figure = corner.corner(samples_array, labels=param_names,
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 12},
                           smooth=1.0,
                           levels=(0.68, 0.95),
                           plot_density=False,
                           plot_datapoints=False,
                           fill_contours=True)
                               
    if true_params is not None:
        true_values = []
        lin_fit_values = []
        for param in param_names:
            if param=='phi':
                true_values.append(true_params[param]%(2*np.pi))
            else:
                true_values.append(true_params[param])
            try:
                lin_fit_values.append(lin_fit_params[param])
            except:
                lin_fit_values.append(-1e6)
        corner.overplot_lines(figure, true_values, color="green", lw=3)

    axes = np.array(figure.axes).reshape((len(param_names), len(param_names)))
    
    # Mass calculation from samples
    mass_samples = samples['thetaE']/(8.144*samples['piE'])
    mass_50, mass_ll, mass_ul = np.percentile(mass_samples, 50), np.percentile(mass_samples, 16), np.percentile(mass_samples, 84) 
    if true_params is not None:
        true_mass = true_params['thetaE']/(8.144*true_params['piE'])
        if verbose:
            print("True (injected) mass: ", np.round(true_mass, 2))
            plt.text(0.5, 0.8, f'true mass: ', fontsize=40, fontfamily='serif', color='green', transform=plt.gcf().transFigure)
            plt.text(0.5, 0.77, rf'${np.round(true_mass, 2)} \ \mathcal{{M}}_\odot$', fontsize=40, fontfamily='serif', color='green', transform=plt.gcf().transFigure)
    if verbose:
        print("Calculated mass: ", np.round(mass_50, 2), "+", np.round(mass_ul-mass_50, 2), "-", np.round(mass_50-mass_ll, 2))
    plt.text(0.5, 0.73, f'mass from joint fit: ', fontsize=40, fontfamily='serif', transform=plt.gcf().transFigure)
    plt.text(0.5, 0.7, rf'${np.round(mass_50, 2)}^{{+{np.round(mass_ul-mass_50, 2)} }}_{{ {np.round(mass_ll-mass_50, 2)} }} \ \mathcal{{M}}_\odot$', fontsize=40, fontfamily='serif', transform=plt.gcf().transFigure)
    
    plt.savefig(f'{output_dir}/{source_id}_corner.png')
