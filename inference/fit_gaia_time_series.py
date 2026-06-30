import yaml
import pandas as pd
import numpy as np
from functools import partial
import argparse
import shutil

import utils
import skypaths

parser = argparse.ArgumentParser()
parser.add_argument('--plot_only', action='store_true', help='Choose if only generating plots, without running the fit')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--source_id', type=str)
parser.add_argument('--overplot_true', action='store_true', help='Choose if true injected parameters are available')

args = parser.parse_args()
data_filename = f'../../clear_underworld/mock_golden/remnants/{args.source_id}.npy'
if args.overplot_true:
    true_filename = f'../../clear_underworld/mock_golden/remnants/{args.source_id}_true_params.npy'
    true_params = np.load(true_filename, allow_pickle='TRUE').item()
else:
    true_params = None
    
# Load data
data = np.load(data_filename, allow_pickle='TRUE').item()
source_id = data['source_id']
    
# Define likelihood/priors
log_likelihood = partial(utils.log_likelihood, obsdata=data)

# Load in config file
with open("config.yaml", "r") as file:
    base_config = yaml.load(file, Loader=yaml.SafeLoader)
    
output_dir = base_config["paths"]["output_directory"]

def run_solution(u0_sign, output_suffix,
    base_config=base_config, output_dir=output_dir, data=data, source_id=source_id, log_likelihood=log_likelihood, plot_only=args.plot_only):

    config_tmp = base_config.copy()
    config_tmp["paths"]["output_directory"] = output_dir + output_suffix 
    config_tmp["priors"]["u0"]["parameters"]["sign"] = u0_sign

    priors = utils.get_all_priors(config=config_tmp, obsdata=data)
    prior_transform = partial(utils.prior_transform, priors=priors)

    if not plot_only:
        dynesty_results = utils.run_dynesty_sampling(log_likelihood, prior_transform, config_tmp)
        utils.save_dynesty_results(results=dynesty_results, source_id=source_id, config=config_tmp)
        
    posterior = utils.get_posterior_samples(source_id=source_id, config=config_tmp, size=None)
    utils.plot_corner(posterior, output_dir=config_tmp["paths"]["output_directory"], source_id=source_id, true_params=true_params)
    median_logl = np.median(dynesty_results.logl)
    return median_logl


# Positive solution
median_pos = run_solution(u0_sign=1, output_suffix="run_positive/")
# Negative solution
median_neg = run_solution(u0_sign=-1, output_suffix="run_negative/")

# Choose the better fit
if median_pos > median_neg:
    solution_dir = output_dir + "run_positive/"
    if args.verbose:
        print("Positive solution chosen")
else:
    solution_dir = output_dir + "run_negative/"
    if args.verbose:
        print("Negative solution chosen")

shutil.copyfile(f'{solution_dir}{source_id}_dynesty.pkl', f'{output_dir}{source_id}_dynesty.pkl')
shutil.copyfile(f'{solution_dir}{source_id}_corner.png', f'{output_dir}{source_id}_corner.png')

posterior = utils.get_posterior_samples(source_id=source_id, path=f'{solution_dir}{source_id}_dynesty.pkl', size=None)

# Create final astrometry plots
skypaths.plotContinuous(posterior=posterior, obsdata=data, output_dir=output_dir, source_id=source_id, true_params=true_params)
skypaths.plotAL(posterior=posterior, obsdata=data, output_dir=output_dir, source_id=source_id, true_params=true_params)
