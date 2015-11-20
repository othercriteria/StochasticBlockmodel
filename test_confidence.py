#!/usr/bin/env python

# Testing confidence interval coverage
# Daniel Klein, 1/16/2013

from __future__ import division

import numpy as np

from Network import Array
from Models import StationaryLogistic, NonstationaryLogistic, FixedMargins
from Models import alpha_zero, alpha_norm, alpha_unif, alpha_gamma

# Parameters
params = { 'M': 20,
           'N': 10,
           'B': 1,
           'beta_fixed': { 'x_0': 2.0, 'x_1': -1.0 },
           'beta_sd': 1.0,
           'alpha_unif_sd': 0.0,
           'alpha_norm_sd': 1.0,
           'alpha_gamma_sd': 0.0,
           'kappa_target': ('row_sum', 5),
           'fit_nonstationary': True,
           'fit_method': 'conditional',
           'covariates_of_interest': ['x_0'],
           'do_large_sample': True,
           'do_biometrika': False,
           'num_reps': 100 }

# Set random seed for reproducible output
np.random.seed(137)

# Initialize array
arr = Array(params['M'], params['N'])

# Generate node-level propensities to extend and receive edges
if params['alpha_norm_sd'] > 0.0:
    alpha_norm(arr, params['alpha_norm_sd'])
elif params['alpha_unif_sd'] > 0.0:
    alpha_unif(arr, params['alpha_unif_sd'])
elif params['alpha_gamma_sd'] > 0.0:
    # Choosing location somewhat arbitrarily to give unit skewness
    alpha_gamma(arr, 4.0, params['alpha_gamma_sd'])
else:
    alpha_zero(arr)

# Generate covariates and associated coefficients
data_model = NonstationaryLogistic()
covariates = []
for b in range(params['B']):
    name = 'x_%d' % b
    covariates.append(name)

    if name in params['beta_fixed']:
        data_model.beta[name] = params['beta_fixed'][name]
    else:
        data_model.beta[name] = np.random.normal(0, params['beta_sd'])

    def f_x(i_1, i_2):
        return np.random.uniform(-np.sqrt(3), np.sqrt(3))
    arr.new_edge_covariate(name).from_binary_function_ind(f_x)
data_model.match_kappa(arr, params['kappa_target'])

# Specify parameter of interest that the confidence interval will try to capture
for c in params['covariates_of_interest']:
    theta_true = data_model.beta[c]
    print '%s theta_true: %.2f' % (c, theta_true)

# Setup fit model
if params['fit_method'] == 'conditional':
    fit_model = StationaryLogistic()
    for c in covariates:
        fit_model.beta[c] = None
    fit_model.generate = fit_model.generate_margins
    fit_model.fit = fit_model.fit_conditional
else:
    if params['fit_nonstationary']:
        fit_model = NonstationaryLogistic()
    else:
        fit_model = StationaryLogistic()
    for c in covariates:
        fit_model.beta[c] = None
    
# Test coverage
methods = []
if params['do_large_sample']:
    methods.extend(['pivotal', 'percentile', 'normal'])
if params['do_biometrika']:
    methods.extend(['harrison'])
covered = { (m,c): np.empty(params['num_reps'])
            for m in methods for c in params['covariates_of_interest'] }
length = { (m,c): np.empty(params['num_reps'])
           for m in methods for c in params['covariates_of_interest'] }
for rep in range(params['num_reps']):
    arr.generate(data_model)

    if params['do_large_sample']:
        fit_model.confidence(arr)
    if params['do_biometrika']:
        for c in params['covariates_of_interest']:
            fit_model.confidence_harrison(arr, c)

    for m in methods:
        print '%s:' % m
        for c in params['covariates_of_interest']:
            ci_l, ci_u = fit_model.conf[c][m]
            print '%s theta_true 95%% CI: [%.2f, %.2f]' % (c, ci_l, ci_u)

            covered[(m,c)][rep] = ci_l <= data_model.beta[c] <= ci_u
            length[(m,c)][rep] = ci_u - ci_l
    print

for m in methods:
    print '%s:' % m
    for c in params['covariates_of_interest']:
        print ' %s:' % c
        print '  Attained coverage: %.2f' % np.mean(covered[(m,c)])
        print '  Median length: %.2f' % np.median(length[(m,c)])
    print

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))
