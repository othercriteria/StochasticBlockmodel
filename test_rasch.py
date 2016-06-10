#!/usr/bin/env python

import json
import signal
from time import time

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

from Array import array_from_data
from BinaryMatrix import approximate_conditional_nll as cond_a_nll_b
from BinaryMatrix import approximate_from_margins_weights as cond_a_sample_b
from BinaryMatrix import clear_cache
from Confidence import invert_test, ci_conservative_generic
from Experiment import Seed
from Models import NonstationaryLogistic, alpha_zero
from Utility import logsumexp, logabsdiffexp


# Parameters
params = { 'fixed_example': 'data/rasch_covariates.json',
           'M': 20,
           'N': 10,
           'theta': 2.0,
           'kappa': -1.628,
           'alpha_min': -0.4,
           'beta_min': -0.86,
           'v_min': -0.6,
           'alpha_level': 0.05,
           'n_MC_levels': [10, 50],#[10, 50, 100, 500],
           'wopt_sort': False,
           'is_T': 50,
           'n_rep': 10,
           'L': 61,
           'theta_l': -6.0,
           'theta_u': 6.0,
           'do_prune': False,
           'random_seed': 137,
           'verbose': True,
           'clear_cache': False }

terminated = False
def sigint_handler(signum, frame):
    print 'Terminating after current trial completes.'
    global terminated
    terminated = True
signal.signal(signal.SIGINT, sigint_handler)

def cond_a_nll(X, w):
    return cond_a_nll_b(X, w, sort_by_wopt_var = params['wopt_sort'])

def cond_a_sample(r, c, w, T = 0):
    return cond_a_sample_b(r, c, w, T, sort_by_wopt_var = params['wopt_sort'])

# Generates (or loads) a particular realization of P_{ij} and then
# repeatedly samples independent Bernoulli random variables according
# to these cell probabilities.
def generate_data(params, seed):
    # Advance random seed for parameter and covariate construction
    seed.next()

    if not params['fixed_example']:
        # Generate parameters and covariates
        M, N = params['M'], params['N']
        alpha = np.random.uniform(size = (M,1)) + params['alpha_min']
        beta = np.random.uniform(size = (1,N)) + params['beta_min']
        kappa = params['kappa']
        v = np.random.uniform(size = (M,N)) + params['v_min']
    else:
        # Load parameters and covariates
        with open(params['fixed_example'], 'r') as example_file:
            example = json.load(example_file)

            v = np.array(example['nu'])
            M, N = v.shape

            alpha = np.array(example['alpha']).reshape((M,1))
            beta = np.array(example['beta']).reshape((1,N))
            kappa = example['kappa']

    # Generate Bernoulli probabilities from logistic regression model
    logit_P = np.zeros((M,N)) + kappa
    logit_P += alpha
    logit_P += beta
    logit_P += params['theta'] * v
    P = 1.0 / (1.0 + np.exp(-logit_P))

    while True:
        # Advance random seed for data generation
        seed.next()

        # Generate data for this trial
        X = np.random.random((M,N)) < P

        # Pruning rows and columns of 0's and 1's; this may improve
        # the quality of the approximation for certain versions of the
        # sampler
        if params['do_prune']:
            X_p = X.copy()
            v_p = v.copy()
            while True:
                r, c = X_p.sum(1), X_p.sum(0)
                r_p = (r == 0) + (r == N)
                c_p = (c == 0) + (c == M)
                pruning = np.any(r_p) or np.any(c_p)

                X_p = X_p[-r_p][:,-c_p]
                v_p = v_p[-r_p][:,-c_p]

                if not pruning:
                    break

            yield X_p.copy(), v_p.copy()

        yield X, v

def timing(func):
    def inner(*args, **kwargs):
        start_time = time()
        val = func(*args, **kwargs)
        elapsed = time() - start_time
        return val, elapsed

    return inner

def fresh_cache(func):
    def inner(*args, **kwargs):
        if params['clear_cache']:
            clear_cache()
        return func(*args, **kwargs)

    return inner

def plot_statistics(ax, theta_grid, test_val, crit):
    # Compute confidence interval from test statistics
    ci_l, ci_u = invert_test(theta_grid, test_val, crit)
    ci_l = max(ci_l, params['theta_l'])
    ci_u = min(ci_u, params['theta_u'])
    cov_alpha = min(1.0, 1.0 / ((1 - params['alpha_level']) * params['n_rep']))

    ax.plot(theta_grid, test_val, color = 'b')
    ax.hlines(crit, theta_grid[0], theta_grid[-1], linestyle = 'dotted')
    ax.hlines(crit, ci_l, ci_u, color = 'r')
    ax.hlines(2.0 * crit, theta_grid[0], theta_grid[-1],
              color = 'w', linewidth = 9, zorder = 99)
    ax.hlines(2.0 * crit, ci_l, ci_u, color = 'r', linewidth = 9, zorder = 100,
              alpha = cov_alpha)
    ax.vlines(ci_l, 2.0 * crit, crit, color = 'r', linestyle = 'dotted')
    ax.vlines(ci_u, 2.0 * crit, crit, color = 'r', linestyle = 'dotted')
    ax.set_ylim(2.0 * crit, 0)

# Set up plots
fig_cmle_a, ax_cmle_a = plt.subplots()
fig_cmle_is, ax_cmle_is = plt.subplots()

@timing
def ci_umle_boot(X, v, alpha_level):
    arr = array_from_data(X, [v])
    arr.offset_extremes()
    alpha_zero(arr)

    fit_model = NonstationaryLogistic()
    fit_model.beta['x_0'] = None
    fit_model.confidence_boot(arr, alpha_level = alpha_level)

    return fit_model.conf['x_0']['pivotal']

@timing
def ci_brazzale(X, v, alpha_level):
    arr = array_from_data(X, [v])
    arr.offset_extremes()
    alpha_zero(arr)

    fit_model = NonstationaryLogistic()
    fit_model.beta['x_0'] = None
    fit_model.fit_brazzale(arr, 'x_0', alpha_level = alpha_level)

    return fit_model.conf['x_0']['brazzale']

@timing
@fresh_cache
def ci_cmle_a(X, v, theta_grid, alpha_level):
    crit = -0.5 * chi2.ppf(1 - alpha_level, 1)

    cmle_a = np.empty_like(theta_grid)
    for l, theta_l in enumerate(theta_grid):
        logit_P_l = theta_l * v
        cmle_a[l] = -cond_a_nll(X, np.exp(logit_P_l))

    plot_statistics(ax_cmle_a, theta_grid, cmle_a - cmle_a.max(), crit)
    return invert_test(theta_grid, cmle_a - cmle_a.max(), crit)

@timing
@fresh_cache
def ci_cmle_is(X, v, theta_grid, alpha_level, T = 100, verbose = False):
    crit = -0.5 * chi2.ppf(1 - alpha_level, 1)

    cmle_is = np.empty_like(theta_grid)
    r = X.sum(1)
    c = X.sum(0)
    for l, theta_l in enumerate(theta_grid):
        logit_P_l = theta_l * v
        w_l = np.exp(logit_P_l)

        z = cond_a_sample(r, c, w_l, T)
        logf = np.empty(T)
        for t in range(T):
            logQ, logP = z[t][1], z[t][2]
            logf[t] = logP - logQ
        logkappa = -np.log(T) + logsumexp(logf)

        if verbose:
            logcvsq = -np.log(T - 1) - 2 * logkappa + \
              logsumexp(2 * logabsdiffexp(logf, logkappa))
            print 'est. cv^2 = %.2f (T = %d)' % (np.exp(logcvsq), T)

        cmle_is[l] = np.sum(np.log(w_l[X])) - logkappa

    plot_statistics(ax_cmle_is, theta_grid, cmle_is - cmle_is.max(), crit)
    return invert_test(theta_grid, cmle_is - cmle_is.max(), crit)

@timing
@fresh_cache
def ci_conservative(X, v, K, theta_grid, alpha_level, corrected,
                    verbose = False):
    M_p, N_p = X.shape
    L = len(theta_grid)

    # Test statistic for CI
    def t(z, theta):
        return log_likelihood(X, theta) - log_likelihood(z, theta)

    # Evaluate log-likelihood at specified parameter value
    def log_likelihood(z, theta):
        return -cond_a_nll(z, np.exp(theta * v))

    # Row and column margins; the part of the data we can use to design Q
    r, c = X.sum(1), X.sum(0)

    # Generate sample from k-th component of mixture proposal distribution
    def sample(theta):
        Y_sparse = cond_a_sample(r, c, np.exp(theta * v))
        Y_dense = np.zeros((M_p,N_p), dtype = np.bool)
        for i, j in Y_sparse:
            if i == -1: break
            Y_dense[i,j] = 1
        return Y_dense

    return ci_conservative_generic(X, K, theta_grid, alpha_level,
                                   log_likelihood, sample, t,
                                   corrected, False, verbose)

def do_experiment(params):
    seed = Seed(params['random_seed'])

    alpha_level = params['alpha_level']
    verbose = params['verbose']

    L = params['L']
    S = len(params['n_MC_levels'])
    T = params['is_T']
    
    # Set up structure and methods for recording results
    results = { 'completed_trials': 0 }
    for method, disp in [('umle', 'UMLE (bootstrap)'),
                         ('brazzale', 'Conditional (Brazzale)'),
                         ('cmle_a', 'CMLE-A'),
                         ('cmle_is', 'CMLE-IS (T = %d)' % T)] + \
                        [('is_c_%d' % s, 'IS-corrected (n = %d)' % n_MC)
                         for s, n_MC in enumerate(params['n_MC_levels'])] + \
                        [('is_u_%d' % s, 'IS-uncorrected (n = %d)' % n_MC)
                         for s, n_MC in enumerate(params['n_MC_levels'])]:
        results[method] = { 'display': disp,
                            'in_interval': [],
                            'length': [],
                            'total_time': 0.0 }
    def do(out, name):
        ci, elapsed = out
        ci_l, ci_u = ci

        result = results[name]

        print '%s (%.2f sec): [%.2f, %.2f]' % \
          (result['display'], elapsed, ci_l, ci_u)

        result['in_interval'].append(ci_l <= params['theta'] <= ci_u)
        result['length'].append(ci_u - ci_l)
        result['total_time'] += elapsed

    # Do experiment
    for X, v in generate_data(params, seed):
        if (results['completed_trials'] == params['n_rep']) or terminated:
            break

        theta_grid = np.linspace(params['theta_l'], params['theta_u'], L)

        do(ci_umle_boot(X, v, alpha_level), 'umle_boot')

        do(ci_brazzale(X, v, alpha_level), 'brazzale')

        do(ci_cmle_a(X, v, theta_grid, alpha_level), 'cmle_a')

        do(ci_cmle_is(X, v, theta_grid, alpha_level, T, verbose), 'cmle_is')

        for s, n_MC in enumerate(params['n_MC_levels']):
            do(ci_conservative(X, v, n_MC, theta_grid,
                               alpha_level, True, verbose),
               'is_c_%d' % s)
            do(ci_conservative(X, v, n_MC, theta_grid,
                               alpha_level, False, verbose),
               'is_u_%d' % s)

        results['completed_trials'] += 1

    # For verifying that same data was generated even if different
    # algorithms consumed a different amount of randomness
    seed.final()

    return results

results = do_experiment(params)

R = results.pop('completed_trials')
print '\nCompleted trials: %d\n\n' % R

for method in results:
    result = results[method]

    print '%s:' % result['display']
    print 'Coverage probability: %.2f' % np.mean(result['in_interval'][0:R])
    print 'Median length: %.2f' % np.median(result['length'][0:R])
    print 'Total time: %.2f sec' % result['total_time']
    print

ax_cmle_a.set_title('CMLE-A confidence intervals')
ax_cmle_is.set_title('CMLE-IS confidence intervals')
#plt.show()
