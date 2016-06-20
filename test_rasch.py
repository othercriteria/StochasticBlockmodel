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
from Models import FixedMargins, StationaryLogistic, NonstationaryLogistic
from Models import alpha_zero
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
           'n_MC_levels': [10, 50, 100, 500],
           'wopt_sort': False,
           'is_T': 50,
           'n_rep': 100,
           'L': 601,
           'theta_l': -6.0,
           'theta_u': 6.0,
           'do_prune': False,
           'random_seed': 137,
           'verbose': True,
           'plot': True,
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

    ax.plot(theta_grid, test_val, color = 'b')
    ax.hlines(crit, theta_grid[0], theta_grid[-1], linestyle = 'dotted')
    ax.hlines(crit, ci_l, ci_u, color = 'r')
    ax.vlines(ci_l, 2.0 * crit, crit, color = 'r', linestyle = 'dotted')
    ax.vlines(ci_u, 2.0 * crit, crit, color = 'r', linestyle = 'dotted')
    ax.set_ylim(2.0 * crit, 0)

def plot_coverage(ax, theta_grid, crit, cis):
    intervals = zip(theta_grid[:-1], theta_grid[1:])
    coverage = np.zeros(len(intervals))
    for ci in cis:
        ci_l, ci_u = ci
        for i, interval in enumerate(intervals):
            i_l, i_u = interval
            if (ci_l <= i_l) and (i_u <= ci_u):
                coverage[i] += 1
    coverage /= len(cis)
        
    ax.hlines(2.0 * crit, theta_grid[0], theta_grid[-1],
              color = 'w', linewidth = 9, zorder = 9)

    for i, interval in enumerate(intervals):
        i_l, i_u = interval
        ax.hlines(2.0 * crit, i_l, i_u, color = 'r',
                  linewidth = 9, zorder = 10, alpha = coverage[i])

# Set up plots
if params['plot']:
    fig_cmle_a, ax_cmle_a = plt.subplots()
    fig_cmle_is, ax_cmle_is = plt.subplots()
    cmle_a_cis = []
    cmle_is_cis = []

# For methods like Wald that can sometimes fail to produce CIs
def safe_ci(model, name, method):
    if name in model.conf:
        if method in model.conf[name]:
            return model.conf[name][method]
    else:
        return (0.0, 0.0)

@timing
def ci_umle_wald(X, v, alpha_level):
    arr = array_from_data(X, [v])
    arr.offset_extremes()
    alpha_zero(arr)

    fit_model = NonstationaryLogistic()
    fit_model.beta['x_0'] = None
    fit_model.confidence_wald(arr, alpha_level = alpha_level)

    return safe_ci(fit_model, 'x_0', 'wald')

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
@fresh_cache
def ci_cmle_wald(X, v, alpha_level):
    arr = array_from_data(X, [v])

    A = arr.as_dense()
    r = A.sum(1)
    c = A.sum(0)
    
    s_model = StationaryLogistic()
    s_model.beta['x_0'] = None
    fit_model = FixedMargins(s_model)
    arr.new_row_covariate('r', np.int)[:] = r
    arr.new_col_covariate('c', np.int)[:] = c
    fit_model.fit = fit_model.base_model.fit_conditional

    fit_model.confidence_wald(arr, alpha_level = alpha_level)

    return safe_ci(fit_model, 'x_0', 'wald')

@timing
@fresh_cache
def ci_cmle_boot(X, v, alpha_level):
    arr = array_from_data(X, [v])

    A = arr.as_dense()
    r = A.sum(1)
    c = A.sum(0)
    
    s_model = StationaryLogistic()
    s_model.beta['x_0'] = None
    fit_model = FixedMargins(s_model)
    arr.new_row_covariate('r', np.int)[:] = r
    arr.new_col_covariate('c', np.int)[:] = c
    fit_model.fit = fit_model.base_model.fit_conditional

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

    return safe_ci(fit_model, 'x_0', 'brazzale')

@timing
@fresh_cache
def ci_cmle_a(X, v, theta_grid, alpha_level):
    crit = -0.5 * chi2.ppf(1 - alpha_level, 1)

    cmle_a = np.empty_like(theta_grid)
    for l, theta_l in enumerate(theta_grid):
        logit_P_l = theta_l * v
        cmle_a[l] = -cond_a_nll(X, np.exp(logit_P_l))

    ci = invert_test(theta_grid, cmle_a - cmle_a.max(), crit)
    if params['plot']:
        plot_statistics(ax_cmle_a, theta_grid, cmle_a - cmle_a.max(), crit)
        cmle_a_cis.append(ci)
        plot_coverage(ax_cmle_a, theta_grid, crit, cmle_a_cis)
    return ci

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

    ci = invert_test(theta_grid, cmle_is - cmle_is.max(), crit)
    if params['plot']:
        plot_statistics(ax_cmle_is, theta_grid, cmle_is - cmle_is.max(), crit)
        cmle_is_cis.append(ci)
        plot_coverage(ax_cmle_is, theta_grid, crit, cmle_is_cis)
    return ci

@timing
@fresh_cache
def ci_cons(X, v, alpha_level, L, theta_l, theta_u,
            K, test = 'lr', corrected = True, verbose = False):
    arr = array_from_data(X, [v])

    fit_model = StationaryLogistic()
    fit_model.beta['x_0'] = None
    fit_model.confidence_cons(arr, 'x_0', alpha_level, K,
                              L, theta_l, theta_u, test, verbose)

    method = 'conservative-%s' % test
    return fit_model.conf['x_0'][method]

def do_experiment(params):
    seed = Seed(params['random_seed'])

    alpha_level = params['alpha_level']
    verbose = params['verbose']

    L = params['L']
    S = len(params['n_MC_levels'])
    T = params['is_T']
    
    # Set up structure and methods for recording results
    results = { 'completed_trials': 0 }
    for method, disp in [('umle_wald', 'UMLE Wald'),
                         ('umle_boot', 'UMLE bootstrap (pivotal)'),
                         ('cmle_wald', 'CMLE Wald'),
                         ('cmle_boot', 'CMLE bootstrap (pivotal)'),
                         ('brazzale', 'Conditional (Brazzale)'),
                         ('cmle_a', 'CMLE-A LR'),
                         ('cmle_is', 'CMLE-IS (T = %d) LR' % T)] + \
                        [('is_sc_c_%d' % n_MC, 'IS-score (n = %d)' % n_MC)
                         for n_MC in params['n_MC_levels']] + \
                        [('is_sc_u_%d' % n_MC, 'IS-score [un] (n = %d)' % n_MC)
                         for n_MC in params['n_MC_levels']] + \
                        [('is_lr_c_%d' % n_MC, 'IS-LR (n = %d)' % n_MC)
                         for n_MC in params['n_MC_levels']] + \
                        [('is_lr_u_%d' % n_MC, 'IS-LR [un] (n = %d)' % n_MC)
                         for n_MC in params['n_MC_levels']]:
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

        do(ci_umle_wald(X, v, alpha_level), 'umle_wald')

        do(ci_umle_boot(X, v, alpha_level), 'umle_boot')

        do(ci_cmle_wald(X, v, alpha_level), 'cmle_wald')

        do(ci_cmle_boot(X, v, alpha_level), 'cmle_boot')

        do(ci_brazzale(X, v, alpha_level), 'brazzale')

        do(ci_cmle_a(X, v, theta_grid, alpha_level), 'cmle_a')

        do(ci_cmle_is(X, v, theta_grid, alpha_level, T, verbose), 'cmle_is')

        for n_MC in params['n_MC_levels']:
            for test in ['lr', 'score']:
                for corrected_str, corrected in [('c', True), ('u', False)]:
                    do(ci_cons(X, v, alpha_level, params['L'],
                               params['theta_l'], params['theta_u'],
                               n_MC, test = test, corrected = corrected,
                               verbose = verbose),
                       'is_%s_%s_%d' % (test[0:2], corrected_str, n_MC))

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
    print 'Average time: %.2f sec' % (result['total_time'] / R)
    print

if params['plot']:
    ax_cmle_a.set_title('CMLE-A confidence intervals')
    ax_cmle_is.set_title('CMLE-IS confidence intervals')
    plt.show()
