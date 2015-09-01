#!/usr/bin/env python

import signal
from time import time

import numpy as np
from scipy.stats import chi2

from BinaryMatrix import approximate_conditional_nll as cond_a_nll_b
from BinaryMatrix import approximate_from_margins_weights as cond_a_sample_b
from Utility import logsumexp, logabsdiffexp
from Experiment import Seed

# Parameters
params = { 'M': 20,
           'N': 10,
           'theta': 2.0,
           'kappa': -1.628,
           'alpha_min': -0.4,
           'beta_min': -0.86,
           'v_min': -0.5,
           'alpha_level': 0.05,
           'n_MC_levels': [10, 50, 100],
           'wopt_sort': True,
           'is_T': 100,
           'n_rep': 100,
           'L': 13,
           'theta_l': -6.0,
           'theta_u': 6.0,
           'do_prune': True,
           'random_seed': 137,
           'verbose': True }

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

def generate_data(params, seed):
    M, N = params['M'], params['N']

    # Generate shared covariate
    v = np.random.uniform(params['v_min'], params['v_min'] + 1, (M,N))

    while True:
        # Advance random seed for data generation
        seed.next()

        # Generate Bernoulli probabilities from logistic regression model
        logit_P = np.zeros((M,N))
        for i in range(1,M):
            logit_P[i,:] += np.random.uniform(params['alpha_min'],
                                              params['alpha_min'] + 1)
        for j in range(1,N):
            logit_P[:,j] += np.random.uniform(params['beta_min'],
                                              params['beta_min'] + 1)
        logit_P += params['kappa']
        logit_P += params['theta'] * v
        P = 1.0 / (1.0 + np.exp(-logit_P))

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

def invert_test(theta_grid, test_val, crit):
    theta_l_min, theta_l_max = min(theta_grid), max(theta_grid)

    C_alpha = theta_grid[test_val > crit]
    if len(C_alpha) == 0:
        return 0, 0

    C_alpha_l, C_alpha_u = np.min(C_alpha), np.max(C_alpha)
    if C_alpha_l == theta_l_min:
        C_alpha_l = -np.inf
    if C_alpha_u == theta_l_max:
        C_alpha_u = np.inf

    return C_alpha_l, C_alpha_u

def timing(func):
    def inner(*args, **kwargs):
        start_time = time()
        val = func(*args, **kwargs)
        elapsed = time() - start_time
        return val, elapsed

    return inner

@timing
def ci_cmle_a(X, v, theta_grid, alpha_level):
    cmle_a = np.empty_like(theta_grid)
    for l, theta_l in enumerate(theta_grid):
        logit_P_l = theta_l * v
        cmle_a[l] = -cond_a_nll(X, np.exp(logit_P_l))

    return invert_test(theta_grid, cmle_a - cmle_a.max(),
                       -0.5 * chi2.ppf(1 - alpha_level, 1))

@timing
def ci_cmle_is(X, v, theta_grid, alpha_level, T = 100, verbose = False):
    cmle_is = np.empty_like(theta_grid)
    for l, theta_l in enumerate(theta_grid):
        logit_P_l = theta_l * v
        w_l = np.exp(logit_P_l)
        r = X.sum(1)
        c = X.sum(0)

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

    return invert_test(theta_grid, cmle_is - cmle_is.max(),
                       -0.5 * chi2.ppf(1 - alpha_level, 1))

@timing
def ci_conservative(X, v, K, theta_grid, alpha_level, verbose = False):
    M_p, N_p = X.shape
    L = len(theta_grid)
    
    # Observed statistic
    t_X = np.sum(X * v)

    # Row and column margins; the part of the data we can use to design Q
    r, c = X.sum(1), X.sum(0)

    # Generate samples from the mixture proposal distribution
    Y = []
    for k in range(K):
        l_k = np.random.randint(L)
        theta_k = theta_grid[l_k]
        logit_P_l = theta_k * v

        Y_sparse = cond_a_sample(r, c, np.exp(logit_P_l))
        Y_dense = np.zeros((M_p,N_p), dtype = np.bool)
        for i, j in Y_sparse:
            if i == -1: break
            Y_dense[i,j] = 1
        Y.append(Y_dense)

    # Statistics for the samples from the proposal distribution only
    # need to be calculated once...
    t_Y = np.zeros(K + 1)
    for k in range(K):
        t_Y[k] = np.sum(Y[k] * v)
    I_t_Y_plus = t_Y >= t_X
    I_t_Y_plus[K] = True
    I_t_Y_minus = -t_Y >= -t_X
    I_t_Y_minus[K] = True

    # Probabilities under each component of the proposal distribution
    # only need to be calculated once...
    log_Q_X = np.empty(L)
    log_Q_Y = np.empty((L,K))
    for l in range(L):
        theta_l = theta_grid[l]

        logit_P_l = theta_l * v
        log_Q_X[l] = -cond_a_nll(X, np.exp(logit_P_l))
        for k in range(K):
            log_Q_Y[l,k] = -cond_a_nll(Y[k], np.exp(logit_P_l))
        if verbose:
            print '%.2f: %.2g, %.2g' % \
              (theta_l, np.exp(log_Q_X[l]), np.exp(log_Q_Y[l].max()))
    log_Q_sum_X = logsumexp(log_Q_X)
    log_Q_sum_Y = np.empty(K)
    for k in range(K):
        log_Q_sum_Y[k] = logsumexp(log_Q_Y[:,k])

    # Step over the grid, calculating approximate p-values
    log_p_plus = np.empty(L)
    log_p_minus = np.empty(L)
    for l in range(L):
        theta_l = theta_grid[l]
        log_w_l = np.empty(K + 1)

        # X contribution
        log_w_l[K] = (theta_l * t_X) - log_Q_sum_X

        # Y contribution
        for k in range(K):
            log_w_l[k] = (theta_l * t_Y[k]) - log_Q_sum_Y[k]

        log_p_num_plus = logsumexp(log_w_l[I_t_Y_plus])
        log_p_num_minus = logsumexp(log_w_l[I_t_Y_minus])
        log_p_denom = logsumexp(log_w_l)

        if verbose:
            print '%.2f: %.2g (%.2g, %.2g)' % \
              (theta_l, log_w_l[K], log_w_l[0:K].min(), log_w_l[0:K].max())

        log_p_plus[l] = log_p_num_plus - log_p_denom
        log_p_minus[l] = log_p_num_minus - log_p_denom

    # p_pm = min(1, 2 * min(p_plus, p_minus))
    log_p_pm = np.fmin(0, np.log(2) + np.fmin(log_p_plus, log_p_minus))
    return invert_test(theta_grid, log_p_pm, np.log(alpha_level))

def do_experiment(params):
    seed = Seed(params['random_seed'])

    alpha = params['alpha_level']
    verbose = params['verbose']

    L = params['L']
    S = len(params['n_MC_levels'])
    T = params['is_T']
    
    # Do experiment
    results = { 'completed_trials': 0 }
    for method, display in [('cmle_a', 'CMLE-A'),
                            ('cmle_is', 'CMLE-IS (T = %d)' % T)] + \
                           [('cons_%d' % s, 'Conservative (n = %d)' % n_MC)
                            for s, n_MC in enumerate(params['n_MC_levels'])]:
        results[method] = { 'display': display,
                            'in_interval': [],
                            'length': [],
                            'total_time': 0.0 }

    def do_and_record(out, name):
        ci, elapsed = out
        ci_l, ci_u = ci

        result = results[name]

        print '%s (%.2f sec): [%.2f, %.2f]' % \
          (result['display'], elapsed, ci_l, ci_u)

        result['in_interval'].append(ci_l <= params['theta'] <= ci_u)
        result['length'].append(ci_u - ci_l)
        result['total_time'] += elapsed

    for X, v in generate_data(params, seed):
        if (results['completed_trials'] == params['n_rep']) or terminated:
            break

        theta_grid = np.linspace(params['theta_l'], params['theta_u'], L)

        do_and_record(ci_cmle_a(X, v, theta_grid, alpha),
                      'cmle_a')

        do_and_record(ci_cmle_is(X, v, theta_grid, alpha, T, verbose),
                      'cmle_is')

        for s, n_MC in enumerate(params['n_MC_levels']):
            do_and_record(ci_conservative(X, v, n_MC, theta_grid, alpha,
                                          verbose),
                          'cons_%d' % s)

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
