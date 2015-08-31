#!/usr/bin/env python

import numpy as np
from scipy.stats import chi2

from BinaryMatrix import approximate_conditional_nll as acnll
from BinaryMatrix import approximate_from_margins_weights as sample
from Utility import logsumexp, logabsdiffexp

from Experiment import Seed

# Parameters
params = { 'M': 50,
           'N': 10,
           'theta': 2.0,
           'kappa': -1.628,
           'alpha_min': -0.4,
           'beta_min': -0.86,
           'alpha_level': 0.05,
           'n_MC_levels': [10, 50],
           'is_T': 100,
           'n_rep': 100,
           'L': 61,
           'theta_l': -6.0,
           'theta_u': 6.0,
           'do_prune': True,
           'random_seed': 137,
           'verbose': True }
    
def generate_data(params, seed):
    M, N = params['M'], params['N']
    
    # Advance random seed for data generation
    seed.next()

    # Generate covariate
    v = np.random.normal(0, 1.0, (M,N))

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
        while True:
            r, c = X.sum(1), X.sum(0)
            r_p = (r == 0) + (r == N)
            c_p = (c == 0) + (c == M)
            pruning = np.any(r_p) or np.any(c_p)
            
            X = X[-r_p][:,-c_p].copy()
            v = v[-r_p][:,-c_p].copy()

            if not pruning:
                break

    return X, v

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

def ci_cmle_a(X, v, theta_grid, alpha_level):
    cmle_a = np.empty_like(theta_grid)
    for l, theta_l in enumerate(theta_grid):
        logit_P_l = theta_l * v
        cmle_a[l] = -acnll(X, np.exp(logit_P_l))

    return invert_test(theta_grid, cmle_a - cmle_a.max(),
                       -0.5 * chi2.ppf(1 - alpha_level, 1))

def ci_cmle_is(X, v, theta_grid, alpha_level, T = 100, verbose = False):
    cmle_is = np.empty_like(theta_grid)
    for l, theta_l in enumerate(theta_grid):
        logit_P_l = theta_l * v
        w_l = np.exp(logit_P_l)
        r = X.sum(1)
        c = X.sum(0)

        z = sample(r, c, w_l, T)
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

        Y_sparse = sample(r, c, np.exp(logit_P_l))
        Y_dense = np.zeros((M_p,N_p), dtype = np.bool)
        for i, j in Y_sparse:
            if i == -1: break
            Y_dense[i,j] = 1
        Y.append(Y_dense)

    # Statistics for the samples from the proposal distribution only
    # need to be calculated once...
    t_Y = np.empty(K)
    for k in range(K):
        t_Y[k] = np.sum(Y[k] * v)
    I_t_Y_plus = t_Y >= t_X
    I_t_Y_minus = -t_Y >= -t_X

    # Probabilities under each component of the proposal distribution
    # only need to be calculated once...
    log_Q_X = np.empty(L)
    log_Q_Y = np.empty((L,K))
    for l in range(L):
        theta_l = theta_grid[l]

        logit_P_l = theta_l * v
        log_Q_X[l] = -acnll(X, np.exp(logit_P_l))
        for k in range(K):
            log_Q_Y[l,k] = -acnll(Y[k], np.exp(logit_P_l))
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

        log_p_num_plus = \
          logsumexp(log_w_l[I_t_Y_plus]) if np.any(I_t_Y_plus) else -np.inf
        log_p_num_minus = \
          logsumexp(log_w_l[I_t_Y_minus]) if np.any(I_t_Y_minus) else -np.inf
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

    R = params['n_rep']
    alpha = params['alpha_level']
    verbose = params['verbose']

    L = params['L']
    S = len(params['n_MC_levels'])
    T = params['is_T']
    
    # Do experiment
    in_interval = np.empty((2+S,R))
    length = np.empty((2+S,R))
    def record(name, m, t, ci_l, ci_u):
        print '%s: [%.2f, %.2f]' % (name, ci_l, ci_u)

        in_interval[m,t] = ci_l <= params['theta'] <= ci_u
        length[m,t] = ci_u - ci_l

    for trial in range(R):
        X, v = generate_data(params, seed)

        theta_grid = np.linspace(params['theta_l'], params['theta_u'], L)
        
        ci_l, ci_u = ci_cmle_a(X, v, theta_grid, alpha)
        record('CMLE-A', 0, trial, ci_l, ci_u)

        ci_l, ci_u = ci_cmle_is(X, v, theta_grid, alpha, T, verbose = verbose)
        record('CMLE-IS (T = %d)' % T, 1, trial, ci_l, ci_u)

        for s, n_MC in enumerate(params['n_MC_levels']):
            ci_l, ci_u = ci_conservative(X, v, n_MC, theta_grid, alpha,
                                         verbose = verbose)
            record('Conservative (n = %d)' % n_MC, (2 + s), trial, ci_l, ci_u)

    # For verifying that same data was generated even if different
    # algorithms consumed a different amount of randomness
    seed.final()

    return in_interval, length

in_interval, length = do_experiment(params)

def report(name, m):
    print '%s:' % name
    print 'Coverage probability: %.2f' % np.mean(in_interval[m])
    print 'Median length: %.2f' % np.median(length[m])
    print

print '\n\n'
report('CMLE-A', 0)
report('CMLE-IS (T = %d)' % params['is_T'], 1)
for s, n_MC in enumerate(params['n_MC_levels']):
    report('Conservative (n = %d)' % n_MC, (2 + s))
