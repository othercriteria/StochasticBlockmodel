#!/usr/bin/env python

# Generic procedures used in the construction of confidence intervals
# Daniel Klein, 2015-11-20

import numpy as np

from Utility import logsumexp

def invert_test(theta_grid, test_val, crit):
    theta_l_min, theta_l_max = theta_grid.min(), theta_grid.max()

    C_alpha = theta_grid[test_val > crit]
    if len(C_alpha) == 0:
        return 0, 0

    C_alpha_l, C_alpha_u = C_alpha.min(), C_alpha.max()
    if C_alpha_l == theta_l_min:
        C_alpha_l = -np.inf
    if C_alpha_u == theta_l_max:
        C_alpha_u = np.inf

    return C_alpha_l, C_alpha_u

def ci_conservative_generic(X, K, theta_grid, alpha_level,
                            suff, log_likelihood, sample, t,
                            corrected = True, two_sided = True,
                            verbose = False):
    L = len(theta_grid)
    
    # Generate samples from the mixture proposal distribution
    Y = [sample(theta_grid[np.random.randint(L)]) for k in range(K)]

    # Test statistic at observation, for each grid point
    t_X = t(X).reshape((L, 1))
    if verbose:
        print 'X: t_min = %.2f, t_max = %.2f' % (t_X.min(), t_X.max())
        
    # Statistics for the samples from the proposal distribution only
    # need to be calculated once...
    t_Y = np.empty((L, K+1))
    t_Y[:,K] = 0.0
    for k in range(K):
        t_Y[:,k] = t(Y[k])
        if verbose:
            print 'Y_%d: t_min = %.2f, t_max = %.2f' % \
                (k, t_Y[:,k].min(), t_Y[:,k].max())
    if two_sided:
        I_t_Y_plus = t_Y >= t_X
        I_t_Y_plus[:,K] = True
    I_t_Y_minus = -t_Y >= -t_X
    I_t_Y_minus[:,K] = True

    # Probabilities under each component of the proposal distribution
    # only need to be calculated once...
    log_Q_X = np.empty(L)
    log_Q_Y = np.empty((L,K))
    for l in range(L):
        theta_l = theta_grid[l]

        log_Q_X[l] = log_likelihood(X, theta_l)
        for k in range(K):
            log_Q_Y[l,k] = log_likelihood(Y[k], theta_l)
        if verbose:
            print '%.2f: %.2g, %.2g' % \
              (theta_l, np.exp(log_Q_X[l]), np.exp(log_Q_Y[l].max()))
    log_Q_sum_X = logsumexp(log_Q_X)
    log_Q_sum_Y = np.empty(K)
    for k in range(K):
        log_Q_sum_Y[k] = logsumexp(log_Q_Y[:,k])

    # Step over the grid, calculating approximate p-values
    if two_sided:
        log_p_plus = np.empty(L)
    log_p_minus = np.empty(L)
    for l in range(L):
        theta_l = theta_grid[l]
        log_w_l = np.empty(K + 1)

        # X contribution
        if corrected:
            log_w_l[K] = theta_l * (suff * X).sum() - log_Q_sum_X
        else:
            log_w_l[K] = -np.inf

        # Y contribution
        for k in range(K):
            log_w_l[k] = theta_l * (suff * Y[k]).sum() - log_Q_sum_Y[k]

        if two_sided:
            log_p_num_plus = logsumexp(log_w_l[I_t_Y_plus[l]])
        log_p_num_minus = logsumexp(log_w_l[I_t_Y_minus[l]])
        log_p_denom = logsumexp(log_w_l)

        if verbose:
            print '%.2f: %.2g (%.2g, %.2g)' % \
              (theta_l, log_w_l[K], log_w_l[0:K].min(), log_w_l[0:K].max())

        if two_sided:
            log_p_plus[l] = log_p_num_plus - log_p_denom
        log_p_minus[l] = log_p_num_minus - log_p_denom

    if two_sided:
        # p_pm = min(1, 2 * min(p_plus, p_minus))
        log_p_vals = np.fmin(0, np.log(2) + np.fmin(log_p_plus, log_p_minus))
    else:
        log_p_vals = np.fmin(0, log_p_minus)

    return invert_test(theta_grid, log_p_vals, np.log(alpha_level))
