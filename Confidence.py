#!/usr/bin/env python

# Generic procedures used in the construction of confidence intervals
# Daniel Klein, 2015-11-20

import numpy as np

from Utility import logsumexp

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

def ci_conservative_generic(X, K, theta_grid, alpha_level,
                            log_likelihood, sample, t,
                            verbose = False):
    L = len(theta_grid)
    
    # Generate samples from the mixture proposal distribution
    Y = []
    for k in range(K):
        l_k = np.random.randint(L)
        theta_k = theta_grid[l_k]
        Y.append(sample(theta_k))

    # Test statistic at observation
    t_X = t(X)
        
    # Statistics for the samples from the proposal distribution only
    # need to be calculated once...
    t_Y = np.zeros(K + 1)
    for k in range(K):
        t_Y[k] = t(Y[k])
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
