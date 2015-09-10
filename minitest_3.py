#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from Array import Array
from Models import NonstationaryLogistic, alpha_norm
from BinaryMatrix import approximate_conditional_nll as cond_a_nll_b
from BinaryMatrix import approximate_from_margins_weights as cond_a_sample_b
from Utility import logsumexp, logabsdiffexp

M = 100
N = 20
theta = 2.0
kappa_target = ('density', 0.1)
T_fit = 20
T = 200
min_error = 0.2
theta_grid_min = 0.0
theta_grid_max = 3.0
theta_grid_G = 121


def cond_a_nll(X, w):
    return cond_a_nll_b(X, w, sort_by_wopt_var = True)

def cond_a_sample(r, c, w, T = 0):
    return cond_a_sample_b(r, c, w, T, sort_by_wopt_var = True)

while True:
    a = Array(M, N)
    alpha_norm(a, 1.0)
    a.new_edge_covariate('x')[:,:] = np.random.normal(0, 1, (M, N))

    d = NonstationaryLogistic()
    d.beta['x'] = theta

    d.match_kappa(a, kappa_target)
    a.generate(d)

    f = NonstationaryLogistic()
    f.beta['x'] = None

    f.fit_conditional(a, T = T_fit, verbose = True)
    abs_err = abs(f.beta['x'] - d.beta['x'])
    if abs_err > min_error:
        print f.beta['x']
        break

theta_vec = np.linspace(theta_grid_min, theta_grid_max, theta_grid_G)
cmle_a_vec = np.empty_like(theta_vec)
cmle_is_vec = np.empty_like(theta_vec)
logkappa_cvsq = np.empty_like(theta_vec)

A = a.array
r = A.sum(1)
c = A.sum(0)
print r
print c
X = a.edge_covariates['x'].matrix()

for l, theta_l in enumerate(theta_vec):
    print l
    logit_P_l = theta_l * X
    w_l = np.exp(logit_P_l)

    cmle_a_vec[l] = -cond_a_nll(A, w_l)
    
    z = cond_a_sample(r, c, w_l, T)
    logf = np.empty(T)
    for t in range(T):
        logQ, logP = z[t][1], z[t][2]
        logf[t] = logP - logQ
    logkappa = -np.log(T) + logsumexp(logf)
    logcvsq = -np.log(T - 1) - 2 * logkappa + \
      logsumexp(2 * logabsdiffexp(logf, logkappa))
    cvsq = np.exp(logcvsq)
    logkappa_cvsq[l] = cvsq
    print 'est. cv^2 = %.2f (T = %d)' % (cvsq, T)
    cmle_is_vec[l] = np.sum(np.log(w_l[A])) - logkappa

print 'CMLE-A: %.2f' % theta_vec[np.argmax(cmle_a_vec)]
print 'CMLE-IS: %.2f' % theta_vec[np.argmax(cmle_is_vec)]
    
plt.figure()
plt.plot(theta_vec, cmle_a_vec)
plt.plot(theta_vec, cmle_is_vec)

plt.figure()
plt.plot(theta_vec, logkappa_cvsq)

plt.show()
