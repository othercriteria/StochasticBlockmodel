#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from Network import Network
from Models import FixedMargins, StationaryLogistic, NonstationaryLogistic
from Models import alpha_norm

N = 25
D = 3
theta = 2.0
kappa_target = ('row_sum', 5)
alpha_sd = 1.0
n_rep = 20
n_boot = 10
alpha_level = 0.05

net = Network(N)
alpha_norm(net, alpha_sd)
for d in range(D):
    net.new_edge_covariate('x_%d' % d)[:,:] = np.random.normal(0, 1, (N, N))

data_model = NonstationaryLogistic()
for d in range(D):
    data_model.beta['x_%d' % d] = np.random.normal(0, 1)
data_model.beta['x_0'] = theta
data_model.match_kappa(net, kappa_target)

s_fit = StationaryLogistic()
ns_fit = NonstationaryLogistic()
for d in range(D):
    s_fit.beta['x_%d' % d] = None
    ns_fit.beta['x_%d' % d] = None

ns_covered = 0
br_covered = 0
cb_covered = 0
ch_covered = 0
for n in range(n_rep):
    net.generate(data_model)
    net.offset_extremes()

    A = net.as_dense()
    r = A.sum(1)
    c = A.sum(0)
    
    ns_fit.confidence(net, n_bootstrap = n_boot, alpha = alpha_level)
    ns_ci_l, ns_ci_u = ns_fit.conf['x_0']['pivotal']
    if ns_ci_l < theta < ns_ci_u:
        ns_covered += 1

    try:
        ns_fit.fit_brazzale(net, 'x_0', alpha_level = alpha_level)
        br_ci_l, br_ci_u = ns_fit.conf['x_0']['brazzale']
    except:
        br_ci_l = br_ci_u = 0.0
    if br_ci_l < theta < br_ci_u:
        br_covered += 1

    c_fit = FixedMargins(s_fit)
    net.new_row_covariate('r', np.int)[:] = r
    net.new_col_covariate('c', np.int)[:] = c
    c_fit.fit = c_fit.base_model.fit_conditional
    c_fit.confidence(net, n_bootstrap = n_boot, alpha = alpha_level)
    cb_ci_l, cb_ci_u = c_fit.conf['x_0']['pivotal']
    if cb_ci_l < theta < cb_ci_u:
        cb_covered += 1
    c_fit.fit(net)
    c_fit.confidence_harrison(net, 'x_0', alpha_level = alpha_level)
    ch_ci_l, ch_ci_u = c_fit.conf['x_0']['harrison']
    if ch_ci_l < theta < ch_ci_u:
        ch_covered += 1

print 'NS: %.2f' % (1.0 * ns_covered / n_rep)
print 'Brazzale: %.2f' % (1.0 * br_covered / n_rep)
print 'Conditional (bootstrap): %.2f' % (1.0 * cb_covered / n_rep)
print 'Conditional (conservative): %.2f' % (1.0 * ch_covered / n_rep)
