#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from Network import Network
from Models import FixedMargins, StationaryLogistic, NonstationaryLogistic
from Models import alpha_norm

N = 50
D = 1
theta = 2.0
kappa_target = ('row_sum', 2)
alpha_sd = 2.0
n_rep = 100
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

def safe_ci(model, name, method):
    if name in model.conf:
        if method in model.conf[name]:
            return model.conf[name][method]
    else:
        return (0.0, 0.0)

braz_covered = 0
ws_covered = 0
wn_covered = 0
wc_covered = 0
bs_covered = 0
bn_covered = 0
bc_covered = 0
cs_covered = 0
cl_covered = 0
for n in range(n_rep):
    new = net.subnetwork(np.arange(N))
    new.generate(data_model)

    try:
        s_fit.fit_brazzale(new, 'x_0', alpha_level = alpha_level)
        braz_ci_l, braz_ci_u = s_fit.conf['x_0']['brazzale']
    except:
        braz_ci_l = braz_ci_u = 0.0
    if braz_ci_l < theta < braz_ci_u:
        braz_covered += 1

    s_fit.confidence_boot(new, n_bootstrap = n_boot,
                          alpha_level = alpha_level)
    bs_ci_l, bs_ci_u = s_fit.conf['x_0']['pivotal']
    if bs_ci_l < theta < bs_ci_u:
        bs_covered += 1

    s_fit.reset_confidence()
    s_fit.confidence_wald(new)
    ws_ci_l, ws_ci_u = safe_ci(s_fit, 'x_0', 'wald')
    if ws_ci_l < theta < ws_ci_u:
        ws_covered += 1

    new.offset_extremes()

    ns_fit.reset_confidence()
    ns_fit.confidence_wald(new)
    wn_ci_l, wn_ci_u = safe_ci(ns_fit, 'x_0', 'wald')
    if wn_ci_l < theta < wn_ci_u:
        wn_covered += 1

    ns_fit.confidence_boot(new, n_bootstrap = n_boot,
                           alpha_level = alpha_level)
    bn_ci_l, bn_ci_u = ns_fit.conf['x_0']['pivotal']
    if bn_ci_l < theta < bn_ci_u:
        bn_covered += 1

    A = new.as_dense()
    r = A.sum(1)
    c = A.sum(0)

    c_fit = FixedMargins(s_fit)
    new.new_row_covariate('r', np.int)[:] = r
    new.new_col_covariate('c', np.int)[:] = c
    c_fit.fit = c_fit.base_model.fit_conditional

    c_fit.reset_confidence()
    c_fit.confidence_wald(new)
    wc_ci_l, wc_ci_u = safe_ci(c_fit, 'x_0', 'wald')
    if wc_ci_l < theta < wc_ci_u:
        wc_covered += 1

    c_fit.confidence_boot(new, n_bootstrap = n_boot, alpha_level = alpha_level)
    bc_ci_l, bc_ci_u = c_fit.conf['x_0']['pivotal']
    if bc_ci_l < theta < bc_ci_u:
        bc_covered += 1

    c_fit.confidence_cons(new, 'x_0', alpha_level = alpha_level, L = 61,
                          test = 'score')
    cs_ci_l, cs_ci_u = c_fit.conf['x_0']['conservative-score']
    if cs_ci_l < theta < cs_ci_u:
        cs_covered += 1

    c_fit.confidence_cons(new, 'x_0', alpha_level = alpha_level, L = 61,
                          test = 'lr')
    cl_ci_l, cl_ci_u = c_fit.conf['x_0']['conservative-lr']
    if cl_ci_l < theta < cl_ci_u:
        cl_covered += 1

print 'Wald (stationary): %.2f' % (1.0 * ws_covered / n_rep)
print 'Wald (nonstationary): %.2f' % (1.0 * wn_covered / n_rep)
print 'Wald (conditional): %.2f' % (1.0 * wc_covered / n_rep)
print 'Bootstrap (stationary): %.2f' % (1.0 * bs_covered / n_rep)
print 'Bootstrap (nonstationary): %.2f' % (1.0 * bn_covered / n_rep)
print 'Bootstrap (conditional): %.2f' % (1.0 * bc_covered / n_rep)
print 'Brazzale: %.2f' % (1.0 * braz_covered / n_rep)
print 'Conservative (score): %.2f' % (1.0 * cs_covered / n_rep)
print 'Conservative (CMLE-A LR): %.2f' % (1.0 * cl_covered / n_rep)
