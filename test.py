#!/usr/bin/env python

# Test of "new style" network inference
# Daniel Klein, 5/16/2012

import numpy as np

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic
from Models import alpha_zero, alpha_norm, alpha_unif, alpha_gamma
from Experiment import RandomSubnetworks, Results, add_network_stats, rel_mse
from Utility import logit

# Parameters
params = { 'N': 220,
           'B': 2,
           'theta_sd': 1.0,
           'theta_fixed': { 'x_0': 2.0, 'x_1': -1.0 },           
           'alpha_unif_sd': 0.0,
           'alpha_norm_sd': 2.0,
           'alpha_gamma_sd': 0.0,
           'kappa_target': ('degree', 2),
           'offset_extremes': True,
           'fisher_information': False,
           'baseline': False,
           'fit_nonstationary': True,
           'fit_method': 'none',
           'num_reps': 10,
           'sampling': 'new',
           'sub_sizes': range(10, 60, 10),
           'verbose': True,
           'plot_mse': True,
           'plot_network': True,
           'plot_fit_info': True }


# Set random seed for reproducible output
np.random.seed(137)

# Initialize full network
net = Network(params['N'])

# Generate node-level propensities to extend and receive edges
if params['alpha_norm_sd'] > 0.0:
    alpha_norm(net, params['alpha_norm_sd'])
elif params['alpha_unif_sd'] > 0.0:
    alpha_unif(net, params['alpha_unif_sd'])
elif params['alpha_gamma_sd'] > 0.0:
    # Choosing location somewhat arbitrarily to give unit skewness
    alpha_gamma(net, 4.0, params['alpha_gamma_sd'])
else:
    alpha_zero(net)

# Generate covariates and associated coefficients
data_model = NonstationaryLogistic()
covariates = []
for b in range(params['B']):
    name = 'x_%d' % b
    covariates.append(name)

    if name in params['theta_fixed']:
        data_model.beta[name] = params['theta_fixed'][name]
    else:
        data_model.beta[name] = np.random.normal(0, params['theta_sd'])

    def f_x(i_1, i_2):
        return np.random.normal(0, 1)
    net.new_edge_covariate(name).from_binary_function_ind(f_x)

# Generate large network, if necessary
if not params['sampling'] == 'new':
    data_model.match_kappa(net, params['kappa_target'])
    net.generate(data_model)
 
if params['fit_nonstationary']:
    fit_model = NonstationaryLogistic()
else:
    fit_model = StationaryLogistic()
for c in covariates:
    fit_model.beta[c] = None

# Set up recording of results from experiment
results = Results(params['sub_sizes'], params['num_reps'])
add_network_stats(results)
if params['sampling'] == 'new':
    results.new('Subnetwork kappa', 'm', lambda d, f: d.kappa)
def true_est_theta_c(c):
    return (lambda d, f: d.beta[c]), (lambda d, f: f.beta[c])
for c in covariates:
    # Need to do this hackily to avoid for-loop/lambda-binding weirdness.
    f_true, f_estimated = true_est_theta_c(c)
    results.new('True theta_{%s}' % c, 'm', f_true)
    results.new('Estimated theta_{%s}' % c, 'm', f_estimated)
if params['offset_extremes']:
    results.new('# Active', 'n', lambda n: np.isfinite(n.offset.matrix()).sum())
else:
    results.new('# Active', 'n', lambda n: n.N ** 2)
if params['fisher_information']:
    def info_theta_c(c):
        def f_info_theta_c(d, f):
            return d.I_inv['theta_{%s}' % c]
        return f_info_theta_c
    for c in covariates:
        results.new('Info theta_{%s}' % c, 'm', info_theta_c(c))
if params['baseline']:
    def rel_mse_p_ij(n, d, f):
        P = d.edge_probabilities(n)
        return rel_mse(f.edge_probabilities(n), f.baseline(n), P)
    results.new('Rel. MSE(P_ij)', 'nm', rel_mse_p_ij)
    if not params['offset_extremes']:
        def rel_mse_logit_p_ij(n, d, f):
            logit_P = logit(d.edge_probabilities(n))
            logit_Q = f.baseline_logit(n)
            return rel_mse(logit(f.edge_probabilities(n)), logit_Q, logit_P)
        results.new('Rel. MSE(logit P_ij)', 'nm', rel_mse_logit_p_ij)

if params['fit_method'] in ['convex_opt', 'conditional']:
    def work(f):
        w = 0
        for work_type in ['nll_evals', 'grad_nll_evals', 'cnll_evals']:
            if work_type in f.fit_info:
                w += f.fit_info[work_type]
        return w
    results.new('Work', 'm', lambda d, f: work(f))
    results.new('Wall time (sec.)', 'm', lambda d, f: f.fit_info['wall_time'])
    results.new('||ET_final - T||_2', 'm',
                lambda d, f: np.sqrt(np.sum((f.fit_info['grad_nll_final'])**2)))

for sub_size in params['sub_sizes']:
    print 'subnetwork size = %d' % sub_size

    if params['sampling'] == 'new':
        gen = RandomSubnetworks(net, sub_size)
    else:
        gen = RandomSubnetworks(net, sub_size, method = params['sampling'])

    for rep in range(params['num_reps']):
        subnet = gen.sample()

        if params['fisher_information']:
            data_model.fisher_information(subnet)
        
        if params['sampling'] == 'new':
            data_model.match_kappa(subnet, params['kappa_target'])
            subnet.generate(data_model)
        
        if params['offset_extremes']:
            if not params['fit_method'] == 'conditional':
                subnet.offset_extremes()

        if params['fit_method'] == 'convex_opt':
            if params['verbose']:
                fit_model.fit_convex_opt(subnet, verbose = True)
                print
            else:
                fit_model.fit_convex_opt(subnet)
        elif params['fit_method'] == 'logistic':
            fit_model.fit_logistic(subnet)
        elif params['fit_method'] == 'logistic_l2':
            fit_model.fit_logistic_l2(subnet, prior_precision = 1.0)
        elif params['fit_method'] == 'mh':
            for c in covariates:
                fit_model.beta[c] = 0.0
            fit_model.fit_mh(subnet)
        elif params['fit_method'] == 'conditional':
            fit_model.fit_conditional(subnet, T = 0, verbose = True)
            if params['offset_extremes']:
                subnet.offset_extremes()
                fit_model.fit_convex_opt(subnet, fix_beta = True)
        elif params['fit_method'] == 'composite':
            fit_model.fit_composite(subnet, T = 100, verbose = True)
            if params['offset_extremes']:
                subnet.offset_extremes()
                fit_model.fit_convex_opt(subnet, fix_beta = True)
        elif params['fit_method'] == 'brazzale':
            fit_model.fit_brazzale(subnet)
        elif params['fit_method'] == 'saddlepoint':
            fit_model.fit_saddlepoint(subnet)
            fit_model.fit_convex_opt(subnet, fix_beta = True)
        elif params['fit_method'] == 'none':
            pass
            
        results.record(sub_size, rep, subnet, data_model, fit_model)

# Compute beta MSEs
covariate_mses = []
for c in covariates:
    name = 'MSE(theta_{%s})' % c
    covariate_mses.append(name)
    results.estimate_mse(name, 'True theta_{%s}' % c, 'Estimated theta_{%s}' % c)
results.summary()

# Plot inference performace, in terms of MSE(theta) and MSE(P_ij)
if params['plot_mse']:
    to_plot = []
    if not params['fit_method'] == 'none':
        to_plot.append((['MSE(theta_i)'] + covariate_mses,
                        {'ymin': 0, 'ymax': 0.5, 'plot_mean': True}))
    if params['baseline']:
        to_plot.append(('Rel. MSE(P_ij)',
                        {'ymin': 0, 'ymax': 2, 'baseline': 1}))
        if not params['offset_extremes']:
            to_plot.append(('Rel. MSE(logit P_ij)',
                           {'ymin':0, 'ymax': 2, 'baseline': 1}))
    to_plot.append(('# Active', {'ymin': 0}))
    if params['fisher_information']:
        to_plot.append((['Info theta_i'] + \
                        ['Info theta_{%s}' % c for c in covariates],
                        {'ymin': 0, 'plot_mean': True}))
    results.plot(to_plot)
  
# Plot network statistics
if params['plot_network']:
    to_plot = [('Average degree', {'ymin': 0, 'plot_mean': True}),
               (['Out-degree', 'Max out-degree', 'Min out-degree'],
                {'ymin': 0, 'plot_mean': True}),
               (['In-degree', 'Max out-degree', 'Min in-degree'],
                {'ymin': 0, 'plot_mean': True}),
               ('Self-loop density', {'ymin': 0, 'plot_mean': True})]
    if params['sampling'] == 'new':
        to_plot.append('Subnetwork kappa')
    results.plot(to_plot)

# Plot convex optimization fitting internal details
if (params['plot_fit_info'] and
    params['fit_method'] in ['convex_opt', 'conditional']):
    results.plot([('Work', {'ymin': 0}),
                  ('Wall time (sec.)', {'ymin': 0}),
                  ('||ET_final - T||_2', {'ymin': 0})])

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))
