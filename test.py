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
params = { 'N': 300,
           'B': 5,
           'beta_sd': 1.0,
           'x_diff_cutoff': 0.3,
           'alpha_unif_sd': 0.0,
           'alpha_norm_sd': 0.0,
           'alpha_gamma_sd': 0.0,
           'kappa_target': ('density', 0.1),
           'offset_extremes': False,
           'fit_nonstationary': True,
           'fit_method': 'convex_opt',
           'num_reps': 5,
           'sub_sizes': range(10, 60, 10),
           'verbose': False,
           'plot_mse': True,
           'plot_network': False,
           'plot_fit_info': False }


# Set random seed for reproducible output
np.random.seed(137)

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))

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

    data_model.beta[name] = np.random.normal(0, params['beta_sd'])

    x_node = np.random.normal(0, 1, params['N'])
    def f_x(i_1, i_2):
        return abs(x_node[i_1] - x_node[i_2]) < params['x_diff_cutoff']
    net.new_edge_covariate(name).from_binary_function_ind(f_x)

if params['fit_nonstationary']:
    fit_model = NonstationaryLogistic()
else:
    fit_model = StationaryLogistic()
for c in covariates:
    fit_model.beta[c] = None

# Set up recording of results from experiment
results = Results(params['sub_sizes'], params['num_reps'])
add_network_stats(results)
results.new('Sample kappa', 'm', lambda d, f: d.kappa)
def f_c(c):
    return (lambda d, f: d.beta[c]), (lambda d, f: f.beta[c])
for c in covariates:
    # Need to do this hackily to avoid for-loop/lambda-binding weirdness.
    f_true, f_estimated = f_c(c)
    results.new('True beta_{%s}' % c, 'm', f_true)
    results.new('Estimated beta_{%s}' % c, 'm', f_estimated)
if params['offset_extremes']:
    results.new('# Active', 'n', lambda n: np.isfinite(n.offset.matrix()).sum())
else:
    results.new('# Active', 'n', lambda n: n.N ** 2)
if params['fit_nonstationary']:
    import scipy.optimize as opt
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
else:
    def rel_mse_p_ij(n, d, f):
        return rel_mse(f.edge_probabilities(n), d.baseline(n),
                       d.edge_probabilities(n))
    results.new('Rel. MSE(P_ij)', 'nm', rel_mse_p_ij)
    def rel_mse_logit_p_ij(n, d, f):
        return rel_mse(logit(f.edge_probabilities(n)), d.baseline_logit(n),
                       logit(d.edge_probabilities(n)))
    results.new('Rel. MSE(logit P_ij)', 'nm', rel_mse_logit_p_ij)

if params['fit_method'] == 'convex_opt':
    results.new('Obj. evals.', 'm', lambda d, f: f.fit_info['obj_evals'])
    results.new('Wall time (sec.)', 'm', lambda d, f: f.fit_info['wall_time'])
    results.new('||ET_final - T||_2', 'm',
                lambda d, f: np.sqrt(np.sum((f.fit_info['grad_final'])**2)))

for sub_size in params['sub_sizes']:
    print 'subnetwork size = %d' % sub_size
    
    gen = RandomSubnetworks(net, sub_size)
    for rep in range(params['num_reps']):
        subnet = gen.sample()
        data_model.match_kappa(subnet, params['kappa_target'])
        subnet.generate(data_model)
        if params['offset_extremes']: subnet.offset_extremes()

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

        results.record(sub_size, rep, subnet, data_model, fit_model)

# Compute beta MSEs
covariate_mses = []
for c in covariates:
    name = 'MSE(beta_{%s})' % c
    covariate_mses.append(name)
    results.estimate_mse(name, 'True beta_{%s}' % c, 'Estimated beta_{%s}' % c)

# Plot inference performace, in terms of MSE(beta) and MSE(P_ij)
if params['plot_mse']:
    to_plot = [(['MSE(beta_i)'] + covariate_mses,
                {'ymin': 0, 'ymax': 3.0, 'plot_mean': True}),
               ('Rel. MSE(P_ij)', {'ymin': 0, 'ymax': 2, 'baseline': 1}),
               ('# Active', {'ymin': 0})]
    if not (params['fit_nonstationary'] and params['offset_extremes']):
        to_plot.insert(2, ('Rel. MSE(logit P_ij)',
                           {'ymin':0, 'ymax': 2, 'baseline': 1}))
    results.plot(to_plot)
  
# Plot network statistics
if params['plot_network']:
    results.plot([('Average degree', {'ymin': 0, 'plot_mean': True}),
                  (['Out-degree', 'Max out-degree', 'Min out-degree'],
                   {'ymin': 0, 'plot_mean': True}),
                  (['In-degree', 'Max out-degree', 'Min in-degree'],
                   {'ymin': 0, 'plot_mean': True}),
                  ('Self-loop density', {'ymin': 0, 'plot_mean': True}),
                  'Sample kappa'])

# Plot convex optimization fitting internal details
if params['plot_fit_info']:
    results.plot([('Obj. evals.', {'ymin': 0}),
                  ('Wall time (sec.)', {'ymin': 0}),
                  ('||ET_final - T||_2', {'ymin': 0})])
