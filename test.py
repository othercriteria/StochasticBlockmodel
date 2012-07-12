#!/usr/bin/env python

# Test of "new style" network inference
# Daniel Klein, 5/16/2012

import numpy as np

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic
from Models import alpha_zero, alpha_norm, alpha_unif, alpha_gamma
from Experiment import RandomSubnetworks, Results, add_network_stats
from Utility import logit

# Parameters
params = { 'N': 200,
           'B': 5,
           'beta_sd': 1.0,
           'x_diff_cutoff': 0.3,
           'alpha_unif': 0.0,
           'alpha_norm_sd': 0.0,
           'alpha_gamma_sd': 0.0,
           'kappa_target': ('degree', 5),
           'fit_nonstationary': False,
           'fit_method': 'convex_opt',
           'num_reps': 10,
           'sub_sizes': range(10, 150, 10),
           'N_test': 5,
           'plot_mse': True,
           'plot_oos': False,
           'plot_network': True }


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
elif params['alpha_unif'] > 0.0:
    alpha_unif(net, params['alpha_unif'])
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
results.new('Subnetwork kappa', 'm', lambda d, f: d.kappa)
def f_c(c):
    return (lambda d, f: d.beta[c]), (lambda d, f: f.beta[c])
for c in covariates:
    # Need to do this hackily to avoid for-loop/lambda-binding weirdness.
    f_true, f_estimated = f_c(c)
    results.new('True beta_{%s}' % c, 'm', f_true)
    results.new('Estimated beta_{%s}' % c, 'm', f_estimated)
results.new('MSE(P_{ij})', 'nm',
            lambda n, d, f: np.mean((d.edge_probabilities(n) - \
                                     f.edge_probabilities(n))**2))
results.new('MSE(logit_P_{ij})', 'nm',
            lambda n, d, f: np.mean((logit(d.edge_probabilities(n)) - \
                                     logit(f.edge_probabilities(n)))**2))

for sub_size in params['sub_sizes']:
    print 'subnetwork size = %d' % sub_size
    
    gen = RandomSubnetworks(net, sub_size, params['N_test'])
    for rep in range(params['num_reps']):
        subnet, subnet_test = gen.sample()
        data_model.match_kappa(subnet, params['kappa_target'])
        subnet.generate(data_model)

        if params['fit_method'] == 'convex_opt':
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

# Plot inference performace, in terms of MSE(beta) and MSE(P_ij); also
# plot kappas chosen for data models
if params['plot_mse']:
    results.plot([(['MSE(beta_i)'] + covariate_mses,
                   {'ymin': 0, 'ymax': 3.0, 'plot_mean': True}),
                  ('MSE(P_{ij})', {'ymin': 0, 'ymax': 1}),
                  ('MSE(logit_P_{ij})', {'ymin': 0, 'ymax': 5}),
                  'Subnetwork kappa'])
  
# Plot network statistics as well as sparsity parameter
if params['plot_network']:
    results.plot([('Average degree', {'ymin': 0, 'plot_mean': True}),
                  (['Out-degree', 'Max out-degree', 'Min out-degree'],
                   {'ymin': 0, 'plot_mean': True}),
                  (['In-degree', 'Max out-degree', 'Min in-degree'],
                   {'ymin': 0, 'plot_mean': True}),
                  ('Self-loop density', {'ymin': 0, 'plot_mean': True})])
