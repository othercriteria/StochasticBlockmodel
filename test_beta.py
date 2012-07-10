#!/usr/bin/env python

# Test of "new style" network inference
#
# Observed networks are generated according to node-specific
# probabilities that are not adequately modeled by covariates.
#
# Daniel Klein, 5/16/2012

import numpy as np

from Network import Network
from Models import IndependentBernoulli
from Models import StationaryLogistic, NonstationaryLogistic
from Experiment import RandomSubnetworks, Results, add_network_stats
from Utility import logit

# Parameters
params = { 'N': 200,
           'B': 5,
           'beta_sd': 1.0,
           'x_diff_cutoff': 0.3,
           'P_beta_params': (1, 9),
           'fit_nonstationary': False,
           'num_reps': 10,
           'sub_sizes': range(10, 60, 10),
           'plot_mse': True,
           'plot_network': True }


# Set random seed for reproducible output
np.random.seed(137)

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))

# Initialize full network
net = Network(params['N'])
data_model = IndependentBernoulli()

# Generate underlying edge probabilities as offsets
P_alpha, P_beta = params['P_beta_params']
P = np.random.beta(P_alpha, P_beta, (params['N'],params['N']))
net.initialize_offset()[:,:] = logit(P)

# Generate covariates and associated coefficients
covariates = []
data_model.beta = {}
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

# Repeatedly generate and fit networks
for sub_size in params['sub_sizes']:
    print 'Subnetwork size = %d' % sub_size

    gen = RandomSubnetworks(net, sub_size)
    for rep in range(params['num_reps']):
        subnet = gen.sample()
        subnet.generate(data_model)

        fit_model.fit(subnet)

        results.record(sub_size, rep, subnet, data_model, fit_model)

# Compute MSEs
covariate_mses = []
for c in covariates:
    name = 'MSE(beta_{%s})' % c
    covariate_mses.append(name)
    results.estimate_mse(name, 'True beta_{%s}' % c, 'Estimated beta_{%s}' % c)

# Plot inference performace, in terms of MSE(beta) and MSE(P_ij)
if params['plot_mse']:
    results.plot([(['MSE(beta_i)'] + covariate_mses,
                   {'ymin': 0, 'ymax': 3.0, 'plot_mean': True}),
                  ('MSE(P_{ij})', {'ymin': 0, 'ymax': 1}),
                  ('MSE(logit_P_{ij})', {'ymin': 0, 'ymax': 5})])
  
# Plot network statistics as well as sparsity parameter
if params['plot_network']:
    results.plot([('Average degree', {'ymin': 0, 'plot_mean': True}),
                  (['Out-degree', 'Max out-degree', 'Min out-degree'],
                   {'ymin': 0, 'plot_mean': True}),
                  (['In-degree', 'Max out-degree', 'Min in-degree'],
                   {'ymin': 0, 'plot_mean': True}),
                  ('Self-loop density', {'ymin': 0, 'plot_mean': True})])
