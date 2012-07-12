#!/usr/bin/env python

# Test of "new style" network inference.
#
# Observed subnetworks are generated approximately from the
# conditional distribution with fixed margins.
#
# Daniel Klein, 6/1/2012

import numpy as np

from Network import Network
from Models import StationaryLogistic, RaschLogistic, StationaryLogisticMargins
from Experiment import RandomSubnetworks, Results, add_network_stats

# Parameters
params = { 'N': 150,
           'B': 5,
           'beta_sd': 1.0,
           'x_diff_cutoff': 0.3,
           'margin_scaling': ('degree', 2),
           'gibbs_cover': 1,
           'fit_nonstationary': False,
           'num_reps': 5,
           'sub_sizes': range(10, 80, 10),
           'plot_mse_beta': True,
           'plot_network': True }


# Set random seed for reproducible output
np.random.seed(137)

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))

# Initialize full network
net = Network(params['N'])

# Generate covariates and associated coefficients
data_model = StationaryLogisticMargins()
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
def f_c(c):
    return (lambda d, f: d.beta[c]), (lambda d, f: f.beta[c])
for c in covariates:
    # Need to do this hackily to avoid for-loop/lambda-binding weirdness.
    f_true, f_estimated = f_c(c)
    results.new('True beta_{%s}' % c, 'm', f_true)
    results.new('Estimated beta_{%s}' % c, 'm', f_estimated)

for sub_size in params['sub_sizes']:
    print 'Subnetwork size = %d' % sub_size

    gen = RandomSubnetworks(net, sub_size)
    for rep in range(params['num_reps']):
        subnet = gen.sample()

        # Determine margins according to specified scaling and subnetwork size
        scaling, value = params['margin_scaling']
        if scaling == 'degree':
            r, c = np.repeat(value, sub_size), np.repeat(value, sub_size)
        elif scaling == 'density':
            m = int(value * sub_size)
            r, c = np.repeat(m, sub_size), np.repeat(m, sub_size)
        subnet.generate(data_model, r, c, params['gibbs_cover'])

        fit_model.fit_convex_opt(subnet)

        results.record(sub_size, rep, subnet, data_model, fit_model)

# Compute beta MSEs
covariate_mses = []
for c in covariates:
    name = 'MSE(beta_{%s})' % c
    covariate_mses.append(name)
    results.estimate_mse(name, 'True beta_{%s}' % c, 'Estimated beta_{%s}' % c)

# Plot inference performace, in terms of MSE(beta)
if params['plot_mse_beta']:
    results.plot([(['MSE(beta_i)'] + covariate_mses,
                   {'ymin': 0, 'ymax': 3.0, 'plot_mean': True})])
  
# Plot network statistics as well as sparsity parameter
if params['plot_network']:
    results.plot([('Average degree', {'ymin': 0, 'plot_mean': True}),
                  (['Out-degree', 'Max out-degree', 'Min out-degree'],
                   {'ymin': 0, 'plot_mean': True}),
                  (['In-degree', 'Max out-degree', 'Min in-degree'],
                   {'ymin': 0, 'plot_mean': True}),
                  ('Self-loop density', {'ymin': 0, 'plot_mean': True})])

