#!/usr/bin/env python

# Test of "new style" network inference.
#
# Observed subnetworks are generated approximately from the
# conditional distribution with fixed margins.
#
# Daniel Klein, 6/1/2012

import numpy as np

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic, FixedMargins
from Experiment import RandomSubnetworks, Results

# Parameters
params = { 'N': 300,
           'B': 5,
           'beta_sd': 1.0,
           'x_diff_cutoff': 0.3,
           'margin_scaling': ('degree', 2),
           'arbitrary_init': False,
           'gibbs_covers': [1.0],
           'fit_nonstationary': False,
           'num_reps': 2,
           'sub_sizes': np.arange(10, 30, 10, dtype=np.int),
           'plot_mse_beta': True }

# Set random seed for reproducible output
np.random.seed(137)

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))

# Initialize full network
net = Network(params['N'])

# Generate covariates and associated coefficients
data_base_model = StationaryLogistic()
covariates = []
for b in range(params['B']):
    name = 'x_%d' % b
    covariates.append(name)

    data_base_model.beta[name] = np.random.normal(0, params['beta_sd'])

    x_node = np.random.normal(0, 1, params['N'])
    def f_x(i_1, i_2):
        return abs(x_node[i_1] - x_node[i_2]) < params['x_diff_cutoff']
    net.new_edge_covariate(name).from_binary_function_ind(f_x)
data_model = FixedMargins(data_base_model)
net.new_node_covariate_int('r')
net.new_node_covariate_int('c')

if params['fit_nonstationary']:
    fit_model = NonstationaryLogistic()
else:
    fit_model = StationaryLogistic()
for c in covariates:
    fit_model.beta[c] = None

# Set up recording of results from experiment
gibbs_results = {}
for gibbs_cover in params['gibbs_covers']:
    results = Results(params['sub_sizes'], params['sub_sizes'],
                      params['num_reps'],
                      'Gibbs cover: %.2f' % gibbs_cover)
    def f_c(c):
        return (lambda d, f: d.base_model.beta[c]), (lambda d, f: f.beta[c])
    for c in covariates:
        # Need to do this hackily to avoid for-loop/lambda-binding weirdness.
        f_true, f_estimated = f_c(c)
        results.new('True beta_{%s}' % c, 'm', f_true)
        results.new('Est. beta_{%s}' % c, 'm', f_estimated)
    gibbs_results[gibbs_cover] = results

for sub_size in params['sub_sizes']:
    size = (sub_size, sub_size)
    print 'Subnetwork size = %s' % str(size)

    gen = RandomSubnetworks(net, size)
    for rep in range(params['num_reps']):
        subnet = gen.sample()

        # Determine margins according to specified scaling and subnetwork size
        scaling, value = params['margin_scaling']
        if scaling == 'degree':
            r, c = np.repeat(value, sub_size), np.repeat(value, sub_size)
        elif scaling == 'density':
            m = int(value * sub_size)
            r, c = np.repeat(m, sub_size), np.repeat(m, sub_size)
        subnet.row_covariates['r'][:] = r
        subnet.col_covariates['c'][:] = c
        
        for gibbs_cover in params['gibbs_covers']:
            data_model.coverage = gibbs_cover
            subnet.generate(data_model,
                            arbitrary_init = params['arbitrary_init'])
            subnet.offset_extremes()
            fit_model.fit_conditional(subnet)
            gibbs_results[gibbs_cover].record(size, rep,
                                              subnet, data_model, fit_model)

# Compute beta MSEs and plot performance in terms of MSE(beta)
for sub, gibbs_cover in enumerate(sorted(gibbs_results)):
    results = gibbs_results[gibbs_cover]
    covariate_mses = []
    for c in covariates:
        name = 'MSE(beta_{%s})' % c
        covariate_mses.append(name)
        results.estimate_mse(name, 'True beta_{%s}' % c, 'Est. beta_{%s}' % c)
    results.plot([(['MSE(beta_i)'] + covariate_mses,
                   {'ymin': 0, 'ymax': 3.0, 'plot_mean': True})])

