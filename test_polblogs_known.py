#!/usr/bin/env python

# Testing blockmodel when real block structure is known...
# Daniel Klein, 2014-03-12

import numpy as np

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic, alpha_zero
from Experiment import RandomSubnetworks, Results, add_network_stats

# Parameters
params = { 'fit_stationary': True,
           'fit_nonstationary': False,
           'fit_conditional': False,
           'fit_conditional_is': False,
           'num_reps': 10,
           'sub_sizes': range(100, 1410, 100),
           'degree_covs': True,
           'sampling': 'node',
           'plot_network': True }


# Set random seed for reproducible output
np.random.seed(137)

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))

# Initialize political blogs network from file. The "value" covariate
# is the ground truth membership to the left-leaning (0) or
# right-leaning (1) class.
net = Network()
net.network_from_file_gml('data/polblogs/polblogs.gml', ['value'])

covariates = []

# Political covariates
for v_1, v_2, name in [(0, 0, 'left_to_left'),
                       (1, 1, 'right_to_right'),
                       (0, 1, 'left_to_right')]:
    covariates.append(name)

    def f_x(i_1, i_2):
        return ((net.node_covariates['value'][i_1] == v_1) and
                (net.node_covariates['value'][i_2] == v_2))

    net.new_edge_covariate(name).from_binary_function_ind(f_x)

# Degree heterogeneity covariates
if params['degree_covs']:
    r = np.array(net.network.asfptype().sum(1),dtype=np.int).flatten()
    c = np.array(net.network.asfptype().sum(0),dtype=np.int).flatten()
    degree = r + c
    med_degree = np.median(degree)
    net.new_node_covariate('low_degree').from_pairs(net.names,
                                                    degree < med_degree)
    for v_1, v_2, name in [(0, 0, 'high_to_high'),
                           (1, 1, 'low_to_low'),
                           (0, 1, 'high_to_low')]:
        covariates.append(name)

        def f_x(i_1, i_2):
            return ((net.node_covariates['low_degree'][i_1] == v_1) and
                    (net.node_covariates['low_degree'][i_2] == v_2))

        net.new_edge_covariate(name).from_binary_function_ind(f_x)
    
# Initialize fitting model
fit_model = StationaryLogistic()
n_fit_model = NonstationaryLogistic()
for c in covariates:
    fit_model.beta[c] = None
    n_fit_model.beta[c] = None

# Set up recording of results from experiment
results = Results(params['sub_sizes'], params['num_reps'], 'Stationary fit')
add_network_stats(results)
def est_theta_c(c):
    return lambda d, f: f.beta[c]
for c in covariates:
    f_est = est_theta_c(c)
    results.new('%s' % c, 'm', f_est)

all_results = {}
if params['fit_stationary']:
    s_results = results.copy()
    s_results.title = 'Stationary fit'
    all_results['s'] = s_results
if params['fit_nonstationary']:
    n_results = results.copy()
    n_results.title = 'Nonstationary fit'
    all_results['n'] = n_results
if params['fit_conditional']:
    c_results = results.copy()
    c_results.title = 'Conditional fit'
    all_results['c'] = c_results
if params['fit_conditional_is']:
    i_results = results.copy()
    i_results.title = 'Conditional (importance sampled) fit'
    all_results['i'] = i_results

for sub_size in params['sub_sizes']:
    print 'subnetwork size = %d' % sub_size
    
    gen = RandomSubnetworks(net, sub_size, method = params['sampling'])
    
    for rep in range(params['num_reps']):
        subnet = gen.sample()

        if params['fit_stationary']:
            fit_model.fit_convex_opt(subnet, verbose = True)
            s_results.record(sub_size, rep, subnet, fit_model = fit_model)
            print

        if params['fit_conditional']:
            fit_model.fit_conditional(subnet, verbose = True)
            c_results.record(sub_size, rep, subnet, fit_model = fit_model)
            print

        if params['fit_conditional_is']:
            fit_model.fit_conditional(subnet, T = 50, verbose = True)
            i_results.record(sub_size, rep, subnet, fit_model = fit_model)
            print
        
        if params['fit_nonstationary']:
            subnet.offset_extremes()
            n_fit_model.fit_convex_opt(subnet, verbose = True)
            n_results.record(sub_size, rep, subnet, fit_model = n_fit_model)
            print

for model in all_results:
    results = all_results[model]
    results.plot(['%s' % c for c in covariates])
  
# Plot network statistics as well as sparsity parameter
if params['plot_network']:
    result = all_results[all_results.keys()[0]]
    result.title = None
    result.plot([('Average out-degree', {'ymin': 0, 'plot_mean': True}),
                 ('Average in-degree', {'ymin': 0, 'plot_mean': True}),
                 (['Out-degree', 'Max out-degree', 'Min out-degree'],
                  {'ymin': 0, 'plot_mean': True}),
                 (['In-degree', 'Max in-degree', 'Min in-degree'],
                  {'ymin': 0, 'plot_mean': True}),
                 ('Self-loop density', {'ymin': 0, 'plot_mean': True})])
