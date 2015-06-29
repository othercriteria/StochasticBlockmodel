#!/usr/bin/env python

# Test of "new style" network inference, finally with blockmodel
# Daniel Klein, 7/24/2013

import numpy as np

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic, Blockmodel
from Experiment import RandomSubnetworks, Results, add_network_stats
from Experiment import minimum_disagreement

# Parameters
params = { 'fit_nonstationary': True,
           'fit_conditional': False,
           'fit_conditional_is': False,
           'blockmodel_fit_method': 'sem',
           'fit_K': 2,
           'num_reps': 1,
           'sub_sizes': [1490], # range(5, 31, 5),
           'sampling': 'node',
           'initialize_true_z': True,
           'cycles': 1,
           'sweeps': 0,
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

# Initialize fitting model
fit_base_model = StationaryLogistic()
fit_model = Blockmodel(fit_base_model, params['fit_K'])
if params['fit_nonstationary']:
    n_fit_base_model = NonstationaryLogistic()
    n_fit_model = Blockmodel(n_fit_base_model, params['fit_K'])
net.new_node_covariate_int('z')
if params['blockmodel_fit_method'] == 'kl':
    fit_model.fit = fit_model.fit_kl
    n_fit_model.fit = n_fit_model.fit_kl

# Set up recording of results from experiment
s_results = Results(params['sub_sizes'], params['num_reps'], 'Stationary fit')
add_network_stats(s_results)
def class_mismatch(n):
    truth = n.node_covariates['value'][:]
    estimated = n.node_covariates['z'][:]
    return minimum_disagreement(truth, estimated)
s_results.new('Class mismatch', 'n', class_mismatch)

all_results = { 's': s_results }
if params['fit_nonstationary']:
    n_results = s_results.copy()
    n_results.title = 'Nonstationary fit'
    all_results['n'] = n_results
if params['fit_conditional']:
    c_results = s_results.copy()
    c_results.title = 'Conditional fit'
    all_results['c'] = c_results
if params['fit_conditional_is']:
    i_results = s_results.copy()
    i_results.title = 'Conditional (importance sampled) fit'
    all_results['i'] = i_results

def initialize(s, f):
    if params['initialize_true_z']:
        s.node_covariates['z'][:] = s.node_covariates['value'][:]
    else:
        s.node_covariates['z'][:] = np.random.randint(0, params['fit_K'], s.N)
        
for sub_size in params['sub_sizes']:
    print 'subnetwork size = %d' % sub_size
    
    gen = RandomSubnetworks(net, sub_size, method = params['sampling'])
    for rep in range(params['num_reps']):
        subnet = gen.sample()
        
        initialize(subnet, fit_model)
        fit_base_model.fit = fit_base_model.fit_convex_opt
        fit_model.fit(subnet, params['cycles'], params['sweeps'])
        s_results.record(sub_size, rep, subnet, fit_model = fit_model)
        print 'S: ', fit_model.Theta
        print

        if params['fit_conditional']:
            initialize(subnet, fit_model)
            fit_base_model.fit = fit_base_model.fit_conditional
            fit_model.fit(subnet, params['cycles'], params['sweeps'])
            c_results.record(sub_size, rep, subnet, fit_model = fit_model)
            print

        if params['fit_conditional_is']:
            initialize(subnet, fit_model)
            fit_base_model.fit = fit_base_model.fit_conditional
            fit_model.fit(subnet, params['cycles'], params['sweeps'], T = 10)
            i_results.record(sub_size, rep, subnet, fit_model = fit_model)
            print
        
        if params['fit_nonstationary']:
            subnet.offset_extremes()
            initialize(subnet, n_fit_model)
            n_fit_model.fit(subnet, params['cycles'], params['sweeps'])
            n_results.record(sub_size, rep, subnet, fit_model = n_fit_model)
            print 'NS: ', n_fit_model.Theta
            print

# Plot inferred class disagreement and report results
for model in all_results:
    results = all_results[model]
    print results.title
    results.summary()
    print
    results.plot([('Class mismatch', {'ymin': 0.0, 'ymax': params['fit_K']})])
  
# Plot network statistics as well as sparsity parameter
if params['plot_network']:
    s_results.title = None
    s_results.plot([('Average out-degree', {'ymin': 0, 'plot_mean': True}),
                    ('Average in-degree', {'ymin': 0, 'plot_mean': True}),
                    (['Out-degree', 'Max out-degree', 'Min out-degree'],
                     {'ymin': 0, 'plot_mean': True}),
                    (['In-degree', 'Max in-degree', 'Min in-degree'],
                     {'ymin': 0, 'plot_mean': True}),
                    ('Self-loop density', {'ymin': 0, 'plot_mean': True})])
