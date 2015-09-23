#!/usr/bin/env python

# Test of "new style" network inference, finally with blockmodel
# Daniel Klein, 7/24/2013

import numpy as np

from Network import network_from_file_gml
from Models import StationaryLogistic, NonstationaryLogistic, Blockmodel
from Experiment import RandomSubnetworks, Results, add_array_stats
from Experiment import minimum_disagreement

# Parameters
params = { 'fit_nonstationary': True,
           'fit_conditional': True,
           'fit_conditional_is': False,
           'blockmodel_fit_method': 'sem',
           'fit_K': 2,
           'num_reps': 2,
           'sub_sizes': np.arange(20, 86, 5, dtype=np.int),
           'sampling': 'link',
           'initialize_true_z': False,
           'cycles': 20,
           'sweeps': 10,
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
net = network_from_file_gml('data/polblogs/polblogs.gml', ['value'])
net.new_node_covariate_int('truth')[:] = net.node_covariates['value'][:]

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
s_results = Results(params['sub_sizes'], params['sub_sizes'],
                    params['num_reps'], 'Stationary fit')
add_array_stats(s_results)
def class_mismatch(n):
    truth = n.node_covariates['truth'][:]
    estimated = n.node_covariates['z'][:]
    return minimum_disagreement(truth, estimated, normalized = False)
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

def initialize(s, f, offset_extremes):
    if params['initialize_true_z']:
        s.node_covariates['z'][:] = s.node_covariates['value'][:]
    else:
        s.node_covariates['z'][:] = np.random.randint(0, params['fit_K'], s.N)
        
    if offset_extremes:
        s.offset_extremes()
    else:
        s.initialize_offset()
        
    for i in range(subnet.N):
        subnet.offset[i,i] = -np.inf
        
for sub_size in params['sub_sizes']:
    size = (sub_size, sub_size)
    print 'subnetwork size = %s' % str(size)
    
    gen = RandomSubnetworks(net, size, method = params['sampling'])
    for rep in range(params['num_reps']):
        subnet = gen.sample(as_network = True)
        
        initialize(subnet, fit_model, offset_extremes = False)
        fit_base_model.fit = fit_base_model.fit_convex_opt
        fit_model.ignore_inner_offset = False
        fit_model.fit(subnet, params['cycles'], params['sweeps'])
        s_results.record(size, rep, subnet, fit_model = fit_model)
        print 'S: ', fit_model.Theta
        print

        if params['fit_conditional']:
            initialize(subnet, fit_model, offset_extremes = False)
            fit_base_model.fit = fit_base_model.fit_conditional
            fit_model.ignore_inner_offset = True
            fit_model.fit(subnet, params['cycles'], params['sweeps'])
            c_results.record(size, rep, subnet, fit_model = fit_model)
            print 'C: ', fit_model.Theta
            print

        if params['fit_conditional_is']:
            initialize(subnet, fit_model, offset_extremes = False)
            fit_base_model.fit = fit_base_model.fit_conditional
            fit_model.ignore_inner_offset = True
            fit_model.fit(subnet, params['cycles'], params['sweeps'], T = 10)
            i_results.record(size, rep, subnet, fit_model = fit_model)
            print 'I: ', fit_model.Theta
            print
        
        if params['fit_nonstationary']:
            initialize(subnet, n_fit_model, offset_extremes = True)
            fit_model.ignore_inner_offset = False
            n_fit_model.fit(subnet, params['cycles'], params['sweeps'])
            n_results.record(size, rep, subnet, fit_model = n_fit_model)
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
    s_results.title = 'Network statistics'
    s_results.plot([('Density', {'ymin': 0, 'plot_mean': True}),
                    (['Out-degree', 'Max row-sum', 'Min row-sum'],
                     {'ymin': 0, 'plot_mean': True}),
                    (['In-degree', 'Max col-sum', 'Min col-sum'],
                     {'ymin': 0, 'plot_mean': True})])
