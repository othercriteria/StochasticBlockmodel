#!/usr/bin/env python

# Check if Gibbs sampling for fixed margin generation is actually
# working and, if so, how quickly...
# Daniel Klein, 2014-03-05

import numpy as np
import matplotlib.pyplot as plt

from Network import Network
from Models import NonstationaryLogistic
from Models import FixedMargins
from Models import alpha_zero

# Parameters
params = { 'N': 100,
           'B': 1,
           'theta_sd': 1.0,
           'theta_fixed': { 'x_0': 2.0, 'x_1': -1.0 },
           'cov_structure': 'multimodal_4_cycles',
           'cov_mult': 2.0,
           'num_reps': 10,
           'coverage_increments': [0.01]*10 + [0.1]*10 + [0.2]*10 + [0.5]*10,
           'arb_init': False }


# Set random seed for reproducible output
np.random.seed(137)

# Initialize full network
net = Network(params['N'])
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

    if params['cov_structure'] == 'none':
        def f_x(i_1, i_2):
            return np.random.uniform(-np.sqrt(3), np.sqrt(3))
    elif params['cov_structure'] == 'unimodal':
        def f_x(i_1, i_2):
            if i_1 == i_2:
                return np.sqrt(3)
            return 0
    elif params['cov_structure'] == 'multimodal':
        def f_x(i_1, i_2):
            if i_1 == i_2:
                return np.sqrt(3)
            if (i_2 - i_1) % params['N'] == 2:
                return params['cov_mult'] * np.sqrt(3)
            return 0
    elif params['cov_structure'] == 'multimodal_4_cycles':
        N = params['N']
        m = N / 4
        def f_x(i_1, i_2):
            if (i_1 - i_2 + 1) % N == 0:
                return np.sqrt(3)

            if (i_1 + 1) % m == 0:
                if (i_1 - i_2 + 1 - 2 * m) % N == 0:
                    return params['cov_mult'] * np.sqrt(3)
            else:
                if (i_1 - i_2 + 1 - m) % N == 0:
                    return params['cov_mult'] * np.sqrt(3)

            return 0
    else:
        print 'Unrecognized covariate structure.'
        import sys; sys.exit()
        
    net.new_edge_covariate(name).from_binary_function_ind(f_x)

# Specify data model as generation permuation networks
net.new_node_covariate_int('r')[:] = 1
net.new_node_covariate_int('c')[:] = 1
data_model = FixedMargins(data_model, 'r', 'c')

coverage_levels = np.append(0.0, np.cumsum(params['coverage_increments']))
traces = { 'wall_time': [],
           'nll': [] }

for rep in range(params['num_reps']):
    net.generate(data_model, arbitrary_init = params['arb_init'])

    wall_time_trace = [net.gen_info['wall_time']]
    nll_trace = [data_model.nll(net)]

    for coverage_inc in params['coverage_increments']:
        data_model.gibbs_improve_perm(net, net.adjacency_matrix(), coverage_inc)

        wall_time_trace.append(net.gen_info['wall_time'])
        nll_trace.append(data_model.nll(net))

    traces['wall_time'].append(wall_time_trace)
    traces['nll'].append(nll_trace)

plt.figure()
plt.title('Computation time')
plt.xlabel('Coverage level')
plt.ylabel('Wall time (msec)')
for rep in range(params['num_reps']):
    plt.plot(coverage_levels, traces['wall_time'][rep], '-')

plt.figure()
plt.title('Generated network quality')
plt.xlabel('Coverage level')
plt.ylabel('NLL')
for rep in range(params['num_reps']):
    plt.plot(coverage_levels, traces['nll'][rep], '-')

plt.figure()
plt.title('Time to quality')
plt.xlabel('Wall time (msec)')
plt.ylabel('NLL')
for rep in range(params['num_reps']):
    plt.plot(traces['wall_time'][rep], traces['nll'][rep], '-')
    
# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, repr(params[field]))

plt.show()    
