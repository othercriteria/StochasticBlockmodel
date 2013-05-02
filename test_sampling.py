#!/usr/bin/env python

# Looking into different sampling schemes to give "sparse scaling"
# (which, paradoxically, is better for small subnetwork inference).
# Daniel Klein, 5/1/2013

import numpy as np

from Network import Network
from Experiment import RandomSubnetworks, Results, add_network_stats

# Parameters
params = { 'N': 300,
           'D': 5,
           'num_reps': 1,
           'sub_sizes': range(10, 110, 10),
           'sampling_methods': ['random_node', 'random_edge'],
           'plot_network': True }


# Set random seed for reproducible output
np.random.seed(137)

# Initialize full network
net = Network(params['N'])
blocks = params['N'] / params['D']
edges = []
for block in range(blocks):
    for i in range(params['D']):
        v_1 = 'n_%d' % (block * params['D'] + i)
        for j in range(params['D']):
            v_2 = 'n_%d' % (((block + 1) * params['D'] + j) % params['N'])
            edges.append((v_1, v_2))
net.network_from_edges(edges)

# Set up recording of results from experiment
results_by_method = { }
for method_name in params['sampling_methods']:
    results = Results(params['sub_sizes'], params['num_reps'])
    add_network_stats(results)
    results.new('# Active', 'n', lambda n: np.isfinite(n.offset.matrix()).sum())
    results_by_method[method_name] = results

for sub_size in params['sub_sizes']:
    print 'subnetwork size = %d' % sub_size

    generators = { 'random_node': RandomSubnetworks(net, sub_size),
                   'random_edge': RandomSubnetworks(net, sub_size,
                                                    method = 'edge') }
    for generator in generators:
        if not generator in params['sampling_methods']: continue
        print generator
        for rep in range(params['num_reps']):
            subnet = generators[generator].sample()

            subnet.offset_extremes()

            results_by_method[generator].record(sub_size, rep, subnet)

# Output results
print
for method_name in params['sampling_methods']:
    print method_name

    results = results_by_method[method_name]
    results.summary()
    if params['plot_network']:
        results.plot([('Average degree', {'ymin': 0, 'plot_mean': True}),
                      (['Out-degree', 'Max out-degree', 'Min out-degree'],
                       {'ymin': 0, 'plot_mean': True}),
                      (['In-degree', 'Max out-degree', 'Min in-degree'],
                       {'ymin': 0, 'plot_mean': True}),
                      ('Self-loop density', {'ymin': 0, 'plot_mean': True}),
                      ('# Active', {'ymin': 0 })])

    print

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))
