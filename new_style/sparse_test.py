#!/usr/bin/env python

# Test of "new style" network inference on sparse data
# Daniel Klein, 5/22/2012

import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from Network import Network
from Models import Stationary, StationaryLogistic, RaschLogistic, alpha_zero

# Parameters
params = { 'file_network': 'cit-HepTh/cit-HepTh.txt',
           'import_limit_edges': 250,
           'file_dates': 'cit-HepTh/cit-HepTh-dates.txt',
           'pub_diff_cutoff': 100 }


# Set random seed for reproducible output
np.random.seed(137)

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))


# Import network data from file
edges = []
for line in open(params['file_network'], 'r').readlines():
    line = line[:-2]
    if line[0] == '#': continue
    if line == '': continue
    n_1, n_2 = line.split('\t')
    edges.append((n_1, n_2))
    if len(edges) >= params['import_limit_edges']: break

# Import covariate data from file
dates = {}
for line in open(params['file_dates'], 'r').readlines():
    line = line[:-2]
    if line[0] == '#': continue
    if line == '': continue
    n, d = line.split('\t')
    if n[:2] == '11':
        n = n[2:]
    d_1, d_2, d_3 = d.split('-')
    dates[n] = date(int(d_1), int(d_2), int(d_3))
 
# Initialize full network
net = Network()
net.network_from_edges(edges)

# Convert covariate data to covariate
cov = net.new_edge_covariate('short_pub_diff')
def f_similar_pub_date(n_1, n_2):
    if (not n_1 in dates) or (not n_2 in dates):
        return False
    return abs(dates[n_1] - dates[n_2]) < timedelta(params['pub_diff_cutoff'])
cov.from_binary_function_name(f_similar_pub_date)

# Fit model
for model_name, model in [('Stationary', Stationary()),
                          ('Stationary Logistic', StationaryLogistic()),
                          ('Rasch Logistic', RaschLogistic())]:
    print model_name
    if not model_name == 'Stationary':
        model.beta['short_pub_diff'] = None
    model.fit_logistic(net)
    if not model_name == 'Stationary':
        print 'short_pub_diff: %.2f' % model.beta['short_pub_diff']
    if model_name == 'Rasch Logistic':
        alpha_out_logistic = net.node_covariates['alpha_out'][:]
        alpha_in_logistic = net.node_covariates['alpha_in'][:]
    print 'kappa: %.2f' % model.kappa
    print '\n'
    model.fit_convex_opt(net)
    if model_name == 'Rasch Logistic':
        alpha_out_convex_opt = net.node_covariates['alpha_out'][:]
        alpha_in_convex_opt = net.node_covariates['alpha_in'][:]
    if not model_name == 'Stationary':
        print 'short_pub_diff: %.2f' % model.beta['short_pub_diff']
    print 'kappa: %.2f' % model.kappa
    print '\n\n'

plt.figure()
plt.subplot('211')
plt.plot(alpha_out_logistic, alpha_out_convex_opt, '.')
plt.subplot('212')
plt.plot(alpha_in_logistic, alpha_in_convex_opt, '.')
plt.show()
