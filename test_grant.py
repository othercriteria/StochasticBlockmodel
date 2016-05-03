#!/usr/bin/env python

# Some tests written in support of Matt's grant-writing.
# Daniel Klein, 10/26/2012

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from datetime import date, timedelta

from Network import network_from_edges
from Models import Stationary, StationaryLogistic, NonstationaryLogistic
from Models import FixedMargins, alpha_zero

# Parameters
params = { 'file_network': 'data/cit-HepTh/cit-HepTh.txt',
           'import_limit_edges': 80,
           'file_dates': 'data/cit-HepTh/cit-HepTh-dates.txt',
           'dir_abstracts': 'data/cit-HepTh/abstracts',
           'pub_diff_classes': [90, 360] }


# Import covariate data from file
dates = {}
for line in open(params['file_dates'], 'r').readlines():
    line = line[:-2]
    if line[0] == '#': continue
    if line == '': continue
    n, d = line.split('\t')
    if len(n) == 9:
        assert(n[:2] == '11')
        n = n[2:]
    d_1, d_2, d_3 = d.split('-')
    dates[n] = date(int(d_1), int(d_2), int(d_3))

# Import network data from file
edges = []
nodes = set()
for line in open(params['file_network'], 'r').readlines():
    line = line[:-2]
    if line[0] == '#': continue
    if line == '': continue
    n_1, n_2 = line.split('\t')
    if not (n_1 in dates and n_2 in dates): continue
    if dates[n_1] < dates[n_2]: continue
    edges.append((n_1, n_2))
    nodes.add(n_1)
    nodes.add(n_2)
    if len(edges) >= params['import_limit_edges']: break
print '# Nodes: %d' % len(nodes)

# Initialize network from citation data
net = network_from_edges(edges)

# Process publication date data in covariates
cov_names = []
for l, u in zip([0] + params['pub_diff_classes'], params['pub_diff_classes']):
    cov_name = 'pub_diff_%d-%d' % (l, u)
    cov_names.append(cov_name)
    cov = net.new_edge_covariate(cov_name)
    def f_pub_date_diff_in_range(n_1, n_2):
        return timedelta(l) <= dates[n_1] - dates[n_2] < timedelta(u)
    cov.from_binary_function_name(f_pub_date_diff_in_range)

# Add publication order node covariate
pub_dates = [dates[p].toordinal() for p in net.names]
net.new_node_covariate('pub_date').from_pairs(net.names, pub_dates)

# Exclude impossible edges using infinite offset
def f_impossible_pub_order(n_1, n_2):
    if dates[n_1] < dates[n_2]:
        return -float('inf')
    else:
        return 0
net.initialize_offset().from_binary_function_name(f_impossible_pub_order)

# Display observed network
o = np.argsort(net.node_covariates['pub_date'][:])
A = net.as_dense()
def heatmap(data, cmap = 'binary'):
    plt.imshow(data[o][:,o]).set_cmap(cmap)
def residuals(data_mean, data_sd):
    r = np.abs((data_mean - A) / data_sd)
    plt.imshow(r[o][:,o], vmin = 0, vmax = 2.0).set_cmap('binary')
plt.figure()
plt.subplot(331)
plt.title('Observed')
heatmap(A)
plt.subplot(332)
plt.title('Covariates')
covariate_levels = np.zeros((net.N, net.N))
num_covariates = len(cov_names)
for i, val in enumerate(np.linspace(1.0, 0.4, num_covariates)):
    covariate_levels[net.edge_covariates[cov_names[i]].matrix() == True] = val
heatmap(covariate_levels, 'Blues')
plt.subplot(333)
plt.title('Network')
graph = nx.DiGraph()
for n1, n2 in edges:
    graph.add_edge(n1, n2)
pos = nx.graphviz_layout(graph, prog = 'neato')
nx.draw(graph, pos, node_size = 10, with_labels = False)

print 'Fitting stationary model'
s_model = StationaryLogistic()
for cov_name in cov_names:
    s_model.beta[cov_name] = None
s_model.fit(net, verbose = True)
print 'NLL: %.2f' % s_model.nll(net)
print 'kappa: %.2f' % s_model.kappa
for cov_name in cov_names:
    print '%s: %.2f' % (cov_name, s_model.beta[cov_name])
print

print 'Fitting nonstationary model'
alpha_zero(net)
ns_model = NonstationaryLogistic()
for cov_name in cov_names:
    ns_model.beta[cov_name] = None
ns_model.fit(net, verbose = True)
print 'NLL: %.2f' % ns_model.nll(net)
print 'kappa: %.2f' % ns_model.kappa
for cov_name in cov_names:
    print '%s: %.2f' % (cov_name, ns_model.beta[cov_name])
print

print 'Fitting conditional model'
c_model = FixedMargins(StationaryLogistic())
for cov_name in cov_names:
    c_model.base_model.beta[cov_name] = None
c_model.base_model.fit_conditional(net, verbose = True)
print 'NLL: %.2f' % c_model.nll(net)
for cov_name in cov_names:
    print '%s: %.2f' % (cov_name, c_model.base_model.beta[cov_name])
print

# Sample typical networks from fit models
reps = 100
s_samples = np.empty((reps, net.N, net.N))
ns_samples = np.empty((reps, net.N, net.N))
c_samples = np.empty((reps, net.N, net.N))
for rep in range(reps):
    s_samples[rep,:,:] = s_model.generate(net)
    ns_samples[rep,:,:] = ns_model.generate(net)
    c_samples[rep,:,:] = c_model.generate(net, coverage = 0.2)

# Calculate sample means and variances
s_samples_mean = np.mean(s_samples, axis = 0)
s_samples_sd = np.sqrt(np.var(s_samples, axis = 0))
ns_samples_mean = np.mean(ns_samples, axis = 0)
ns_samples_sd = np.sqrt(np.var(ns_samples, axis = 0))
c_samples_mean = np.mean(c_samples, axis = 0)
c_samples_sd = np.sqrt(np.var(c_samples, axis = 0))

# Finish plotting
plt.subplot(334)
plt.title('Stationary')
heatmap(s_samples_mean)
plt.subplot(337)
residuals(s_samples_mean, s_samples_sd)
plt.subplot(335)
plt.title('Nonstationary')
heatmap(ns_samples_mean)
plt.subplot(338)
residuals(ns_samples_mean, ns_samples_sd)
plt.subplot(336)
plt.title('Conditional')
heatmap(c_samples_mean)
plt.subplot(339)
residuals(c_samples_mean, c_samples_sd)
plt.show()
