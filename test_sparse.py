#!/usr/bin/env python

# Test of "new style" network inference on sparse data
# Daniel Klein, 5/22/2012

from datetime import date, timedelta

from Network import Network
from Models import Stationary, StationaryLogistic, NonstationaryLogistic

# Parameters
params = { 'file_network': 'data/cit-HepTh/cit-HepTh.txt',
           'import_limit_edges': 500,
           'file_dates': 'data/cit-HepTh/cit-HepTh-dates.txt',
           'pub_diff_cutoff': 100 }


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

# Import network data from file
edges = []
for line in open(params['file_network'], 'r').readlines():
    line = line[:-2]
    if line[0] == '#': continue
    if line == '': continue
    n_1, n_2 = line.split('\t')
    if not (n_1 in dates and n_2 in dates): continue
    edges.append((n_1, n_2))
    if len(edges) >= params['import_limit_edges']: break

# Initialize full network
net = Network()
net.network_from_edges(edges)
net.show_degree_histograms()

# Convert covariate data to covariate
cov = net.new_edge_covariate('short_pub_diff')
def f_similar_pub_date(n_1, n_2):
    return abs(dates[n_1] - dates[n_2]) < timedelta(params['pub_diff_cutoff'])
cov.from_binary_function_name(f_similar_pub_date)

# Add publication order node covariate
pub_dates = [dates[p].toordinal() for p in net.names]
net.new_node_covariate('pub_dates').from_pairs(net.names, pub_dates)
net.show_heatmap('pub_dates')

# Exclude impossible edges using infinite offset
def f_impossible_pub_order(n_1, n_2):
    if dates[n_1] < dates[n_2]:
        return -float('inf')
    else:
        return 0
net.initialize_offset().from_binary_function_name(f_impossible_pub_order)

# Fit model
for model_name, model in [('Stationary', Stationary()),
                          ('Stationary Logistic', StationaryLogistic()),
                          ('Nonstationary Logistic', NonstationaryLogistic())]:
    print model_name
    if not model_name == 'Stationary':
        model.beta['short_pub_diff'] = None
    model.fit_convex_opt(net)
    if not model_name == 'Stationary':
        print 'short_pub_diff: %.2f' % model.beta['short_pub_diff']
    print 'kappa: %.2f' % model.kappa
    print '\n'

# Redisplay heatmap, ordered by estimated alphas from NonstationaryLogistic fit
net.show_heatmap('alpha_out')
net.show_heatmap('alpha_in')
