#!/usr/bin/env python

# Test of "new style" network inference on sparse data
# Daniel Klein, 5/22/2012

from datetime import date, timedelta

from Network import Network
from Models import Stationary, StationaryLogistic, NonstationaryLogistic

# Parameters
params = { 'file_network': 'data/cit-HepTh/cit-HepTh.txt',
           'import_limit_edges': 1200,
           'file_dates': 'data/cit-HepTh/cit-HepTh-dates.txt',
           'pub_diff_classes': [30, 60, 90, 180, 360, 720] }


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
    if dates[n_1] < dates[n_2]: continue
    edges.append((n_1, n_2))
    if len(edges) >= params['import_limit_edges']: break

# Initialize full network
net = Network()
net.network_from_edges(edges)
net.show_degree_histograms()

# Convert covariate data to covariates
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
net.new_node_covariate('pub_dates').from_pairs(net.names, pub_dates)
net.node_covariates['pub_dates'].show_histogram()
net.show_heatmap('pub_dates')

# Exclude impossible edges using infinite offset
def f_impossible_pub_order(n_1, n_2):
    if dates[n_1] < dates[n_2]:
        return -float('inf')
    else:
        return 0
net.initialize_offset().from_binary_function_name(f_impossible_pub_order)

# Fit model
for name, model, use_covs in [('Stationary', Stationary(), False),
                              ('Stationary', StationaryLogistic(), True),
                              ('Nonstationary', NonstationaryLogistic(), False),
                              ('Nonstationary', NonstationaryLogistic(), True)]:
    print name
    if use_covs:
        for cov_name in cov_names:
            model.beta[cov_name] = None
    model.fit_convex_opt(net)
    print 'NLL: %.2f' % model.nll(net)
    print 'kappa: %.2f' % model.kappa
    if use_covs:
        for cov_name in cov_names:
            print '%s: %.2f' % (cov_name, model.beta[cov_name])
    print '\n'

# Redisplay heatmap, ordered by estimated alphas from last fit, i.e.,
# NonstationaryLogistic with publication date difference covariates
net.show_heatmap('alpha_out')
net.show_heatmap('alpha_in')
