#!/usr/bin/env python

# Test of "new style" network inference on sparse data
# Daniel Klein, 5/22/2012

from datetime import date, timedelta

from Network import Network
from Models import Stationary, StationaryLogistic, NonstationaryLogistic
from Web import dump_to_json

# Parameters
params = { 'file_network': 'data/cit-HepTh/cit-HepTh.txt',
           'import_limit_edges': 300,
           'file_dates': 'data/cit-HepTh/cit-HepTh-dates.txt',
           'pub_diff_classes': [30, 60, 90, 180, 360, 720],
           'offset_extremes': True,
           'plot': False }


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
for line in open(params['file_network'], 'r').readlines():
    line = line[:-2]
    if line[0] == '#': continue
    if line == '': continue
    n_1, n_2 = line.split('\t')
    if not (n_1 in dates and n_2 in dates): continue
    if dates[n_1] < dates[n_2]: continue
    edges.append((n_1, n_2))
    if len(edges) >= params['import_limit_edges']: break

# Plot the raw data
if params['plot']:
    import matplotlib.pyplot as plt
    plt.figure()
    x, y = [], []
    for n_1, n_2 in edges:
        x.append(dates[n_1].toordinal())
        y.append(dates[n_2].toordinal())
    plt.plot(x, y, '.')
    plt.show()

# Initialize network from citation data
net = Network()
net.network_from_edges(edges)
if params['plot']: net.show_degree_histograms()

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
if params['plot']:
    net.node_covariates['pub_date'].show_histogram()
    net.show_heatmap('pub_date')

# Exclude impossible edges using infinite offset
def f_impossible_pub_order(n_1, n_2):
    if dates[n_1] < dates[n_2]:
        return -float('inf')
    else:
        return 0
net.initialize_offset().from_binary_function_name(f_impossible_pub_order)

# Fit model
def fit_and_summarize(name, fit_model, use_covs):
    print name
    if use_covs:
        for cov_name in cov_names:
            fit_model.beta[cov_name] = None
    fit_model.fit_convex_opt(net, verbose = True)
    print 'NLL: %.2f' % fit_model.nll(net)
    print 'kappa: %.2f' % fit_model.kappa
    if use_covs:
        for cov_name in cov_names:
            print '%s: %.2f' % (cov_name, fit_model.beta[cov_name])
    print '\n'
fit_and_summarize('Stationary', Stationary(), False)
fit_and_summarize('Stationary', StationaryLogistic(), True)
if params['offset_extremes']:
    print 'Detecting subnetworks associated with infinite parameter estimates.\n'
    net.offset_extremes()
    if params['plot']: net.show_offset('pub_date')
fit_and_summarize('Nonstationary', NonstationaryLogistic(), False)
fit_and_summarize('Nonstationary', NonstationaryLogistic(), True)

# Redisplay heatmap, ordered by estimated alphas from last fit, i.e.,
# NonstationaryLogistic with publication date difference covariates
if params['plot']:
    net.show_heatmap('alpha_out')
    net.show_heatmap('alpha_in')

outfile = open('scratch.json', 'w')
outfile.write(dump_to_json(net))
outfile.close()
