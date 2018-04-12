#!/usr/bin/env python

# Test of "new style" network inference, minimal code to demo something...
# Daniel Klein, 5/16/2012

from Network import Network
from Models import NonstationaryLogistic, alpha_unif
from Experiment import RandomSubnetworks
from numpy.random import normal, seed

# Seed random number for reproducible results
seed(137)

# Initialize full network
N = 100
net = Network(N)
alpha_unif(net, 0.5)

# Initialize the data model; generate covariates and associated coefficients
data_model = NonstationaryLogistic()
data_model.kappa = -1.0
covariates = ['x_%d' % i for i in range(1)]
for covariate in covariates:
    data_model.beta[covariate] = normal(0, 1.0)

    x_node = normal(0, 1.0, N)
    def f_x(i_1, i_2):
        return abs(x_node[i_1] - x_node[i_2]) < 0.6
    net.new_edge_covariate(covariate).from_binary_function_ind(f_x)
net.generate(data_model)
net.offset_extremes()
net.show()
print 'True theta_0: %.2f' % data_model.beta['x_0']

# Initialize the fit model; specify which covariates it should have terms for
fit_model = NonstationaryLogistic()
for covariate in covariates:
    fit_model.beta[covariate] = None

# Set up random subnetwork generator, and run fitting experiments
gen = RandomSubnetworks(net, (40, 40))
for rep in range(5):
    subnet = gen.sample()

    fit_model.fit_brazzale(subnet, 'x_0')
    print 'Estimated theta_0: %.2f' % fit_model.beta['x_0']

    fit_model.confidence_boot(subnet, n_bootstrap = 10)
    cis = fit_model.conf['x_0']
    print 'Brazzale CI for theta_0: (%.2f, %.2f)' % cis['brazzale']
    print 'Pivotal CI for theta_0: (%.2f, %.2f)' % cis['pivotal']
