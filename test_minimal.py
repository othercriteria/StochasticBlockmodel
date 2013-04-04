#!/usr/bin/env python

# Test of "new style" network inference, minimal code to demo something...
# Daniel Klein, 5/16/2012

from Network import Network
from Models import NonstationaryLogistic, alpha_unif
from Experiment import RandomSubnetworks
from numpy.random import normal

# Initialize full network
N = 300
net = Network(N)
alpha_unif(net, 0.5)

# Initialize the data model; generate covariates and associated coefficients
data_model = NonstationaryLogistic()
data_model.kappa = -7.0
covariates = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5']
for covariate in covariates:
    data_model.beta[covariate] = normal(0, 1.0)

    x_node = normal(0, 1.0, N)
    def f_x(i_1, i_2):
        return abs(x_node[i_1] - x_node[i_2]) < 0.3
    net.new_edge_covariate(covariate).from_binary_function_ind(f_x)
net.generate(data_model)
net.show()
print 'True beta_1: %.2f' % data_model.beta['x_1']

# Initialize the fit model; specify which covariates it should have terms for
fit_model = NonstationaryLogistic()
for covariate in covariates:
    fit_model.beta[covariate] = None

# Set up random subnetwork generator, and run fitting experiments
gen = RandomSubnetworks(net, 200)
for rep in range(5):
    subnet = gen.sample()

    fit_model.fit(subnet)
    print 'Estimated beta_1: %.2f' % fit_model.beta['x_1']
