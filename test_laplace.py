#!/usr/bin/env python

# Test of "new style" network inference, playing with Laplace
# approximation confidence intervals
# Daniel Klein, 5/16/2012

import numpy as np
import matplotlib.pyplot as plt

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic, alpha_unif
from Experiment import RandomSubnetworks
from Utility import draw_confidence

# Initialize full network
N = 300
net = Network(N)
alpha_unif(net, 0.5)

# Initialize the data model; generate covariates and associated coefficients
data_model = NonstationaryLogistic()
data_model.kappa = -7.0
covariates = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5']
for covariate in covariates:
    data_model.beta[covariate] = np.random.normal(0, 1.0)

    x_node = np.random.normal(0, 1.0, N)
    def f_x(i_1, i_2):
        return abs(x_node[i_1] - x_node[i_2]) < 0.3
    net.new_edge_covariate(covariate).from_binary_function_ind(f_x)
net.generate(data_model)
print 'True beta_1: %.2f' % data_model.beta['x_1']

# Initialize the fit model; specify which covariates it should have terms for
fit_model = StationaryLogistic()
for covariate in covariates:
    fit_model.beta[covariate] = None

# Set up plotting
f = plt.figure()
ax = f.add_subplot(1, 1, 1, aspect = 'equal')
ax.set_title('95% confidence regions')
ax.set_xlim(data_model.beta['x_1'] - 2.0, data_model.beta['x_1'] + 2.0)
ax.set_ylim(data_model.beta['x_2'] - 2.0, data_model.beta['x_2'] + 2.0)
ax.set_xlabel('beta_1')
ax.set_ylabel('beta_2')

# Set up random subnetwork generator, and run fitting experiments
gen = RandomSubnetworks(net, 100)
for rep in range(10):
    subnet = gen.sample()

    fit_model.fit_logistic_l2(subnet, prior_precision = 1.0,
                              variance_covariance = True)
    print 'Estimated beta_1: %.2f' % fit_model.beta['x_1']
    
    m = np.array([fit_model.beta['x_1'], fit_model.beta['x_2']])
    vc = fit_model.variance_covariance
    S = np.array([[vc[('x_1','x_1')], vc['x_1','x_2']],
                  [vc[('x_2','x_1')], vc['x_2','x_2']]])
    draw_confidence(ax, m, S)
plt.show()
