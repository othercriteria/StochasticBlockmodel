#!/usr/bin/env python

# Check SEM's ability to stay in the neighborhood of the (label) truth
# when initialized at the (label) truth.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic, Blockmodel
from Models import alpha_zero, alpha_norm
from Experiment import minimum_disagreement

# Parameters
N = 30
alpha_sd = 1.0
from_truth = True
steps = 100

# Set random seed for reproducible outputs
np.random.seed(137)

net = Network(N)

net.new_node_covariate('value').from_pairs(net.names, [0]*(N/2) + [1]*(N/2))
for v_1, v_2, name in [(0, 0, 'll'),
                       (1, 1, 'rr'),
                       (0, 1, 'lr')]:
    def f_x(i_1, i_2):
        return ((net.node_covariates['value'][i_1] == v_1) and
                (net.node_covariates['value'][i_2] == v_2))

    net.new_edge_covariate(name).from_binary_function_ind(f_x)

def f_x(i_1, i_2):
    return np.random.uniform(-np.sqrt(3), np.sqrt(3))
net.new_edge_covariate('x').from_binary_function_ind(f_x)
        
data_model = NonstationaryLogistic()
data_model.beta['x'] = 3.0
for name, theta in [('ll', 4.0),
                    ('rr', 3.0),
                    ('lr', -2.0)]:
    data_model.beta[name] = theta
alpha_norm(net, alpha_sd)
data_model.match_kappa(net, ('row_sum', 2))
net.generate(data_model)
net.show_heatmap()
net.offset_extremes()

fit_base_model = NonstationaryLogistic()
fit_base_model.beta['x'] = None
fit_model = Blockmodel(fit_base_model, 2)
#fit_model.base_model.fit = fit_model.base_model.fit_conditional

# Initialize block assignments
net.new_node_covariate_int('z')
if from_truth:
    net.node_covariates['z'][:] = net.node_covariates['value'][:]
else:
    net.node_covariates['z'][:] = np.random.random(N) < 0.5

# Calculate NLL at truth
fit_model.fit_sem(net, cycles = 1, sweeps = 0,
                  use_best = False, store_all = True)
baseline_nll = fit_model.sem_trace[0][0]

nll_trace = []
z_trace = np.empty((steps,N))
disagreement_trace = []
theta_trace = []

for step in range(steps):
    print step
    fit_model.fit_sem(net, 1, 2, store_all = True)
    #fit_model.fit_kl(net, 1)
    nll_trace.append(fit_model.nll(net))
    z_trace[step,:] = net.node_covariates['z'][:]
    disagreement = minimum_disagreement(net.node_covariates['value'][:],
                                        net.node_covariates['z'][:])
    disagreement_trace.append(disagreement)
    theta_trace.append(fit_model.base_model.beta['x'])

# Eliminate symmetry of 'z'
for step in range(steps):
    if np.mean(z_trace[step,:]) < 0.5:
        z_trace[step,:] = 1 - z_trace[step,:]
z_trace += np.random.normal(0, 0.01, (steps, N))
                    
nll_trace = np.array(nll_trace)
nll_trace -= baseline_nll
disagreement_trace = np.array(disagreement_trace)

plt.figure()
plt.plot(np.arange(steps), theta_trace)
plt.xlabel('step')
plt.ylabel('theta')

plt.figure()
plt.plot(np.arange(steps), nll_trace)
plt.xlabel('step')
plt.ylabel('NLL')

plt.figure()
plt.plot(np.arange(steps), disagreement_trace)
plt.xlabel('step')
plt.ylabel('normalized disagreement')

plt.figure()
nll_trimmed = nll_trace[nll_trace <= np.percentile(nll_trace, 90)]
plt.hist(nll_trimmed, bins = 50)
plt.xlabel('NLL')
plt.title('Trimmed histogram of NLL')

try:    
    pca = PCA(z_trace)

    plt.figure()
    plt.plot(np.arange(steps), pca.Y[:,0])
    plt.xlabel('step')
    plt.ylabel('z (PC1)')

    plt.figure()
    plt.subplot(211)
    plt.plot(pca.Y[:,0], nll_trace, '.')
    plt.xlabel('z (PC1)')
    plt.ylabel('NLL')
    plt.subplot(212)
    plt.plot(pca.Y[:,1], nll_trace, '.')
    plt.xlabel('z (PC2)')
    plt.ylabel('NLL')

    plt.figure()
    plt.subplot(211)
    plt.plot(pca.Y[:,0], disagreement_trace, '.')
    plt.xlabel('z (PC1)')
    plt.ylabel('normalized disagreement')
    plt.subplot(212)
    plt.plot(pca.Y[:,1], disagreement_trace, '.')
    plt.xlabel('z (PC2)')
    plt.ylabel('normalized_disagreement')
    
    plt.figure()
    plt.plot(pca.Y[:,0], pca.Y[:,1])
    plt.xlabel('z (PC1)')
    plt.ylabel('z (PC2)')
except:
    print 'PCA failed; maybe no variation in z or steps < N?'

plt.show()    
