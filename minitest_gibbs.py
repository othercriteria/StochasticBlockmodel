#!/usr/bin/env python

# Check SEM's ability to stay in the neighborhood of the (label) truth
# when initialized at the (label) truth, in the absence of degree
# heterogeneity...

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic, Blockmodel
from Models import alpha_zero, alpha_norm
from Experiment import minimum_disagreement

N = 100
reps = 200

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
alpha_norm(net, 1.0)
data_model.match_kappa(net, ('degree', 2))
net.generate(data_model)
net.show_heatmap()
net.offset_extremes()


fit_base_model = StationaryLogistic()
fit_base_model.beta['x'] = None
fit_model = Blockmodel(fit_base_model, 2)
#fit_model.base_model.fit = fit_model.base_model.fit_conditional

# Initialize from truth...
net.new_node_covariate_int('z')
net.node_covariates['z'][:] = net.node_covariates['value'][:]

# Calculate NLL at truth
fit_model.fit_sem(net, 1, 0, use_best = False)
baseline_nll = fit_model.sem_trace[0][0]

nll_trace = []
z_trace = np.empty((reps,N))
disagreement_trace = []

for rep in range(reps):
    fit_model.fit_sem(net, 1, 5)
    # fit_model.fit_kl(net, 1)
    nll_trace.append(fit_model.nll(net))
    z_trace[rep,:] = net.node_covariates['z'][:]
    disagreement = minimum_disagreement(net.node_covariates['value'][:],
                                        net.node_covariates['z'][:])
    disagreement_trace.append(disagreement)

# Eliminate symmetry of 'z'
for rep in range(reps):
    if np.mean(z_trace[rep,0]) < 0.5:
        z_trace[rep,:] = 1 - z_trace[rep,:]
z_trace += np.random.normal(0, 0.01, (reps, N))
                    
nll_trace = np.array(nll_trace)
nll_trace -= baseline_nll
disagreement_trace = np.array(disagreement_trace)
    
plt.figure()
plt.plot(np.arange(reps), nll_trace)
plt.xlabel('step')
plt.ylabel('NLL')

plt.figure()
plt.plot(np.arange(reps), disagreement_trace)
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
    plt.plot(np.arange(reps), pca.Y[:,0])
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
    print 'PCA failed; maybe no variation in z?'

plt.show()    
