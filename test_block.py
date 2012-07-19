#!/usr/bin/env python

# Test of "new style" network inference, finally with blockmodel
# Daniel Klein, 7/8/2012

import numpy as np

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic, Blockmodel
from Models import alpha_zero, alpha_norm, alpha_unif, alpha_gamma
from Experiment import RandomSubnetworks, Results, add_network_stats
from Experiment import minimum_disagreement
from Utility import logit

# Parameters
params = { 'N': 200,
           'K': 3,
           'class_conc': 10.0,
           'Theta_mean': 0.0,
           'Theta_sd': 3.0,
           'B': 5,
           'beta_sd': 1.0,
           'x_diff_cutoff': 0.3,
           'alpha_unif': 0.7,
           'alpha_norm_sd': 0.0,
           'alpha_gamma_sd': 0.0,
           'kappa_target': ('density', 0.1),
           'fit_nonstationary': False,
           'fit_blockmodel_K': 3,
           'num_reps': 6,
           'sub_sizes': range(10, 110, 10),
           'plot_mse': True,
           'plot_network': True }


# Set random seed for reproducible output
np.random.seed(136)

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))

# Initialize full network
net = Network(params['N'])

# Generate node-level propensities to extend and receive edges
if params['alpha_norm_sd'] > 0.0:
    alpha_norm(net, params['alpha_norm_sd'])
elif params['alpha_unif'] > 0.0:
    alpha_unif(net, params['alpha_unif'])
elif params['alpha_gamma_sd'] > 0.0:
    # Choosing location somewhat arbitrarily to give unit skewness
    alpha_gamma(net, 4.0, params['alpha_gamma_sd'])
else:
    alpha_zero(net)

# Generate covariates and associated coefficients
data_base_model = NonstationaryLogistic()
covariates = []
for b in range(params['B']):
    name = 'x_%d' % b
    covariates.append(name)

    data_base_model.beta[name] = np.random.normal(0, params['beta_sd'])

    x_node = np.random.normal(0, 1, params['N'])
    def f_x(i_1, i_2):
        return abs(x_node[i_1] - x_node[i_2]) < params['x_diff_cutoff']
    net.new_edge_covariate(name).from_binary_function_ind(f_x)

# Initialize data (block)model from base model
class_probs = np.random.dirichlet(np.repeat(params['class_conc'], params['K']))
z = np.where(np.random.multinomial(1, class_probs, params['N']) == 1)[1]
net.new_node_covariate_int('z_true')[:] = z
data_model = Blockmodel(data_base_model, params['K'], 'z_true')
Theta = np.random.normal(params['Theta_mean'], params['Theta_sd'],
                         (params['K'],params['K']))
Theta -= np.mean(Theta)
data_model.Theta = Theta

net.generate(data_model)
net.show_heatmap('z_true')

# Initialize fitting model
fit_base_model = StationaryLogistic()
for c in covariates:
    fit_base_model.beta[c] = None
fit_model = Blockmodel(fit_base_model, params['fit_blockmodel_K'])
if params['fit_nonstationary']:
    ns_fit_base_model = NonstationaryLogistic()
    for c in covariates:
        ns_fit_base_model.beta[c] = None
    ns_fit_model = Blockmodel(ns_fit_base_model, params['fit_blockmodel_K'])
net.new_node_covariate_int('z')

# Set up recording of results from experiment
s_results = Results(params['sub_sizes'], params['num_reps'], 'Stationary fit')
add_network_stats(s_results)
s_results.new('Subnetwork kappa', 'm', lambda d, f: d.base_model.kappa)
def f_c(c):
    return ((lambda d, f: d.base_model.beta[c]),
            (lambda d, f: f.base_model.beta[c]))
for c in covariates:
    # Need to do this hackily to avoid for-loop/lambda-binding weirdness.
    f_true, f_estimated = f_c(c)
    s_results.new('True beta_{%s}' % c, 'm', f_true)
    s_results.new('Estimated beta_{%s}' % c, 'm', f_estimated)
s_results.new('Class mismatch', 'n',
              lambda n: minimum_disagreement(n.node_covariates['z_true'][:], \
                                             n.node_covariates['z'][:]))
s_results.new('MSE(P_{ij})', 'nm',
              lambda n, d, f: np.mean((d.edge_probabilities(n) - \
                                       f.edge_probabilities(n))**2))
s_results.new('MSE(logit_P_{ij})', 'nm',
              lambda n, d, f: np.mean((logit(d.edge_probabilities(n)) - \
                                       logit(f.edge_probabilities(n)))**2))
all_results = [s_results]
if params['fit_nonstationary']:
    ns_results = s_results.copy()
    ns_results.title = 'Nonstationary fit'
    all_results.append(ns_results)

for sub_size in params['sub_sizes']:
    print 'subnetwork size = %d' % sub_size
    
    gen = RandomSubnetworks(net, sub_size)
    for rep in range(params['num_reps']):
        subnet = gen.sample()
        data_model.match_kappa(subnet, params['kappa_target'])
        subnet.generate(data_model)

        fit_model.Theta[:,:] = 0.0
        subnet.node_covariates['z'][:] = 0
        fit_model.fit_sem(subnet, cycles = 20, sweeps = 2)
        s_results.record(sub_size, rep, subnet, data_model, fit_model)

        if params['fit_nonstationary']:
            ns_fit_model.Theta[:,:] = 0.0
            subnet.node_covariates['z'][:] = 0
            ns_fit_model.fit_sem(subnet, cycles = 10, sweeps = 2)
            ns_results.record(sub_size, rep, subnet, data_model, ns_fit_model)

# Compute beta MSEs
covariate_mses = []
for c in covariates:
    name = 'MSE(beta_{%s})' % c
    covariate_mses.append(name)
    for results in all_results:
        results.estimate_mse(name,
                             'True beta_{%s}' % c, 'Estimated beta_{%s}' % c)

# Plot inference performace, in terms of MSE(beta), MSE(P_ij), and
# inferred class disagreement; also plot kappas chosen for data models
if params['plot_mse']:
    for results in all_results:
        results.plot([(['MSE(beta_i)'] + covariate_mses,
                       {'ymin': 0, 'ymax': 3.0, 'plot_mean': True}),
                      ('MSE(P_{ij})', {'ymin': 0, 'ymax': 1}),
                      ('MSE(logit_P_{ij})', {'ymin': 0, 'ymax': 5}),
                      ('Class mismatch', {'ymin': 0, 'ymax': 1}),
                      'Subnetwork kappa'])
  
# Plot network statistics as well as sparsity parameter
if params['plot_network']:
    s_results.title = None
    s_results.plot([('Average degree', {'ymin': 0, 'plot_mean': True}),
                    (['Out-degree', 'Max out-degree', 'Min out-degree'],
                     {'ymin': 0, 'plot_mean': True}),
                    (['In-degree', 'Max out-degree', 'Min in-degree'],
                     {'ymin': 0, 'plot_mean': True}),
                    ('Self-loop density', {'ymin': 0, 'plot_mean': True})])
