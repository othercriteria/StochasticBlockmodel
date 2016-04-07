#!/usr/bin/env python

# Test of "new style" network inference, finally with blockmodel
# Daniel Klein, 7/8/2012

import numpy as np

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic, Blockmodel
from Models import alpha_zero, alpha_norm, alpha_unif, alpha_gamma
from Experiment import RandomSubnetworks, Results, add_array_stats
from Experiment import minimum_disagreement, rel_mse

# Parameters
params = { 'N': 100,
           'K': 2,
           'class_conc': 10.0,
           'Theta_diag': 4.0,
           'Theta_mean': 0.0,
           'Theta_sd': 1.0,
           'B': 1,
           'beta_sd': 1.0,
           'alpha_unif': 0.0,
           'alpha_norm_sd': 2.0,
           'alpha_gamma_sd': 0.0,
           'kappa_target': ('row_sum', 5),
           'fit_nonstationary': True,
           'fit_conditional': True,
           'fit_K': 2,
           'initialize_true_z': False,
           'cycles': 5,
           'sweeps': 1,
           'verbose': False,
           'num_reps': 5,
           'sub_sizes': np.arange(10, 60, 5),
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
for b in range(params['B']):
    name = 'x_%d' % b

    data_base_model.beta[name] = np.random.normal(0, params['beta_sd'])

    def f_x(i_1, i_2):
        return np.random.uniform(-np.sqrt(3), np.sqrt(3))
    net.new_edge_covariate(name).from_binary_function_ind(f_x)
    
# Initialize data (block)model from base model
class_probs = np.random.dirichlet(np.repeat(params['class_conc'], params['K']))
z = np.where(np.random.multinomial(1, class_probs, params['N']) == 1)[1]
net.new_node_covariate_int('z_true')[:] = z
data_model = Blockmodel(data_base_model, params['K'], 'z_true')
Theta = np.random.normal(params['Theta_mean'], params['Theta_sd'],
                         (params['K'],params['K']))
Theta += params['Theta_diag'] * np.identity(params['K'])
Theta -= np.mean(Theta)
data_model.Theta = Theta

net.generate(data_model)
if params['plot_network']:
    net.show_heatmap('z_true')

# Initialize fitting model
fit_base_model = StationaryLogistic()
for b in data_base_model.beta:
    fit_base_model.beta[b] = None
fit_model = Blockmodel(fit_base_model, params['fit_K'])
if params['fit_nonstationary']:
    n_fit_base_model = NonstationaryLogistic()
    for b in data_base_model.beta:
        n_fit_base_model.beta[b] = None
    n_fit_model = Blockmodel(n_fit_base_model, params['fit_K'])
net.new_node_covariate_int('z')

# Set up recording of results from experiment
s_results = Results(params['sub_sizes'], params['sub_sizes'],
                    params['num_reps'], 'Stationary fit')
add_array_stats(s_results)
s_results.new('Subnetwork kappa', 'm', lambda d, f: d.base_model.kappa)
def f_b(b):
    return ((lambda d, f: d.base_model.beta[b]),
            (lambda d, f: f.base_model.beta[b]))
for b in data_base_model.beta:
    # Need to do this hackily to avoid for-loop/lambda-binding weirdness.
    f_true, f_estimated = f_b(b)
    s_results.new('True beta_{%s}' % b, 'm', f_true)
    s_results.new('Estimated beta_{%s}' % b, 'm', f_estimated)
s_results.new('Class mismatch', 'n',
              lambda n: minimum_disagreement(n.node_covariates['z_true'][:], \
                                             n.node_covariates['z'][:]))
def rel_mse_p_ij(n, d, f):
    P = d.edge_probabilities(n)
    return rel_mse(f.edge_probabilities(n), f.baseline(n), P)
s_results.new('Rel. MSE(P)', 'nm', rel_mse_p_ij)
def rel_mse_logit_p_ij(n, d, f):
    logit_P = d.edge_probabilities(n, logit = True)
    logit_Q = f.baseline_logit(n)
    return rel_mse(f.edge_probabilities(n, logit = True), logit_Q, logit_P)
s_results.new('Rel. MSE(logit_P)', 'nm', rel_mse_logit_p_ij)

all_results = { 's': s_results }
if params['fit_nonstationary']:
    n_results = s_results.copy()
    n_results.title = 'Nonstationary fit'
    all_results['n'] = n_results
if params['fit_conditional']:
    c_results = s_results.copy()
    c_results.title = 'Conditional fit'
    all_results['c'] = c_results

def initialize(s, f):
    if params['initialize_true_z']:
        s.node_covariates['z'][:] = s.node_covariates['z_true'][:]
    else:
        s.node_covariates['z'][:] = np.random.randint(0, params['fit_K'], s.N)

for sub_size in params['sub_sizes']:
    size = (sub_size, sub_size)
    print 'subnetwork size = %s' % str(size)
    
    gen = RandomSubnetworks(net, size)
    for rep in range(params['num_reps']):
        subnet = gen.sample(as_network = True)
        data_model.match_kappa(subnet, params['kappa_target'])
        subnet.generate(data_model)
        
        initialize(subnet, fit_model)
        fit_model.fit(subnet, params['cycles'], params['sweeps'],
                      verbose = params['verbose'])
        s_results.record(size, rep, subnet, data_model, fit_model)
        print

        if params['fit_conditional']:
            initialize(subnet, fit_model)
            fit_base_model.fit = fit_base_model.fit_conditional
            fit_model.fit(subnet, params['cycles'], params['sweeps'])
            c_results.record(size, rep, subnet, data_model, fit_model)
            print
        
        if params['fit_nonstationary']:
            subnet.offset_extremes()
            initialize(subnet, n_fit_model)
            n_fit_model.fit(subnet, params['cycles'], params['sweeps'],
                            verbose = params['verbose'])
            n_results.record(size, rep, subnet, data_model, n_fit_model)
            print

# Compute beta MSEs
covariate_mses = []
for b in fit_base_model.beta:
    name = 'MSE(beta_{%s})' % b
    covariate_mses.append(name)
    for model in all_results:
        results = all_results[model]
        results.estimate_mse(name,
                             'True beta_{%s}' % b, 'Estimated beta_{%s}' % b)

for model in all_results:
    results = all_results[model]
    print results.title
    results.summary()
    print
            
# Plot inference performace, in terms of MSE(beta), MSE(P_ij), and
# inferred class disagreement; also plot kappas chosen for data models
if params['plot_mse']:
    for model in all_results:
        results = all_results[model]
        to_plot = [(['MSE(beta_i)'] + covariate_mses,
                    {'ymin': 0, 'ymax': 0.5, 'plot_mean': True}),
                   ('Rel. MSE(P)', {'ymin': 0, 'ymax': 2, 'baseline': 1}),
                   ('Rel. MSE(logit_P)', {'ymin': 0, 'ymax': 2, 'baseline': 1}),
                   ('Class mismatch', {'ymin': 0, 'ymax': 2})]
        if model == 'n': to_plot.pop(2)
        results.plot(to_plot)
  
# Plot network statistics as well as sparsity parameter
if params['plot_network']:
    s_results.title = None
    
    s_results.plot([('Average row-sum', {'ymin': 0, 'plot_mean': True}),
                    ('Average col-sum', {'ymin': 0, 'plot_mean': True}),
                    (['row-sum', 'Max row-sum', 'Min row-sum'],
                     {'ymin': 0, 'plot_mean': True}),
                    (['col-sum', 'Max row-sum', 'Min col-sum'],
                     {'ymin': 0, 'plot_mean': True}),
                    'Subnetwork kappa'])
