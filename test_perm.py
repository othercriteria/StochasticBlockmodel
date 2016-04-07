#!/usr/bin/env python

# Test of "new style" network inference
# Daniel Klein, 5/16/2012

import numpy as np

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic
from Models import FixedMargins
from Models import alpha_zero
from Experiment import RandomSubnetworks, Results, add_array_stats, rel_mse
from Utility import l2, logit

# Parameters
params = { 'N': 300,
           'B': 2,
           'theta_sd': 1.0,
           'theta_fixed': { 'x_0': 2.0, 'x_1': -1.0 },
           'cov_unif_sd': 0.0,
           'cov_norm_sd': 0.0,
           'cov_disc_sd': 1.0,
           'fisher_information': False,
           'baseline': True,
           'fit_nonstationary': True,
           'fit_method': 'logistic_l2',
           'ignore_separation': False,
           'separation_samples': 10,
           'num_reps': 15,
           'sub_sizes': np.floor(np.logspace(1.0, 2.1, 20)),
           'verbose': True,
           'plot_mse': True,
           'plot_network': False,
           'plot_fit_info': True }


# Set random seed for reproducible output
np.random.seed(137)

# Initialize full network
net = Network(params['N'])
alpha_zero(net)

# Generate covariates and associated coefficients
data_model = NonstationaryLogistic()
for b in range(params['B']):
    name = 'x_%d' % b

    if name in params['theta_fixed']:
        data_model.beta[name] = params['theta_fixed'][name]
    else:
        data_model.beta[name] = np.random.normal(0, params['theta_sd'])

    if params['cov_unif_sd'] > 0.0:
        c = np.sqrt(12) / 2
        def f_x(i_1, i_2):
            return np.random.uniform(-c * params['cov_unif_sd'],
                                     c * params['cov_unif_sd'])
    elif params['cov_norm_sd'] > 0.0:
        def f_x(i_1, i_2):
            return np.random.normal(0, params['cov_norm_sd'])
    elif params['cov_disc_sd'] > 0.0:
        def f_x(i_1, i_2):
            return (params['cov_disc_sd'] *
                    (np.sign(np.random.random() - 0.5)))
    else:
        print 'Error: no covariate distribution specified.'
        sys.exit()

    net.new_edge_covariate(name).from_binary_function_ind(f_x)

# Specify data model as generation of permuation networks
net.new_node_covariate_int('r')[:] = 1
net.new_node_covariate_int('c')[:] = 1
data_model = FixedMargins(data_model, 'r', 'c', coverage = 2.0)

if params['fit_nonstationary']:
    fit_model = NonstationaryLogistic()
else:
    fit_model = StationaryLogistic()
for b in data_model.base_model.beta:
    fit_model.beta[b] = 0.0

# Set up recording of results from experiment
results = Results(params['sub_sizes'], params['sub_sizes'], params['num_reps'])
add_array_stats(results)
def true_est_theta_b(b):
    return (lambda d, f: d.base_model.beta[b]), (lambda d, f: f.beta[b])
for b in fit_model.beta:
    # Need to do this hackily to avoid for-loop/lambda-binding weirdness.
    f_true, f_est = true_est_theta_b(b)
    results.new('True theta_{%s}' % b, 'm', f_true)
    results.new('Est. theta_{%s}' % b, 'm', f_est)
results.new('# Active', 'n', lambda n: n.N ** 2)
results.new('Separated', 'm', lambda d, f: f.fit_info['separated'])
if params['fisher_information']:
    def info_theta_b(b):
        def f_info_theta_b(d, f):
            return d.base_model.I_inv['theta_{%s}' % b]
        return f_info_theta_b
    for b in fit_model.beta:
        results.new('Info theta_{%s}' % b, 'm', info_theta_b(b))
if params['baseline']:
    def rel_mse_p_ij(n, d, f):
        P = d.edge_probabilities(n)
        return rel_mse(f.edge_probabilities(n), f.baseline(n), P)
    results.new('Rel. MSE(P_ij)', 'nm', rel_mse_p_ij)
    def rel_mse_logit_p_ij(n, d, f):
        logit_P = d.edge_probabilities(n, logit = True)
        logit_Q = f.baseline_logit(n)
        return rel_mse(f.edge_probabilities(n, logit = True), logit_Q, logit_P)
    results.new('Rel. MSE(logit P_ij)', 'nm', rel_mse_logit_p_ij)

if params['fit_method'] in ['convex_opt', 'conditional',
                            'irls', 'conditional_is']:
    results.new('Wall time (sec.)', 'm', lambda d, f: f.fit_info['wall_time'])
if params['fit_method'] in ['convex_opt', 'conditional', 'conditional_is']:
    def work(f):
        w = 0
        for work_type in ['nll_evals', 'grad_nll_evals', 'cnll_evals']:
            if work_type in f.fit_info:
                w += f.fit_info[work_type]
        return w
    results.new('Work', 'm', lambda d, f: work(f))
    results.new('||ET_final - T||_2', 'm',
                lambda d, f: l2(f.fit_info['grad_nll_final']))

for sub_size in params['sub_sizes']:
    size = (sub_size, sub_size)
    print 'subnetwork size = %s' % str(size)

    gen = RandomSubnetworks(net, size)

    for rep in range(params['num_reps']):
        subnet = gen.sample()
        subnet.generate(data_model)

        if params['fisher_information']:
            data_model.base_model.fisher_information(subnet)

        if params['fit_method'] in ('conditional', 'conditional_is',
                                    'brazzale', 'saddlepoint'):
            fixed_model = FixedMargins(base_model = fit_model, coverage = 2.0)
            fixed_model.check_separated(subnet,
                                        samples = params['separation_samples'])
        else:
            fit_model.check_separated(subnet)
        
        if not params['ignore_separation'] and fit_model.fit_info['separated']:
            print 'Separated, defaulting to theta = 0.'
            for b in data_model.base_model.beta:
                fit_model.beta[b] = 0.0
            fit_model.fit_convex_opt(subnet, fix_beta = True)
        elif params['fit_method'] == 'convex_opt':
            if params['verbose']:
                fit_model.fit_convex_opt(subnet, verbose = True)
                print
            else:
                fit_model.fit_convex_opt(subnet)
        elif params['fit_method'] == 'irls':
            fit_model.fit_irls(subnet)
        elif params['fit_method'] == 'logistic':
            fit_model.fit_logistic(subnet)
        elif params['fit_method'] == 'logistic_l2':
            fit_model.fit_logistic_l2(subnet, prior_precision = 1.0)
        elif params['fit_method'] == 'conditional':
            fit_model.fit_conditional(subnet, verbose = params['verbose'])
        elif params['fit_method'] == 'conditional_is':
            fit_model.fit_conditional(subnet, T = 50, one_sided = True,
                                      verbose = params['verbose'])
        elif params['fit_method'] == 'composite':
            fit_model.fit_composite(subnet, T = 100, verbose = True)
        elif params['fit_method'] == 'brazzale':
            fit_model.fit_brazzale(subnet, 'x_0', verbose = params['verbose'])
        elif params['fit_method'] == 'saddlepoint':
            fit_model.fit_saddlepoint(subnet, verbose = params['verbose'])
            fit_model.fit_convex_opt(subnet, fix_beta = True)
        elif params['fit_method'] == 'none':
            pass
            
        results.record(size, rep, subnet, data_model, fit_model)

# Compute beta MSEs, MAEs
covariate_mses = []
covariate_maes = []
for b in fit_model.beta:
    name = 'MSE(theta_{%s})' % b
    covariate_mses.append(name)
    results.estimate_mse(name, 'True theta_{%s}' % b, 'Est. theta_{%s}' % b)

    name = 'MAE(theta_{%s})' % b
    covariate_maes.append(name)
    results.estimate_mae(name, 'True theta_{%s}' % b, 'Est. theta_{%s}' % b)

# Dump summary results
results.summary()

# Plot inference performance, in terms of MSE(theta), MAE(theta), MSE(P_ij)
if params['plot_mse']:
    to_plot = []
    if not params['fit_method'] == 'none':
        to_plot.append((['MSE(theta_i)'] + covariate_mses,
                        {'ymin': 0, 'ymax': 0.5, 'plot_mean': True}))
    if params['baseline']:
        to_plot.append(('Rel. MSE(P_ij)',
                        {'ymin': 0, 'ymax': 2, 'baseline': 1}))
        to_plot.append(('Rel. MSE(logit P_ij)',
                        {'ymin':0, 'ymax': 2, 'baseline': 1}))
    to_plot.append(('# Active', {'ymin': 0}))
    to_plot.append(('Separated', {'ymin': 0, 'ymax': 1, 'plot_mean': True}))
    if params['fisher_information']:
        to_plot.append((['Info theta_i'] + \
                        ['Info theta_{%s}' % b for b in fit_model.beta],
                        {'ymin': 0, 'plot_mean': True}))
    results.plot(to_plot)

    to_plot = []
    to_plot.append((['MSE(theta_i)'] + covariate_mses,
                    {'loglog': True, 'plot_mean': True}))
    if params['fisher_information']:
        to_plot.append((['Info theta_i'] + \
                        ['Info theta_{%s}' % b for b in fit_model.beta],
                        {'plot_mean': True, 'loglog': True}))
    results.plot(to_plot)

    results.plot([(['MAE(theta_i)'] + covariate_maes,
                   {'loglog': True, 'plot_mean': True})])

# Plot network statistics
if params['plot_network']:
    to_plot = [('Density', {'ymin': 0, 'plot_mean': True}),
               (['Row-sum', 'Max row-sum', 'Min row-sum'],
                {'ymin': 0, 'plot_mean': True}),
               (['Col-sum', 'Max col-sum', 'Min col-sum'],
                {'ymin': 0, 'plot_mean': True})]
    results.plot(to_plot)

# Plot convex optimization fitting internal details
if (params['plot_fit_info'] and params['fit_method'] == 'irls'):
    results.plot([('Wall time (sec.)', {'ymin': 0})])
if (params['plot_fit_info'] and
    params['fit_method'] in ['convex_opt', 'conditional', 'conditional_is']):
    results.plot([('Work', {'ymin': 0}),
                  ('Wall time (sec.)', {'ymin': 0}),
                  ('||ET_final - T||_2', {'ymin': 0})])

# Report parameters for the run
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, repr(params[field]))
