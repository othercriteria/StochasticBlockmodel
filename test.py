#!/usr/bin/env python

# Test of "new style" network inference
# Daniel Klein, 5/16/2012

import sys
import json
import pickle

# Putting this in front of expensive imports
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('inputs', help = 'list of JSON files to set parameters',
                    type = argparse.FileType('r'), nargs = '*')
args = parser.parse_args()

import numpy as np
import matplotlib.backends.backend_pdf as pltpdf

from Network import Network
from Models import StationaryLogistic, NonstationaryLogistic
from Models import alpha_zero, alpha_norm, alpha_unif, alpha_gamma
from Experiment import RandomSubnetworks, Seed, \
     Results, add_array_stats, rel_mse
from BinaryMatrix import approximate_conditional_nll as acnll
from Utility import logit

# Parameters
params = { 'N': 130,
           'B': 1,
           'theta_sd': 1.0,
           'theta_fixed': { 'x_0': 2.0, 'x_1': -1.0 },           
           'alpha_unif_sd': 0.0,
           'alpha_norm_sd': 0.0,
           'alpha_gamma_sd': 0.0,
           'cov_unif_sd': 0.0,
           'cov_norm_sd': 0.0,
           'cov_disc_sd': 0.0,
           'kappa_target': ('density', 0.1),
           'pre_offset': False,
           'post_fit': False,
           'fisher_information': False,
           'baseline': False,
           'fit_nonstationary': True,
           'fit_method': 'convex_opt',
           'is_T': 100,
           'num_reps': 3,
           'sampling': 'new',
           'sub_sizes_r': np.floor(0.2 * (np.floor(np.logspace(1.0, 2.1, 30)))),
           'sub_sizes_c': np.floor(np.logspace(1.0, 2.1, 30)),
           'find_good': 0.0,
           'find_bad': 0.0,
           'verbose': True,
           'plot_xaxis': 'c',
           'plot_mse': True,
           'plot_nll': False,
           'plot_network': True,
           'plot_fit_info': True,
           'random_seed': 137,
           'dump_fits': None,
           'load_fits': None,
           'interactive': False }

# Convenience functions for (un)pickling
pick = lambda x: pickle.dumps(x, protocol = 0)
unpick = lambda x: pickle.loads(x)

def do_experiment(params):
    if params['dump_fits'] and params['load_fits']:
        print 'Warning: simultaneously dumping and loading is a bad idea.'
        
    if params['dump_fits']:
        fits = []

    if params['load_fits']:
        with open(params['load_fits'], 'r') as fits_file:
            loaded_params_pick, loaded_fits = json.load(fits_file)

        loaded_params = dict([(k,unpick(v)) for (k,v) in loaded_params_pick])

        # Compare on parameters that control data generation and inference
        run_params = ['N', 'B', 'theta_sd', 'theta_fixed',
                      'alpha_unif_sd', 'alpha_norm_sd', 'alpha_gamma_sd',
                      'cov_unif_sd', 'cov_norm_sd', 'cov_disc_sd',
                      'kappa_target', 'pre_offset', 'post_fit',
                      'fit_nonstationary', 'fit_method', 'num_reps',
                      'is_T', 'sampling', 'sub_sizes_r', 'sub_sizes_c',
                      'random_seed']

        for p in run_params:
            if not np.all(loaded_params[p] == params[p]):
                print 'Warning: load mismatch on', p
    
    # Set random seed for reproducible output
    seed = Seed(params['random_seed'])

    # Initialize full network
    arr = Network(params['N'])

    # Generate node-level propensities to extend and receive edges
    if params['alpha_norm_sd'] > 0.0:
        alpha_norm(arr, params['alpha_norm_sd'])
    elif params['alpha_unif_sd'] > 0.0:
        alpha_unif(arr, params['alpha_unif_sd'])
    elif params['alpha_gamma_sd'] > 0.0:
        # Choosing location somewhat arbitrarily to give unit skewness
        alpha_gamma(arr, 4.0, params['alpha_gamma_sd'])
    else:
        alpha_zero(arr)

    # Generate covariates and associated coefficients
    data_model = NonstationaryLogistic()
    covariates = []
    for b in range(params['B']):
        name = 'x_%d' % b
        covariates.append(name)

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

        arr.new_edge_covariate(name).from_binary_function_ind(f_x)

    # Generate large network, if necessary
    if not params['sampling'] == 'new':
        data_model.match_kappa(arr, params['kappa_target'])
        arr.generate(data_model)

    if params['fit_nonstationary']:
        fit_model = NonstationaryLogistic()
    else:
        fit_model = StationaryLogistic()
    for c in covariates:
        fit_model.beta[c] = None

    # Set up recording of results from experiment
    results = Results(params['sub_sizes_r'], params['sub_sizes_c'],
                      params['num_reps'], interactive = params['interactive'])
    add_array_stats(results)
    if params['plot_nll']:
        from scipy.stats import chi2
        results.new('UMLE diff.', 'nm',
                    lambda n, d, f: f.nll(n) - NonstationaryLogistic().nll(n))
        results.new('UMLE sig.', 'n',
                    lambda n: -0.5 * chi2.ppf(0.95, n.M + n.N - 1))
        results.new('CMLE-A diff.', 'nm',
                    lambda n, d, f: (acnll(n.as_dense(),
                                           np.exp(f.edge_probabilities(n))) - \
                                     acnll(n.as_dense(),
                                           np.ones_like(n.as_dense()))))
        results.new('CMLE-IS diff.', 'nm',
                    lambda n, d, f: (f.fit_conditional(n, evaluate = True, T = 100) -\
                                     NonstationaryLogistic().fit_conditional(n, evaluate = True, T = 100)))
        results.new('C-CMLE diff.', 'nm',
                    lambda n, d, f: (f.fit_c_conditional(n, evaluate = True) - \
                                     NonstationaryLogistic().fit_c_conditional(n, evaluate = True)))
    if params['sampling'] == 'new':
        results.new('Subnetwork kappa', 'm', lambda d, f: d.kappa)
    def true_est_theta_c(c):
        return (lambda d, f: d.beta[c]), (lambda d, f: f.beta[c])
    for c in covariates:
        # Need to do this hackily to avoid for-loop/lambda-binding weirdness.
        f_true, f_est = true_est_theta_c(c)
        results.new('True theta_{%s}' % c, 'm', f_true)
        results.new('Est. theta_{%s}' % c, 'm', f_est)
    if params['pre_offset'] or params['post_fit']:
        results.new('# Active', 'n',
                    lambda n: np.isfinite(n.offset.matrix()).sum())
    else:
        results.new('# Active', 'n', lambda n: n.M * n.N)
    if params['fisher_information']:
        def info_theta_c(c):
            def f_info_theta_c(d, f):
                return d.I_inv['theta_{%s}' % c]
            return f_info_theta_c
        for c in covariates:
            results.new('Info theta_{%s}' % c, 'm', info_theta_c(c))
    if params['baseline']:
        def rel_mse_p_ij(n, d, f):
            P = d.edge_probabilities(n)
            return rel_mse(f.edge_probabilities(n), f.baseline(n), P)
        results.new('Rel. MSE(P_ij)', 'nm', rel_mse_p_ij)
        if not (params['pre_offset'] or params['post_fit']):
            def rel_mse_logit_p_ij(n, d, f):
                logit_P = logit(d.edge_probabilities(n))
                logit_Q = f.baseline_logit(n)
                return rel_mse(logit(f.edge_probabilities(n)),
                               logit_Q, logit_P)
            results.new('Rel. MSE(logit P_ij)', 'nm', rel_mse_logit_p_ij)

    if params['fit_method'] in ['convex_opt', 'conditional', 'c_conditional',
                                'irls', 'conditional_is']:
        results.new('Wall time (sec.)', 'm',
                    lambda d, f: f.fit_info['wall_time'])
    if params['fit_method'] in ['convex_opt',
                                'conditional', 'conditional_is']:
        def work(f):
            w = 0
            for work_type in ['nll_evals', 'grad_nll_evals', 'cnll_evals']:
                if work_type in f.fit_info:
                    w += f.fit_info[work_type]
            return w
        results.new('Work', 'm', lambda d, f: work(f))
        l2 = lambda x: np.sqrt(np.sum(x ** 2))
        results.new('||ET_final - T||_2', 'm',
                    lambda d, f: l2(f.fit_info['grad_nll_final']))

    for sub_size in zip(results.M_sizes, results.N_sizes):
        print 'subnetwork size =', sub_size

        if params['sampling'] == 'new':
            gen = RandomSubnetworks(arr, sub_size)
        else:
            gen = RandomSubnetworks(arr, sub_size,
                                    method = params['sampling'])

        for rep in range(params['num_reps']):
            seed.next()
            sub = gen.sample()

            if params['fisher_information']:
                data_model.fisher_information(sub)

            if params['sampling'] == 'new':
                data_model.match_kappa(sub, params['kappa_target'])
                sub.generate(data_model)

            if params['load_fits']:
                fit, loaded_fits = loaded_fits[0], loaded_fits[1:]
                fit_model.beta = unpick(fit['theta'])
                if 'alpha' in fit:
                    sub.row_covariates['alpha_out'] = unpick(fit['alpha'])
                if 'beta' in fit:
                    sub.col_covariates['alpha_in'] = unpick(fit['beta'])
                if 'kappa' in fit:
                    fit_model.kappa = fit['kappa']
                if 'offset' in fit:
                    sub.offset = unpick(fit['offset'])
                if 'fit_info' in fit:
                    fit_model.fit_info = unpick(fit['fit_info'])
            else:
                if params['pre_offset']:
                    sub.offset_extremes()

                if params['fit_method'] == 'convex_opt':
                    fit_model.fit_convex_opt(sub,
                                             verbose = params['verbose'])
                elif params['fit_method'] == 'irls':
                    fit_model.fit_irls(sub, verbose = params['verbose'])
                elif params['fit_method'] == 'logistic':
                    fit_model.fit_logistic(sub)
                elif params['fit_method'] == 'logistic_l2':
                    fit_model.fit_logistic_l2(sub, prior_precision = 1.0)
                elif params['fit_method'] == 'mh':
                    for c in covariates:
                        fit_model.beta[c] = 0.0
                    fit_model.fit_mh(sub)
                elif params['fit_method'] == 'conditional':
                    fit_model.fit_conditional(sub,
                                              verbose = params['verbose'])
                elif params['fit_method'] == 'conditional_is':
                    fit_model.fit_conditional(sub, T = params['is_T'],
                                              verbose = params['verbose'])
                elif params['fit_method'] == 'c_conditional':
                    fit_model.fit_c_conditional(sub,
                                                verbose = params['verbose'])
                elif params['fit_method'] == 'composite':
                    fit_model.fit_composite(sub, T = 100,
                                            verbose = params['verbose'])
                elif params['fit_method'] == 'brazzale':
                    fit_model.fit_brazzale(sub)
                elif params['fit_method'] == 'saddlepoint':
                    fit_model.fit_saddlepoint(sub)
                elif params['fit_method'] == 'none':
                    pass

                if params['post_fit']:
                    sub.offset_extremes()
                    fit_model.fit_convex_opt(sub, fix_beta = True)

                if params['dump_fits']:
                    fit = {}
                    fit['theta'] = pick(fit_model.beta)
                    if 'alpha_out' in sub.row_covariates:
                        fit['alpha'] = pick(sub.row_covariates['alpha_out'])
                    if 'alpha_in' in sub.row_covariates:
                        fit['beta'] = pick(sub.col_covariates['alpha_in'])
                    if not fit_model.kappa is None:
                        fit['kappa'] = fit_model.kappa
                    if not sub.offset is None:
                        sub.offset.dirty()
                        fit['offset'] = pick(sub.offset)
                    if not fit_model.fit_info is None:
                        fit['fit_info'] = pick(fit_model.fit_info)

                    fits.append(fit)

            if params['find_good'] > 0:
                abs_err = abs(fit_model.beta['x_0'] - data_model.beta['x_0'])
                if abs_err < params['find_good']:
                    print abs_err

                    sub.offset = None
                    fit_model.fit_conditional(sub, T = 1000,
                                              verbose = True)
                    print fit_model.beta['x_0']
                    print fit_model.fit_info

                    f = file('goodmat.mat', 'wb')
                    import scipy.io
                    Y = np.array(sub.adjacency_matrix(), dtype=np.float)
                    X = sub.edge_covariates['x_0'].matrix()
                    scipy.io.savemat(f, { 'Y': Y, 'X': X })
                    sys.exit()

            if params['find_bad'] > 0:
                abs_err = abs(fit_model.beta['x_0'] - data_model.beta['x_0'])
                if abs_err > params['find_bad']:
                    print abs_err

                    sub.offset = None
                    fit_model.fit_conditional(sub, T = 1000,
                                              verbose = True)
                    print fit_model.beta['x_0']
                    print fit_model.fit_info

                    f = file('badmat.mat', 'wb')
                    import scipy.io
                    Y = np.array(sub.adjacency_matrix(), dtype=np.float)
                    X = sub.edge_covariates['x_0'].matrix()
                    scipy.io.savemat(f, { 'Y': Y, 'X': X })
                    sys.exit()

            results.record(sub_size, rep, sub, data_model, fit_model)

            if params['verbose']:
                print

    if params['dump_fits']:
        with open(params['dump_fits'], 'w') as outfile:
            json.dump(([(p, pick(params[p])) for p in params], fits), outfile)

    # Compute beta MSEs
    covariate_naming = []
    for c in covariates:
        mse_name = 'MSE(theta_{%s})' % c
        true_name = 'True theta_{%s}' % c
        est_name = 'Est. theta_{%s}' % c
        results.estimate_mse(mse_name, true_name, est_name)
        covariate_naming.append((c, mse_name, true_name, est_name))

    # Report parameters for the run
    print 'Parameters:'
    for field in params:
        print '%s: %s' % (field, str(params[field]))

    # Should not vary between runs with the same seed and same number
    # of arrays tested
    seed.next()
    print 'URN from Seed:', np.random.random()

    results.summary()

    return results, covariate_naming

def do_plots(results, covariate_naming, params):
    if not params['interactive']:
        pdf = pltpdf.PdfPages('out.pdf')

    # Plot inference performace, in terms of MSE(theta) and MSE(P_ij)
    if params['plot_mse']:
        covariates = [c[0] for c in covariate_naming]
        covariate_mse_names = [c[1] for c in covariate_naming]

        to_plot = []
        if not params['fit_method'] == 'none':
            to_plot.append((['MSE(theta_i)'] + covariate_mse_names,
                            {'ymin': 0, 'ymax': 0.5, 'plot_mean': True}))
        if params['baseline']:
            to_plot.append(('Rel. MSE(P_ij)',
                            {'ymin': 0, 'ymax': 2, 'baseline': 1}))
            if not (params['pre_offset'] or params['post_fit']):
                to_plot.append(('Rel. MSE(logit P_ij)',
                               {'ymin':0, 'ymax': 2, 'baseline': 1}))
        to_plot.append(('# Active', {'ymin': 0}))
        if params['fisher_information']:
            to_plot.append((['Info theta_i'] + \
                            ['Info theta_{%s}' % c for c in covariates],
                            {'ymin': 0, 'plot_mean': True}))
        results.plot(to_plot, {'xaxis': params['plot_xaxis']})
        if not params['interactive']:
            pdf.savefig()

        to_plot = []
        to_plot.append((['MSE(theta_i)'] + covariate_mse_names,
                        {'plot_mean': True, 'loglog': True}))
        if params['fisher_information']:
            to_plot.append((['Info theta_i'] + \
                            ['Info theta_{%s}' % c for c in covariates],
                            {'plot_mean': True, 'loglog': True}))
        results.plot(to_plot, {'xaxis': params['plot_xaxis']})
        if not params['interactive']:
            pdf.savefig()

    # Plot change in NLLs from initialization
    if params['plot_nll']:
        results.plot(['UMLE diff.', 'UMLE sig.'],
                     {'xaxis': params['plot_xaxis']})
        if not params['interactive']:
            pdf.savefig()
        results.plot(['CMLE-A diff.'],
                     {'xaxis': params['plot_xaxis']})
        if not params['interactive']:
            pdf.savefig()
        results.plot(['CMLE-IS diff.'],
                     {'xaxis': params['plot_xaxis']})
        if not params['interactive']:
            pdf.savefig()
        results.plot(['C-CMLE diff.'],
                     {'xaxis': params['plot_xaxis']})
        if not params['interactive']:
            pdf.savefig()
        
    # Plot network statistics
    if params['plot_network']:
        to_plot = [('Density', {'ymin': 0, 'plot_mean': True}),
                   (['Row-sum', 'Max row-sum', 'Min row-sum'],
                    {'ymin': 0, 'plot_mean': True}),
                   (['Col-sum', 'Max col-sum', 'Min col-sum'],
                    {'ymin': 0, 'plot_mean': True})]
        if params['sampling'] == 'new':
            to_plot.append('Subnetwork kappa')
        results.plot(to_plot, {'xaxis': params['plot_xaxis']})
        if not params['interactive']:
            pdf.savefig()

    # Plot convex optimization fitting internal details
    if (params['plot_fit_info'] and params['fit_method'] == 'irls'):
        results.plot([('Wall time (sec.)', {'ymin': 0})],
                     {'xaxis': params['plot_xaxis']})
    if (params['plot_fit_info'] and
        params['fit_method'] in ['convex_opt',
                                 'conditional', 'conditional_is']):
        results.plot([('Work', {'ymin': 0}),
                      ('Wall time (sec.)', {'ymin': 0}),
                      ('||ET_final - T||_2', {'ymin': 0})],
                     {'xaxis': params['plot_xaxis']})
        if not params['interactive']:
            pdf.savefig()

    if not params['interactive']:
        pdf.close()

if len(args.inputs) > 0:
    results = None
    for params_file in args.inputs:
        new_params_pick = json.load(params_file)
        new_params = dict([(k,unpick(v)) for (k,v) in new_params_pick])

        print 'Setting parameters from %s:' % params_file
        for k in new_params:
            print k
            print 'old:', params[k]
            print 'new:', new_params[k]
            print
            params[k] = new_params[k]

        new_results, covariate_naming = do_experiment(params)
        if results:
            results.merge(new_results)
        else:
            results = new_results

        print

    print 'Combined results:\n'

    # Recompute MSEs over all the runs
    for c, mse_name, true_name, est_name in covariate_naming:
        results.estimate_mse(mse_name, true_name, est_name)

    results.summary()

    do_plots(results, covariate_naming, params)
else:
    results, covariate_naming = do_experiment(params)
    do_plots(results, covariate_naming, params)
