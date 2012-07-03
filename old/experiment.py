#!/usr/bin/env python

import numpy as np
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from common import infer_block, plot_block, neg_log_likelihood

# Parameters
params = { 'N': 50,
           'N_subs': range(5,50,5),
           'K': 3,
           'conc': 10.0,
           'alpha_sd': 0.5,
           'beta_shank': 2.0,
           'beta_self': 3.0,
           'num_shanks': 8,
           'Theta_mean': -1.0,
           'Theta_sd': 3.0,
           'num_fits': 30,
           'K_fit': 3,
           'steps': 50,
           'sweeps': 2,
           'C': 100.0,
           'init_from_true': False,
           'true_z': False,
           'full_network_fit': False,
           'subnetwork_fit': True,
           'plot': False }


# Set random seed for reproducible output
np.random.seed(137)

# Generate random network, given block structure
def make_network(Theta):
    class_probs = np.random.dirichlet([params['conc']] * params['K'])
    z = np.where(np.random.multinomial(1, class_probs, params['N']) == 1)[1]
    alpha = np.random.normal(0, params['alpha_sd'], (2,params['N']))
    alpha[0] -= np.mean(alpha[0])
    alpha[1] -= np.mean(alpha[1])
    beta = np.array([params['beta_shank'],params['beta_self']])
    shank = np.random.randint(0, params['K'], params['N'])
    x = np.empty((params['N'],params['N'],2))
    for i in range(params['N']):
        for j in range(params['N']):
            x[i,j,0] = shank[i] == shank[j]
            x[i,j,1] = i == j

    logit_P = np.empty((params['N'],params['N']))
    for i in range(params['N']):
        for j in range(params['N']):
            logit_P[i,j] = (Theta[z[i],z[j]] +
                            alpha[0,i] + alpha[1,j] +
                            np.dot(x[i,j], beta))
    P = 1.0 / (np.exp(-logit_P) + 1.0)
    A = np.random.random((params['N'],params['N'])) < P

    return z, x, alpha, beta, P, A

Theta = np.random.normal(params['Theta_mean'], params['Theta_sd'],
                         (params['K'],params['K']))
print Theta
z, x, alpha, beta, P, A = make_network(Theta)
print 'True NLL: %.2f' % neg_log_likelihood(A, z, Theta, alpha, beta, x)

# Plot network with true blocking
if params['plot']:
    plot_block('true_blocking', A, z, params['K'], alpha = alpha)

# Repeatedly fit and plot inferred blocking
if params['full_network_fit']:
    for num_fit in range(params['num_fits']):
        init_z, init_theta, init_alpha, init_beta = None, None, None, None
        if params['init_from_true']:
            init_z, init_theta, init_alpha, init_beta = z, Theta, alpha, beta
        fit = infer_block(A, x,
                          params['K_fit'], params['steps'], params['sweeps'],
                          params['C'],
                          zero_alpha = False, true_z = None, init_z = init_z,
                          init_alpha = init_alpha, init_beta = init_beta,
                          init_theta = init_theta)
        print fit['beta']
        fit_nll = neg_log_likelihood(A, fit['z'], fit['Theta'],
                                     fit['alpha'], fit['beta'], x)
        print 'Fit NLL: %.2f' % fit_nll

        if params['plot']:
            plot_block('fit_blocking_%d' % num_fit, A, fit['z'], params['K_fit'],
                       alpha = fit['alpha'])

# Fit model on subnetworks and assess performance
if params['subnetwork_fit']:
    bias = np.empty((3,3,len(params['N_subs'])))
    variance = np.empty((3,3,len(params['N_subs'])))
    inds = np.arange(params['N'])
    for n, N_sub in enumerate(params['N_subs']):
        print 'N_sub = %d' % N_sub
        estimate = np.empty((3,3,params['num_fits']))
        for num_fit in range(params['num_fits']):
            np.random.shuffle(inds)
            sub = inds[0:N_sub]

            A_sub = A[sub][:,sub]
            x_sub = x[sub][:,sub]
            z_sub = z[sub]
            alpha_sub = alpha[:,sub]

            sub_nll = neg_log_likelihood(A_sub, z_sub, Theta,
                                         alpha_sub, beta, x_sub)
            print 'True subnetwork NLL: %.2f' % sub_nll

            init_z, init_theta, init_alpha, init_beta = None, None, None, None
            true_z = None
            if params['init_from_true']:
                init_z, init_alpha = z_sub, alpha_sub
                init_theta, init_beta = Theta, beta
            if params['true_z']:
                true_z = z_sub

            # Fit with alpha free
            fit = infer_block(A_sub, x_sub,
                              params['K_fit'], params['steps'],
                              params['sweeps'], params['C'],
                              zero_alpha = False,
                              true_z = true_z, init_z = init_z,
                              init_alpha = init_alpha, init_beta = init_beta,
                              init_theta = init_theta)
            fit_nll = neg_log_likelihood(A_sub, fit['z'], fit['Theta'],
                                         fit['alpha'], fit['beta'], x_sub)
            print 'Fit subnetwork NLL: %.2f' % fit_nll
            estimate[0,0,num_fit] = fit['beta'][0]
            estimate[1,0,num_fit] = fit['beta'][1]
            estimate[2,0,num_fit] = eigvals(fit['Theta'])[0].real

            # Fit with alpha fixed to zero
            fit = infer_block(A_sub, x_sub,
                              params['K_fit'], params['steps'],
                              params['sweeps'], params['C'],
                              zero_alpha = True,
                              true_z = true_z, init_z = init_z,
                              init_alpha = None, init_beta = init_beta,
                              init_theta = init_theta)
            fit_nll = neg_log_likelihood(A_sub, fit['z'], fit['Theta'],
                                         fit['alpha'], fit['beta'], x_sub)
            print 'Fit subnetwork (zero alpha) NLL: %.2f' % fit_nll
            estimate[0,1,num_fit] = fit['beta'][0]
            estimate[1,1,num_fit] = fit['beta'][1]
            estimate[2,1,num_fit] = eigvals(fit['Theta'])[0].real

            # Fit beta parameters alone using logistic regression
            lr = LogisticRegression(C = 1.0, penalty = 'l2')
            X = x_sub.reshape((N_sub*N_sub,x_sub.shape[2]))
            y = A_sub.reshape((N_sub*N_sub,))
            lr.fit(X, y)
            fit = lr.coef_[0]
            estimate[0,2,num_fit] = fit[0]
            estimate[1,2,num_fit] = fit[1]
            estimate[2,2,num_fit] = 0.0

        true_eig = eigvals(Theta)[0].real
        for metric, true_val in enumerate([beta[0], beta[1], true_eig]):
            for m in range(3):
                bias[metric,m,n] = np.mean(estimate[metric,m] - true_val)
                variance[metric,m,n] = np.var(estimate[metric,m])
    bias_sq = bias ** 2
    mse = variance + bias_sq

    plt.figure()
    for metric, (name, val) in enumerate([('beta_shank', beta[0]),
                                          ('beta_self', beta[1]),
                                          ('theta_eig', true_eig)]):
        plt.subplot(3,1,(metric+1))
        plt.plot(params['N_subs'], bias_sq[metric,0], 'b--')
        plt.plot(params['N_subs'], variance[metric,0], 'b-.')
        plt.plot(params['N_subs'], mse[metric,0], 'b')
        plt.plot(params['N_subs'], bias_sq[metric,1], 'r--')
        plt.plot(params['N_subs'], variance[metric,1], 'r-.')
        plt.plot(params['N_subs'], mse[metric,1], 'r')
        if not metric == 2:
            plt.plot(params['N_subs'], bias_sq[metric,2], 'y--')
            plt.plot(params['N_subs'], variance[metric,2], 'y-.')
            plt.plot(params['N_subs'], mse[metric,2], 'y')
        plt.plot(params['N_subs'], [val ** 2] * len(params['N_subs']), 'k:')
        plt.ylim(ymax = 1.2 * val ** 2)
        plt.title(name)
    plt.show()

for field in params:
    print '%s: %s' % (field, str(params[field]))
