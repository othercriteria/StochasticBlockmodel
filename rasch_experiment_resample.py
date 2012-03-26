#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Parameters
params = { 'N': 400,
           'alpha_sd': 0.0,
           'alpha_unif': 1.0,
           'beta_shank': 3.0,
           'num_shank': 8,
           'beta_self': 4.0,
           'kappa_base': -5.0,
           'kappa_coef_1': 1.0,
           'kappa_coef_2': 0.0,
           'reg_C': 10.0,
           'N_subs': range(10, 400, 20),
           'num_fits': 10,
           'do_inference': True }


# Set random seed for reproducible output
np.random.seed(137)

# Define mapping from covariates and parameters alpha, beta, kappa to
# edge probabilties
def edge_probabilities(alpha, beta, kappa, x):
    N = x.shape[0]
    logit_P = np.zeros((N,N))
    for i in range(N):
        logit_P[i,:] += alpha[0,i]
    for j in range(N):
        logit_P[:,j] += alpha[0,j]
    logit_P += np.dot(x, beta)
    logit_P += kappa

    return 1.0 / (np.exp(-logit_P) + 1.0)

# Define negative log-likelihood
def neg_log_likelihood(alpha, beta, kappa, A, x):
    P = edge_probabilities(alpha, beta, kappa, x)
    return -np.sum(np.log(P ** A) + np.log((1.0 - P) ** (1.0 - A)))

# Procedure to find MLE via logistic regression
def infer(A, x, fit_alpha = False):
    N = A.shape[0]
    B = x.shape[2]

    lr = LogisticRegression(fit_intercept = True,
                            C = params['reg_C'], penalty = 'l2')
    y = A.reshape((N*N,))
    if fit_alpha:
        Phi = np.zeros((N*N,(B + 2*N)))
    else:
        Phi = np.zeros((N*N,B))
    Phi[:,0] = 1.0
    for b in range(B):
        Phi[:,b] = x[:,:,b].reshape((N*N,))
    if fit_alpha:
        for i in range(N):
            phi_row = np.zeros((N,N))
            phi_row[i,:] = 1.0
            Phi[:,B + i] = phi_row.reshape((N*N,))
        for j in range(N):
            phi_col = np.zeros((N,N))
            phi_col[:,j] = 1.0
            Phi[:,B + N + j] = phi_col.reshape((N*N,))
    lr.fit(Phi, y)
    coefs = lr.coef_[0]
    intercept = lr.intercept_[0]

    alpha = np.zeros((2,N))
    out = {'alpha': alpha, 'beta': coefs[0:B], 'kappa': intercept}
    if fit_alpha:
        out['alpha'][0] = coefs[B:(B + N)]
        out['alpha'][1] = coefs[(B + N):(B + 2*N)]
    return out

# Generate latent parameters
if params['alpha_sd'] > 0.0:
    alpha = np.random.normal(0, params['alpha_sd'], (2,params['N']))
elif params['alpha_unif'] > 0.0:
    alpha = np.random.uniform(-params['alpha_unif'], params['alpha_unif'],
                              (2,params['N']))
else:
    alpha = np.zeros((2,params['N']))
alpha[0] -= np.mean(alpha[0])
alpha[1] -= np.mean(alpha[1])
beta = np.array([params['beta_shank'], params['beta_self']])
shank = np.random.randint(0, params['num_shank'], params['N'])
x = np.empty((params['N'],params['N'],2))
for i in range(params['N']):
    for j in range(params['N']):
        x[i,j,0] = (shank[i] == shank[j])
        x[i,j,1] = (i == j)

# Procedure for generating random subnetwork
def subnetwork(n):
    inds = np.arange(params['N'])
    np.random.shuffle(inds)
    sub = inds[0:n]
    
    alpha_sub = alpha[:,sub]
    x_sub = x[sub][:,sub]
    kappa_sub = (params['kappa_base'] -
                 params['kappa_coef_1'] * np.log(n) -
                 params['kappa_coef_2'] * np.log(np.log(n)))
    
    P_sub = edge_probabilities(alpha_sub, beta, kappa_sub, x_sub)
    A_sub = np.random.random((n,n)) < P_sub
    return alpha_sub, kappa_sub, A_sub, x_sub

# Some pre-analysis output
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))

# Fit model on partially observed subnetworks and assess performance
bias = np.empty((2,2,len(params['N_subs'])))
variance = np.empty((2,2,len(params['N_subs'])))
network = np.empty((4,len(params['N_subs'])))
for n, N_sub in enumerate(params['N_subs']):
    print 'N_sub = %d' % N_sub
    estimate = np.empty((2,2,params['num_fits']))
    network_obs = np.empty((4,params['num_fits']))
    for num_fit in range(params['num_fits']):
        alpha_sub, kappa_sub, A_sub, x_sub = subnetwork(N_sub)
        network_obs[0,num_fit] = 1.0 * np.sum(A_sub) / N_sub
        network_obs[1,num_fit] = np.max(np.sum(A_sub, axis = 1))
        network_obs[2,num_fit] = np.max(np.sum(A_sub, axis = 0))
        network_obs[3,num_fit] = 1.0 * np.sum(np.diagonal(A_sub)) / N_sub

        if not params['do_inference']:
            continue

        sub_nll = neg_log_likelihood(alpha_sub, beta, kappa_sub, A_sub, x_sub)
        print 'True subnetwork NLL: %.2f' % sub_nll

        # Fit full model
        fit = infer(A_sub, x_sub, fit_alpha = True)
        fit_nll = neg_log_likelihood(fit['alpha'], fit['beta'], fit['kappa'],
                                     A_sub, x_sub)
        print 'Fit subnetwork NLL: %.2f' % fit_nll
        estimate[0,0,num_fit] = fit['beta'][0]
        estimate[1,0,num_fit] = fit['beta'][1]

        # Fit model with zero alpha
        fit = infer(A_sub, x_sub, fit_alpha = False)
        fit_nll = neg_log_likelihood(fit['alpha'], fit['beta'], fit['kappa'],
                                     A_sub, x_sub)
        print 'Fit subnetwork (zero alpha) NLL: %.2f' % fit_nll
        estimate[0,1,num_fit] = fit['beta'][0]
        estimate[1,1,num_fit] = fit['beta'][1]

    if params['do_inference']:
        for metric, true_val in enumerate([beta[0], beta[1]]):
            for m in range(2):
                bias[metric,m,n] = np.mean(estimate[metric,m] - true_val)
                variance[metric,m,n] = np.var(estimate[metric,m])
    for metric in range(4):
        network[metric,n] = np.mean(network_obs[metric])
if params['do_inference']:
    bias_sq = bias ** 2
    mse = variance + bias_sq

plt.figure()
if params['do_inference']:
    for metric, (name, val) in enumerate([('beta_shank', beta[0]),
                                          ('beta_self', beta[1])]):
        plt.subplot(6,1,(metric+1))
        plt.plot(params['N_subs'], mse[metric,0], 'b')
        plt.plot(params['N_subs'], variance[metric,0], 'b--')
        plt.plot(params['N_subs'], mse[metric,1], 'r')
        plt.plot(params['N_subs'], variance[metric,1], 'r--')
        plt.plot(params['N_subs'], [val ** 2] * len(params['N_subs']), 'k:')
        plt.ylim(ymax = 1.2 * val ** 2)
        plt.title(name)

for metric, name in enumerate(['average degree',
                               'max out-degree',
                               'max in-degree',
                               'self-edge probability']):
    plt.subplot(6,1,(metric+3))
    plt.plot(params['N_subs'], network[metric], '-')
    plt.ylabel(name)
    plt.ylim(ymin = 0)

plt.show()

