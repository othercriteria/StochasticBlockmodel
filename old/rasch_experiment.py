#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Parameters
params = { 'N': 200,
           'alpha_sd': 0.001,
           'beta_shank': 3.0,
           'num_shank': 8,
           'beta_self': 4.0,
           'kappa': -5.0,
           'reg_C': 10.0,
           'edge_thinning_exp': 1.0,
           'edge_thinning_coef': 2.0,
           'edge_thinning_scheme': 'prop',
           'N_subs': range(10, 210, 10),
           'num_fits': 15 }


# Set random seed for reproducible output
np.random.seed(137)

# Define mapping from covariates and parameters alpha, beta, kappa to
# edge probabilties
def edge_probabilities(alpha, beta, kappa, x):
    N = x.shape[0]
    logit_P = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            logit_P[i,j] = alpha[0,i] + alpha[1,j] + np.dot(x[i,j], beta)
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

# Generate random network
if params['alpha_sd'] > 0.0:
    alpha = np.random.normal(0, params['alpha_sd'], (2,params['N']))
    alpha[0] -= np.mean(alpha[0])
    alpha[1] -= np.mean(alpha[1])
else:
    alpha = np.zeros((2,params['N']))
beta = np.array([params['beta_shank'], params['beta_self']])
shank = np.random.randint(0, params['num_shank'], params['N'])
x = np.empty((params['N'],params['N'],2))
for i in range(params['N']):
    for j in range(params['N']):
        x[i,j,0] = (shank[i] == shank[j])
        x[i,j,1] = (i == j)
kappa = params['kappa']
P = edge_probabilities(alpha, beta, kappa, x)
A = np.random.random((params['N'],params['N'])) < P

# Some pre-analysis output
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))
print
print 'Ground truth network:'
print np.asarray(A, dtype=np.int)
print
print 'Ground truth NLL: %.2f' % neg_log_likelihood(alpha, beta, kappa, A, x)

# Fit model on partially observed subnetworks and assess performance
bias = np.empty((2,2,len(params['N_subs'])))
variance = np.empty((2,2,len(params['N_subs'])))
inds = np.arange(params['N'])
for n, N_sub in enumerate(params['N_subs']):
    thinning = int(params['edge_thinning_coef'] *
                   N_sub ** params['edge_thinning_exp'])
    print 'N_sub = %d (thinning to %d edges)' % (N_sub, thinning)
    estimate = np.empty((2,2,params['num_fits']))
    for num_fit in range(params['num_fits']):
        np.random.shuffle(inds)
        sub = inds[0:N_sub]

        # Sample subnetwork
        A_sub = A[sub][:,sub]
        x_sub = x[sub][:,sub]
        alpha_sub = alpha[:,sub]

        sub_nll = neg_log_likelihood(alpha_sub, beta, kappa, A_sub, x_sub)
        print 'True subnetwork NLL: %.2f' % sub_nll

        # Thin observations to fixed number of edges
        A_sub_thin = A_sub.copy()
        sub_num_edges = np.sum(A_sub)
        if sub_num_edges > thinning:
            if params['edge_thinning_scheme'] == 'uniform':
                thin = np.arange(sub_num_edges)
                np.random.shuffle(thin)
                thin = thin[0:thinning]
                w_row, w_col = np.where(A_sub == True)
                A_sub_thin[:,:] = False
                A_sub[w_row[thin], w_col[thin]] = True
            elif params['edge_thinning_scheme'] == 'prop':
                probs = edge_probabilities(alpha, beta, kappa, x_sub)
                probs = np.reshape(probs, (N_sub*N_sub,))
                obs = np.random.multinomial(thinning, probs / np.sum(probs))
                A_sub_thin[:,:] = np.reshape((obs > 0), (N_sub,N_sub))

        # Fit full model
        fit = infer(A_sub_thin, x_sub, fit_alpha = True)
        fit_nll = neg_log_likelihood(fit['alpha'], fit['beta'], kappa,
                                     A_sub, x_sub)
        print 'Fit subnetwork NLL: %.2f' % fit_nll
        estimate[0,0,num_fit] = fit['beta'][0]
        estimate[1,0,num_fit] = fit['beta'][1]

        # Fit model with zero alpha
        fit = infer(A_sub_thin, x_sub, fit_alpha = False)
        fit_nll = neg_log_likelihood(fit['alpha'], fit['beta'], kappa,
                                     A_sub, x_sub)
        print 'Fit subnetwork (zero alpha) NLL: %.2f' % fit_nll
        estimate[0,1,num_fit] = fit['beta'][0]
        estimate[1,1,num_fit] = fit['beta'][1]

    for metric, true_val in enumerate([beta[0], beta[1]]):
        for m in range(2):
            bias[metric,m,n] = np.mean(estimate[metric,m] - true_val)
            variance[metric,m,n] = np.var(estimate[metric,m])
bias_sq = bias ** 2
mse = variance + bias_sq

plt.figure()
for metric, (name, val) in enumerate([('beta_shank', beta[0]),
                                      ('beta_self', beta[1])]):
    plt.subplot(2,1,(metric+1))
    plt.plot(params['N_subs'], mse[metric,0], 'b')
    plt.plot(params['N_subs'], variance[metric,0], 'b--')
    plt.plot(params['N_subs'], mse[metric,1], 'r')
    plt.plot(params['N_subs'], variance[metric,1], 'r--')
    plt.plot(params['N_subs'], [val ** 2] * len(params['N_subs']), 'k:')
    plt.ylim(ymax = 1.2 * val ** 2)
    plt.title(name)
plt.show()
