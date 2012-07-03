#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.linear_model import LogisticRegression

# Parameters
params = { 'N': 200,
           'edge_precision': 1.0,
           'prior_precision': 0.1,
           'alpha_sd': 1.2,
           'beta_shank': 3.0,
           'num_shank': 8,
           'beta_self': 4.0,
           'kappa_base': -5.0,
           'kappa_coef_1': 0.0,
           'kappa_coef_2': 0.0,
           'N_subs': [10, 40, 70, 100, 130],
           'num_fits': 10,
           'logistic_fit_alpha': True }

# Set random seed for reproducible output
np.random.seed(137)

# Calculate edge means from parameters and covariates
def edge_means(alpha, beta, kappa, x):
    N = x.shape[0]
    mu = np.zeros((N,N))
    for i in range(N):
        mu[i,:] += alpha[0,i]
    for j in range(N):
        mu[:,j] += alpha[0,j]
    mu += np.dot(x, beta)
    mu += kappa
    return mu

# Inverse-logit
def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))

# Procedure to find posterior mean and covariance via Bayesian inference
def infer_normal(A, x):
    N = A.shape[0]
    B = x.shape[2]
    
    t = A.reshape((N*N,))
    Phi = np.zeros((N*N,(B + 1 + 2 * N_sub)))
    Phi_trans = np.transpose(Phi)
    for b in range(B):
        Phi[:,b] = x_sub[:,:,b].reshape((N*N,))
    Phi[:,B] = 1.0
    for i in range(N):
        phi_row = np.zeros((N,N))
        phi_row[i,:] = 1.0
        Phi[:,B + 1 + i] = phi_row.reshape((N*N,))
    for j in range(N_sub):
        phi_col = np.zeros((N,N))
        phi_col[:,j] = 1.0
        Phi[:,B + 1 + N + j] = phi_col.reshape((N*N,))
    S_N_inv = (params['prior_precision'] * np.eye(B + 1 + 2 * N) +
               params['edge_precision'] * np.dot(Phi_trans, Phi))
    S_N = np.linalg.inv(S_N_inv)
    m_N = params['edge_precision'] * np.dot(S_N, np.dot(Phi_trans, t))

    return m_N, S_N

# Procedure to find MLE via logistic regression
def infer_logistic(A, x, fit_alpha = False):
    N = A.shape[0]
    B = x.shape[2]

    lr = LogisticRegression(fit_intercept = True,
                            C = 1.0 / params['prior_precision'], penalty = 'l2')
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

    # Compute posterior covariance via Laplace approximation
    if fit_alpha:
        S_0_inv = params['prior_precision'] * np.eye(B + 1 + 2*N)
        Phi_kappa = np.empty((N*N,(B + 1 + 2*N)))
        Phi_kappa[:,(B + 1):(B + 1 + 2*N)] = Phi[:,B:(B + 2*N)]
        w = np.empty(B + 1 + 2*N)
        w[(B + 1):(B + 1 + 2*N)] = coefs[B:(B + 2*N)]
    else:
        S_0_inv = params['prior_precision'] * np.eye(B + 1)
        Phi_kappa = np.empty((N*N,(B + 1)))
        w = np.empty(B + 1)
    Phi_kappa[:,0:B] = Phi[:,0:B]
    Phi_kappa[:,B] = 1.0
    w[0:B] = coefs[0:B]
    w[B] = intercept
    C = 0.0
    for i in range(N*N):
        y = sigma(np.dot(w, Phi_kappa[i,:]))
        C += y * (1.0 - y) * (np.outer(Phi_kappa[i,:], Phi_kappa[i,:]))
    S_N = np.linalg.inv(S_0_inv + C)
    out['S_N'] = S_N
 
    return out

# Generate random network, using randomly generated latent parameters
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
    
    mu = edge_means(alpha_sub, beta, kappa_sub, x_sub)
    A_n_sub = np.random.normal(mu, np.sqrt(1.0 / params['edge_precision']))
    A_l_sub = np.random.random((n,n)) < sigma(mu)
    return A_n_sub, A_l_sub, x_sub

# Convenience functions for plotting
#
# Finding the right settings for Ellipse is surprisingly tricky so I follow:
#   http://scikit-learn.org/stable/auto_examples/plot_lda_qda.html
def make_axis(f, n, title):
    ax = f.add_subplot(2, len(params['N_subs']), (n+1), aspect = 'equal')
    ax.set_xlim(beta[0] - 2.0, beta[0] + 2.0)
    ax.set_ylim(beta[1] - 2.0, beta[1] + 2.0)
    ax.set_xlabel('beta_shank')
    ax.set_ylabel('beta_self')
    ax.set_title(title)
    return ax
def draw_ellipse(a, m, S):
    v, w = np.linalg.eigh(S)
    u = w[0] / np.linalg.norm(w[0])
    angle = (180.0 / np.pi) * np.arctan(u[1] / u[0])
    e = Ellipse(m, 2.0 * np.sqrt(v[0]), 2.0 * np.sqrt(v[1]),
                180.0 + angle, color = 'k')
    a.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.5)

# Fit model to subset of data, displaying beta posterior
fig = plt.figure()
inds = np.arange(params['N'])
for n, N_sub in enumerate(params['N_subs']):
    for num_fit in range(params['num_fits']):
        A_n_sub, A_l_sub, x_sub = subnetwork(N_sub)

        # Fit normal model
        m_N, S_N = infer_normal(A_n_sub, x_sub)
        ax = make_axis(fig, n, 'Normal (N_sub = %d)' % N_sub)
        draw_ellipse(ax, m_N[0:2], S_N[0:2,0:2])

        # Fit logistic model
        fit = infer_logistic(A_l_sub, x_sub, params['logistic_fit_alpha'])
        ax = make_axis(fig, len(params['N_subs']) + n, 'Logistic')
        draw_ellipse(ax, fit['beta'], fit['S_N'][0:2,0:2])

# Display all pending graphs
plt.show()
