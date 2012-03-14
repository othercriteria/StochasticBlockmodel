#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Parameters
params = { 'N': 200,
           'edge_precision': 1.0,
           'prior_precision': 1.0,
           'alpha_sd': 20.0,
           'beta_shank': 3.0,
           'num_shank': 8,
           'beta_self': 4.0,
           'kappa': -5.0,
           'N_subs': [5, 20, 50, 100],
           'num_fits': 10,
           'plot_heatmap': False }

# Set random seed for reproducible output
np.random.seed(137)

# Calculate edge means from parameters and covariates
def edge_means(alpha, beta, kappa, x):
    N = x.shape[0]
    mu = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            mu[i,j] = alpha[0,i] + alpha[1,j] + np.dot(x[i,j], beta)
    mu += kappa
    return mu

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
kappa = params['kappa']
mu = edge_means(alpha, beta, kappa, x)
A = np.random.normal(mu, np.sqrt(1.0 / params['edge_precision']))

# Show heatmap of the underlying network
if params['plot_heatmap']:
    plt.figure()
    plt.imshow(A)
    plt.title('Unordered')

    plt.figure()
    o = np.argsort(shank)
    plt.imshow(A[o][:,o])
    plt.title('Grouped by shank')

    plt.figure()
    o = np.argsort(alpha[0])
    plt.imshow(A[o][:,o])
    plt.title('Ordered by alpha_out')

    plt.figure()
    o = np.argsort(alpha[1])
    plt.imshow(A[o][:,o])
    plt.title('Ordered by alpha_in')

# Fit model to subset of data, displaying beta posterior
fig = plt.figure()
inds = np.arange(params['N'])
for n, N_sub in enumerate(params['N_subs']):
    ax = fig.add_subplot(1, len(params['N_subs']), (n+1), aspect = 'equal')
    ax.set_xlim(beta[0] - 2.0, beta[0] + 2.0)
    ax.set_ylim(beta[1] - 2.0, beta[1] + 2.0)
    ax.set_xlabel('beta_shank')
    ax.set_xlabel('beta_self')
    ax.set_title('N_sub = %d' % N_sub)
    for num_fit in range(params['num_fits']):
        np.random.shuffle(inds)
        sub = inds[0:N_sub]

        # Sample subnetwork
        A_sub = A[sub][:,sub]
        x_sub = x[sub][:,sub]
        alpha_sub = alpha[:,sub]

        # Fit model
        t = A_sub.reshape((N_sub*N_sub,))
        Phi = np.zeros((N_sub*N_sub,(2 + 1 + 2 * N_sub)))
        Phi_trans = np.transpose(Phi)
        for b in range(2):
            Phi[:,b] = x_sub[:,:,b].reshape((N_sub*N_sub,))
        Phi[:,2] = 1.0
        for i in range(N_sub):
            phi_row = np.zeros((N_sub,N_sub))
            phi_row[i,:] = 1.0
            Phi[:,2 + 1 + i] = phi_row.reshape((N_sub*N_sub,))
        for j in range(N_sub):
            phi_col = np.zeros((N_sub,N_sub))
            phi_col[:,j] = 1.0
            Phi[:,2 + 1 + N_sub + j] = phi_col.reshape((N_sub*N_sub,))
        S_N_inv = (params['prior_precision'] * np.eye(2 + 1 + 2 * N_sub) +
                   params['edge_precision'] * np.dot(Phi_trans, Phi))
        S_N = np.linalg.inv(S_N_inv)
        m_N = params['edge_precision'] * np.dot(S_N, np.dot(Phi_trans, t))

        # Plot posterior. Finding the right settings for Ellipse is
        # surprisingly tricky so I follow:
        #  http://scikit-learn.org/stable/auto_examples/plot_lda_qda.html
        S_N_beta = S_N[0:2,0:2]
        v, w = np.linalg.eigh(S_N_beta)
        u = w[0] / np.linalg.norm(w[0])
        angle = (180.0 / np.pi) * np.arctan(u[1] / u[0])
        e = Ellipse(m_N[0:2], 2.0 * np.sqrt(v[0]), 2.0 * np.sqrt(v[1]),
                    180.0 + angle, color = 'k')
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.5)

# Display all pending graphs
plt.show()
