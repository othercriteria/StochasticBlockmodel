#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from os import system

# Parameters
params = { 'output_folder': 'figs' }


# Wrapper for common GraphViz output
def graphviz(file_stem, lines, engine = 'neato'):
    dot_name = '%s/%s.dot' % (params['output_folder'], file_stem)
    ps_name = '%s/%s.ps' % (params['output_folder'], file_stem)
    pdf_name = '%s/%s.pdf' % (params['output_folder'], file_stem)
    outfile = open(dot_name, 'w')
    outfile.write('digraph G {\n')
    outfile.write('size="6,8";\n')
    outfile.write('orientation=landscape;\n')
    outfile.writelines(lines)
    outfile.write('}\n')
    outfile.close()
    system('%s -Tps2 %s -o %s' % (engine, dot_name, ps_name))
    system('ps2pdf %s %s' % (ps_name, pdf_name))

# Calculate negative log-likelihood
def neg_log_likelihood(A, z, Theta, alpha, beta, x):
    N = A.shape[0]
    logit_P = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            logit_P[i,j] = (Theta[z[i],z[j]] +
                            alpha[0,i] + alpha[1,j] +
                            np.dot(x[i,j], beta))
    P = 1.0 / (np.exp(-logit_P) + 1.0)
    return -np.sum(np.log(P ** A) + np.log((1.0 - P) ** (1.0 - A)))

# Use SEM to infer a stochastic blockmodel fit
def infer_block(A, x, K = 1,
                steps = 10, sweeps = 5, C = 1.0,
                zero_alpha = False,
                true_z = None, init_z = None,
                init_theta = None, init_alpha = None, init_beta = None):
    N = A.shape[0]
    B = x.shape[2]

    if not init_theta is None:
        Theta = init_theta.copy()
    else:
        Theta = np.zeros((K,K))

    if not init_alpha is None:
        alpha = init_alpha.copy()
    else:
        alpha = np.zeros((2,N))

    if not init_beta is None:
        beta = init_beta.copy()
    else:
        beta = np.zeros(B)
        
    if not true_z is None:
        z = true_z.copy()
    elif not init_z is None:
        z = init_z.copy()
    else:
        z = np.random.randint(0, K, N)
    
    for step in range(steps):
        # Stochastic E-step
        if true_z is None:
            for gibbs_step in range(sweeps*N):
                l = np.random.randint(N)
                logprobs = np.empty(K)
                for k in range(K):
                    logprobs[k] = (np.dot(Theta[z,k], A[:,l]) +
                                   np.dot(Theta[k,z], A[l,:]) -
                                   (Theta[k,k] * A[l,l]))
                logprobs -= np.max(logprobs)
                probs = np.exp(logprobs)
                probs /= np.sum(probs)
                z[l] = np.where(np.random.multinomial(1, probs) == 1)[0][0]
        init_nll = neg_log_likelihood(A, z, Theta, alpha, beta, x)
    
        # M-step
        if zero_alpha:
            lr = LogisticRegression(C = C, penalty = 'l2')
            y = A.reshape((N*N,))
            X = np.zeros((N*N,(K**2 + B)))
            for i in range(N):
                for j in range(N):
                    X[N*i + j, K*z[i] + z[j]] = 1.0
            for b in range(B):
                X[:,K**2 + b] = x[:,:,b].reshape((N**2,))

            lr.fit(X, y)
            fit = lr.coef_[0]
        
            Theta = fit[0:(K**2)].reshape((K,K))
            beta = fit[(K**2):(K**2 + B)]
        else:
            lr = LogisticRegression(C = C, penalty = 'l2')
            y = A.reshape((N*N,))
            X = np.zeros((N*N,(K**2 + 2*N + B)))
            for i in range(N):
                for j in range(N):
                    X[N*i + j, K*z[i] + z[j]] = 1.0
            for i in range(N):
                x_row = np.zeros((N,N))
                x_row[i,:] = 1.0
                X[:,K**2 + i] = x_row.reshape((N**2,))
            for j in range(N):
                x_col = np.zeros((N,N))
                x_col[:,j] = 1.0
                X[:,K**2 + N + j] = x_col.reshape((N**2,))
            for b in range(B):
                X[:,K**2 + 2*N + b] = x[:,:,b].reshape((N**2,))

            lr.fit(X, y)
            fit = lr.coef_[0]
            # print 'alpha-out mean: %.2f' % np.mean(fit[K**2:(K**2 + N)])
            # print 'alpha-in mean: %.2f' % np.mean(fit[(K**2 + N):(K**2 + 2*N)])
        
            Theta = fit[0:(K**2)].reshape((K,K))
            alpha[0] = fit[(K**2):(K**2 + N)]
            alpha[1] = fit[(K**2 + N):(K**2 + 2*N)]
            beta = fit[(K**2 + 2*N):(K**2 + 2*N + B)]

    return { 'z': z, 'Theta': Theta, 'alpha': alpha, 'beta': beta }

# Plot block assignments
def plot_block(file_name, A, z, K, labels = None, alpha = None):
    N = A.shape[0]
    
    lines = []
    for block in range(K):
        lines.append('subgraph cluster_%d {' % block)
        lines.append('label = "block %d";' % block)
        for n in range(N):
            if z[n] == block:
                name = n
                if not labels is None:
                    name = labels[n]
                if not alpha is None:
                    lines.append('%d [label="%d\\n[%.2f,%.2f]"];' %
                                 (name, name, alpha[0,n], alpha[1,n]))
                else:
                    lines.append('%d;' % name)
        lines.append('}')
    for i in range(N):
        for j in range(N):
            if A[i,j]:
                if not labels is None:
                    lines.append('%d -> %d;' % (labels[i],labels[j]))
                else:
                    lines.append('%d -> %d;' % (i,j))
    graphviz(file_name, lines, engine = 'fdp')
