#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from os import system

# Parameters
params = { 'epsilon_min': 0.0001,
           'output_folder': 'figs' }


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
                steps = 10, sweeps = 5, miters = 50, epsilon = 0.1, term = 0.05,
                zero_alpha = False, zero_beta = False,
                true_z = None, init_z = None,
                init_theta = None, init_alpha = None, init_beta = None):
    N = A.shape[0]

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
        beta = np.zeros(x.shape[2])
        
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
        T_theta = np.zeros((K,K))
        T_beta = np.zeros(x.shape[2])
        for i in range(N):
            a = z[i]
            for j in range(N):
                b = z[j]
                if A[i,j]:
                    T_theta[a,b] += 1.0
                    T_beta += x[i,j]

        T_alpha = np.zeros((2,N))
        T_alpha[0] = np.sum(A, axis = 1)
        T_alpha[1] = np.sum(A, axis = 0)
        
        for miter in range(miters):
            ET_theta = np.zeros((K,K))
            ET_alpha = np.zeros((2,N))
            ET_beta = np.zeros(x.shape[2])
            for i in range(N):
                a = z[i]
                for j in range(N):
                    b = z[j]
                    logit_P = (Theta[a,b] +
                               alpha[0,i] + alpha[1,j] +
                               np.dot(x[i,j], beta))
                    P = 1.0 / (np.exp(-logit_P) + 1.0)
                    ET_theta[a,b] += P
                    ET_alpha[0,i] += P
                    ET_alpha[1,j] += P
                    ET_beta += P * x[i,j]

            grad_theta = T_theta - ET_theta
            grad_alpha = T_alpha - ET_alpha
            grad_beta = T_beta - ET_beta

            # Adaptively set scale of gradient moves
            if miter == 0:
                curr_epsilon = np.min([epsilon,
                                       1.0 / np.max(np.abs(grad_theta)),
                                       1.0 / np.max(np.abs(grad_alpha)),
                                       1.0 / np.max(np.abs(grad_beta))])
                while True:
                    new_nll = \
                        neg_log_likelihood(A, z,
                                           Theta + curr_epsilon * grad_theta,
                                           alpha + curr_epsilon * grad_alpha,
                                           beta + curr_epsilon * grad_beta,
                                           x)
                    if new_nll < init_nll or \
                        curr_epsilon < params['epsilon_min']:
                        break
                    curr_epsilon /= 2.0
            
            diff_theta = curr_epsilon * grad_theta
            diff_alpha = curr_epsilon * grad_alpha
            diff_beta = curr_epsilon * grad_beta

            max_diff = np.max(np.abs(diff_theta))
            Theta += diff_theta
            if not zero_alpha:
                max_diff = max(max_diff, np.max(np.abs(diff_alpha)))
                alpha += diff_alpha
                alpha_mean = np.mean(alpha, axis = 1)
                alpha[0] -= alpha_mean[0]
                alpha[1] -= alpha_mean[1]
                Theta += (alpha_mean[0] + alpha_mean[1])
            if not zero_beta:
                max_diff = max(max_diff, np.max(np.abs(diff_beta)))
                beta += diff_beta

            if max_diff < term:
                break

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
