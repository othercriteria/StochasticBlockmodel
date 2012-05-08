#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scikits.statsmodels.api as sm
import scipy.optimize as opt
import scipy.linalg as la

# Parameters
params = { 'N': 400,
           'alpha_sd': 0.0,
           'alpha_unif': 0.0,
           'B': 5,
           'beta_sd': 1.0,
           'x_discrete': False,
           'target': ('density', 0.2),
           'N_subs': range(15, 60, 5),
           'num_fits': 25,
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
def nll(alpha, beta, kappa, A, x):
    P = edge_probabilities(alpha, beta, kappa, x)
    return -np.sum(np.log(P ** A) + np.log((1.0 - P) ** (1.0 - A)))

# Numerical search for kappa to provide the right expected graph properties
def target_to_kappa(d, alpha, beta, x):
    target, val = d
    N = x.shape[0]
    def obj(kappa):
        exp_edges = np.sum(edge_probabilities(alpha, beta, kappa, x))
        if target == 'degree':
            exp_degree = exp_edges / (1.0 * N)
            return abs(exp_degree - val)
        elif target == 'density':
            exp_density = exp_edges / (1.0 * N ** 2)
            return abs(exp_density - val)
    return opt.golden(obj)

# Procedure to find MLE via logistic regression
def infer(A, x, fit_alpha = False):
    N = A.shape[0]
    B = x.shape[2]

    if fit_alpha:
        # (Separately) sort rows and columns of A by increasing sums
        r_ord = np.argsort(np.sum(A, axis = 1))
        c_ord = np.argsort(np.sum(A, axis = 0))
        A = A[r_ord][:,c_ord]
        x = x[r_ord][:,c_ord]
        
        # Recursively examine for submatrices that will send
        # corresponding EMLE parameter estimates to infinity and
        # assemble list of "active" submatrices to retain
        to_screen = [(np.arange(N), np.arange(N))]
        act = []
        while len(to_screen) > 0:
            r_act, c_act = to_screen.pop()

            A_act = A[r_act][:,c_act]
            n_act = A_act.shape
            violation = False
            trivial = [(0,0), (0,n_act[1]), (n_act[0],0), (n_act[0],n_act[1])]
            for i,j in [(i,j)
                        for i in range(n_act[0] + 1)
                        for j in range(n_act[1] + 1)]:
                if (i,j) in trivial: continue
                if np.any(A_act[:i][:,:j]): continue
                if not np.all(A_act[i:][:,j:]): continue
                if i > 0 and j < n_act[1]:
                    A_sub = A_act[:i][:,j:]
                    if (np.any(A_sub) and (not np.all(A_sub))):
                        to_screen.append((r_act[np.arange(i)],
                                          c_act[np.arange(j, n_act[1])]))
                if i < n_act[0] and j > 0:
                    A_sub = A_act[i:][:,:j]
                    if (np.any(A_sub) and (not np.all(A_sub))):
                        to_screen.append((r_act[np.arange(i, n_act[0])],
                                          c_act[np.arange(j)]))
                violation = True
                break

            if not violation:
                act.append((r_act, c_act))

        if len(act) == 0:
            return { 'beta': np.zeros(B), 'N_act_r': 0, 'N_act_c': 0 }

        # Calculate size of design matrix and outcome vector, then
        # construct them
        N_act_r, N_act_c = 0, B
        for r_act, c_act in act:
            N_act_r += len(r_act) * len(c_act)
            N_act_c += len(r_act) + len(c_act)
        y = np.zeros((N_act_r,))
        Phi = np.zeros((N_act_r, N_act_c))
        i_offset, j_offset = 0, B
        for r_act, c_act in act:
            i_inc = len(r_act) * len(c_act)
            j_inc_r, j_inc_c = len(r_act), len(c_act)

            A_act = A[r_act][:,c_act]
            x_act = x[r_act][:,c_act]

            y[i_offset:(i_offset + i_inc)] = A_act.reshape((i_inc,))

            for b in range(B):
                Phi[i_offset:(i_offset + i_inc), b] = \
                    x_act[:,:,b].reshape((i_inc,))

            for r in range(j_inc_r):
                phi_row = np.zeros((j_inc_r,j_inc_c))
                phi_row[r,:] = 1.0
                Phi[i_offset:(i_offset + i_inc), j_offset + r] = \
                    phi_row.reshape((i_inc,))
            j_offset += j_inc_r

            for c in range(j_inc_c):
                phi_col = np.zeros((j_inc_r,j_inc_c))
                phi_col[:,c] = 1.0
                Phi[i_offset:(i_offset + i_inc), j_offset + c] = \
                    phi_col.reshape((i_inc,))
            j_offset += j_inc_c
            
            i_offset += i_inc

        try:
            coefs = sm.Logit(y, Phi).fit().params
        except:
            print y
            print Phi
            print Phi.shape
            return { 'beta': np.zeros(B), 'N_act_r': 0, 'N_act_c': 0 }
        return { 'beta': coefs[0:B], 'N_act_r': N_act_r, 'N_act_c': N_act_c }
        
    else:
        y = A.reshape((N*N,))
        Phi = np.zeros((N*N,B + 1))
        Phi[:,B] = 1.0
        for b in range(B):
            Phi[:,b] = x[:,:,b].reshape((N*N,))
        coefs = sm.Logit(y, Phi).fit().params
        return { 'beta': coefs[0:B], 'N_act_r': N*N, 'N_act_c': (B + 1) }
    
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
beta = np.random.normal(0, params['beta_sd'], params['B'])
if params['x_discrete']:
    x = 2 * np.random.binomial(1, 0.5, (params['N'], params['N'], params['B']))
else:
    x = np.random.normal(0, 1, (params['N'], params['N'], params['B']))

# Procedure for generating random subnetwork
def subnetwork(n):
    inds = np.arange(params['N'])
    np.random.shuffle(inds)
    sub = inds[0:n]
    
    alpha_sub = alpha[:,sub]
    x_sub = x[sub][:,sub]
    kappa_sub = target_to_kappa(params['target'], alpha_sub, beta, x_sub)
    
    P_sub = edge_probabilities(alpha_sub, beta, kappa_sub, x_sub)
    A_sub = np.random.random((n,n)) < P_sub
    return alpha_sub, kappa_sub, A_sub, x_sub

# Some pre-analysis output
print 'Parameters:'
for field in params:
    print '%s: %s' % (field, str(params[field]))

# Fit model on partially observed subnetworks and assess performance
bias = np.empty((params['B'],2,len(params['N_subs'])))
variance = np.empty((params['B'],2,len(params['N_subs'])))
act_ratio = np.empty((len(params['N_subs']),params['num_fits']))
network = np.empty((4,len(params['N_subs'])))
kappas = np.empty((len(params['N_subs']),params['num_fits']))
for n, N_sub in enumerate(params['N_subs']):
    print 'N_sub = %d' % N_sub
    estimate = np.empty((params['B'],2,params['num_fits']))
    network_obs = np.empty((4,params['num_fits']))
    for num_fit in range(params['num_fits']):
        alpha_sub, kappa_sub, A_sub, x_sub = subnetwork(N_sub)
        print 'kappa_sub = %.2f' % kappa_sub
                
        network_obs[0,num_fit] = 1.0 * np.sum(A_sub) / N_sub
        network_obs[1,num_fit] = np.max(np.sum(A_sub, axis = 1))
        network_obs[2,num_fit] = np.max(np.sum(A_sub, axis = 0))
        network_obs[3,num_fit] = 1.0 * np.sum(np.diagonal(A_sub)) / N_sub
        kappas[n,num_fit] = kappa_sub
                
        if params['do_inference']:
            sub_nll = nll(alpha_sub, beta, kappa_sub, A_sub, x_sub)
            print 'True subnetwork NLL: %.2f' % sub_nll

            # Fit full model
            fit = infer(A_sub, x_sub, fit_alpha = True)
            for b in range(params['B']):
                estimate[b,0,num_fit] = fit['beta'][b]
            if fit['N_act_r'] > 0:
                act_ratio[n,num_fit] = fit['N_act_r'] / fit['N_act_c']
            else:
                act_ratio[n,num_fit] = 0

            # Fit model with zero alpha, i.e., stationary model
            fit = infer(A_sub, x_sub, fit_alpha = False)
            for b in range(params['B']):
                estimate[b,1,num_fit] = fit['beta'][b]
                    
    if params['do_inference']:
        for b, true_val in enumerate(beta):
            for m in range(2):
                bias[b,m,n] = np.mean(estimate[b,m] - true_val)
                variance[b,m,n] = np.var(estimate[b,m])
    for metric in range(4):
        network[metric,n] = np.mean(network_obs[metric])
if params['do_inference']:
    bias_sq = bias ** 2
    mse = variance + bias_sq

# Plot inference performace, as well as data/parameter ratio
if params['do_inference']:
    plt.figure()
    plt.subplot(2,1,1)
    for b, true_val in enumerate(beta):
        plt.plot(params['N_subs'], mse[b,0] / abs(true_val), 'b')
        plt.plot(params['N_subs'], mse[b,1] / abs(true_val), 'r')
        plt.ylim(0, 3.0)
        plt.title('Scaled MSE of beta')

    plt.subplot(2,1,2)
    for n in range(params['num_fits']):
        plt.plot(params['N_subs'], act_ratio[:,n], 'k.', hold = True)
    plt.ylabel('N_act_r / N_act_c')

# Plot network statistics as well as sparsity parameter
plt.figure()
for metric, name in enumerate(['average degree',
                               'max out-degree',
                               'max in-degree',
                               'self-edge probability']):
    plt.subplot(5,1,(metric+1))
    plt.plot(params['N_subs'], network[metric], '-')
    plt.ylabel(name)
    plt.ylim(ymin = 0)
plt.subplot(5,1,5)
for n in range(params['num_fits']):
    plt.plot(params['N_subs'], kappas[:,n], 'k.', hold = True)
plt.ylabel('kappa_sub')

plt.show()


