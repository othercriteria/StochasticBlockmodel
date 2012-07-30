#!/usr/bin/env python

# Models for generating and fitting networks
# Daniel Klein, 5/11/2012

from __future__ import division
import numpy as np
import scipy.optimize as opt
from time import time

from Utility import logit, inv_logit, logit_mean
from BinaryMatrix import arbitrary_from_margins, approximate_from_margins_weights

# It's a bit weird to have NonstationaryLogistic as a subclass of strictly
# less general StationaryLogistic model, but I believe this is how
# inheritance is supposed to work in Python.

# Some of the inference routines and the Kappa picking routine rely on
# a side-effect (modifying an instance variable of the enclosing
# class) inside the objective function. This is inelegant and seems
# error-prone, and so should probably be fixed...

# P_{ij} = Logit^{-1}(o_{ij})
class IndependentBernoulli:
    def edge_probabilities(self, network):
        N = network.N

        if network.offset:
            logit_P = network.offset.matrix()
            return inv_logit(logit_P)
        else:
            return np.tile(0.5, (N,N))
    
    def nll(self, network):
        P = self.edge_probabilities(network)
        A = network.adjacency_matrix()

        # Check for impossible data for the cells with 0/1 probabilities
        ind_P_zero, ind_P_one = (P == 0.0), (P == 1.0)
        if np.any(A[ind_P_zero]) or not np.all(A[ind_P_one]):
            return np.Inf

        # Compute the negative log-likelihood for the rest of the cells
        ind_P_rest = -(ind_P_zero + ind_P_one)
        P, A = P[ind_P_rest], A[ind_P_rest]
        nll = -np.sum(np.log(P ** A) + np.log((1.0 - P) ** (1.0 - A)))

        # Edge cases and numerical weirdness can occur; better to pass
        # on infinity that at least carries the sign of the blowup
        if np.isnan(nll):
            return np.Inf
        return nll

    def generate(self, network):
        N = network.N
        
        P = self.edge_probabilities(network)
        return np.random.random((N,N)) < P

    # Generate sample (approximately) from the conditional
    # distribution with fixed margins.
    #
    # There's no obvious stopping criterion for the Gibbs sampler used
    # to generate random networks conditioned on the degree sequence,
    # i.e., the adjacency matrix margins. As a heuristic, I let the
    # user specify a degree of "cover" and then ensure that the
    # sampler makes a non-trivial proposal on average "cover" many
    # times concerning each possible edge.
    #
    # Initializing the Gibbs sampler from the approximate conditional
    # distribution requires a bit more time initially but takes many
    # fewer steps to reach the stationary distribution, so it is
    # enabled by default.
    def generate_margins(self, network, r, c,
                         coverage = 1, arbitrary_init = False):
        N = network.N
        windows = N // 2
        coverage_target = coverage * N**2 / 4

        # Precomputing for diagonal/anti-diagonal check in Gibbs sampler
        diag = np.array([[True,False],[False,True]])
        adiag = np.array([[False,True],[True,False]])
        valid = set([diag.data[0:4], adiag.data[0:4]])
        
        if arbitrary_init:
            # Initialize from an arbitrary matrix with the requested margins
            gen = arbitrary_from_margins(r, c)
        else:
            # Initialize from an approximate sample from the
            # conditional distribution
            p = self.edge_probabilities(network)
            w = p / (1.0 - p)
            gen_sparse = approximate_from_margins_weights(r, c, w)
            gen = np.zeros((N,N), dtype=np.bool)
            for i, j in gen_sparse:
                if i == -1: break 
                gen[i,j] = 1

        # Gibbs sampling to match the "location" of the edges to where
        # they are likely under the conditional distribution
        #
        # Scheduling Gibbs sweeps in a checkerboard-like manner to
        # simplify picking distinct random indices.
        P_full = self.edge_probabilities(network)
        coverage_attained = 0
        inds = np.arange(N)
        while coverage_attained < coverage_target:
            # Pick i-ranges and j-ranges of the randomly chosen 2x2
            # subnetworks in which to propose moves
            np.random.shuffle(inds)
            i_props = [inds[2*window:2*(window+1)] for window in range(windows)]
            np.random.shuffle(inds)
            j_props = [inds[2*window:2*(window+1)] for window in range(windows)]

            active = []
            for i_prop in i_props:
                gen_prop_i = gen[i_prop]
                for j_prop in j_props:
                    gen_prop = gen_prop_i[:,j_prop]

                    # Margin-preserving moves only possible if
                    # n_proposal is diagonal or anti-diagonal
                    if not (gen_prop.data[0:4] in valid): continue
                    active.append(np.ix_(i_prop, j_prop))
            A = len(active)
            coverage_attained += A

            # Calculate individual edge probabilities
            P = np.empty((2*A,2))
            for a, ij_prop in enumerate(active):
                P[2*a:2*(a+1)] = P_full[ij_prop]

            # Normalize probabilities of allowed configurations to get
            # conditional probabilities
            l_diag, l_adiag = np.empty(A), np.empty(A)
            for a in range(A):
                P_a = P[2*a:2*(a+1)]
                l_diag[a] = P_a[0,0] * P_a[1,1] * (1-P_a[0,1]) * (1-P_a[1,0])
                l_adiag[a] = P_a[0,1] * P_a[1,0] * (1-P_a[0,0]) * (1-P_a[1,1])
            p_diag = l_diag / (l_diag + l_adiag)

            # Update n according to calculated probabilities
            to_diags = np.random.random(A) < p_diag
            for to_diag, ij_prop in zip(to_diags, active):
                if to_diag:
                    gen[ij_prop] = diag
                else:
                    gen[ij_prop] = adiag

        return gen

# P_{ij} = Logit^{-1}(kappa + o_{ij})
class Stationary(IndependentBernoulli):
    def __init__(self):
        self.kappa = 0.0
        self.fit = self.fit_convex_opt

    def edge_probabilities(self, network):
        N = network.N

        if network.offset:
            logit_P = network.offset.matrix().copy()
        else:
            logit_P = np.zeros((N,N))
        logit_P += self.kappa

        return inv_logit(logit_P)

    def match_kappa(self, network, kappa_target):
        N = network.N

        # kappa_target should be a tuple of the following form:
        #  ('edges', 0 < x)
        #  ('degree', 0 < x)
        #  ('density', 0 < x < 1) 
        target, val = kappa_target
        def obj(kappa):
            self.kappa = kappa
            P = self.edge_probabilities(network)
            exp_edges = np.sum(P)
            if target == 'edges':
                return abs(exp_edges - val)
            elif target == 'degree':
                exp_degree = exp_edges / (1.0 * N)
                return abs(exp_degree - val)
            elif target == 'density':
                exp_density = exp_edges / (1.0 * N ** 2)
                return abs(exp_density - val)
        self.kappa = opt.golden(obj)

    # Mechanical specialization of the StationaryLogistic code, so not
    # really ideal for this one-dimensional problem.
    def fit_convex_opt(self, network, verbose = False):
        self.fit_info = { 'obj_evals': 0, 'grad_evals': 0,
                          'grad_final': np.empty(1) }
        start_time = time()
        
        # Calculate observed sufficient statistic
        T = np.empty(1)
        A = network.adjacency_matrix()
        T[0] = np.sum(A, dtype=np.int)

        theta = np.empty(1)
        theta[0] = logit(A.sum(dtype=np.int) / network.N ** 2)
        if network.offset:
            theta[0] -= logit_mean(network.offset.matrix())
        def obj(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing objective for nan-containing vector.'
                return np.Inf
            self.kappa = theta[0]
            nll = self.nll(network)
            self.fit_info['obj_evals'] += 1
            return nll
        def grad(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing gradient for nan-containing vector.'
                return np.zeros(1)
            self.kappa = theta[0]
            ET = np.empty(1)
            P = self.edge_probabilities(network)
            ET[0] = np.sum(P)
            grad = ET - T
            self.fit_info['grad_evals'] += 1
            self.fit_info['grad_final'][:] = grad
            if verbose:
                print '|ET - T|: %.2f' % abs(grad[0])
            return grad

        bounds = [(-15,15)]
        theta_opt = opt.fmin_l_bfgs_b(obj, theta, grad, bounds = bounds)[0]
        self.kappa = theta_opt[0]

        self.fit_info['wall_time'] = time() - start_time

    def fit_logistic(self, network):
        import statsmodels.api as sm

        N = network.N

        y = network.adjacency_matrix().reshape((N*N,))
        Phi = np.zeros((N*N,1))
        Phi[:,0] = 1.0
        if network.offset:
            offset = network.offset.matrix().reshape((N*N,))
            coefs = sm.GLM(y, Phi, sm.families.Binomial(), offset).fit().params
        else:
            coefs = sm.Logit(y, Phi).fit().params

        self.kappa = coefs[0]

# P_{ij} = Logit^{-1}(\sum_b x_{bij}*beta_b + kappa + o_{ij}) 
class StationaryLogistic(Stationary):
    def __init__(self):
        Stationary.__init__(self)
        self.beta = {}
        self.fit = self.fit_convex_opt

    def edge_probabilities(self, network):
        N = network.N
        
        if network.offset:
            logit_P = network.offset.matrix().copy()
        else:
            logit_P = np.zeros((N,N))            
        for b in self.beta:
            logit_P += self.beta[b] * network.edge_covariates[b].matrix()
        logit_P += self.kappa

        return inv_logit(logit_P)

    def fit_convex_opt(self, network, verbose = False):
        B = len(self.beta)

        self.fit_info = { 'obj_evals': 0, 'grad_evals': 0,
                          'grad_final': np.empty(B + 1) }
        start_time = time()

        # Calculate observed sufficient statistics
        T = np.empty(B + 1)
        A = np.array(network.adjacency_matrix())
        for b, b_n in enumerate(self.beta):
            T[b] = np.sum(A * network.edge_covariates[b_n].matrix())
        T[B] = np.sum(A, dtype=np.int)

        # Initialize theta
        theta = np.zeros(B + 1)
        theta[B] = logit(A.sum(dtype=np.int) / network.N ** 2)
        if network.offset:
            theta[B] -= logit_mean(network.offset.matrix())

        def obj(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing objective for nan-containing vector.'
                return np.Inf
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            self.kappa = theta[B]
            nll = self.nll(network)
            self.fit_info['obj_evals'] += 1
            return nll
        def grad(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing gradient for nan-containing vector.'
                return np.zeros(B + 1)
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            self.kappa = theta[B]
            ET = np.empty(B + 1)
            P = self.edge_probabilities(network)
            for b, b_n in enumerate(self.beta):
                ET[b] = np.sum(P * network.edge_covariates[b_n].matrix())
            ET[B] = np.sum(P)
            grad = ET - T
            self.fit_info['grad_evals'] += 1
            self.fit_info['grad_final'][:] = grad
            if verbose:
                abs_grad = np.abs(ET - T)
                print '|ET - T|: %.2f, %.2f, %.2f (min, mean, max)' % \
                    (np.min(abs_grad), np.mean(abs_grad), np.max(abs_grad))
            return grad

        bounds = [(-8,8)] * B + [(-15,15)]
        theta_opt = opt.fmin_l_bfgs_b(obj, theta, grad, bounds = bounds)[0]
        if (np.any(theta_opt == [b[0] for b in bounds]) or
            np.any(theta_opt == [b[1] for b in bounds])):
            print 'Warning: some constraints active in model fitting.'
        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = theta_opt[b]
        self.kappa = theta_opt[B]

        self.fit_info['wall_time'] = time() - start_time

    def fit_logistic(self, network):
        import statsmodels.api as sm

        N = network.N
        B = len(self.beta)

        y = network.adjacency_matrix().reshape((N*N,))
        Phi = np.zeros((N*N,B + 1))
        Phi[:,B] = 1.0
        for b, b_n in enumerate(self.beta):
            Phi[:,b] =  network.edge_covariates[b_n].matrix().reshape((N*N,))
        if network.offset:
            offset = network.offset.matrix().reshape((N*N,))
            coefs = sm.GLM(y, Phi, sm.families.Binomial(), offset).fit().params
        else:
            coefs = sm.Logit(y, Phi).fit().params

        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = coefs[b]
        self.kappa = coefs[B]

    # Doing this fit endows the model with a posterior
    # variance/covariance matrix estimated via the Laplace
    # approximation (see Bishop for details)
    def fit_logistic_l2(self, network, prior_precision = 1.0,
                        variance_covariance = False):
        if network.offset:
            print 'Regularized logistic regression with offset not supported.'
            raise
        
        from sklearn.linear_model import LogisticRegression

        N = network.N
        B = len(self.beta)

        y = network.adjacency_matrix().reshape((N*N,))
        Phi = np.zeros((N*N,B))
        for b, b_n in enumerate(self.beta):
            Phi[:,b] =  network.edge_covariates[b_n].matrix().reshape((N*N,))
        lr = LogisticRegression(fit_intercept = True, C = 1.0 / prior_precision,
                                penalty = 'l2')
        lr.fit(Phi, y)
        coefs, intercept = lr.coef_[0], lr.intercept_[0]

        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = coefs[b]
        self.kappa = intercept

        if variance_covariance:
            S_0_inv = prior_precision * np.eye(B + 1)
            Phi_kappa = np.hstack([Phi, np.ones((N*N,1))])
            w = np.empty(B + 1)
            w[0:B] = coefs
            w[B] = intercept
            C = 0.0
            for i in range(N*N):
                y = inv_logit(np.dot(w, Phi_kappa[i,:]))
                C += y * (1.0 - y) * (np.outer(Phi_kappa[i,:], Phi_kappa[i,:]))
            S_N = np.linalg.inv(S_0_inv + C)

            # Convert variance/covariance matrix into a dictionary
            # indexed by pairs of parameters
            self.variance_covariance = {}
            parameters = self.beta.keys() + ['kappa']
            for r, p_1 in enumerate(parameters):
                for s, p_2 in enumerate(parameters):
                    self.variance_covariance[(p_1,p_2)] = S_N[r,s]

# P_{ij} = Logit^{-1}(alpha_out_i + alpha_in_j + \sum_b x_{bij}*beta_b + kappa +
#                     o_{ij})
# Constraints: \sum_i alpha_out_i = 0, \sum_j alpha_in_j = 0
class NonstationaryLogistic(StationaryLogistic):
    def __init__(self):
        StationaryLogistic.__init__(self)
        self.fit = self.fit_convex_opt
        
    def edge_probabilities(self, network):
        N = network.N
        alpha_out = network.node_covariates['alpha_out']
        alpha_in = network.node_covariates['alpha_in']
        
        if network.offset:
            logit_P = network.offset.matrix().copy()
        else:
            logit_P = np.zeros((N,N))
        for i in range(N):
            logit_P[i,:] += alpha_out[i]
        for j in range(N):
            logit_P[:,j] += alpha_in[j]
        for b in self.beta:
            logit_P += self.beta[b] * network.edge_covariates[b].matrix()
        logit_P += self.kappa

        return inv_logit(logit_P)

    def fit_convex_opt(self, network, verbose = False):
        N = network.N
        B = len(self.beta)

        self.fit_info = { 'obj_evals': 0, 'grad_evals': 0,
                          'grad_final': np.empty(B + 1 + 2*(N-1)) }
        start_time = time()

        if network.offset:
            O = network.offset.matrix()
        alpha_zero(network)

        # Calculate observed sufficient statistics
        T = np.empty(B + 1 + 2*(N-1))
        A = np.array(network.adjacency_matrix())
        r = np.sum(A, axis = 1, dtype=np.int)[0:(N-1)]
        c = np.sum(A, axis = 0, dtype=np.int)[0:(N-1)]
        T[(B + 1):(B + 1 + (N-1))] = r
        T[(B + 1 + (N-1)):(B + 1 + 2*(N-1))] = c
        for b, b_n in enumerate(self.beta):
            T[b] = np.sum(A * network.edge_covariates[b_n].matrix())
        T[B] = np.sum(A, dtype=np.int)

        # Initialize theta
        theta = np.zeros(B + 1 + 2*(N-1))
        theta[B] = logit(A.sum(dtype=np.int) / N ** 2)
        if network.offset:
            theta[B] -= logit_mean(O)
        theta[(B + 1):(B + 1 + 2*(N-1))] = -theta[B]
        for i in range(N-1):
            theta[B + 1 + i] += logit((A[i,:].sum(dtype=np.int)+1)/(N+1))
            if network.offset:
                o_row = logit_mean(O[i,:])
                if np.isfinite(o_row):
                    theta[B + 1 + i] -= o_row
        for j in range(N-1):
            theta[B + 1 + (N-1) + j] += logit((A[:,j].sum(dtype=np.int)+1)/(N+1))
            if network.offset:
                o_col = logit_mean(O[:,j])
                if np.isfinite(o_col):
                    theta[B + 1 + (N-1) + j] -= o_col

        alpha_out = network.node_covariates['alpha_out']
        alpha_in = network.node_covariates['alpha_in']
        def obj(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing objective for nan-containing vector.'
                return np.Inf
            alpha_out[0:N-1] = theta[(B + 1):(B + 1 + (N-1))]
            alpha_in[0:N-1] = theta[(B + 1 + (N-1)):(B + 1 + 2*(N-1))]
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            self.kappa = theta[B]
            nll = self.nll(network)
            self.fit_info['obj_evals'] += 1
            return nll
        def grad(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing gradient for nan-containing vector.'
                return np.zeros(B + 1 + 2*(N-1))
            alpha_out[0:N-1] = theta[(B + 1):(B + 1 + (N-1))]
            alpha_in[0:N-1] = theta[(B + 1 + (N-1)):(B + 1 + 2*(N-1))]
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            self.kappa = theta[B]
            ET = np.empty(B + 1 + 2*(N-1))
            P = self.edge_probabilities(network)
            Er = np.sum(P, axis = 1)[0:(N-1)]
            Ec = np.sum(P, axis = 0)[0:(N-1)]
            ET[(B + 1):(B + 1 + (N-1))] = Er
            ET[(B + 1 + (N-1)):(B + 1 + 2*(N-1))] = Ec
            for b, b_n in enumerate(self.beta):
                ET[b] = np.sum(P * network.edge_covariates[b_n].matrix())
            ET[B] = np.sum(P)
            grad = ET - T
            self.fit_info['grad_evals'] += 1
            self.fit_info['grad_final'][:] = grad
            if verbose:
                abs_grad = np.abs(ET - T)
                print '|ET - T|: %.2f, %.2f, %.2f (min, mean, max)' % \
                    (np.min(abs_grad), np.mean(abs_grad), np.max(abs_grad))
            return grad

        bounds = [(-8,8)] * B + [(-15,15)] + [(-8,8)] * (2*(N-1))
        theta_opt = opt.fmin_l_bfgs_b(obj, theta, grad, bounds = bounds)[0]
        if (np.any(theta_opt == [b[0] for b in bounds]) or
            np.any(theta_opt == [b[1] for b in bounds])):
            print 'Warning: some constraints active in model fitting.'
        alpha_out[0:N-1] = theta_opt[(B + 1):(B + 1 + (N-1))]
        alpha_in[0:N-1] = theta_opt[(B + 1 + (N-1)):(B + 1 + 2*(N-1))]
        alpha_out_mean = np.mean(alpha_out[:])
        alpha_in_mean = np.mean(alpha_in[:])
        alpha_out[:] -= alpha_out_mean
        alpha_in[:] -= alpha_in_mean
        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = theta_opt[b]
        self.kappa = theta_opt[B] + alpha_out_mean + alpha_in_mean

        self.fit_info['wall_time'] = time() - start_time

    def fit_logistic(self, network):
        import statsmodels.api as sm

        N = network.N
        B = len(self.beta)
        alpha_zero(network)

        y = network.adjacency_matrix().reshape((N*N,))
        Phi = np.zeros((N*N,B + 1 + 2*(N-1)))
        for b, b_n in enumerate(self.beta):
            Phi[:,b] =  network.edge_covariates[b_n].matrix().reshape((N*N,))
        Phi[:,B] = 1.0
        for r in range(N-1):
            phi_row = np.zeros((N,N))
            phi_row[r,:] = 1.0
            Phi[:,B + 1 + r] = phi_row.reshape((N*N,))
        for c in range(N-1):
            phi_col = np.zeros((N,N))
            phi_col[:,c] = 1.0
            Phi[:,B + 1 + (N-1) + c] = phi_col.reshape((N*N,))
        if network.offset:
            offset = network.offset.matrix().reshape((N*N,))
            coefs = sm.GLM(y, Phi, sm.families.Binomial(), offset).fit().params
        else:
            coefs = sm.Logit(y, Phi).fit().params

        alpha_out = network.node_covariates['alpha_out']
        alpha_in = network.node_covariates['alpha_in']
        alpha_out[0:N-1] = coefs[(B + 1):(B + 1 + (N-1))]
        alpha_in[0:N-1] = coefs[(B + 1 + (N-1)):(B + 1 + 2*(N-1))]
        alpha_out_mean = np.mean(alpha_out[:])
        alpha_in_mean = np.mean(alpha_in[:])
        alpha_out[:] -= alpha_out_mean
        alpha_in[:] -= alpha_in_mean
        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = coefs[b]
        self.kappa = coefs[B] + alpha_out_mean + alpha_in_mean

# P_{ij} = Logit^{-1}(base_model(i,j) + Theta_{z_i,z_j})
# Constraints: \sum_{i,j} z_{i,j} = 0
class Blockmodel(IndependentBernoulli):
    def __init__(self, base_model, K, block_name = 'z'):
        self.base_model = base_model
        self.K = K
        self.Theta = np.zeros((K,K))
        self.block_name = block_name

    def apply_to_offset(self, network):
        N = network.N
        z = network.node_covariates[self.block_name]
        for i in range(N):
            for j in range(N):
                network.offset[i,j] += self.Theta[z[i], z[j]]

    def edge_probabilities(self, network):
        if network.offset:
            old_offset = network.offset.copy()
        else:
            network.initialize_offset()
            old_offset = None
        self.apply_to_offset(network)

        P = self.base_model.edge_probabilities(network)

        if old_offset:
            network.offset = old_offset
        else:
            network.offset = None

        return P

    def match_kappa(self, network, kappa_target):
        if network.offset:
            old_offset = network.offset.copy()
        else:
            network.initialize_offset()
            old_offset = None
        self.apply_to_offset(network)

        self.base_model.match_kappa(network, kappa_target)

        if old_offset:
            network.offset = old_offset
        else:
            network.offset = None

    def fit(self, network):
        self.fit_sem(network)

    # Stochastic EM fitting with `sweeps` Gibbs sweeps in the E-step
    # and `cycles` repetitions of the entire E-step/M-step operation
    #
    # This fitting procedure requires that `base_model` can handle
    # edge covariate effects and a kappa term.
    def fit_sem(self, network, cycles = 20, sweeps = 5):        
        # Local aliases for convenience
        K, Theta = self.K, self.Theta
        N = network.N
        z = network.node_covariates[self.block_name]
        A = network.adjacency_matrix()
        
        for cycle in range(cycles):
            # Stochastic E-step
            for gibbs_step in range(sweeps * N):
                l = np.random.randint(N)
                logprobs = np.empty(K)
                for k in range(K):
                    logprobs[k] = (np.dot(Theta[z[:],k], A[:,l]) +
                                   np.dot(Theta[k,z[:]], A[l,:]) +
                                   (Theta[k,k] * A[l,l]))
                logprobs -= np.max(logprobs)
                probs = np.exp(logprobs)
                probs /= np.sum(probs)
                z[l] = np.where(np.random.multinomial(1, probs) == 1)[0][0]

            # M-step
            cov_name_to_inds = {}
            for s in range(K):
                for t in range(K):
                    cov_name = '_%d_%d' % (s,t)
                    cov_name_to_inds[cov_name] = (s,t)
                    cov = network.new_edge_covariate(cov_name)
                    def f_edge_class(i_1, i_2):
                        return (z[i_1] == s) and (z[i_2] == t)
                    cov.from_binary_function_ind(f_edge_class)
                    self.base_model.beta[cov_name] = None

            self.base_model.fit(network)

            for cov_name in cov_name_to_inds:
                s, t = cov_name_to_inds[cov_name]
                self.Theta[s,t] = self.base_model.beta[cov_name]
                network.edge_covariates.pop(cov_name)
                self.base_model.beta.pop(cov_name)
            Theta_mean = np.mean(Theta)
            Theta -= Theta_mean
            self.base_model.kappa += Theta_mean

# Endow an existing model with fixed row and column margins
class FixedMargins(IndependentBernoulli):
    def __init__(self, base_model, r_name = 'r', c_name = 'c'):
        self.base_model = base_model
        self.r_name = r_name
        self.c_name = c_name

    def generate(self, network, **opts):
        r = network.node_covariates[self.r_name][:]
        c = network.node_covariates[self.c_name][:]

        return self.base_model.generate_margins(network, r, c, **opts)
     
# Generate alpha_out/in for an existing Network
def center(x):
    return x - np.mean(x)

def alpha_zero(network):
    network.new_node_covariate('alpha_out')
    network.new_node_covariate('alpha_in')
    
def alpha_norm(network, alpha_sd):
    a = np.random.normal(0, alpha_sd, (2,network.N))
    a[0] = center(a[0])
    a[1] = center(a[1])

    network.new_node_covariate('alpha_out')[:] = a[0]
    network.new_node_covariate('alpha_in')[:] = a[1]

def alpha_unif(network, alpha_sd):
    c = np.sqrt(12) / 2
    a = np.random.uniform(-alpha_sd * c, alpha_sd * c, (2,network.N))
    a[0] = center(a[0])
    a[1] = center(a[1])
    
    network.new_node_covariate('alpha_out')[:] = a[0]
    network.new_node_covariate('alpha_in')[:] = a[1]

def alpha_gamma(network, alpha_loc, alpha_scale):
    a = np.random.gamma(alpha_loc, alpha_scale, (2,network.N))
    a[0] = center(a[0])
    a[1] = center(a[1])
    
    network.new_node_covariate('alpha_out')[:] = a[0]
    network.new_node_covariate('alpha_in')[:] = a[1]
