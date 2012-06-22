#!/usr/bin/env python

# Models for generating and fitting networks
# Daniel Klein, 5/11/2012

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse

from Utility import inv_logit
from BinaryMatrix import arbitrary_from_margins

# It's a bit weird to have NonstationaryLogistic as a subclass of strictly
# less general StationaryLogistic model, but I believe this is how
# inheritance is supposed to work in Python.

# Some of the inference routines and the Kappa picking routine rely on
# a side-effect (modifying an instance variable of the enclosing
# class) inside the objective function. This is inelegant and seems
# error-prone, and so should probably be fixed...

# P_{ij} arbitrary, given by some function of the Network
class IndependentBernoulli:
    def __init__(self, edge_probabilities):
        self.edge_probabilities = edge_probabilities
    
    def nll(self, network):
        P = self.edge_probabilities(network)
        if np.any(P == 0) or np.any(P == 1):
            return np.Inf
        A = network.adjacency_matrix()
        return -np.sum(np.log(P ** A) + np.log((1.0 - P) ** (1.0 - A)))

    def generate(self, network):
        N = network.N
        
        P = self.edge_probabilities(network)
        return np.random.random((N,N)) < P

# P_{ij} = Logit^{-1}(kappa)
class Stationary(IndependentBernoulli):
    def __init__(self):
        self.kappa = 0.0

    def edge_probabilities(self, network):
        N = network.N

        logit_P = np.zeros((N,N))
        logit_P += self.kappa

        return inv_logit(logit_P)

    def snll_part(self, network, train, test):
        P = self.edge_probabilities(network)
        pass

    # Mechanical specialization of the StationaryLogistic code, so not
    # really ideal for this one-dimensional problem.
    def fit_convex_opt(self, network):
        # Calculate observed sufficient statistic
        T = np.zeros(1)
        A = network.adjacency_matrix()
        T[0] = np.sum(A)

        theta = np.zeros(1)
        def obj(theta):
            self.kappa = theta[0]
            return self.nll(network)
        def grad(theta):
            self.kappa = theta[0]
            ET = np.zeros(1)
            P = self.edge_probabilities(network)
            ET[0] = np.sum(P)
            return ET - T

        bounds = [(-20,20)]
        theta_opt = opt.fmin_l_bfgs_b(obj, theta, grad, bounds = bounds)[0]
        self.kappa = theta_opt[0]

    def fit_logistic(self, network):
        import scikits.statsmodels.api as sm

        N = network.N

        y = network.adjacency_matrix().reshape((N*N,))
        Phi = np.zeros((N*N,1))
        Phi[:,0] = 1.0
        coefs = sm.Logit(y, Phi).fit().params

        self.kappa = coefs[0]

# P_{ij} = Logit^{-1}(\sum_b x_{bij}*beta_b + kappa)
class StationaryLogistic(Stationary):
    def __init__(self):
        Stationary.__init__(self)
        self.beta = {}

    def edge_probabilities(self, network):
        N = network.N
        
        logit_P = np.zeros((N,N))
        for b in self.beta:
            logit_P += self.beta[b] * network.edge_covariates[b].matrix()
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

    def fit_convex_opt(self, network):
        B = len(self.beta)

        # Calculate observed sufficient statistics
        T = np.zeros(B + 1)
        A = network.adjacency_matrix()
        for b, b_n in enumerate(self.beta):
            T[b] = np.sum(A * network.edge_covariates[b_n].matrix())
        T[B] = np.sum(A)

        theta = np.zeros(B + 1)
        def obj(theta):
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            self.kappa = theta[B]
            return self.nll(network)
        def grad(theta):
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            self.kappa = theta[B]
            ET = np.zeros(B + 1)
            P = self.edge_probabilities(network)
            for b, b_n in enumerate(self.beta):
                ET[b] = np.sum(P * network.edge_covariates[b_n].matrix())
            ET[B] = np.sum(P)
            return ET - T

        bounds = [(-10,10)] * B + [(-15,15)]
        theta_opt = opt.fmin_l_bfgs_b(obj, theta, grad, bounds = bounds)[0]
        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = theta_opt[b]
        self.kappa = theta_opt[B]

    def fit_logistic(self, network):
        import scikits.statsmodels.api as sm

        N = network.N
        B = len(self.beta)

        y = network.adjacency_matrix().reshape((N*N,))
        Phi = np.zeros((N*N,B + 1))
        Phi[:,B] = 1.0
        for b, b_n in enumerate(self.beta):
            Phi[:,b] =  network.edge_covariates[b_n].matrix().reshape((N*N,))
        coefs = sm.Logit(y, Phi).fit().params

        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = coefs[b]
        self.kappa = coefs[B]

    # Doing this fit endows the model with a posterior
    # variance/covariance matrix estimated via the Laplace
    # approximation (see Bishop for details)
    def fit_logistic_l2(self, network, prior_precision = 1.0,
                        variance_covariance = False):
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

# P_{ij} = Logit^{-1}(alpha_out_i + alpha_in_j + \sum_b x_{bij}*beta_b + kappa)
# Constraints: \sum_i alpha_out_i = 0, \sum_j alpha_in_j = 0
class NonstationaryLogistic(StationaryLogistic):
    def edge_probabilities(self, network):
        N = network.N
        alpha_out = network.node_covariates['alpha_out']
        alpha_in = network.node_covariates['alpha_in']
        
        logit_P = np.zeros((N,N))
        for i in range(N):
            logit_P[i,:] += alpha_out[i]
        for j in range(N):
            logit_P[:,j] += alpha_in[j]
        for b in self.beta:
            logit_P += self.beta[b] * network.edge_covariates[b].matrix()
        logit_P += self.kappa

        return inv_logit(logit_P)

    def fit_convex_opt(self, network):
        N = network.N
        B = len(self.beta)
        alpha_zero(network)

        # Calculate observed sufficient statistics
        T = np.zeros(B + 1 + 2*(N-1))
        A = np.array(network.adjacency_matrix())
        r = np.sum(A, axis = 1)[0:(N-1)]
        c = np.sum(A, axis = 0)[0:(N-1)]
        T[(B + 1):(B + 1 + (N-1))] = r
        T[(B + 1 + (N-1)):(B + 1 + 2*(N-1))] = c
        for b, b_n in enumerate(self.beta):
            T[b] = np.sum(A * network.edge_covariates[b_n].matrix())
        T[B] = np.sum(A)
            
        theta = np.zeros(B + 1 + 2*(N-1))
        alpha_out = network.node_covariates['alpha_out']
        alpha_in = network.node_covariates['alpha_in']
        def obj(theta):
            alpha_out[0:N-1] = theta[(B + 1):(B + 1 + (N-1))]
            alpha_in[0:N-1] = theta[(B + 1 + (N-1)):(B + 1 + 2*(N-1))]
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            self.kappa = theta[B]
            return self.nll(network)
        def grad(theta):
            alpha_out[0:N-1] = theta[(B + 1):(B + 1 + (N-1))]
            alpha_in[0:N-1] = theta[(B + 1 + (N-1)):(B + 1 + 2*(N-1))]
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            self.kappa = theta[B]
            ET = np.zeros(B + 1 + 2*(N-1))
            P = self.edge_probabilities(network)
            Er = np.sum(P, axis = 1)[0:(N-1)]
            Ec = np.sum(P, axis = 0)[0:(N-1)]
            ET[(B + 1):(B + 1 + (N-1))] = Er
            ET[(B + 1 + (N-1)):(B + 1 + 2*(N-1))] = Ec
            for b, b_n in enumerate(self.beta):
                ET[b] = np.sum(P * network.edge_covariates[b_n].matrix())
            ET[B] = np.sum(P)
            return ET - T

        bounds = [(-10,10)] * B + [(-15,15)] + [(-6,6)] * (2*(N-1))
        theta_opt = opt.fmin_l_bfgs_b(obj, theta, grad, bounds = bounds)[0]
        alpha_out[0:N-1] = theta_opt[(B + 1):(B + 1 + (N-1))]
        alpha_in[0:N-1] = theta_opt[(B + 1 + (N-1)):(B + 1 + 2*(N-1))]
        alpha_out_mean = np.mean(alpha_out[:])
        alpha_in_mean = np.mean(alpha_in[:])
        alpha_out[:] -= alpha_out_mean
        alpha_in[:] -= alpha_in_mean
        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = theta_opt[b]
        self.kappa = theta_opt[B] + alpha_out_mean + alpha_in_mean

    def fit_logistic(self, network):
        import scikits.statsmodels.api as sm

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
        # Dealing with weird bug in scikits?
        # dummy_val = np.where(Phi.var(0) == 0)[0][0] == B
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

# Identical to StationaryLogistic for inference, but generated
# (approximately) from the conditional distribution with fixed margins.
#
# There's no obvious stopping criterion for the Gibbs sampler used to
# generate random networks conditioned on the degree sequence, i.e.,
# the adjacency matrix margins. As a heuristic, I let the user specify
# a degree of "cover" and then ensure that the sampler makes a
# non-trivial proposal on average "cover" many times concerning each
# possible edge.
class StationaryLogisticMargins(StationaryLogistic):
    # Precomputing for diagonal/anti-diagonal check
    diag = np.array([[True,False],[False,True]])
    adiag = np.array([[False,True],[True,False]])
    valid = set([diag.data[0:4], adiag.data[0:4]])
    
    def generate(self, network, r, c, coverage = 100):
        N = network.N
        windows = N // 2
        coverage_target = coverage * N**2 / 4
        
        # Initialize from an arbitrary matrix with the requested margins
        gen = arbitrary_from_margins(r, c)

        # Gibbs sampling to match the "location" of the edges to where
        # they are likely under the conditional distribution
        #
        # Scheduling Gibbs sweeps in a checkerboard-like manner to
        # simplify picking distinct random indices.
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
                    if not (gen_prop.data[0:4] in self.valid): continue
                    active.append(np.ix_(i_prop, j_prop))
            A = len(active)
            coverage_attained += A

            # Calculate individual edge probabilities; because of
            # normalization, only beta/covariate contribution matters
            logit_P = np.zeros((2*A,2))
            for b in self.beta:
                cov = network.edge_covariates[b].matrix()
                for a, ij_prop in enumerate(active):
                    logit_P[2*a:2*(a+1)] += self.beta[b] * cov[ij_prop]
            P = inv_logit(logit_P)

            # Normalize probabilities of allowed configurations to get
            # conditional probabilities
            l_diag, l_adiag = np.empty(A), np.empty(A)
            for a in range(A):
                P_a = P[2*a:2*(a+1)]
                l_diag[a] = P_a[0,0] * P_a[1,1] * (1-P_a[0,1]) * (1-P_a[1,0])
                l_adiag[a] = P_a[0,1] * P_a[1,0] * (1-P_a[0,0]) * (1-P_a[1,1])
            p_diag = l_diag / (l_diag + l_adiag)

            # Update n according to calculated probabilities
            to_diag = np.random.random(A) < p_diag
            for diag, ij_prob in zip(to_diag, active):
                if diag:
                    gen[ij_prop] = self.diag
                else:
                    gen[ij_prop] = self.adiag

        return gen

    
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

def alpha_unif(network, alpha_max):
    a = np.random.uniform(-alpha_max, alpha_max, (2,network.N))
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
