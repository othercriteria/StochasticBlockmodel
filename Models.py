#!/usr/bin/env python

# Models for generating and fitting networks
# Daniel Klein, 5/11/2012

from __future__ import division
import numpy as np
import scipy.optimize as opt
from time import time
from itertools import permutations

from Utility import logit, inv_logit, logit_mean, logsumexp, logabsdiffexp
from BinaryMatrix import arbitrary_from_margins
from BinaryMatrix import approximate_from_margins_weights
from BinaryMatrix import approximate_conditional_nll

# It's a bit weird to have NonstationaryLogistic as a subclass of strictly
# less general StationaryLogistic model, but I believe this is how
# inheritance is supposed to work in Python.

# Some of the inference routines and the Kappa picking routine rely on
# a side-effect (modifying an instance variable of the enclosing
# class) inside the objective function. This is inelegant and seems
# error-prone, and so should probably be fixed...

# P_{ij} = Logit^{-1}(o_{ij})
class IndependentBernoulli:
    def edge_probabilities(self, network, submatrix = None):
        N = network.N
        if submatrix:
            i_sub, j_sub = submatrix

        if network.offset:
            logit_P = network.offset.matrix()
            if submatrix:
                logit_P = logit[i_sub][:,j_sub]
            return inv_logit(logit_P)
        else:
            if submatrix:
                return np.tile(0.5, (len(i_sub),len(j_sub)))
            else:
                return np.tile(0.5, (N,N))

    def nll(self, network, submatrix = None):
        P = self.edge_probabilities(network, submatrix)
        A = np.asarray(network.adjacency_matrix())
        if submatrix:
            i_sub, j_sub = submatrix
            A = A[i_sub][:,j_sub]

        # Check for impossible data for the cells with 0/1 probabilities
        ind_P_zero, ind_P_one = (P == 0.0), (P == 1.0)
        if np.any(A[ind_P_zero]) or not np.all(A[ind_P_one]):
            return np.Inf

        # Compute the negative log-likelihood for the rest of the cells
        ind_P_rest = -(ind_P_zero + ind_P_one)
        P, A = P[ind_P_rest], A[ind_P_rest]
        nll = -np.sum(A * np.log(P) + (1.0 - A) * np.log1p(-P))

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
                         coverage = 0, arbitrary_init = False):
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
        self.fit_info = None

    def edge_probabilities(self, network, submatrix = None):
        N = network.N
        if submatrix:
            sub_i, sub_j = submatrix

        if network.offset:
            logit_P = network.offset.matrix().copy()
            if submatrix:
                logit_P = logit_P[sub_i][:,sub_j]
        else:
            if submatrix:
                logit_P = np.zeros((len(sub_i),len(sub_j)))
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
        if not self.fit_info:
            self.fit_info = {}
        self.fit_info['nll_evals'] = 0
        self.fit_info['grad_nll_evals'] = 0
        self.fit_info['grad_nll_final'] = np.empty(1)
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
            self.fit_info['nll_evals'] += 1
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
            self.fit_info['grad_nll_evals'] += 1
            self.fit_info['grad_nll_final'][:] = grad
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

    def fit_mh(self, network):
        mh = self.mh(network)
        mh.run()
        self.kappa = np.mean(mh.results['kappa'])
        self.fit_info = {'nll': mh.results['nll']}
    
    def mh(self, network, kappa_prior = (0,1), kappa_prop_sd = 0.2):
        from scipy.stats import norm
        log_prior = lambda x: np.log(norm(*kappa_prior).pdf(x))
        
        def update(n, m):
            kappa_curr = m.kappa
            kappa_prop = kappa_curr + np.random.normal(0, kappa_prop_sd)

            nll_curr = m.nll(n)
            log_post_curr = -nll_curr + log_prior(kappa_curr)
            
            m.kappa = kappa_prop
            nll_prop = m.nll(n)
            log_post_prop = -nll_prop + log_prior(kappa_prop)

            p_acc = np.exp(log_post_prop - log_post_curr)
            if p_acc >= 1.0 or np.random.random() < p_acc:
                return nll_prop
            else:
                m.kappa = kappa_curr
                return nll_curr

        return Sampler(network, self, update, {'kappa': lambda n, m: m.kappa})

# P_{ij} = Logit^{-1}(\sum_b x_{bij}*beta_b + kappa + o_{ij}) 
class StationaryLogistic(Stationary):
    def __init__(self):
        Stationary.__init__(self)
        self.beta = {}
        self.fit = self.fit_convex_opt

    def edge_probabilities(self, network, submatrix = None):
        N = network.N
        if submatrix:
            i_sub, j_sub = submatix
        
        if network.offset:
            logit_P = network.offset.matrix().copy()
            if submatrix:
                logit_P = logit_P[i_sub][:,j_sub]
        else:
            if submatrix:
                logit_P = np.zeros((len(i_sub),len(j_sub)))
            else:
                logit_P = np.zeros((N,N))            
        for b in self.beta:
            ec_b = network.edge_covariates[b].matrix()
            if submatrix:
                ec_b = ec_b[i_sub][:,j_sub]
            logit_P += self.beta[b] * ec_b
        logit_P += self.kappa

        return inv_logit(logit_P)

    def baseline(self, network):
        return np.mean(self.edge_probabilities(network))

    def baseline_logit(self, network):
        return np.mean(logit(self.edge_probabilities(network)))

    def fit_convex_opt(self, network, verbose = False, fix_beta = False):
        B = len(self.beta)

        if not self.fit_info:
            self.fit_info = {}
        self.fit_info['nll_evals'] = 0
        self.fit_info['grad_nll_evals'] = 0
        self.fit_info['grad_nll_final'] = np.empty(B + 1)
        
        start_time = time()

        # Calculate observed sufficient statistics
        T = np.empty(B + 1)
        A = np.array(network.adjacency_matrix())
        for b, b_n in enumerate(self.beta):
            T[b] = np.sum(A * network.edge_covariates[b_n].matrix())
        T[B] = np.sum(A, dtype=np.int)

        # Initialize theta
        theta = np.zeros(B + 1)
        if fix_beta:
            for b, b_n in enumerate(self.beta):
                theta[b] = self.beta[b_n]
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
            self.fit_info['nll_evals'] += 1
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
            if fix_beta:
                grad[0:B] = 0.0
            self.fit_info['grad_nll_evals'] += 1
            self.fit_info['grad_nll_final'][:] = grad
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

    def fit_conditional(self, network,
                        fit_grid = False, verbose = False, T = 0):
        B = len(self.beta)
        if fit_grid and not B in [1,2]:
            print 'Can only grid search B = 1, 2. Defaulting to minimizer.'
            fit_grid = False

        if not self.fit_info:
            self.fit_info = {}
        self.fit_info['cnll_evals'] = 0

        start_time = time()

        A = np.array(network.adjacency_matrix())
        r, c = A.sum(1), A.sum(0)

        # Initialize theta
        theta = np.zeros(B)

        def obj(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing objective for nan-containing vector.'
                return np.Inf
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            P = self.edge_probabilities(network)
            w = P / (1.0 - P)

            if T == 0:
                cnll = approximate_conditional_nll(A, w,
                                                   sort_by_wopt_var = False)
            else:
                z = approximate_from_margins_weights(r, c, w, T,
                                                     sort_by_wopt_var = False)
                logf = np.empty(T)
                for t in range(T):
                    logf[t] = z[t][2] - z[t][1]
                logkappa = -np.log(T) + logsumexp(logf)
                logcvsq = -np.log(T - 1) - 2 * logkappa + \
                    logsumexp(2 * logabsdiffexp(logf, logkappa))
                if verbose:
                    print 'est. cv^2 = %.2f (T = %d)' % (np.exp(logcvsq), T)

                cnll = logkappa - np.sum(np.log(w[A]))

            self.fit_info['cnll_evals'] += 1
            if verbose:
                print cnll, theta
            return cnll

        if fit_grid:
            if B == 1:
                G = 100
                theta_0 = np.linspace(-6.0, 6.0, G)
                cnll = np.empty(G)
                for g in range(G):
                    cnll[g] = obj(np.array([theta_0[g]]))
                theta_0_opt_ind = np.argmin(cnll)
                self.beta[self.beta.keys()[0]] = theta_0[theta_0_opt_ind]
            if B == 2:
                G = 15
                theta_0 = np.linspace(-6.0, 6.0, G)
                theta_1 = np.linspace(-6.0, 6.0, G)
                cnll = np.empty((G,G))
                for g_0 in range(G):
                    for g_1 in range(G):
                        t_0, t_1 = theta_0[g_0], theta_1[g_1]
                        cnll[g_0,g_1] = obj(np.array([t_0,t_1]))
                cnll_min = np.min(cnll)
                theta_opt_ind = np.where(cnll == cnll_min)
                theta_0_opt = theta_0[theta_opt_ind[0][0]]
                theta_1_opt = theta_1[theta_opt_ind[1][0]]
                self.beta[self.beta.keys()[0]] = theta_0_opt
                self.beta[self.beta.keys()[1]] = theta_1_opt
        else:
            if T > 0:
                # Use Kiefer-Wolfowitz stochastic approximation
                for n in range(1, 16):
                    a_n = 0.02 * n ** (-1.0)
                    c_n = 0.5 * n ** (-1.0 / 3)
                    grad = np.empty(B)
                    for b in range(B):
                        e = np.zeros(B)
                        e[b] = 1.0
                        y_p = obj(theta + c_n * e)
                        y_m = obj(theta - c_n * e)
                        grad[b] = (y_p - y_m) / c_n
                    theta -= a_n * grad
                theta_opt = theta
            else:
                theta_opt = opt.fmin(obj, theta)

            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta_opt[b]

        self.fit_convex_opt(network, fix_beta = True)

        self.fit_info['wall_time'] = time() - start_time

    def fit_composite(self, network, T = 100, verbose = False):
        B = len(self.beta)

        if not self.fit_info:
            self.fit_info = {}

        start_time = time()

        A = np.array(network.adjacency_matrix())
        r, c = A.sum(1), A.sum(0)

        # Initialize theta
        theta = np.zeros(B)

        def obj(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing objective for nan-containing vector.'
                return np.Inf
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            P = self.edge_probabilities(network)
            w = P / (1.0 - P)

            cnll = 0.0
            for t in range(T):
                # Pick a random pair of rows 
                a, b = 0, 0
                while a == b:
                    a, b = np.random.randint(B), np.random.randint(B)

                # Extract subnetwork
                A_sub = A[[a,b],:]
                w_sub = w[[a,b],:]

                # Extract nontrivial columns
                nontrivial = A_sub.sum(0) == 1
                A_sub = A_sub[:,nontrivial]
                w_sub = w_sub[:,nontrivial]

                # Check if all permutations are trivial, so no change to cnll
                active = A_sub.shape[1]
                if np.sum(A_sub[0,:]) in [active, 0]:
                    continue

                logkappa = 0.0
                seen = set()
                for perm in permutations(range(active)):
                    A_sub_perm = A_sub[:,perm]

                    # Skip redundant permutations
                    rep = A_sub_perm[0,:].dumps()
                    if rep in seen: continue
                    seen.add(rep)

                    logp_perm = np.sum(np.log(w_sub[A_sub_perm]))
                    logkappa = np.logaddexp(logkappa, logp_perm)
                
                cnll += logkappa - np.sum(np.log(w_sub[A_sub]))

            if verbose:
                print cnll, theta
            return cnll

        # Use Kiefer-Wolfowitz stochastic approximation
        for n in range(1, 16):
            a_n = 0.002 * n ** (-1.0)
            c_n = 0.5 * n ** (-1.0 / 3)
            grad = np.empty(B)
            for b in range(B):
                e = np.zeros(B)
                e[b] = 1.0
                y_p = obj(theta + c_n * e)
                y_m = obj(theta - c_n * e)
                grad[b] = (y_p - y_m) / c_n
            theta -= a_n * grad
        theta_opt = theta

        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = theta_opt[b]

        self.fit_convex_opt(network, fix_beta = True)

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

    def fit_mh(self, network):
        mh = self.mh(network)
        mh.run()
        self.kappa = np.mean(mh.results['kappa'])
        for b_n in self.beta:
            self.beta[b_n] = np.mean(mh.results[b_n])
        self.fit_info = {'nll': mh.results['nll']}
    
    def mh(self, network, kappa_prior = (0,1), kappa_prop_sd = 0.2,
           beta_prior = (0,1), beta_prop_sd = 0.2):
        from scipy.stats import norm
        def log_prior(kappa, beta):
            lp = np.log(norm(*kappa_prior).pdf(kappa))
            for b_n in beta:
                lp += np.log(norm(*beta_prior).pdf(beta[b_n]))
            return lp
        
        def update(n, m):
            kappa_curr = m.kappa
            beta_curr = m.beta
            kappa_prop = kappa_curr + np.random.normal(0, kappa_prop_sd)
            beta_prop = m.beta.copy()
            for b_n in beta_prop:
                beta_prop[b_n] += np.random.normal(0, beta_prop_sd)

            nll_curr = m.nll(n)
            log_post_curr = -nll_curr + log_prior(kappa_curr, beta_curr)
            
            m.kappa = kappa_prop
            m.beta = beta_prop
            nll_prop = m.nll(n)
            log_post_prop = -nll_prop + log_prior(kappa_prop, beta_prop)

            p_acc = np.exp(log_post_prop - log_post_curr)
            if p_acc >= 1.0 or np.random.random() < p_acc:
                return nll_prop
            else:
                m.kappa = kappa_curr
                m.beta = beta_curr
                return nll_curr

        record = {'kappa': lambda n, m: m.kappa}
        for b_n in self.beta:
            record[b_n] = lambda n, m: m.beta[b_n]

        return Sampler(network, self, update, record)

# P_{ij} = Logit^{-1}(alpha_out_i + alpha_in_j + \sum_b x_{bij}*beta_b + kappa +
#                     o_{ij})
# Constraints: \sum_i alpha_out_i = 0, \sum_j alpha_in_j = 0
class NonstationaryLogistic(StationaryLogistic):
    def __init__(self):
        StationaryLogistic.__init__(self)
        self.fit = self.fit_convex_opt
        
    def edge_probabilities(self, network, submatrix = None):
        N = network.N
        if submatrix:
            i_sub, j_sub = submatrix

        alpha_out = network.node_covariates['alpha_out']
        alpha_in = network.node_covariates['alpha_in']
        if submatrix:
            alpha_out = alpha_out[i_sub]
            alpha_in = alpha_in[j_sub]
        
        if network.offset:
            logit_P = network.offset.matrix().copy()
            if submatrix:
                logit_P = logit_P[i_sub][:,j_sub]
        else:
            if submatrix:
                logit_P = np.zeros((len(i_sub),len(j_sub)))
            else:
                logit_P = np.zeros((N,N))
        for i, a in enumerate(alpha_out):
            logit_P[i,:] += a
        for j, a in enumerate(alpha_in):
            logit_P[:,j] += a
        for b in self.beta:
            ec_b = network.edge_covariates[b].matrix()
            if submatrix:
                ec_b = ec_b[i_sub][:,j_sub]
            logit_P += self.beta[b] * ec_b
        logit_P += self.kappa

        return inv_logit(logit_P)

    def baseline(self, network):
        N = network.N
        P = self.edge_probabilities(network)
        def params_to_Q(params):
            a = params[0:N]
            b = params[N:(2*N)]
            c = params[2*N]
            logit_Q = np.zeros((N,N))
            for i in range(N):
                logit_Q[i,:] += a[i]
            for j in range(N):
                logit_Q[:,j] += b[j]
            logit_Q += c
            return inv_logit(logit_Q)
        params = np.zeros(2*N + 1)
        def obj(params):
            Q = params_to_Q(params)
            return np.reshape((P - Q), (N*N,))
        best_params = opt.leastsq(obj, params)[0]
        Q = params_to_Q(best_params)
        return Q

    def baseline_logit(self, network):
        N = network.N
        logit_P = logit(self.edge_probabilities(network))
        a, b = logit_P.mean(1), logit_P.mean(0)
        a_mean, b_mean = a.mean(), b.mean()
        a -= a_mean
        b -= b_mean
        c = a_mean + b_mean
        logit_Q = np.zeros((N,N))
        for i in range(N):
            logit_Q[i,:] += a[i]
        for j in range(N):
            logit_Q[:,j] += b[j]
        logit_Q += c
        return logit_Q

    def fit_convex_opt(self, network, verbose = False, fix_beta = False):
        N = network.N
        B = len(self.beta)

        if not self.fit_info:
            self.fit_info = {}
        self.fit_info['nll_evals'] = 0
        self.fit_info['grad_nll_evals'] = 0
        self.fit_info['grad_nll_final'] = np.empty(B + 1 + 2*(N-1))

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
        if fix_beta:
            for b, b_n in enumerate(self.beta):
                theta[b] = self.beta[b_n]
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
            self.fit_info['nll_evals'] += 1
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
            if fix_beta:
                grad[0:B] = 0.0
            self.fit_info['grad_nll_evals'] += 1
            self.fit_info['grad_nll_final'][:] = grad
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

    def fit_mh(self, network):
        mh = self.mh(network)
        mh.run()
        self.kappa = np.mean(mh.results['kappa'])
        for b_n in self.beta:
            self.beta[b_n] = np.mean(mh.results[b_n])
        self.fit_info = {'nll': mh.results['nll']}
    
    def mh(self, network, kappa_prior = (0,1), kappa_prop_sd = 0.2,
           beta_prior = (0,1), beta_prop_sd = 0.2,
           alpha_base = (0,1), alpha_conc = 10.0, alpha_steps = 10):
        from scipy.stats import norm
        def log_prior(kappa, beta):
            lp = np.log(norm(*kappa_prior).pdf(kappa))
            for b_n in beta:
                lp += np.log(norm(*beta_prior).pdf(beta[b_n]))
            return lp
        def sample_from_base():
            return np.random.normal(*alpha_base)
        
        def update(n, m):
            # Gibbs steps for alphas
            vars = [('i', i) for i in range(n.N)] + \
                [('j', j) for j in range(n.N)]
            np.random.shuffle(vars)
            for type, ind in vars[0:alpha_steps]:
                # Get current table assignment
                alpha_name = {'i': 'alpha_out', 'j': 'alpha_in'}[type]
                alphas = n.node_covariates[alpha_name]
                tables = {}
                for k, a in enumerate(alphas):
                    if k == ind: continue
                    if not a in tables:
                        tables[a] = 0
                    tables[a] += 1

                # Use current table assignment to compute prior term
                T = len(tables)
                atoms = np.empty(T + 1)
                table_prob_unscaled = np.empty(T + 1)
                for t, a in enumerate(tables):
                    atoms[t] = a
                    table_prob_unscaled[t] = tables[a]
                atoms[T] = sample_from_base()
                table_prob_unscaled[T] = alpha_conc

                # Get posterior term for all possible assignments
                net_prob = np.empty(T + 1)
                for t in range(T + 1):
                    alphas[ind] = atoms[t]
                    if type == 'i':
                        submatrix = (np.array([ind]), np.arange(n.N))
                    if type == 'j':
                        submatrix = (np.arange(n.N), np.array([ind]))
                    net_prob[t] = m.nll(n, submatrix)

                # Resample according to conditional distribution
                cond_prob_unscaled = table_prob_unscaled * net_prob
                cond_prob = cond_prob_unscaled / np.sum(cond_prob_unscaled)
                t_new = np.random.multinomial(1, cond_prob).argmax()
                a_new = atoms[t_new]
                alphas[ind] = a_new
            
            kappa_curr = m.kappa
            beta_curr = m.beta
            kappa_prop = kappa_curr + np.random.normal(0, kappa_prop_sd)
            beta_prop = m.beta.copy()
            for b_n in beta_prop:
                beta_prop[b_n] += np.random.normal(0, beta_prop_sd)

            nll_curr = m.nll(n)
            log_post_curr = -nll_curr + log_prior(kappa_curr, beta_curr)
            
            m.kappa = kappa_prop
            m.beta = beta_prop
            nll_prop = m.nll(n)
            log_post_prop = -nll_prop + log_prior(kappa_prop, beta_prop)

            p_acc = np.exp(log_post_prop - log_post_curr)
            if p_acc >= 1.0 or np.random.random() < p_acc:
                return nll_prop
            else:
                m.kappa = kappa_curr
                m.beta = beta_curr
                return nll_curr

        record = {'kappa': lambda n, m: m.kappa}
        for b_n in self.beta:
            record[b_n] = lambda n, m: m.beta[b_n]

        return Sampler(network, self, update, record)

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

    def baseline(self, network):
        return self.base_model.baseline(network)

    def baseline_logit(self, network):
        return self.base_model.baseline_logit(network)

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
    def __init__(self, base_model, r_name = 'r', c_name = 'c', coverage = 0):
        self.base_model = base_model
        self.r_name = r_name
        self.c_name = c_name
        self.coverage = coverage

    def generate(self, network, **opts):
        if not self.r_name in network.node_covariates:
            print 'Covariate "%s" not found.' % self.r_name
            r = np.asarray(network.adjacency_matrix()).sum(1)
        else:
            r = network.node_covariates[self.r_name][:]

        if not self.c_name in network.node_covariates:
            print 'Covariate "%s" not found.' % self.c_name
            c = np.asarray(network.adjacency_matrix()).sum(0)
        else:
            c = network.node_covariates[self.c_name][:]

        return self.base_model.generate_margins(network, r, c, coverage = 0,
                                                **opts)

# Handles the state, updates, and recording for a (usually MCMC) sampler
class Sampler:
    def __init__(self, network, model, update, record = {}):
        self.network = network
        self.model = model
        self.update = update
        self.record = record

        self.results = {'nll_full': [], 'nll': []}
        for var in self.record:
            self.results[var] = []

    def step(self):
        return self.update(self.network, self.model)

    def run(self, samples = 1000, burnin = 1000, thinning = 10):
        for b in range(burnin):
            nll = self.step()
            self.results['nll_full'].append(nll)

        for s in range(samples):
            nll = self.step()
            self.results['nll_full'].append(nll)
            if s % thinning == 0:
                self.results['nll'].append(nll)
                for var in self.record:
                    recorder = self.record[var]
                    val = recorder(self.network, self.model)
                    self.results[var].append(val)
      
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
