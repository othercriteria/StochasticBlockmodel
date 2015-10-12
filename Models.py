#!/usr/bin/env python

# Models for generating and fitting networks
# Daniel Klein, 5/11/2012

from __future__ import division
import numpy as np
import scipy.optimize as opt
from scipy.stats import norm
from scipy.linalg import inv, solve
from time import time
from itertools import permutations
import hashlib

from Utility import logit, inv_logit, logit_mean
from BinaryMatrix import arbitrary_from_margins
from BinaryMatrix import approximate_from_margins_weights
from BinaryMatrix import approximate_conditional_nll as acnll
from BinaryMatrix import p_margins_saddlepoint
from BinaryMatrix import log_partition_is

# See if embedded R process can be started; this should be done once,
# globally, to reduce overhead.
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    r_interface_cond = importr('cond')

    r_interface_started = True
except:
    print 'Can\'t open R interface. Some inference methods may be unavailable.'
    r_interface_started = False

# It's a bit weird to have NonstationaryLogistic as a subclass of strictly
# less general StationaryLogistic model, but I believe this is how
# inheritance is supposed to work in Python.

# Some of the inference routines and the kappa picking routine rely on
# a side-effect (modifying an instance variable of the enclosing
# class) inside the objective function. This is inelegant and seems
# error-prone, and so should probably be fixed...

# P_{ij} = Logit^{-1}(o_{ij})
class IndependentBernoulli:
    def edge_probabilities(self, network, submatrix = None,
                           ignore_offset = False):
        N = network.N
        if submatrix:
            i_sub, j_sub = submatrix
            m, n = len(i_sub), len(j_sub)
        else:
            m, n = N, N

        if (not ignore_offset) and network.offset:
            logit_P = network.offset.matrix()
            if submatrix:
                logit_P = logit[i_sub][:,j_sub]
            return inv_logit(logit_P)
        else:
            return np.tile(0.5, (m, n))

    def nll(self, network, submatrix = None, ignore_offset = False):
        P = self.edge_probabilities(network, submatrix, ignore_offset)
        A = np.asarray(network.as_dense())
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
        M = network.M
        N = network.N
        
        P = self.edge_probabilities(network)
        return np.random.random((M,N)) < P

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
    def generate_margins(self, network, r = None, c = None,
                         coverage = 0, arbitrary_init = False,
                         optimize_perm = True):
        network.gen_info = { 'wall_time': 0.0,
                             'coverage': 0.0, }
        
        N = network.N
        if r is None:
            r = np.asarray(network.as_dense()).sum(1)
        if c is None:
            c = np.asarray(network.as_dense()).sum(0)

        start_time = time()
        if arbitrary_init:
            # Initialize from an arbitrary matrix with the requested margins
            gen = arbitrary_from_margins(r, c)
        else:
            # Initialize from an approximate sample from the
            # conditional distribution
            P = self.edge_probabilities(network)
            w = P / (1.0 - P)
            gen_sparse = approximate_from_margins_weights(r, c, w)
            gen = np.zeros((N,N), dtype=np.bool)
            for i, j in gen_sparse:
                if i == -1: break 
                gen[i,j] = 1
        network.gen_info['wall_time'] += time() - start_time

        if optimize_perm and np.all(r[:] == 1) and np.all(c[:] == 1):
            return self.gibbs_improve_perm(network, gen, coverage)
        else:
            return self.gibbs_improve(network, gen, coverage)                

    def gibbs_improve(self, network, gen, coverage):
        N = network.N
        windows = N // 2
        coverage_target = coverage * N**2 / 4

        start_time = time()
        
        # Precomputing for diagonal/anti-diagonal check in Gibbs sampler
        diag = np.array([[True,False],[False,True]])
        adiag = np.array([[False,True],[True,False]])
        valid = set([diag.data[0:4], adiag.data[0:4]])

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

        network.gen_info['wall_time'] += time() - start_time
        network.gen_info['coverage'] += coverage
                    
        return gen
        
    def gibbs_improve_perm(self, network, gen, coverage):
        N = network.N
        windows = N // 2
        coverage_target = coverage * N**2 / 4

        start_time = time()
        
        # Precomputing for diagonal/anti-diagonal check in Gibbs sampler
        diag = np.array([[True,False],[False,True]])
        adiag = np.array([[False,True],[True,False]])

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
            j_props = [np.where(gen[i_prop] == 1)[1] for i_prop in i_props]

            active = [np.ix_(i_prop, j_prop)
                      for i_prop, j_prop in zip(i_props, j_props)]
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

        network.gen_info['wall_time'] += time() - start_time
        network.gen_info['coverage'] += coverage
                    
        return gen
        
# P_{ij} = Logit^{-1}(kappa + o_{ij})
class Stationary(IndependentBernoulli):
    def __init__(self):
        self.kappa = 0.0
        self.fit = self.fit_convex_opt
        self.fit_info = None
        self.conf = None

    def edge_probabilities(self, network, submatrix = None,
                           ignore_offset = False):
        N = network.N
        if submatrix:
            sub_i, sub_j = submatrix
            m, n = len(i_sub), len(j_sub)

        if (not ignore_offset) and network.offset:
            logit_P = network.offset.matrix().copy()
            if submatrix:
                logit_P = logit_P[sub_i][:,sub_j]
        else:
            logit_P = np.zeros((m,n))
        logit_P += self.kappa

        return inv_logit(logit_P)

    def match_kappa(self, network, kappa_target):
        M = network.M
        N = network.N

        # kappa_target should be a tuple of the following form:
        #  ('sum', 0 < x)
        #  ('row_sum', 0 < x)
        #  ('col_sum', 0 < x)
        #  ('density', 0 < x < 1) 
        target, val = kappa_target
        def obj(kappa):
            self.kappa = kappa
            P = self.edge_probabilities(network)
            exp_edges = np.sum(P)
            if target == 'sum':
                return abs(exp_edges - val)
            elif target in 'row_sum':
                exp_degree = exp_edges / (1.0 * M)
                return abs(exp_degree - val)
            elif target == 'col_sum':
                exp_degree = exp_edges / (1.0 * N)
                return abs(exp_degree - val)
            elif target == 'density':
                exp_density = exp_edges / (1.0 * M * N)
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
        A = network.as_dense()
        T[0] = np.sum(A, dtype=np.int)

        theta = np.empty(1)
        theta[0] = logit(A.sum(dtype=np.int) / (1.0 * network.M * network.N))
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

        y = network.as_dense().reshape((N*N,))
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

    def edge_probabilities(self, network, submatrix = None,
                           ignore_offset = False):
        M = network.M
        N = network.N
        if submatrix:
            i_sub, j_sub = submatix
            m, n = len(i_sub), len(j_sub)
        else:
            m, n = M, N
        
        if (not ignore_offset) and network.offset:
            logit_P = network.offset.matrix().copy()
            if submatrix:
                logit_P = logit_P[i_sub][:,j_sub]
        else:
            logit_P = np.zeros((m,n))
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
        A = np.array(network.as_dense())
        for b, b_n in enumerate(self.beta):
            T[b] = np.sum(A * network.edge_covariates[b_n].matrix())
        T[B] = np.sum(A, dtype=np.int)

        # Initialize theta
        theta = np.zeros(B + 1)
        if fix_beta:
            for b, b_n in enumerate(self.beta):
                theta[b] = self.beta[b_n]
        theta[B] = logit(A.sum(dtype=np.int) / (1.0 * network.M * network.N))
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
        
    def fit_saddlepoint(self, network, verbose = False):
        B = len(self.beta)
        N = network.N
        A = np.array(network.as_dense())

        if not self.fit_info:
            self.fit_info = {}
        self.fit_info['cnll_evals'] = 0
        
        start_time = time()

        # Identify non-extreme sub-matrix on which saddlepoint
        # approximation is well-defined.
        i_nonextreme = range(N)
        j_nonextreme = range(N)
        while True:
            found = False
            A_j_nonextreme = A[:,j_nonextreme]
            for i in i_nonextreme:
                r = np.sum(A_j_nonextreme[i,:])
                if r == 0 or r == len(j_nonextreme):
                    found = True
                    i_nonextreme.remove(i)
            if found:
                continue

            found = False
            A_i_nonextreme = A[i_nonextreme,:]
            for j in j_nonextreme:
                c = np.sum(A_i_nonextreme[:,j])
                if c == 0 or c == len(i_nonextreme):
                    found = True
                    j_nonextreme.remove(j)
            if found:
                continue

            break

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
            A_non = A[i_nonextreme][:,j_nonextreme]
            r_non = np.sum(A_non,1)
            c_non = np.sum(A_non,0)
            P = self.edge_probabilities(network)
            P_non = P[i_nonextreme][:,j_nonextreme]
            p_denom = np.log(p_margins_saddlepoint(r_non, c_non, P_non))
            cnll = nll + p_denom
            self.fit_info['cnll_evals'] += 1
            return cnll

        bounds = [(-8,8)] * B + [(-15,15)]
        theta_opt = opt.fmin_l_bfgs_b(obj, theta, bounds = bounds)[0]
        if (np.any(theta_opt == [b[0] for b in bounds]) or
            np.any(theta_opt == [b[1] for b in bounds])):
            print 'Warning: some constraints active in model fitting.'
        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = theta_opt[b]
        self.kappa = theta_opt[B]

        self.fit_info['wall_time'] = time() - start_time

    def fit_brazzale(self, network):
        B = len(self.beta)
        if not (B == 1):
            print 'Method only applicable to scalar parameter of interest.'
            return

        N = network.N
        A = np.array(network.as_dense())
        o = network.offset.matrix()
        x = network.edge_covariates[self.beta.keys()[0]].matrix()

        y_vec = robjects.FloatVector(A.flatten())
        x_vec = robjects.FloatVector(x.flatten())
        o_vec = robjects.FloatVector(o.flatten())
        row_vec = robjects.IntVector(np.repeat(range(N), N))
        col_vec = robjects.IntVector(range(N) * N)

        dat = robjects.DataFrame({'y': y_vec, 'x': x_vec, 'o': o_vec,
                                  'row': row_vec, 'col': col_vec})
        robjects.globalenv['dat'] = dat
        robjects.r('dat <- dat[is.finite(dat$o),]')
        spec = 'glm(y ~ x + factor(row) + factor(col), ' + \
            'data=dat, family=binomial)'
        dat_glm = robjects.r(spec)
        robjects.globalenv['dat.glm'] = dat_glm
        robjects.r('write.csv(dat, file = "debug.csv")')
        dat_cond = robjects.r('cond(dat.glm, x, from=-2.5, to=2.5, n=250)')
        robjects.globalenv['dat.cond'] = dat_cond
        theta_opt = robjects.r('summary(dat.cond)$coefficients[2,1]')[0]

        self.beta[self.beta.keys()[0]] = theta_opt

        self.fit_convex_opt(network, fix_beta = True)

    def fit_conditional(self, network,
                        fit_grid = False, verbose = False, T = 0,
                        evaluate = False):
        B = len(self.beta)
        if fit_grid and not B in (1,2):
            print 'Can only grid search B = 1, 2. Defaulting to minimizer.'
            fit_grid = False

        if not self.fit_info:
            self.fit_info = {}
        self.fit_info['cnll_evals'] = 0

        start_time = time()

        A = np.array(network.as_dense())
        r, c = A.sum(1, dtype=np.int), A.sum(0, dtype=np.int)

        def obj(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing objective for nan-containing vector.'
                return np.Inf
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            P = self.edge_probabilities(network)
            w = P / (1.0 - P)

            if T == 0:
                if (np.all(w == 0.0) or np.all(w == np.Inf) or
                    np.any(np.isnan(w))):
                    cnll = np.Inf
                else:
                    cnll = acnll(A, w, sort_by_wopt_var = False)
            else:
                z = approximate_from_margins_weights(r, c, w, T,
                                                     sort_by_wopt_var = True)
                if not verbose:
                    logkappa = log_partition_is(z)
                else:
                    logkappa, logcvsq = log_partition_is(z, cvsq = True)
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
            if evaluate:
                theta = np.zeros(B)
                for b, b_n in enumerate(self.beta):
                    theta[b] = self.beta[b_n]

                cnll = obj(theta)
                self.fit_info['wall_time'] = time() - start_time

                return cnll
            else:
                # Initialize theta
                theta = np.zeros(B)

                if T > 0:
                    # Use Kiefer-Wolfowitz stochastic approximation
                    scale = 1.0 / obj(np.repeat(0, B))
                    for n in range(1, 40):
                        a_n = 2.0 * scale * n ** (-1.0)
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
                else:
                    if B == 1:
                        obj_scalar = lambda x: obj(np.array([x]))
                        res = opt.minimize_scalar(obj_scalar,
                                                  method = 'bounded',
                                                  bounds = (-10, 10))
                        self.beta[self.beta.keys()[0]] = res.x
                    else:
                        theta_opt = opt.fmin(obj, theta)
                        for b, b_n in enumerate(self.beta):
                            self.beta[b_n] = theta_opt[b]

        self.fit_convex_opt(network, fix_beta = True)

        self.fit_info['wall_time'] = time() - start_time

    def fit_c_conditional(self, network, verbose = False, evaluate = False):
        M = network.M
        N = network.N
        B = len(self.beta)

        if not self.fit_info:
            self.fit_info = {}
        self.fit_info['c_cnll_evals'] = 0

        start_time = time()

        # Pre-compute column statistics
        T_c = {}
        A = np.array(network.as_dense())
        c = np.sum(A, axis = 0, dtype=np.int)
        for b, b_n in enumerate(self.beta):
            T_b = A * network.edge_covariates[b_n].matrix()
            for j in range(N):
                T_c[(j,b)] = T_b[:,j]
        z = {}
        for j in range(N):
            z[j] = A[:,j].flatten()

        def obj(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing objective for vector with nan.'
                return np.Inf
            c_cnll = 0
            alpha = np.zeros(M)
            alpha[0:(M-1)] = theta[B:(B + (M-1))]
            for j in range(N):
                c_j = c[j]

                if ((c_j == 0) or (c_j == M)):
                    continue

                logit_P = np.zeros(M)

                logit_P += alpha
                for b in range(B):
                    logit_P += theta[b] * T_c[(j,b)]
                P = inv_logit(logit_P)
                if np.sum(P) == 0:
                    continue

                log_P = np.log(P)
                log1m_P = np.log1p(-P)

                # Probability of column sum equal to c_j
                log_p_c = np.zeros(c_j + 1)
                for k in range(0, M):
                    log_bk, log1m_bk = log_P[k], log1m_P[k]

                    if k <= c_j:
                        i_init = k
                        if k < c_j:
                            log_p_c[k+1] = log_p_c[k] + log_bk
                    else:
                        i_init = c_j

                    for i in range(i_init, 0, -1):
                        x = log_p_c[i-1] + log_bk
                        y = log_p_c[i] + log1m_bk
                        x, y = min(x, y), max(x, y)
                        log_p_c[i] = y + np.log1p(np.exp(x-y))

                    log_p_c[0] += log1m_bk

                c_cnll += log_p_c[c_j]

                # Marginal probability of columns with given column sum
                z_j = z[j]
                c_cnll -= np.sum(log_P[z_j]) + np.sum(log1m_P[-z_j])

            self.fit_info['c_cnll_evals'] += 1
            if verbose:
                print c_cnll, theta
            return c_cnll

        if evaluate:
            theta = np.zeros(B + (M-1))
            for b, b_n in enumerate(self.beta):
                theta[b] = self.beta[b_n]
            theta[B:(B + (M-1))] = network.row_covariates['alpha_out'][0:(M-1)]
            c_cnll = obj(theta)
            self.fit_info['wall_time'] = time() - start_time

            return c_cnll
        else:
            # Initialize theta
            theta = np.zeros(B + (M-1))

            bounds = [(-8,8)] * B + [(-6,6)] * (M-1)
            theta_opt = opt.fmin_l_bfgs_b(obj, theta, bounds = bounds,
                                          approx_grad = True)[0]
            if (np.any(theta_opt == [b[0] for b in bounds]) or
                np.any(theta_opt == [b[1] for b in bounds])):
                print 'Warning: some constraints active in model fitting.'
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta_opt[b]

            self.fit_convex_opt(network, fix_beta = True)

            self.fit_info['wall_time'] = time() - start_time
        
    def fit_composite(self, network, T = 100, verbose = False):
        B = len(self.beta)

        if not self.fit_info:
            self.fit_info = {}

        start_time = time()

        A = np.array(network.as_dense())
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
                if active > 8:
                    print 'Skipping row pair with excessive active columns.'
                    continue
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

        y = network.as_dense().reshape((N*N,))
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

        M = network.M
        N = network.N
        B = len(self.beta)

        y = network.as_dense().reshape((M*N,))
        Phi = np.zeros((M*N,B))
        for b, b_n in enumerate(self.beta):
            Phi[:,b] =  network.edge_covariates[b_n].matrix().reshape((M*N,))
        lr = LogisticRegression(fit_intercept = True,
                                C = 1.0 / prior_precision,
                                penalty = 'l2')
        try:
            lr.fit(Phi, y)
            coefs, intercept = lr.coef_[0], lr.intercept_[0]
        except:
            print 'Warning: regularized logistic fit failed.'
            coefs, intercept = np.zeros(B), 0

        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = coefs[b]
        self.kappa = intercept

        if variance_covariance:
            S_0_inv = prior_precision * np.eye(B + 1)
            Phi_kappa = np.hstack([Phi, np.ones((M*N,1))])
            w = np.empty(B + 1)
            w[0:B] = coefs
            w[B] = intercept
            C = 0.0
            for i in range(M*N):
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

    def confidence(self, network, num_bootstrap = 50, alpha = 0.05,
                   **fit_options):
        # Point estimate
        self.fit(network, **fit_options)
        theta_hats = { b: self.beta[b] for b in self.beta }

        # Parametric bootstrap to characterize uncertainty in point estimate
        network_samples = []
        for n in range(num_bootstrap):
            network_samples.append(self.generate(network))
        network_original = network.network.copy()
        theta_hat_bootstraps = { b: np.empty(num_bootstrap) for b in self.beta }
        for n in range(num_bootstrap):
            network.network = network_samples[n]
            self.fit(network, **fit_options)
            for b in theta_hat_bootstraps:
                theta_hat_bootstraps[b][n] = self.beta[b]
        network.network = network_original

        # Initialize data structure to hold confidence intervals
        if not self.conf:
            self.conf = {}
        for b in self.beta:
            if not b in self.conf:
                self.conf[b] = {}

        # Construct (asymptotically valid) confidence interval
        for b in self.conf:
            p_l, p_u = alpha / 2.0, 1.0 - alpha / 2.0
            theta_hat = theta_hats[b]
            theta_hat_bootstrap = theta_hat_bootstraps[b]
            self.conf[b]['percentile'] = \
                (np.percentile(theta_hat_bootstrap, 100.0 * p_l),
                 np.percentile(theta_hat_bootstrap, 100.0 * p_u))
            self.conf[b]['pivotal'] = \
                (2*theta_hat - np.percentile(theta_hat_bootstrap, 100.0 * p_u),
                 2*theta_hat - np.percentile(theta_hat_bootstrap, 100.0 * p_l))
            z_score = norm().ppf(p_u)
            theta_hat_se = np.sqrt(np.mean((theta_hat_bootstrap-theta_hat)**2))
            self.conf[b]['normal'] = \
                (theta_hat - z_score * theta_hat_se,
                 theta_hat + z_score * theta_hat_se)

    # Implementation of ideas from "Conservative Hypothesis Tests and
    # Confidence Intervals using Importance Sampling" (Harrison, 2012).
    def confidence_harrison(self, network, b, alpha_level = 0.05, n_MC = 100,
                            L = 121, beta_l_min = -6.0, beta_l_max = 6.0):
        N = network.N
        A = np.array(network.as_dense())

        x = network.edge_covariates[b].matrix()

        # Generate beta grid for inference
        beta_grid = np.linspace(beta_l_min, beta_l_max, L)

        # Observed statistic
        t_X = np.sum(A * x)

        # Row and column margins; the part of the data we can use to design Q
        r, c = A.sum(1), A.sum(0)

        # Generate samples from the mixture proposal distribution
        Y = []
        for n in range(n_MC):
            l = np.random.randint(L)
            logit_P_l = beta_grid[l] * x
        
            Y_sparse = approximate_from_margins_weights(r, c, np.exp(logit_P_l))
            Y_dense = np.zeros((N,N), dtype = np.bool)
            for i, j in Y_sparse:
                if i == -1: break
                Y_dense[i,j] = 1
            Y.append(Y_dense)

        # Statistics for the samples from the proposal distribution only
        # need to be calculated once...
        t_Y = np.empty(n_MC)
        for n in range(n_MC):
            t_Y[n] = np.sum(Y[n] * x)
        I_t_Y_plus = t_Y >= t_X
        I_t_Y_minus = -t_Y >= -t_X

        # Probabilities under each component of the proposal distribution
        # only need to be calculated once...
        log_Q_X = np.empty(L)
        log_Q_Y = np.empty((L,n_MC))
        for l in range(L):
            logit_P_l = beta_grid[l] * x
            log_Q_X[l] = -acnll(A, np.exp(logit_P_l))
            for n in range(n_MC):
                log_Q_Y[l,n] = -acnll(Y[n], np.exp(logit_P_l))
        Q_sum_X = np.exp(np.logaddexp.reduce(log_Q_X))
        Q_sum_Y = np.empty(n_MC)
        for n in range(n_MC):
            Q_sum_Y[n] = np.exp(np.logaddexp.reduce(log_Q_Y[:,n]))

        # Step over the grid, calculating approximate p-values
        p_plus = np.empty(L)
        p_minus = np.empty(L)
        for l in range(L):
            beta_l = beta_grid[l]

            p_num_plus, p_num_minus, p_denom = 0.0, 0.0, 0.0

            # X contribution
            w_X = np.exp(beta_l * t_X) / Q_sum_X
            p_num_plus += w_X
            p_num_minus += w_X
            p_denom += w_X

            # Y contribution
            for n in range(n_MC):
                w_Y = np.exp(beta_l * t_Y[n]) / Q_sum_Y[n]
                if I_t_Y_plus[n]: p_num_plus += w_Y
                if I_t_Y_minus[n]: p_num_minus += w_Y
                p_denom += w_Y

            p_plus[l] = p_num_plus / p_denom
            p_minus[l] = p_num_minus / p_denom

        p_plus_minus = np.fmin(1, 2 * np.fmin(p_plus, p_minus))

        C_alpha = beta_grid[p_plus_minus > alpha_level]

        if not self.conf:
            self.conf = {}
        if not b in self.conf:
            self.conf[b] = {}
        l, u = np.min(C_alpha), np.max(C_alpha)
        if l == beta_l_min:
            l = -np.inf
        if u == beta_l_max:
            u = np.inf
        self.conf[b]['harrison'] = (l, u)

    # Generalization of conservative importance sampling confidence
    # intervals to two parameters.
    def confidence_harrison_2(self, network, a, alpha_level = 0.05, n_MC = 20,
                              L = 20, beta_l_min = -6.0, beta_l_max = 6.0):
        N = network.N
        A = network.as_dense()
        b_1, b_2 = self.beta.keys()

        x_1 = network.edge_covariates[b_1].matrix()
        x_2 = network.edge_covariates[b_2].matrix()

        # Generate beta grid for inference
        beta_grid_1 = np.linspace(beta_l_min, beta_l_max, L)
        beta_grid_2 = np.linspace(beta_l_min, beta_l_max, L)

        # Observed statistic
        t_X = np.sum(A * (x_1 + x_2))

        # Row and column margins; the part of the data we can use to design Q
        r, c = A.sum(1), A.sum(0)

        # Generate samples from the mixture proposal distribution
        Y = []
        for n in range(n_MC):
            l = np.random.randint(L)
            logit_P_l = beta_grid[l] * x
        
            Y_sparse = approximate_from_margins_weights(r, c, np.exp(logit_P_l))
            Y_dense = np.zeros((N,N), dtype = np.bool)
            for i, j in Y_sparse:
                if i == -1: break
                Y_dense[i,j] = 1
            Y.append(Y_dense)

        # Statistics for the samples from the proposal distribution only
        # need to be calculated once...
        t_Y = np.empty(n_MC)
        for n in range(n_MC):
            t_Y[n] = np.sum(Y[n] * x)
        I_t_Y_plus = t_Y >= t_X
        I_t_Y_minus = -t_Y >= -t_X

        # Probabilities under each component of the proposal distribution
        # only need to be calculated once...
        log_Q_X = np.empty(L)
        log_Q_Y = np.empty((L,n_MC))
        for l in range(L):
            logit_P_l = beta_grid[l] * x
            log_Q_X[l] = -acnll(A, np.exp(logit_P_l))
            for n in range(n_MC):
                log_Q_Y[l,n] = -acnll(Y[n], np.exp(logit_P_l))
        Q_sum_X = np.exp(np.logaddexp.reduce(log_Q_X))
        Q_sum_Y = np.empty(n_MC)
        for n in range(n_MC):
            Q_sum_Y[n] = np.exp(np.logaddexp.reduce(log_Q_Y[:,n]))

        # Step over the grid, calculating approximate p-values
        p_plus = np.empty(L)
        p_minus = np.empty(L)
        for l in range(L):
            beta_l = beta_grid[l]

            p_num_plus, p_num_minus, p_denom = 0.0, 0.0, 0.0

            # X contribution
            w_X = np.exp(beta_l * t_X) / Q_sum_X
            p_num_plus += w_X
            p_num_minus += w_X
            p_denom += w_X

            # Y contribution
            for n in range(n_MC):
                w_Y = np.exp(beta_l * t_Y[n]) / Q_sum_Y[n]
                if I_t_Y_plus[n]: p_num_plus += w_Y
                if I_t_Y_minus[n]: p_num_minus += w_Y
                p_denom += w_Y

            p_plus[l] = p_num_plus / p_denom
            p_minus[l] = p_num_minus / p_denom

        p_plus_minus = np.fmin(1, 2 * np.fmin(p_plus, p_minus))

        C_alpha = beta_grid[p_plus_minus > alpha_level]
        return np.min(C_alpha), np.max(C_alpha)
    
# P_{ij} = Logit^{-1}(alpha_out_i + alpha_in_j + \sum_b x_{bij}*beta_b + kappa +
#                     o_{ij})
# Constraints: \sum_i alpha_out_i = 0, \sum_j alpha_in_j = 0
class NonstationaryLogistic(StationaryLogistic):
    def __init__(self):
        StationaryLogistic.__init__(self)
        self.fit = self.fit_convex_opt
        
    def edge_probabilities(self, network, submatrix = None,
                           ignore_offset = False):
        M = network.M
        N = network.N
        if submatrix:
            i_sub, j_sub = submatrix
            m, n = len(i_sub), len(j_sub)
        else:
            m, n = M, N

        alpha_out = network.row_covariates['alpha_out']
        alpha_in = network.col_covariates['alpha_in']
        if submatrix:
            alpha_out = alpha_out[i_sub]
            alpha_in = alpha_in[j_sub]
        
        if (not ignore_offset) and network.offset:
            logit_P = network.offset.matrix().copy()
            if submatrix:
                logit_P = logit_P[i_sub][:,j_sub]
        else:
            logit_P = np.zeros((m,n))
        np.add(logit_P, alpha_out[:].reshape((-1,1)), logit_P)
        np.add(logit_P, alpha_in[:].reshape((1,-1)), logit_P)
        for b in self.beta:
            ec_b = network.edge_covariates[b].matrix()
            if submatrix:
                ec_b = ec_b[i_sub][:,j_sub]
            logit_P += self.beta[b] * ec_b
        logit_P += self.kappa
        
        return inv_logit(logit_P)

    def baseline(self, network):
        M = network.M
        N = network.N
        P = self.edge_probabilities(network)
        def params_to_Q(params):
            a = params[0:M]
            b = params[M:(M + N)]
            c = params[(M+N)]
            logit_Q = np.zeros((M,N))
            for i in range(M):
                logit_Q[i,:] += a[i]
            for j in range(N):
                logit_Q[:,j] += b[j]
            logit_Q += c
            return inv_logit(logit_Q)
        params = np.zeros(M + N + 1)
        def obj(params):
            Q = params_to_Q(params)
            return np.reshape((P - Q), (M*N,))
        best_params = opt.leastsq(obj, params)[0]
        Q = params_to_Q(best_params)
        return Q

    def baseline_logit(self, network):
        M = network.M
        N = network.N
        logit_P = logit(self.edge_probabilities(network))
        a, b = logit_P.mean(1), logit_P.mean(0)
        a_mean, b_mean = a.mean(), b.mean()
        a -= a_mean
        b -= b_mean
        c = a_mean + b_mean
        logit_Q = np.zeros((M,N))
        for i in range(M):
            logit_Q[i,:] += a[i]
        for j in range(N):
            logit_Q[:,j] += b[j]
        logit_Q += c
        return logit_Q

    def fit_convex_opt(self, network, verbose = False, fix_beta = False):
        M = network.M
        N = network.N
        B = len(self.beta)

        if not self.fit_info:
            self.fit_info = {}
        self.fit_info['nll_evals'] = 0
        self.fit_info['grad_nll_evals'] = 0
        self.fit_info['grad_nll_final'] = np.empty(B + 1 + (M-1) + (N-1))

        start_time = time()

        if network.offset:
            O = network.offset.matrix()
        alpha_zero(network)

        # Calculate observed sufficient statistics
        T = np.empty(B + 1 + (M-1) + (N-1))
        A = np.array(network.as_dense())
        r = np.sum(A, axis = 1, dtype=np.int)[0:(M-1)]
        c = np.sum(A, axis = 0, dtype=np.int)[0:(N-1)]
        T[(B + 1):(B + 1 + (M-1))] = r
        T[(B + 1 + (M-1)):(B + 1 + (M-1) + (N-1))] = c
        for b, b_n in enumerate(self.beta):
            T[b] = np.sum(A * network.edge_covariates[b_n].matrix())
        T[B] = np.sum(A, dtype=np.int)

        # Initialize theta
        theta = np.zeros(B + 1 + (M-1) + (N-1))
        if fix_beta:
            for b, b_n in enumerate(self.beta):
                theta[b] = self.beta[b_n]
        theta[B] = logit(A.sum(dtype=np.int) / (1.0 * M * N))
        if network.offset:
            theta[B] -= logit_mean(O)
        theta[(B + 1):(B + 1 + (M-1) + (N-1))] = -theta[B]
        for i in range(M-1):
            theta[B + 1 + i] += \
              logit((A[i,:].sum(dtype=np.int) + 1.0) / (N + 1.0))
            if network.offset:
                o_row = logit_mean(O[i,:])
                if np.isfinite(o_row):
                    theta[B + 1 + i] -= o_row
        for j in range(N-1):
            theta[B + 1 + (M-1) + j] += \
              logit((A[:,j].sum(dtype=np.int) + 1.0) / (M + 1.0))
            if network.offset:
                o_col = logit_mean(O[:,j])
                if np.isfinite(o_col):
                    theta[B + 1 + (M-1) + j] -= o_col

        alpha_out = network.row_covariates['alpha_out']
        alpha_in = network.col_covariates['alpha_in']
        def obj(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing objective for nan-containing vector.'
                return np.Inf
            alpha_out[0:M-1] = theta[(B + 1):(B + 1 + (M-1))]
            alpha_in[0:N-1] = theta[(B + 1 + (M-1)):(B + 1 + (M-1) + (N-1))]
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            self.kappa = theta[B]
            nll = self.nll(network)
            self.fit_info['nll_evals'] += 1
            return nll
        def grad(theta):
            if np.any(np.isnan(theta)):
                print 'Warning: computing gradient for nan-containing vector.'
                return np.zeros(B + 1 + (M-1) + (N-1))
            alpha_out[0:M-1] = theta[(B + 1):(B + 1 + (M-1))]
            alpha_in[0:N-1] = theta[(B + 1 + (M-1)):(B + 1 + (M-1) + (N-1))]
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta[b]
            self.kappa = theta[B]
            ET = np.empty(B + 1 + (M-1) + (N-1))
            P = self.edge_probabilities(network)
            Er = np.sum(P, axis = 1)[0:(M-1)]
            Ec = np.sum(P, axis = 0)[0:(N-1)]
            ET[(B + 1):(B + 1 + (M-1))] = Er
            ET[(B + 1 + (M-1)):(B + 1 + (M-1) + (N-1))] = Ec
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

        bounds = [(-8,8)] * B + [(-15,15)] + [(-8,8)] * ((M-1) + (N-1))
        theta_opt = opt.fmin_l_bfgs_b(obj, theta, grad, bounds = bounds)[0]
        if (np.any(theta_opt == [b[0] for b in bounds]) or
            np.any(theta_opt == [b[1] for b in bounds])):
            print 'Warning: some constraints active in model fitting.'
        alpha_out[0:M-1] = theta_opt[(B + 1):(B + 1 + (M-1))]
        alpha_in[0:N-1] = theta_opt[(B + 1 + (M-1)):(B + 1 + (M-1) + (N-1))]
        alpha_out_mean = np.mean(alpha_out[:])
        alpha_in_mean = np.mean(alpha_in[:])
        alpha_out[:] -= alpha_out_mean
        alpha_in[:] -= alpha_in_mean
        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = theta_opt[b]
        self.kappa = theta_opt[B] + alpha_out_mean + alpha_in_mean

        self.fit_info['wall_time'] = time() - start_time

    def fit_irls(self, network, verbose = False, perturb = 1e-4):
        M = network.M
        N = network.N
        B = len(self.beta)
        P = B + 1 + (M-1) + (N-1)

        if not self.fit_info:
            self.fit_info = {}

        start_time = time()

        alpha_zero(network)
        alpha_out = network.row_covariates['alpha_out']
        alpha_in = network.col_covariates['alpha_in']
        
        # Construct response and design matrices
        y = np.asarray(network.as_dense(), dtype='float64')
        y = y.reshape((M*N,1))
        X = np.zeros((M*N,P))
        for b, b_n in enumerate(self.beta):
            X[:,b] =  network.edge_covariates[b_n].matrix().reshape((M*N,))
        X[:,B] = 1.0
        for r in range(M-1):
            X_row = np.zeros((M,N))
            X_row[r,:] = 1.0
            X[:,B + 1 + r] = X_row.reshape((M*N,))
        for c in range(N-1):
            X_col = np.zeros((M,N))
            X_col[:,c] = 1.0
            X[:,B + 1 + (M-1) + c] = X_col.reshape((M*N,))

        theta = np.zeros((P,1))

        def fitted_p(theta):
            theta_vec = np.reshape(theta, (P,))
            alpha_out[0:M-1] = theta_vec[(B + 1):(B + 1 + (M-1))]
            alpha_in[0:N-1] = theta_vec[(B + 1 + (M-1)):P]
            for b, b_n in enumerate(self.beta):
                self.beta[b_n] = theta_vec[b]
            self.kappa = theta_vec[B]
            return self.edge_probabilities(network).reshape((M*N,1))

        for iter in range(10):
            p = fitted_p(theta)
            X_tilde = X * p
            del p
            for j in range(P):
                X_tilde[:,j] += np.random.uniform(-perturb, perturb, M*N)
            X_t = np.transpose(X)
            X_t_X_tilde = np.dot(X_t, X_tilde)
            del X_tilde
            hat = solve(X_t_X_tilde, X_t, overwrite_a = True)
            p = fitted_p(theta)
            theta += np.dot(hat, (y - p))

        theta_vec = np.reshape(theta, (P,))
        alpha_out[0:M-1] = theta_vec[(B + 1):(B + 1 + (M-1))]
        alpha_in[0:N-1] = theta_vec[(B + 1 + (M-1)):P]
        alpha_out_mean = np.mean(alpha_out[:])
        alpha_in_mean = np.mean(alpha_in[:])
        alpha_out[:] -= alpha_out_mean
        alpha_in[:] -= alpha_in_mean
        for b, b_n in enumerate(self.beta):
            self.beta[b_n] = theta_vec[b]
        self.kappa = theta_vec[B] + alpha_out_mean + alpha_in_mean

        self.fit_info['wall_time'] = time() - start_time
        
    def fit_logistic(self, network):
        import statsmodels.api as sm

        M = network.M
        N = network.N
        B = len(self.beta)
        alpha_zero(network)

        y = network.as_dense().reshape((M*N,))
        Phi = np.zeros((M*N,B + 1 + (M-1) + (N-1)))
        for b, b_n in enumerate(self.beta):
            Phi[:,b] =  network.edge_covariates[b_n].matrix().reshape((M*N,))
        Phi[:,B] = 1.0
        for r in range(M-1):
            phi_row = np.zeros((M,N))
            phi_row[r,:] = 1.0
            Phi[:,B + 1 + r] = phi_row.reshape((M*N,))
        for c in range(N-1):
            phi_col = np.zeros((M,N))
            phi_col[:,c] = 1.0
            Phi[:,B + 1 + (M-1) + c] = phi_col.reshape((M*N,))
        if network.offset:
            offset = network.offset.matrix().reshape((M*N,))
            coefs = sm.GLM(y, Phi, sm.families.Binomial(), offset).fit().params
        else:
            coefs = sm.Logit(y, Phi).fit().params

        alpha_out = network.row_covariates['alpha_out']
        alpha_in = network.col_covariates['alpha_in']
        alpha_out[0:M-1] = coefs[(B + 1):(B + 1 + (M-1))]
        alpha_in[0:N-1] = coefs[(B + 1 + (M-1)):(B + 1 + (M-1) + (N-1))]
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

    # The network is needed for its covariates and degree
    # heterogeneity terms, not for the observed pattern of edges, etc.
    #
    # Typically, the inverse Fisher information matrix will be more
    # useful (it gives a lower bound on the variances/covariances of
    # an unbised estimator), so that is calculated by default.
    def fisher_information(self, network, inverse = True):
        M = network.M
        N = network.N
        B = len(self.beta)

        P = self.edge_probabilities(network)

        x = np.empty((B,M,N))
        for i, b in enumerate(self.beta):
            x[i] = network.edge_covariates[b].matrix()

        P_bar = P * (1.0 - P)

        I = np.zeros(((M-1) + (N-1) + 1 + B, (M-1) + (N-1) + 1 + B))
        for i in range(M-1):
            for j in range(N-1):
                v = P_bar[i,j]
                I[i,(M-1)+j] = v
                I[(M-1)+j,i] = v
        for i in range(M-1):
            v = np.sum(P_bar[i,:])
            I[i,i] = v
            I[(M-1) + (N-1),i] = v
            I[i,(M-1) + (N-1)] = v
            for b in range(B):
                v = np.sum(x[b,i,:] * P_bar[i,:])
                I[(M-1) + (N-1) + 1 + b,i] = v
                I[i,(M-1) + (N-1) + 1 + b] = v
        for j in range(N-1):
            v = np.sum(P_bar[:,j])
            I[(M-1) + j,(M-1) + j] = v
            I[(M-1) + (N-1),(M-1) + j] = v
            I[(M-1) + j,(M-1) + (N-1)] = v
            for b in range(B):
                v = np.sum(x[b,:,j] * P_bar[:,j])
                I[(M-1) + (N-1) + 1 + b,(M-1) + j] = v
                I[(M-1) + j,(M-1) + (N-1) + 1 + b] = v
        I[(M-1) + (N-1),(M-1) + (N-1)] = np.sum(P_bar)
        for b in range(B):
            v = np.sum(x[b] * P_bar)
            I[(M-1) + (N-1) + 1 + b,(M-1) + (N-1)] = v
            I[(M-1) + (N-1),(M-1) + (N-1) + 1 + b] = v
        for b_1 in range(B):
            for b_2 in range(B):
                v = np.sum(x[b_1] * x[b_2] * P_bar)
                I[(M-1) + (N-1) + 1 + b_1,(M-1) + (N-1) + 1 + b_2] = v
                I[(M-1) + (N-1) + 1 + b_2,(M-1) + (N-1) + 1 + b_1] = v

        if inverse: I_inv = inv(I)

        I_names = ['alpha_{%s}' % n for n in network.rnames[0:(M-1)]] + \
            ['beta_{%s}' % n for n in network.cnames[0:(N-1)]] + \
            ['kappa'] + \
            ['theta_{%s}' % b for b in self.beta]
        self.I = {}
        if inverse: self.I_inv = {}
        for i in range((M-1) + (N-1) + 1 + B):
            for j in range((M-1) + (N-1) + 1 + B):
                if i == j:
                    self.I[I_names[i]] = I[i,i]
                    if inverse: self.I_inv[I_names[i]] = I_inv[i,i]
                else:
                    self.I[(I_names[i],I_names[j])] = I[i,j]
                    if inverse: self.I_inv[(I_names[i],I_names[j])] = I_inv[i,j]

# P_{ij} = Logit^{-1}(base_model(i,j) + Theta_{z_i,z_j})
# Constraints: \sum_{i,j} z_{i,j} = 0
class Blockmodel(IndependentBernoulli):
    def __init__(self, base_model, K, block_name = 'z',
                 ignore_inner_offset = False):
        self.base_model = base_model
        self.K = K
        self.Theta = np.zeros((K,K))
        self.block_name = block_name
        self.fit = self.fit_sem
        self.ignore_inner_offset = ignore_inner_offset
        
    def apply_to_offset(self, network):
        N = network.N
        z = network.node_covariates[self.block_name]
        for i in range(N):
            for j in range(N):
                network.offset[i,j] += self.Theta[z[i], z[j]]

    def edge_probabilities(self, network, submatrix = None,
                           ignore_offset = None):
        if ignore_offset is None:
           ignore_inner_offset = self.ignore_inner_offset
        else:
            ignore_inner_offset = ignore_offset

        old_offset = network.offset.copy()
        if network.offset is None:
            network.initialize_offset()
        self.apply_to_offset(network)

        # If outer ignore_offset, then the inner offset is just block
        # effects, and so should be used. If not outer ignore_offset,
        # then the offset should be used anyways.
        P = self.base_model.edge_probabilities(network, submatrix, \
              ignore_offset = ignore_inner_offset)

        network.offset = old_offset

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

    # Stochastic EM fitting with `sweeps` Gibbs sweeps in the E-step
    # and `cycles` repetitions of the entire E-step/M-step operation
    #
    # This fitting procedure requires that `base_model` can handle
    # edge covariate effects and a kappa term.
    def fit_sem(self, network, cycles = 20, sweeps = 5, store_all = False,
                use_best = True, **base_fit_options):
        # Local aliases for convenience
        K, Theta = self.K, self.Theta
        N = network.N
        z = network.node_covariates[self.block_name]
        A = np.array(network.as_dense())

        self.sem_trace = []

        cov_name_to_inds = {}
        def fit_at_z(z, Theta):
            for s in range(K):
                for t in range(K):
                    if s == 0 and t == 0: continue
                    cov_name = '_%d_%d' % (s,t)
                    cov_name_to_inds[cov_name] = (s,t)
                    cov = network.new_edge_covariate(cov_name)
                    def f_edge_class(i_1, i_2):
                        return (z[i_1] == s) and (z[i_2] == t)
                    cov.from_binary_function_ind(f_edge_class)
                    self.base_model.beta[cov_name] = None
                    
            self.base_model.fit(network, **base_fit_options)

            Theta[0,0] = 0.0
            for cov_name in cov_name_to_inds:
                s, t = cov_name_to_inds[cov_name]
                Theta[s,t] = self.base_model.beta[cov_name]
                network.edge_covariates.pop(cov_name)
                self.base_model.beta.pop(cov_name)
            Theta_mean = np.mean(Theta)
            Theta -= Theta_mean
            self.base_model.kappa += Theta_mean

        for cycle in range(cycles):
            # M-step
            fit_at_z(z, Theta)
            
            # Stochastic E-step
            for gibbs_step in range(sweeps * N):
                l = np.random.randint(N)
                logprobs = np.empty(K)
                for k in range(K):
                    logprobs[k] = (np.dot(Theta[k,z[:]], A[l,:]) +
                                   np.dot(Theta[z[:],k], A[:,l]) -
                                   (Theta[k,k] * A[l,l]))
                logprobs -= np.max(logprobs)
                probs = np.exp(logprobs)
                probs /= np.sum(probs)
                z[l] = np.where(np.random.multinomial(1, probs) == 1)[0][0]

            nll = self.nll(network, ignore_offset = self.ignore_inner_offset)
            if use_best:
                self.sem_trace.append((nll, z.copy()))

        if use_best:
            best_nll, best_c = np.inf, 0
            for c in range(cycles):
                if self.sem_trace[c][0] < best_nll:
                    best_nll, best_c = self.sem_trace[c][0], c
            z[:] = self.sem_trace[best_c][1][:]
            fit_at_z(self.sem_trace[best_c][1], Theta)

    # Blockmodel fitting using the algorithm given in Karrer and
    # Newman (2011) that the authors claim is inspired by the
    # Kernighan-Lin algorithm.
    #
    # Note that the `cycles` in this algorithm may require as much as
    # O(K*N^2) times as many model fits as the `cycles` in the SEM
    # algorithm! It's doubtful that this can be improved in the most
    # general case...
    def fit_kl(self, network, cycles = 5, sweeps = None, **base_fit_options):
        ## FIXME: "sweeps" argument is ignored!

        K, Theta = self.K, self.Theta
        N = network.N
        z = network.node_covariates[self.block_name]
        A = np.array(network.as_dense())

        z_to_nll_cache = {}
        cov_name_to_inds = {}
        def fit_at_z(z):
            z_hash = hashlib.sha1(z[:].view(np.uint8)).hexdigest()
            if z_hash in z_to_nll_cache:
                return z_to_nll_cache[z_hash]
            
            for s in range(K):
                for t in range(K):
                    if s == 0 and t == 0: continue
                    cov_name = '_%d_%d' % (s,t)
                    cov_name_to_inds[cov_name] = (s,t)
                    cov = network.new_edge_covariate(cov_name)
                    def f_edge_class(i_1, i_2):
                        return (z[i_1] == s) and (z[i_2] == t)
                    cov.from_binary_function_ind(f_edge_class)
                    self.base_model.beta[cov_name] = None
                    
            self.base_model.fit(network, **base_fit_options)

            nll = self.nll(network, ignore_offset = self.ignore_offset)
            z_to_nll_cache[z_hash] = nll
            return nll
            
        for cycle in range(cycles):
            z_states = []
            unmoved = range(N)
            np.random.shuffle(unmoved)
            while len(unmoved) > 0:
                # Defaults may actually be used when NLL calculation
                # behaves poorly...
                best_nll, best_m, best_z_m = np.inf, unmoved[0], 0
                for m in unmoved:
                    z_m_current = z[m]
                    for z_m in range(K):
                        if z_m == z_m_current:
                            continue
                        z[m] = z_m
                        nll = fit_at_z(z)
                        if nll < best_nll:
                            best_nll, best_m, best_z_m = nll, m, z_m
                    z[m] = z_m_current
                z[best_m] = best_z_m
                unmoved.remove(best_m)
                z_states.append((best_nll, z[:].copy()))

            best_nll, best_i = np.inf, 0
            for i in range(N):
                nll = z_states[i][0]
                if nll < best_nll:
                    best_nll, best_i = nll, i
            z[:] = z_states[best_i][1]
            
        Theta[0,0] = 0.0
        for cov_name in cov_name_to_inds:
            s, t = cov_name_to_inds[cov_name]
            Theta[s,t] = self.base_model.beta[cov_name]
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
            r = np.asarray(network.as_dense()).sum(1)
        else:
            r = network.node_covariates[self.r_name][:]

        if not self.c_name in network.node_covariates:
            print 'Covariate "%s" not found.' % self.c_name
            c = np.asarray(network.as_dense()).sum(0)
        else:
            c = network.node_covariates[self.c_name][:]

        return self.base_model.generate_margins(network, r, c, self.coverage,
                                                **opts)

    def nll(self, network, **opts):
        return self.base_model.nll(network, **opts)

    def edge_probabilities(self, network, submatrix = None):
        return self.base_model.edge_probabilities(network, submatrix)
    
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

def alpha_f(network, f):
    a = f(network.M)
    b = f(network.N)

    network.new_row_covariate('alpha_out')[:] = a
    network.new_col_covariate('alpha_in')[:] = b

def alpha_zero(network):
    alpha_f(network, lambda l: np.tile(0.0, l))
    
def alpha_norm(network, alpha_sd):
    alpha_f(network, lambda l: center(np.random.normal(0, alpha_sd, l)))

def alpha_unif(network, alpha_sd):
    c = np.sqrt(12) / 2
    alpha_f(network,
            lambda l: center(np.random.uniform(-alpha_sd*c, alpha_sd * c, l)))

def alpha_gamma(network, alpha_loc, alpha_scale):
    alpha_f(network,
            lambda l: center(np.random.gamma(alpha_loc, alpha_scale, l)))
