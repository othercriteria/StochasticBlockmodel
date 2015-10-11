#!/usr/bin/env python

# Algorithms for working for random binary matrices with margin constraints
# Daniel Klein, 5/11/2012

from __future__ import division
import numpy as np
from scipy import optimize as opt
import hashlib

from Utility import logsumexp, logabsdiffexp

# Get machine precision; used to prevent divide by zero
eps0 = np.spacing(0)

# See if C support code can be loaded
try:
    import ctypes

    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_int = ctypes.c_int
    c_double = ctypes.c_double

    support_library = ctypes.cdll.LoadLibrary('./support.so')

    support_library.fill_G.argtypes = [c_int_p, c_int, c_int, c_int,
                                       c_double_p, c_double_p, c_double_p]
    support_library.fill_G.restype = ctypes.c_void_p
    def fill_G(r, r_max, m, n, wopt, logwopt, G):
        arr_r = np.ascontiguousarray(r, dtype='int32')
        arr_wopt = np.ascontiguousarray(wopt, dtype='float64')
        arr_logwopt = np.ascontiguousarray(logwopt, dtype='float64')
        arr_G = np.ascontiguousarray(G, dtype='float64')

        support_library.fill_G(arr_r.ctypes.data_as(c_int_p), r_max, m, n,
                               arr_wopt.ctypes.data_as(c_double_p),
                               arr_logwopt.ctypes.data_as(c_double_p),
                               arr_G.ctypes.data_as(c_double_p))

    support_library.core_cnll.argtypes = [c_int_p,
                                          c_int, c_int, c_int,
                                          c_int_p, c_int_p, c_int_p,
                                          c_int_p, c_int_p, c_int_p,
                                          c_double_p,
                                          c_double_p, c_double_p, c_int_p]
    support_library.core_cnll.restype = c_double
    def core_cnll(A,
                  count, m, n, r, rndx, irndx, csort, cndx, cconj, G,
                  S, SS, B):
        arr_A = np.ascontiguousarray(A, dtype='int32')
        arr_r = np.ascontiguousarray(r, dtype='int32')
        arr_rndx = np.ascontiguousarray(rndx, dtype='int32')
        arr_irndx = np.ascontiguousarray(irndx, dtype='int32')
        arr_csort = np.ascontiguousarray(csort, dtype='int32')
        arr_cndx = np.ascontiguousarray(cndx, dtype='int32')
        arr_cconj = np.ascontiguousarray(cconj, dtype='int32')
        arr_G = np.ascontiguousarray(G, dtype='float64')
        arr_S = np.ascontiguousarray(S, dtype='float64')
        arr_SS = np.ascontiguousarray(SS, dtype='float64')
        arr_B = np.ascontiguousarray(B, dtype='int32')
        return support_library.core_cnll(arr_A.ctypes.data_as(c_int_p),
                                         count, m, n,
                                         arr_r.ctypes.data_as(c_int_p),
                                         arr_rndx.ctypes.data_as(c_int_p),
                                         arr_irndx.ctypes.data_as(c_int_p),
                                         arr_csort.ctypes.data_as(c_int_p),
                                         arr_cndx.ctypes.data_as(c_int_p),
                                         arr_cconj.ctypes.data_as(c_int_p),
                                         arr_G.ctypes.data_as(c_double_p),
                                         arr_S.ctypes.data_as(c_double_p),
                                         arr_SS.ctypes.data_as(c_double_p),
                                         arr_B.ctypes.data_as(c_int_p))

    support_library.core_sample.argtypes = [c_double_p,
                                            c_int, c_int, c_int,
                                            c_int_p, c_int_p, c_int_p,
                                            c_int_p, c_int_p, c_int_p,
                                            c_double_p, c_double_p,
                                            c_double_p, c_double_p,
                                            c_int_p, c_double_p, c_double_p]
    support_library.core_sample.restype = ctypes.c_void_p
    def core_sample(logw,
                    count, m, n, r, rndx, irndx, csort, cndx, cconj,
                    G, rvs, S, SS, B, logQ, logP):
        arr_logw = np.ascontiguousarray(logw, dtype='float64')
        arr_r = np.ascontiguousarray(r, dtype='int32')
        arr_rndx = np.ascontiguousarray(rndx, dtype='int32')
        arr_irndx = np.ascontiguousarray(irndx, dtype='int32')
        arr_csort = np.ascontiguousarray(csort, dtype='int32')
        arr_cndx = np.ascontiguousarray(cndx, dtype='int32')
        arr_cconj = np.ascontiguousarray(cconj, dtype='int32')
        arr_G = np.ascontiguousarray(G, dtype='float64')
        arr_rvs = np.ascontiguousarray(rvs, dtype='float64')
        arr_S = np.ascontiguousarray(S, dtype='float64')
        arr_SS = np.ascontiguousarray(SS, dtype='float64')
        arr_B = np.ascontiguousarray(B, dtype='int32')
        arr_logQ = np.ascontiguousarray(logQ, dtype='float64')
        arr_logP = np.ascontiguousarray(logP, dtype='float64')
        support_library.core_sample(arr_logw.ctypes.data_as(c_double_p),
                                    count, m, n,
                                    arr_r.ctypes.data_as(c_int_p),
                                    arr_rndx.ctypes.data_as(c_int_p),
                                    arr_irndx.ctypes.data_as(c_int_p),
                                    arr_csort.ctypes.data_as(c_int_p),
                                    arr_cndx.ctypes.data_as(c_int_p),
                                    arr_cconj.ctypes.data_as(c_int_p),
                                    arr_G.ctypes.data_as(c_double_p),
                                    arr_rvs.ctypes.data_as(c_double_p),
                                    arr_S.ctypes.data_as(c_double_p),
                                    arr_SS.ctypes.data_as(c_double_p),
                                    arr_B.ctypes.data_as(c_int_p),
                                    arr_logQ.ctypes.data_as(c_double_p),
                                    arr_logP.ctypes.data_as(c_double_p))
        return (arr_B, arr_logQ[0], arr_logP[0])
    
    c_support_loaded = True
except:
    print 'C support code can\'t load. Falling back to Python.'
    c_support_loaded = False

# Saddlepoint approximation to P(R = r, C = c) for a matrix with
# independent Bernoulli(p_{ij}) entries. Because of linear dependence
# in the distribution of the margins (which leads to a singular
# K''(s_hat, t_hat), I introduce the additional random variable of the
# sum of all the cells and then only consider the distribution of the
# first (m-1) row and first (n-1) column margins.
#
# Behavior not well-defined if any margins are extreme or if the
# margins don't satisfy the Gale-Ryser conditions.
def p_margins_saddlepoint(r, c, p):
    m, n = len(r), len(c)
    
    a = np.sum(r)
    r = r[0:(m-1)]
    c = c[0:(n-1)]

    # Utility code: unpack vectorized (s, t, u)
    def unpack_s_t_u(x):
        s = x[0:(m-1)]
        t = x[(m-1):(m+n-2)]
        u = x[(m+n-2)]
        return s, t, u

    # The "E matrix" is used repeatedly in the calculations for K, K',
    # and K''. Vectorized for efficiency.
    def E_mat(s, t, u):
        E = np.zeros((m,n))
        for i in xrange(m-1):
            E[i,:] += s[i]
        for j in xrange(n-1):
            E[:,j] += t[j]
        E[:,:] += u
        E = np.exp(E)
        return E
    
    # The Hessian K''(s, t, u) is needed in both the saddlepoint
    # approximation and in the numerical solution for the saddlepoint
    # equations.
    def K_prime_prime(x):
        s, t, u = unpack_s_t_u(x)

        E = E_mat(s, t, u)
        K_prime_prime_mat = p * (1.0 - p) * E / \
            ((p * E + (1.0 - p)) ** 2)
        
        out = np.zeros(((m+n-1),(m+n-1)))

        for i in xrange(m-1):
            val = np.sum(K_prime_prime_mat[i,:])
            out[i,i] = val
            out[m+n-2,i] = val
            out[i,m+n-2] = val

        for j in xrange(n-1):
            val = np.sum(K_prime_prime_mat[:,j])
            out[m-1+j,m-1+j] = val
            out[m+n-2,m-1+j] = val
            out[m-1+j,m+n-2] = val

        for i in xrange(m-1):
            for j in xrange(n-1):
                out[i,m-1+j] = K_prime_prime_mat[i,j]
                out[m-1+j,i] = K_prime_prime_mat[i,j]

        out[m+n-2,m+n-2] = np.sum(K_prime_prime_mat)

        return out
    
    # Solve saddlepoint equations K'(s_hat, t_hat, u_hat) = (r, c, a)
    def s_t_hat_eq(x):
        s, t, u = unpack_s_t_u(x)

        E = E_mat(s, t, u)
        K_prime_mat = p * E / (p * E + (1.0 - p))

        out = np.empty((m+n-1))

        for i in xrange(m-1):
            out[i] = np.sum(K_prime_mat[i,:]) - r[i]

        for j in xrange(n-1):
            out[m-1+j] = np.sum(K_prime_mat[:,j]) - c[j]

        out[m+n-2] = np.sum(K_prime_mat) - a

        return out

    x_hat = opt.fsolve(s_t_hat_eq, np.zeros((m+n-1)), fprime = K_prime_prime)
    s_hat, t_hat, u_hat = unpack_s_t_u(x_hat)

    # Compute saddlepoint approximation
    E = E_mat(s_hat, t_hat, u_hat)
    K_hat = np.sum(np.log(p * E + (1.0 - p)))

    s_hat_dot_r = np.dot(s_hat, r)
    t_hat_dot_c = np.dot(t_hat, c)
    u_hat_dot_a = u_hat * a

    K_prime_prime_hat = K_prime_prime(x_hat)

    return (2 * np.pi) ** (-(m+n-1) / 2.0) * \
        np.linalg.det(K_prime_prime_hat) ** (-0.5) * \
        np.exp(K_hat - s_hat_dot_r - t_hat_dot_c - u_hat_dot_a)

##############################################################################
# Adapting a Matlab routine provided by Jeff Miller
# (jeffrey_miller@brown.edu), which implements an algorithm suggested
# in Manfred Krause's "A Simple Proof of the Gale-Ryser Theorem".
##############################################################################

# Necessary and sufficient check for the existence of a binary matrix
# with the specified row and column margins.
#
# Failure here throws an exception that should stop the calling
# function before anything weird happens.
def check_margins(r, c):
    # Check for conforming input
    assert(np.all(r >= 0))
    assert(np.all(c >= 0))
    assert(r.dtype.kind == 'i')
    assert(c.dtype.kind == 'i')
    assert(np.sum(r) == np.sum(c))

    # Check whether a satisfying matrix exists (Gale-Ryser conditions)
    cc = conjugate(r, max(len(r),len(c)))
    cd = c[np.argsort(-c)]
    assert(np.sum(c) == np.sum(cc))
    l = min(len(cc), len(cd))
    assert(np.all(np.cumsum(cd)[0:l] <= np.cumsum(cc)[0:l]))

# Eliminating the column margin nonincreasing condition by sorting and
# then undoing the sorting after the target matrix is generated.
#
# Return an arbitrary binary matrix with specified margins.
# Inputs:
#   r: row margins, length m
#   c: column margins, length n
# Output:
#   (m x n) binary matrix
def arbitrary_from_margins(r, c):
    check_margins(r, c)

    m = len(r)
    n = len(c)

    # Sort column margins and prepare for unsorting
    o = np.argsort(-c)
    oo = np.argsort(o)
    c = c[o]
    c_unsorted = c[oo]
    assert(np.all(np.diff(c) <= 0))

    # Construct the maximal matrix and the conjugate
    A = np.zeros((m,n), dtype = np.bool)
    for i in xrange(m):
        A[i,0:r[i]] = True
    col = np.sum(A, axis = 0)

    # Convert the maximal matrix into one with column sums c
    # (This procedure is guaranteed to terminate.)
    while not np.all(col == c):
        j = np.where(col > c)[0][0]
        k = np.where(col < c)[0][0]
        i = np.where(A[:,j] > A[:,k])[0][0]
        A[i,j] = False
        A[i,k] = True
        col[j] -= 1
        col[k] += 1

    # Undo the sort
    A = A[:,oo]
 
    # Verify that the procedure found a satisfying matrix
    assert(np.all(r == np.sum(A, axis = 1)))
    assert(np.all(c_unsorted == np.sum(A, axis = 0)))

    return A

##############################################################################
# Adapting Matlab code provided by Matt Harrison (matt_harrison@brown.edu).
##############################################################################

# Find row and column scalings to balance a matrix, using new "rc" method.
_dict_canonical_scalings = {}
def canonical_scalings(w, r, c):
    w_hash = hashlib.sha1(w.view(np.uint8)).hexdigest()
    r_hash = hashlib.sha1(r.flatten().view(np.uint8)).hexdigest()
    c_hash = hashlib.sha1(c.flatten().view(np.uint8)).hexdigest()
    hash = (w_hash, r_hash, c_hash)
    if hash in _dict_canonical_scalings:
        return _dict_canonical_scalings[hash]

    max_iter = 1000
    tol = 1e-8

    m, n = w.shape
    r = r.reshape((m,1))
    c = c.reshape((1,n))
    
    def sw_sums(a, b):
        abw = a * w * b
        sw = abw / (1 + abw)
        sw[np.isnan(sw)] = 1
        swr = sw.sum(1).reshape((m,1))
        swc = sw.sum(0).reshape((1,n))
        return swr, swc

    p_i_dot = (1 / n) * w.sum(1).reshape((m,1))
    p_dot_j = (1/  m) * w.sum(0).reshape((1,n))
    a = np.sqrt((r / n) / (p_i_dot * (1 - (r / n))))
    b = np.sqrt((c / m) / (p_dot_j * (1 - (c / m))))
    a[np.isnan(a)] = 1
    b[np.isnan(b)] = 1
    swr, swc = sw_sums(a, b)

    tol_check = np.Inf
    iter = 0
    while tol_check > tol and iter < max_iter:
        a = a * r / (swr + eps0)
        b = b * c / (swc + eps0)
        swr, swc = sw_sums(a, b)

        tol_check = np.max(np.abs(swr - r)) + np.max(np.abs(swc - c))
        iter += 1

    _dict_canonical_scalings[hash] = (a, b)
    return a, b

# Suppose c is a sequence of nonnegative integers. Returns c_conj where:
#   c_conj(k) := sum(c > k),    k = 0, ..., (n-1)
def conjugate(c, n):
    cc = np.zeros(n, dtype = np.int);

    for j, k in enumerate(c):
        if k >= n:
            cc[n-1] += 1
        elif k >= 1:
            cc[k-1] += 1

    s = cc[n-1]
    for j in xrange(n-2,-1,-1):
        s += cc[j]
        cc[j] = s

    return cc

# Eliminate extreme rows and columns recursively until all remaining
# rows and columns are non-extreme. Perform the matching pruning on
# supplied arrays
def prune(r, c, *arrays):
    r = r.copy()
    c = c.copy()
    arrays = list([a.copy() for a in arrays])
    A = len(arrays)

    unprune_ones = []
    r_unprune = np.arange(len(r))
    c_unprune = np.arange(len(c))
    while True:
        m, n = len(r), len(c)

        r_0 = (r == 0)
        if np.any(r_0):
            r = r[-r_0]
            for a in range(A):
                arrays[a] = arrays[a][-r_0]
            r_unprune = r_unprune[-r_0]
            continue

        r_n = (r == n)
        if np.any(r_n):
            r = r[-r_n]
            unprune_ones.extend([[r_u,c_u]
                                 for r_u in r_unprune[r_n]
                                 for c_u in c_unprune])
            c -= np.sum(r_n)
            for a in range(A):
                arrays[a] = arrays[a][-r_n]
            r_unprune = r_unprune[-r_n]
            continue

        c_0 = (c == 0)
        if np.any(c_0):
            c = c[-c_0]
            for a in range(A):
                arrays[a] = arrays[a][:,-c_0]
            c_unprune = c_unprune[-c_0]
            continue

        c_m = (c == m)
        if np.any(c_m):
            c = c[-c_m]
            unprune_ones.extend([[r_u,c_u]
                                 for r_u in r_unprune
                                 for c_u in c_unprune[c_n]])
            r -= np.sum(c_m)
            for a in range(A):
                arrays[a] = arrays[a][:,-c_m]
            c_unprune = c_unprune[-c_n]
            continue

        break

    unprune_ones = np.array(unprune_ones)
    def unprune(x):
        if unprune_ones.shape[0] == 0:
            return x
        else:
            x_a, x_b, x_c = x
            return (np.vstack([x_a, unprune_ones]), x_b, x_c)

    # Copy (views of) arrays to put them in C-contiguous form
    return r, c, [a.copy() for a in arrays], unprune

# Return a binary matrix (or a list of binary matrices) sampled
# approximately according to the specified Bernoulli weights,
# conditioned on having the specified margins.
# Inputs:
#   r: row margins, length m
#   c: column margins, length n
#   w: weight matrix, (m x n) matrix with values in (0, +infty)
#   T: number of matrices to sample
#   sort_by_wopt_var: when enabled, column ordering depends on w
# Output:
#   B_sample_sparse: (T default) sparse representation of (m x n) binary matrix
#                    (T >= 1) list of (sparse binary matrices, logQ, logP)
#
# More explicitly, consider independent Bernoulli random variables
# B(i,j) arranged as an m x n matrix B given the m-vector of row sums
# r and the n-vector of column sums c of the sample, i.e., given that
# sum(B_sample, 1) = r and sum(B_sample, 0) = c.
#
# An error is generated if no binary matrix agrees with r and c.
#
# B(i,j) is Bernoulli(p(i,j)) where p(i,j) = w(i,j)/(1+w(i,j)), i.e.,
# w(i,j) = p(i,j)/(1-p(i,j)).  [The case p(i,j) = 1 must be handled by
# the user in a preprocessing step, by converting to p(i,j) = 0 and
# decrementing the row and column sums appropriately.]
#
# The sparse representation used for output is a matrix giving the
# locations of the ones in the sample. If d = sum(r) = sum(c), then
# B_sample_sparse has dimensions (d x 2). If something goes wrong (due
# to undetected improper input), some of the rows of B_sample_sparse
# may [-1,-1], indicating no entry of B_sample.
# 
# B_sample can be recovered from B_sample_sparse via:
#
#     B_sample = np.zeros((m,n), dtype=np.bool)
#     for i, j in B_sample_sparse:
#         if i == -1: break 
#         B_sample[i,j] = 1
def approximate_from_margins_weights(r, c, w, T = None,
                                     sort_by_wopt_var = True):
    r_prune, c_prune, arrays_prune, unprune = prune(r, c, w)
    w_prune = arrays_prune[0]

    check_margins(r_prune, c_prune)

    ### Preprocessing

    # Sizing (making copies of m and n, as they are mutated during sampling)
    r_init = r_prune.copy()
    m, n = len(r_prune), len(c_prune)
    m_init, n_init = m, n
    assert((m,n) == w_prune.shape)

    # Sort the row margins (descending)
    rndx_init = np.argsort(-r_prune)
    rsort = r_prune[rndx_init]

    # Balance the weights
    a_scale, b_scale = canonical_scalings(w_prune, r_prune, c_prune)
    wopt = a_scale * w_prune * b_scale

    # Reorder the columns
    if sort_by_wopt_var:
        cndx = np.lexsort((-wopt.var(0), c_prune))
    else:
        cndx = np.argsort(c_prune)
    csort = c_prune[cndx];
    wopt = wopt[:,cndx]

    # Precompute log weights
    logw = np.log(w_prune)

    # Compute G
    G = compute_G(r_prune, m, n, wopt)

    # Generate the inverse index for the row orders to facilitate fast
    # sorting during the updating
    irndx_init = np.argsort(rndx_init)

    # Compute the conjugate of c
    cconj_init = conjugate(csort, m)

    # Get the running total of number of ones to assign
    count_init = np.sum(rsort)

    def do_sample():
        sample_prune = compute_sample(logw,
                                      count_init, m_init, n_init,
                                      r_init, rndx_init, irndx_init,
                                      csort, cndx, cconj_init,
                                      G)
        return unprune(sample_prune)
    
    if T:
        return [do_sample() for t in xrange(T)]
    else:
        return do_sample()[0]
    
# Return the approximate nll of an observed binary matrix given
# specified Bernoulli weights, conditioned on having the observed
# margins.
#
# Inputs:
#   A: observed data, (m x n) binary matrix
#   w: weight matrix, (m x n) matrix with values in (0, +infty)
# Output:
#   ncll: negative conditional log-likelihood
def approximate_conditional_nll(A, w, sort_by_wopt_var = True):
    assert(A.shape == w.shape)

    # FIXME: this will probably break if A is a matrix (as opposed to an array)
    r = A.sum(1, dtype=np.int)
    c = A.sum(0, dtype=np.int)

    r, c, arrays, _ = prune(r, c, A, w)
    A, w = arrays

    # Sizing
    m, n = len(r), len(c)
    if (m == 0) or (n == 0):
        return 0.0

    # Sort the row margins (descending)
    rndx = np.argsort(-r)
    rsort = r[rndx]

    # Balance the weights
    a_scale, b_scale = canonical_scalings(w, r, c)
    wopt = a_scale * w * b_scale
    if np.any(np.isnan(wopt)):
        wopt = w

    # Reorder the columns
    if sort_by_wopt_var:
        cndx = np.lexsort((-wopt.var(0), c))
    else:
        cndx = np.argsort(c)
    csort = c[cndx];
    wopt = wopt[:,cndx]

    # Compute G
    G = compute_G(r, m, n, wopt)

    return compute_cnll(A, r, rsort, rndx, csort, cndx, m, n, G)
    

def compute_G(r, m, n, wopt):
    logwopt = np.log(wopt)
    r_max = max(1, np.max(r))

    G = np.tile(-np.inf, (r_max+1, m, n-1))
    G[0,:,:] = 0.0
    G[1,:,n-2] = logwopt[:,n-1]
    if c_support_loaded:
        fill_G(r, r_max, m, n, wopt, logwopt, G)
    else:
        for i, ri in enumerate(r):
            for j in xrange(n-2, 0, -1):
                wij = logwopt[i,j]
                for k in xrange(1, ri+1):
                    b = G[k-1,i,j] + wij
                    a = G[k,i,j]
                    if a == -np.inf and b == -np.inf: continue
                    if a > b:
                        G[k,i,j-1] = a + np.log(1.0 + np.exp(b-a))
                    else:
                        G[k,i,j-1] = b + np.log(1.0 + np.exp(a-b))
            for j in xrange(n-1):
                for k in xrange(r_max):
                    Gk_num = G[k,i,j]
                    Gk_den = G[k+1,i,j]
                    if np.isinf(Gk_den):
                        G[k,i,j] = -1.0
                    else:
                        G[k,i,j] = wopt[i,j] * np.exp(Gk_num-Gk_den) * \
                            ((n - j - k - 1.0) / (k + 1.0))
                if np.isinf(Gk_den):
                    G[r_max,i,j] = -1.0
    return G

def compute_cnll(A, r, rsort, rndx, csort, cndx, m, n, G):
    # Generate the inverse index for the row orders to facilitate fast
    # sorting during the updating
    irndx = np.argsort(rndx)

    # Compute the conjugate of c
    cconj = conjugate(csort, m)

    # Get the running total of number of ones to assign
    count = np.sum(rsort)

    # Initialize B_sample_sparse
    B_sample_sparse = -np.ones((count,2), dtype=np.int)
    
    # Initialize intermediate storage
    #
    # Index 0 corresponds to -1, index 1 corresponds to 0, index 2
    # corresponds to 1, ..., index M-1 corresponds to c[0]+1
    M = csort[-1] + 3
    S = np.zeros((M,m))
    SS = np.zeros(M)
    
    if c_support_loaded:
        return core_cnll(A,
                         count, m, n,
                         r, rndx, irndx, csort, cndx, cconj, G,
                         S, SS, B_sample_sparse)
    else:
        # Most recent assigned column in B_sample_sparse
        place = -1

        # Initialize nll
        cnll = 0.0

        # Loop over columns for column-wise sampling
        #
        # Warning: things that "should" be fixed are modified in this
        # loop, e.g., n, the number of columns!
        for c1 in xrange(n):
            ### Sample the next column

            # Remember the starting point for this column in B_sample_sparse
            placestart = place + 1

            # Inspect column
            clabel, colval = cndx[c1], csort[c1]
            if count == 0: break

            # Update the conjugate
            cconj[0:colval] -= 1

            # Update the number of columns remaining
            n -= 1

            ### DP initialization

            # Variables used inside DP
            smin, smax = colval, colval
            cumsums, cumconj = count, count - colval

            # Update the count
            count -= colval

            # Start filling SS (indices of colval-1, colval, colval+1)
            SS[colval:(colval+3)] = [0,1,0]
                
            ### DP

            # Loop over (remaining and sorted descending) rows in reverse
            for i in reversed(range(m)):
                # Get the value for this row, for use in computing the
                # probability of a 1 for this row/column pair
                rlabel = rndx[i]
                val = r[rlabel]

                # Combinatorial approximations to N(r,c) are used in
                # the importance sampler cell probabilities.
                p = val / (n + 1.0)
                q = 1.0 - p

                # Incorporate weights
                if n > 0 and val > 0:
                    Gk = G[val-1,rlabel,c1]
                    if (Gk < 0) or (q <= 0.0) or (p >= 1.0):
                        q = 0.0
                        p = 1.0
                    else:
                        p = p / (1.0 - p) * Gk
                        p = p / (1.0 + p)
                        q = 1.0 - p

                # Update the feasibility constraints
                cumsums -= val
                cumconj -= cconj[i]

                # Incorporate the feasibility constraints into bounds on
                # the running column sum
                sminold, smaxold = smin, smax
                smin = max(0, max(cumsums - cumconj, sminold - 1))
                smax = min(smaxold, i)

                # DP iteration (only needed parts of SS updated)
                SSS = 0.0
                SS[smin] = 0.0
                for j in xrange(smin+1,smax+2):
                    a = SS[j] * q
                    b = SS[j+1] * p
                    apb = a + b
                    SSS += apb
                    SS[j] = apb
                    S[j,i] = b / (apb + eps0)
                SS[smax+2] = 0.0

                # Check for impossible; if so, jump out of inner loop
                if SSS <= 0: break

                # Normalize to prevent overflow/underflow
                SS[(smin+1):(smax+2)] /= SSS

            # Check for impossible; if so, jump out of outer loop
            if SSS <= 0: break

            ### Updating cnll

            # Running total and target of how many entries filled (offset
            # to match S)
            j, jmax = 1, colval + 1

            # Skip assigning anything when colval = 0
            if j < jmax:
                for i in xrange(m):
                    # 1's are generated according to the transition probability
                    p = S[j,i]
                    rlabel = rndx[i]
                    if A[rlabel,clabel]:
                        # Decrement row total
                        val = r[rlabel]
                        r[rlabel] -= 1

                        # Record the entry
                        place += 1
                        cnll -= np.log(p)
                        B_sample_sparse[place,:] = [rlabel,clabel]
                        j += 1

                        # Break the loop early, since all the remaining
                        # p's must be 0
                        if j == jmax: break
                    else:
                        cnll -= np.log1p(-p)

            # Everything is updated except the re-sorting, so skip if possible
            if count == 0: break

            ### Re-sort row sums

            # Essentially, we only need to re-sort the assigned rows. In
            # greater detail, we take each row that was assigned to the
            # list and either leave it in place or swap it with the last
            # row that matches its value; this leaves the rows sorted
            # (descending) since each row was decremented by only 1.

            # Looping in reverse ensures that least rows are swapped first
            for j in xrange(place, placestart-1, -1):
                # Get the row label, its new value, and its inverse index
                k = B_sample_sparse[j,0]
                val = r[k]
                irndxk = irndx[k]

                # See if the list is still sorted
                irndxk1 = irndxk + 1
                if irndxk1 >= m or r[rndx[irndxk1]] <= val:
                    continue

                # Find the first place where k can be inserted
                irndxk1 += 1
                while irndxk1 < m and r[rndx[irndxk1]] > val:
                    irndxk1 += 1
                irndxk1 -= 1

                # Perform swap
                rndxk1 = rndx[irndxk1]
                rndx[irndxk] = rndxk1
                rndx[irndxk1] = k
                irndx[k] = irndxk1
                irndx[rndxk1] = irndxk
            import sys; sys.exit()

            ### Recursion

            # At this point:
            #   r[rndx] is sorted and represents row margins
            #   rndx[irndx] = 0:m
            #   c[cndx[(c1+1):]] is sorted and represents column margins
            #   m, n, count, cconj, etc. are valid
            #   place points to new entries in B_sample_sparse

        return cnll

def compute_sample(logw,
                   count_init, m_init, n_init,
                   r_init, rndx_init, irndx_init,
                   csort, cndx, cconj_init,
                   G):
    ### Initialization

    # Make local copy of variables to be mutated
    r = r_init.copy()
    m, n = m_init, n_init
    rndx, irndx = rndx_init.copy(), irndx_init.copy()
    cconj = cconj_init.copy()
    count = count_init

    # Pre-generate uniform random variates
    rvs = np.random.random(m * n)
        
    # Initialize intermediate storage
    #
    # Index 0 corresponds to -1, index 1 corresponds to 0, index 2
    # corresponds to 1, ..., index M-1 corresponds to c[0]+1
    M = csort[-1] + 3
    S = np.zeros((M,m))
    SS = np.zeros(M)

    # Initialize B_sample_sparse
    B_sample_sparse = -np.ones((count,2), dtype=np.int)

    if c_support_loaded:
        logQ, logP = np.zeros(1), np.zeros(1)
        return core_sample(logw,
                           count, m, n,
                           r, rndx, irndx, csort, cndx, cconj, G, rvs,
                           S, SS, B_sample_sparse, logQ, logP)
    else:
        # Initialize logP, logQ, and random variates pointer
        logQ, logP = 0.0, 0.0
        rand_p = 0

        # Most recent assigned column in B_sample_sparse
        place = -1

        # Loop over columns for column-wise sampling
        #
        # Warning: things that "should" be fixed are modified in this
        # loop, e.g., n, the number of columns!
        for c1 in xrange(n):
            ### Sample the next column

            # Remember the starting point for this column in B_sample_sparse
            placestart = place + 1

            # Inspect column
            clabel, colval = cndx[c1], csort[c1]
            if colval == 0 or count == 0: break

            # Update the conjugate
            cconj[0:colval] -= 1

            # Update the number of columns remaining
            n -= 1

            ### DP initialization

            # Variables used inside DP
            smin, smax = colval, colval
            cumsums, cumconj = count, count - colval

            # Update the count
            count -= colval

            # Start filling SS (indices corresponding to colval-1 ... colval+1)
            SS[colval:(colval+3)] = [0,1,0]

            ### DP

            # Loop over (remaining and sorted descending) rows in reverse
            for i in reversed(range(m)):
                # Get the value for this row, for use in computing the
                # probability of a 1 for this row/column pair
                rlabel = rndx[i]
                val = r[rlabel]

                # Combinatorial approximations to N(r,c) are used in
                # the importance sampler cell probabilities.
                p = val / (n + 1.0)
                q = 1.0 - p

                # Incorporate weights
                if n > 0 and val > 0:
                    Gk = G[val-1,rlabel,c1]
                    if (Gk < 0) or (q <= 0.0) or (p >= 1.0):
                        q = 0.0
                        p = 1.0
                    else:
                        p = p / (1.0 - p) * Gk
                        p = p / (1.0 + p)
                        q = 1.0 - p
                                
                # Update the feasibility constraints
                cumsums -= val
                cumconj -= cconj[i]

                # Incorporate the feasibility constraints into bounds on
                # the running column sum
                sminold, smaxold = smin, smax
                smin = max(0, max(cumsums - cumconj, sminold - 1))
                smax = min(smaxold, i)

                # DP iteration (only needed parts of SS updated)
                SSS = 0.0
                SS[smin] = 0.0
                for j in xrange(smin+1,smax+2):
                    a = SS[j] * q
                    b = SS[j+1] * p
                    apb = a + b
                    SSS += apb
                    SS[j] = apb
                    S[j,i] = b / (apb + eps0)
                SS[smax+2] = 0.0

                # Check for impossible; if so, jump out of inner loop
                if SSS <= 0: break

                # Normalize to prevent overflow/underflow
                SS[(smin+1):(smax+2)] /= SSS

            # Check for impossible; if so, jump out of outer loop
            if SSS <= 0: break

            ### Sampling

            # Running total and target of how many entries filled (offset
            # to match S)
            j, jmax = 1, colval + 1

            # Skip assigning anything when colval = 0
            if j < jmax:
                for i in xrange(m):
                    # Generate a one according to the transition probability
                    p = S[j,i]
                    rv = rvs[rand_p]
                    rand_p += 1
                    if rv < p:
                        # Decrement row total
                        rlabel = rndx[i]
                        val = r[rlabel]
                        r[rlabel] -= 1

                        # Record the entry
                        place += 1
                        B_sample_sparse[place,:] = [rlabel,clabel]
                        j += 1

                        # Record contribution to logQ, logP
                        logQ += np.log(p)
                        logP += logw[rlabel,clabel]

                        # Break the loop early, since all the remaining
                        # p's must be 0
                        if j == jmax: break
                    else:
                        logQ += np.log1p(-p)

            # Everything is updated except the re-sorting, so skip if possible
            if count == 0: break

            ### Re-sort row sums

            # Essentially, we only need to re-sort the assigned rows. In
            # greater detail, we take each row that was assigned to the
            # list and either leave it in place or swap it with the last
            # row that matches its value; this leaves the rows sorted
            # (descending) since each row was decremented by only 1.

            # Looping in reverse ensures that least rows are swapped first
            for j in xrange(place, placestart-1, -1):
                # Get the row label, its new value, and its inverse index
                k = B_sample_sparse[j,0]
                val = r[k]
                irndxk = irndx[k]

                # See if the list is still sorted
                irndxk1 = irndxk + 1
                if irndxk1 >= m or r[rndx[irndxk1]] <= val:
                    continue

                # Find the first place where k can be inserted
                irndxk1 += 1
                while irndxk1 < m and r[rndx[irndxk1]] > val:
                    irndxk1 += 1
                irndxk1 -= 1

                # Perform swap
                rndxk1 = rndx[irndxk1]
                rndx[irndxk] = rndxk1
                rndx[irndxk1] = k
                irndx[k] = irndxk1
                irndx[rndxk1] = irndxk

            ### Recursion

            # At this point:
            #   r[rndx] sorted, represents unassigned row margins
            #   rndx[irndx] = 0:m
            #   c[cndx[(c1+1):]] sorted, represents unassigned column margins
            #   m, n, count, cconj, etc. are valid
            #   place points to new entries in B_sample_sparse
            #
            # In other words, it is as if Initialization had just
            # completed for sampling a submatrix of B_sample.
                
        return (B_sample_sparse, logQ, logP)

    
##############################################################################
# End of adapted code
##############################################################################

def log_partition_is(z, cvsq = False):
    """From importance-weighted sampled, estimate log-partition function."""
    T = len(z)

    logf = np.empty(T)
    for t in range(T):
        logf[t] = z[t][2] - z[t][1]
    logkappa = -np.log(T) + logsumexp(logf)
    if not cvsq:
        return logkappa
    else:
        logcvsq = -np.log(T - 1) - 2 * logkappa + \
          logsumexp(2 * logabsdiffexp(logf, logkappa))
        return logkappa, logcvsq

if __name__ == '__main__':
    # Test of binary matrix generation code
    m = np.random.random(size=(12,10)) < 0.3
    r, c = np.sum(m, axis = 1), np.sum(m, axis = 0)
    print r, c
    A = arbitrary_from_margins(r, c)
    print np.sum(A, axis = 1), np.sum(A, axis = 0)

    # Test of "rc" balancing
    m = np.random.normal(10, 1, size = (6,5))
    r, c = np.ones((6,1)), np.ones((1,5))
    c[0] = 2
    a, b = canonical_scalings(m, r, c)
    m_canonical = a * m * b
    print m_canonical.sum(1)
    print m_canonical.sum(0)

    # Test of conjugate
    print conjugate([1,1,1,1,2,8], 10)

    # Test of approximate margins-conditional sampling
    N = 5;
    a_out = np.random.normal(0, 1, N)
    a_in = np.random.normal(0, 1, N)
    x = np.random.normal(0, 1, (N,N))
    theta = 0.8
    logit_P = np.zeros((N,N))
    for i, a in enumerate(a_out):
        logit_P[i,:] += a
    for j, a in enumerate(a_in):
        logit_P[:,j] += a
    logit_P += theta * x
    w = np.exp(logit_P)
    r, c = np.repeat(2, N), np.repeat(2, N)
    B_sample_sparse = approximate_from_margins_weights(r, c, w)
    B_sample = np.zeros((N,N), dtype=np.bool)
    for i, j in B_sample_sparse:
        if i == -1: break
        B_sample[i,j] = 1
    print B_sample.sum(1)
    print B_sample.sum(0)
    print B_sample[x < -1.0].sum(), B_sample[x > 1.0].sum()

    # Test of approximate conditional likelihood
    print approximate_conditional_nll(B_sample, w)
    print approximate_conditional_nll(B_sample, np.tile(1.0, (N,N)))

    # Test repeated sampling
    T = 10
    B_samples_sparse = approximate_from_margins_weights(r, c, w, T)
    from Utility import logsumexp, logabsdiffexp
    logf = np.empty(T)
    for t in range(T):
        B_sample = np.zeros((N,N), dtype = np.bool)
        for i, j in B_samples_sparse[t][0]:
            if i == -1: break
            B_sample[i,j] = 1
        print B_sample[x < -1.0].sum(), B_sample[x > 1.0].sum()
        logf[t] = B_samples_sparse[t][2] - B_samples_sparse[t][1]
    logkappa = -np.log(T) + logsumexp(logf)
    logcvsq = -np.log(T - 1) - 2 * logkappa + \
        logsumexp(2 * logabsdiffexp(logf, logkappa))
    print np.exp(logcvsq)
