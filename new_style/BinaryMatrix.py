#!/usr/bin/env python

# Algorithms for producing random binary matrices with margin constraints
# Daniel Klein, 5/11/2012

import numpy as np

# Adapting a Matlab routine provided by Jeff Miller
# (jeffrey_miller@brown.edu), which implements an algorithm suggested
# in Manfred Krause's "A Simple Proof of the Gale-Ryser Theorem".
#
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
    m = len(r)
    n = len(c)

    # Check for conforming input
    assert(np.all(r >= 0))
    assert(np.all(c >= 0))
    assert(r.dtype.kind == 'i')
    assert(c.dtype.kind == 'i')

    # Sort column margins and prepare for unsorting
    o = np.argsort(-c)
    oo = np.argsort(o)
    c = c[o]
    c_unsorted = c[oo]
    assert(np.all(np.diff(c) <= 0))

    # Construct the maximal matrix and the conjugate
    A = np.zeros((m,n), dtype = np.bool)
    for i in range(m):
        A[i,0:r[i]] = True
    col = np.sum(A, axis = 0)

    # Check whether a satisfying matrix exists (Gale-Ryser conditions)
    assert(np.sum(c) == np.sum(col))
    assert(np.all(np.cumsum(c) <= np.cumsum(col)))

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

# Find row and column scalings to balance a matrix, using the
# Sinkhorn(-Knopp) algorithm
def canonical_scalings(w):
    tol = 1e-8;

    # Balancing is only meaningful for a nonnegative matrix
    assert(np.all(w >= 0.0))

    m, n = w.shape
    M, N = n * np.ones((m,1)), m * np.ones((1,n))
    r, c = w.sum(1).reshape((m,1)), w.sum(0).reshape((1,n))

    a = M / r
    a /= np.mean(a)
    b = N / (a * w).sum(0)

    tol_check = np.Inf
    while tol_check > tol:
        a_new = M / (w * b).sum(1, keepdims = True)
        a_new /= np.mean(a_new)
        b_new = N / (a_new * w).sum(0, keepdims = True)

        # "L1"-ish tolerance in change during the last iteration
        tol_check = np.sum(np.abs(a - a_new)) + np.sum(np.abs(b - b_new))
        a, b = a_new, b_new

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
    for j in range(n-2,-1,-1):
        s += cc[j]
        cc[j] = s

    return cc



##############################################################################
# End of adapted code
##############################################################################


if __name__ == '__main__':
    # Test of binary matrix generation code
    m = np.random.random(size=(12,10)) < 0.3
    r, c = np.sum(m, axis = 1), np.sum(m, axis = 0)
    print r, c
    A = arbitrary_from_margins(r, c)
    print np.sum(A, axis = 1), np.sum(A, axis = 0)

    # Test of Sinkhorn balancing
    m = np.random.normal(10, 1, size = (6,5))
    a, b = canonical_scalings(m)
    m_canonical = a * m * b
    print m_canonical.sum(1)
    print m_canonical.sum(0)

    # Test of conjugate
    print conjugate([1,1,1,1,2,8], 10)
