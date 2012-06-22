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
 
if __name__ == '__main__':
    # Test of binary matrix generation code
    m = np.random.random(size=(12,10)) < 0.3
    r, c = np.sum(m, axis = 1), np.sum(m, axis = 0)
    print r, c
    A = arbitrary_from_margins(r, c)
    print np.sum(A, axis = 1), np.sum(A, axis = 0)
