#!/usr/bin/env python

import numpy as np

from BinaryMatrix import approximate_from_margins_weights as samp

def fill(m, n, sparse):
    Y = np.zeros((m,n), dtype = np.bool)
    for i, j in sparse:
        if i == -1: break
        Y[i,j] = 1
    return Y

m = 8
n = 6
s = 10

t = 5

while True:
    r = np.random.multinomial(s, np.ones(m) / m)
    c = np.random.multinomial(s, np.ones(n) / n)
    if ((r.min() == 0) or (r.max() >= n) or (c.min() == 0) or (c.max() >= m)):
        continue
    try:
        p = np.random.random((m,n))
        w = p / (1 - p)
        Z = samp(r, c, w, T = (t + 1))

        z0 = fill(m, n, Z[t][0])
        refuted = False
        for k in range(t):
            z = fill(m, n, Z[k][0])
            print (z == z0).sum()
            if not (z == z0).all():
                refuted = True
        if refuted: continue

        print r
        print c
        print w
        print z0
        break
    except:
        pass
