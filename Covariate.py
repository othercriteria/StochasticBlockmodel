#!/usr/bin/env python

# Representation for covariate data
# Daniel Klein, 5/10/2012

import numpy as np
import scipy.sparse as sparse

class NodeCovariate:
    def __init__(self, names, dtype = np.float):
        self.names = names
        self.dtype = dtype
        self.data = np.zeros(len(names), dtype = dtype)

    def __str__(self):
        return '<NodeCovariate\n%s\n%s>' % (repr(self.names),repr(self.data))

    def __getitem__(self, index):
        return self.data.__getitem__(index)

    def __setitem__(self, index, x):
        self.data.__setitem__(index, x)

    def subset(self, inds):
        sub_names = self.names[inds]
        sub_dtype = self.dtype
        sub = NodeCovariate(sub_names, sub_dtype)
        sub.data[:] = self.data[inds]

        return sub

    def from_pairs(self, names, values):
        n_to_ind = {}
        for i, n in enumerate(self.names):
            n_to_ind[n] = i

        for n, v in zip(names, values):
            if not n in n_to_ind: continue
            self.data[n_to_ind[n]] = v

    def show_histogram(self):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(self.data, bins = 50)
        plt.show()

    def copy(self):
        new = NodeCovariate(self.names, self.dtype)
        new.data = self.data.copy()
        return new

class EdgeCovariate:
    def __init__(self, rnames, cnames):
        self.rnames = rnames
        self.cnames = cnames
        self.data = sparse.lil_matrix((len(self.rnames),len(self.cnames)))
        self.dirty()

    def __str__(self):
        return '<EdgeCovariate\n%s\n%s\n%s>' % \
              (repr(self.rnames), repr(self.cnames), repr(self.data))

    def __getitem__(self, index):
        return self.data.__getitem__(index)

    def __setitem__(self, index, x):
        self.data.__setitem__(index, x)
        self.dirty()

    def copy(self):
        new = EdgeCovariate(self.rnames, self.cnames)
        new.data = self.data.copy()

        return new

    def tocsr(self):
        self.data = self.data.tocsr()

    # Indicate that matrix should not used a cached version
    def dirty(self):
        self.is_dirty = True
        self.cached_matrix = None

    def matrix(self):
        if self.is_dirty:
            self.cached_matrix = self.data.toarray()
            self.is_dirty = False

        return self.cached_matrix

    def sparse_matrix(self):
        return self.data

    def subset(self, rinds, cinds):
        # TODO: Check if this is actually necessary.
        self.tocsr()
        
        sub = EdgeCovariate(self.rnames[rinds], self.cnames[cinds])
        sub.data[:,:] = self.data[rinds][:,cinds]

        return sub

    def from_binary_function_name(self, f):
        for i, n_1 in enumerate(self.rnames):
            for j, n_2 in enumerate(self.cnames):
                val = f(n_1, n_2)
                if val != 0:
                    self.data[i,j] = val
        self.dirty()

    def from_binary_function_ind(self, f):
        for i in range(len(self.rnames)):
            for j in range(len(self.cnames)):
                val = f(i, j)
                if val != 0:
                    self.data[i,j] = val
        self.dirty()
