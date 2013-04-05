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

class EdgeCovariate:
    def __init__(self, names):
        self.names = names
        self.data = sparse.lil_matrix((len(names),len(names)))
        self.dirty()

    def __str__(self):
        return '<EdgeCovariate\n%s\n%s>' % (repr(self.names),repr(self.data))

    def __getitem__(self, index):
        return self.data.__getitem__(index)

    def __setitem__(self, index, x):
        self.data.__setitem__(index, x)
        self.dirty()

    def copy(self):
        new = EdgeCovariate(self.names)
        new.data = self.data.copy()
        return new

    def tocsr(self):
        self.data = self.data.tocsr()

    # Indicate that matrix should not used a cached version
    def dirty(self):
        self.is_dirty = True

    def matrix(self):
        if self.is_dirty:
            self.cached_matrix = self.data.toarray()
            self.is_dirty = False
            return self.cached_matrix
        else:
            return self.cached_matrix

    def sparse_matrix(self):
        return self.data

    def from_existing(self, edge_covariate, inds):
        self.names = edge_covariate.names[inds]
        
        edge_covariate.tocsr()
        self.data = edge_covariate.data[inds][:,inds]
        self.dirty()

    def from_binary_function_name(self, f):
        for i, n_1 in enumerate(self.names):
            for j, n_2 in enumerate(self.names):
                val = f(n_1, n_2)
                if val != 0:
                    self.data[i, j] = val
        self.dirty()

    def from_binary_function_ind(self, f):
        for i in range(len(self.names)):
            for j in range(len(self.names)):
                val = f(i, j)
                if val != 0:
                    self.data[i, j] = val
        self.dirty()
 
