#!/usr/bin/env python

# Representation and basic operations for array data; more general
# than a Network to allow for testing, applications to bipartite graphs, etc.
# Daniel Klein, 4/4/2013

import numpy as np
import scipy.sparse as sparse

from Covariate import NodeCovariate, EdgeCovariate

class Array:
    def __init__(self, M, N):
        self.M = M
        self.N = N

        self.rnames = np.array(['m_%d' % m for m in range(self.M)])
        self.cnames = np.array(['n_%d' % n for n in range(self.N)])
        
        self.array = sparse.lil_matrix((self.M, self.N), dtype = np.bool)

        # Offset initially set to None so simpler offset-free model
        # code can be used by default
        self.offset = None

        self.row_covariates = {}
        self.col_covariates = {}
        self.edge_covariates = {}

    def __setitem__(self, index, x):
        self.array.__setitem__(index, x)

    def new_row_covariate(self, name):
        self.row_covariates[name] = NodeCovariate(self.rnames)
        return self.row_covariates[name]

    def new_col_covariate(self, name):
        self.col_covariates[name] = NodeCovariate(self.cnames)
        return self.col_covariates[name]

    def new_edge_covariate(self, name):
        self.edge_covariates[name] = EdgeCovariate(self.rnames, self.cnames)
        return self.edge_covariates[name]

    def initialize_offset(self):
        self.offset = EdgeCovariate(self.rnames, self.cnames)
        return self.offset

    def subarray(self, rinds = None, cinds = None):
        if rinds is None:
            rinds = np.arange(self.M)
        if cinds is None:
            cinds = np.arange(self.N)

        sub_M = len(rinds)
        sub_N = len(cinds)
        sub = Array(sub_M, sub_N)
        sub.rnames = self.rnames[rinds]
        sub.cnames = self.cnames[cinds]

        sub.array = self.array[rinds][:,cinds]

        for row_covariate in self.row_covariates:
            src = self.row_covariates[row_covariate]
            sub.row_covariates[row_covariate] = src.subset(rinds)
        for col_covariate in self.col_covariates:
            src = self.col_covariates[col_covariate]
            sub.col_covariates[col_covariate] = src.subset(cinds)
        for edge_covariate in self.edge_covariates:
            src = self.edge_covariates[edge_covariate]
            sub.edge_covariates[edge_covariate] = src.subset(rinds, cinds)

        if self.offset:
            sub.offset = self.offset.subset(rinds, cinds)

        return sub

    # Syntactic sugar to make array generation look like object mutation
    def generate(self, model, **opts):
        self.array = model.generate(self, **opts)

    def is_sparse(self):
        return sparse.issparse(self.array)
        
    def as_dense(self):
        if self.is_sparse():
            return self.array.todense()
        else:
            return self.array

    def offset_extremes(self):
        if self.offset is None:
            self.initialize_offset()

        # (Separately) sort rows and columns of adjacency matrix by
        # increasing sum
        A = np.asarray(self.as_dense())
        r_ord = np.argsort(A.sum(1))
        c_ord = np.argsort(A.sum(0))
        A = A[r_ord][:,c_ord]

        # Convenience function to set blocks of the offset
        def set_offset_block(r, c, val):
            # XXX: Replace this with working vectorized code
            for i in r_ord[r]:
                for j in c_ord[c]:
                    self.offset[i,j] = val

        # Recursively examine for submatrices that will send
        # corresponding EMLE parameter estimates to infinity
        to_screen = [(np.arange(self.M), np.arange(self.N))]
        while len(to_screen) > 0:
            r_act, c_act = to_screen.pop()

            if len(r_act) == 0 or len(c_act) == 0:
                continue

            A_act = A[r_act][:,c_act]
            n_act = A_act.shape

            # Cumulative sums allow for efficient bad substructure search
            C_inc = np.cumsum(np.cumsum(A_act,1),0)
            def reverse(x):
                return x[range(n_act[0]-1,-1,-1)][:,range(n_act[1]-1,-1,-1)]
            C_dec = reverse(np.cumsum(np.cumsum(reverse(1-A_act),1),0))

            if C_inc[-1,-1] == 0:
                set_offset_block(r_act, c_act, -np.inf)
                continue
            if C_dec[0,0] == 0:
                set_offset_block(r_act, c_act, np.inf)
                continue

            align = np.zeros((n_act[0]+1,n_act[1]+1))
            align[1:(n_act[0]+1)][:,1:(n_act[1]+1)] += (C_inc == 0)
            align[0:n_act[0]][:,0:n_act[1]] *= (C_dec == 0)
            splits = zip(*np.where(align))
            if len(splits) == 0:
                continue
            i_split, j_split = splits[0]

            set_offset_block(r_act[:i_split], c_act[:j_split], -np.inf)
            set_offset_block(r_act[i_split:], c_act[j_split:], np.inf)

            to_screen.append((r_act[i_split:], c_act[:j_split]))
            to_screen.append((r_act[:i_split], c_act[j_split:]))
