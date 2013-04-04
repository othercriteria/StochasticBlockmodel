#!/usr/bin/env python

# Network representation and basic operations
# Daniel Klein, 5/10/2012

import numpy as np
import scipy.sparse as sparse
import networkx as nx
import matplotlib.pyplot as plt

from Covariate import NodeCovariate, EdgeCovariate

class Network:
    def __init__(self, N = None):
        if N:
            self.N = N
            self.network = sparse.lil_matrix((self.N,self.N), dtype=np.bool)
            self.names = np.array(['n_%d' % n for n in range(N)])
        else:
            self.N = 0
            self.network = None
            self.names = None

        # Offset intitially set to None so simpler offset-free model
        # code can be used by default
        self.offset = None
             
        # Maps from names to 1-D and 2-D arrays, respectively
        self.node_covariates = {}
        self.edge_covariates = {}

    def tocsr(self):
        if self.is_sparse():
            self.network = self.network.tocsr()
        else:
            print 'Attempting CSR conversion of a non-sparse network.'
            raise
            
    def new_node_covariate(self, name, as_int = False):
        self.node_covariates[name] = NodeCovariate(self.names)
        return self.node_covariates[name]

    def new_node_covariate_int(self, name):
        self.node_covariates[name] = NodeCovariate(self.names, dtype = np.int)
        return self.node_covariates[name]

    def new_edge_covariate(self, name):
        self.edge_covariates[name] = EdgeCovariate(self.names)
        return self.edge_covariates[name]

    def initialize_offset(self):
        self.offset = EdgeCovariate(self.names)
        return self.offset

    def subnetwork(self, inds):
        sub_N = len(inds)
        sub = Network()

        if self.is_sparse():
            self.tocsr()
        sub.network = self.network[inds][:,inds]
        sub.names = self.names[inds]
        sub.N = sub_N

        sub.node_covariates = {}
        for node_covariate in self.node_covariates:
            src = self.node_covariates[node_covariate]
            sub.new_node_covariate(node_covariate).from_existing(src, inds)
        sub.edge_covariates = {}
        for edge_covariate in self.edge_covariates:
            src = self.edge_covariates[edge_covariate]
            sub.new_edge_covariate(edge_covariate).from_existing(src, inds)

        if self.offset:
            sub.initialize_offset().from_existing(self.offset, inds)
            
        return sub

    # Syntactic sugar to make network generation look like object mutation
    def generate(self, model, **opts):
        self.network = model.generate(self, **opts)

    def network_from_file_gexf(self, path):
        in_network = nx.read_gexf(path)
        self.N = in_network.number_of_nodes()
        self.names = np.array(in_network.nodes())
        self.network = sparse.lil_matrix((self.N,self.N), dtype=np.bool)

        name_to_index = {}
        for i, n in enumerate(self.names):
            name_to_index[n] = i
        for s, t in in_network.edges():
            self.network[name_to_index[s],name_to_index[t]] = True

    def network_from_edges(self, edges):
        # First pass over edges to determine names and number of nodes
        names = set()
        N = 0
        for n_1, n_2 in edges:
            if not n_1 in names:
                names.add(n_1)
                N += 1
            if not n_2 in names:
                names.add(n_2)
                N += 1

        # Process list of names and assign indices
        self.N = N
        self.network = sparse.lil_matrix((self.N,self.N), dtype=np.bool)
        self.names = np.array(list(names))
        name_to_index = {}
        for i, n in enumerate(self.names):
            name_to_index[n] = i

        # Second pass over edges to populate network
        for n_1, n_2 in edges:
            self.network[name_to_index[n_1],name_to_index[n_2]] = True

    def nodes(self):
        return self.names

    def adjacency_matrix(self):
        if self.is_sparse():
            return self.network.todense()
        else:
            return self.network

    def is_sparse(self):
        return sparse.issparse(self.network)

    def sparse_adjacency_matrix(self):
        if self.is_sparse():
            return np.array(self.network)
        else:
            print 'Asked for sparse adjacency matrix of non-sparse network.'
            raise

    def offset_extremes(self):
        if not self.offset:
            self.initialize_offset()

        # (Separately) sort rows and columns of adjacency matrix by
        # increasing sum
        A = np.asarray(self.adjacency_matrix())
        r_ord = np.argsort(A.sum(1))
        c_ord = np.argsort(A.sum(0))
        A = A[r_ord][:,c_ord]

        # Convenience function to set blocks of the offset
        def set_offset_block(r, c, val):
            self.offset[r_ord[r],c_ord[c]] = val

        # Recursively examine for submatrices that will send
        # corresponding EMLE parameter estimates to infinity
        to_screen = [(np.arange(self.N), np.arange(self.N))]
        while len(to_screen) > 0:
            r_act, c_act = to_screen.pop()

            A_act = A[r_act][:,c_act]
            n_act = A_act.shape
            trivial = [(0,0), (0,n_act[1]), (n_act[0],0), (n_act[0],n_act[1])]
            for (i,j) in [(i,j)
                         for i in range(n_act[0] + 1)
                         for j in range(n_act[1] + 1)]:
                if (i,j) in trivial: continue
                if np.any(A_act[:i][:,:j]): continue
                if not np.all(A_act[i:][:,j:]): continue
                set_offset_block(r_act[:i], c_act[:j], -np.inf)
                set_offset_block(r_act[i:], c_act[j:], np.inf)
                if i > 0 and j < n_act[1]:
                    A_sub = A_act[:i][:,j:]
                    if not np.any(A_sub):
                        set_offset_block(r_act[:i], c_act[j:], -np.inf)
                    elif np.all(A_sub):
                        set_offset_block(r_act[:i], c_act[j:], np.inf)
                    else:
                        to_screen.append((r_act[np.arange(i)],
                                          c_act[np.arange(j, n_act[1])]))
                if i < n_act[0] and j > 0:
                    A_sub = A_act[i:][:,:j]
                    if not np.any(A_sub):
                        set_offset_block(r_act[i:], c_act[:j], -np.inf)
                    elif np.all(A_sub):
                        set_offset_block(r_act[i:], c_act[:j], np.inf)
                    else:
                        to_screen.append((r_act[np.arange(i, n_act[0])],
                                          c_act[np.arange(j)]))
                break

    def show(self):
        graph = nx.DiGraph()

        for n in self.nodes():
            graph.add_node(n)

        if self.is_sparse():
            nonzeros = set()
            nz_i, nz_j = self.network.nonzero()
            for n in range(self.network.nnz):
                graph.add_edge(self.names[nz_i[n]],self.names[nz_j[n]])
        else:
            for i in range(self.N):
                for j in range(self.N):
                    if self.network[i,j]:
                        graph.add_edge(self.names[i], self.names[j])
        
        nx.draw_graphviz(graph)
        plt.show()

    def show_heatmap(self, order_by = None):
        if order_by:
            title = 'Adjacency matrix ordered by node covariate\n"%s"' % order_by
            o = np.argsort(self.node_covariates[order_by][:])
        else:
            title, o = 'Unordered adjacency matrix', np.arange(self.N)

        f, (ax_im, ax_ord) = plt.subplots(2, sharex = True)
        f.set_figwidth(3)
        f.set_figheight(6)
        A = self.adjacency_matrix()
        ax_im.imshow(A[o][:,o]).set_cmap('binary')
        ax_im.set_ylim(0, self.N - 1)
        ax_im.set_xticks([])
        ax_im.set_yticks([])
        ax_im.set_title(title)
        #plt.setp([ax_im.get_xticklabels(), ax_im.get_yticklabels()],
        #         visible = False)
        if order_by:
            ax_ord.scatter(np.arange(self.N), self.node_covariates[order_by][o])
            ax_ord.set_xlim(0, self.N - 1)
            ax_ord.set_ylim(self.node_covariates[order_by][o[0]],
                            self.node_covariates[order_by][o[-1]])
        plt.show()

    def show_offset(self, order_by = None):
        if order_by:
            title = 'Offsets ordered by node covariate\n"%s"' % order_by
            o = np.argsort(self.node_covariates[order_by][:])
        else:
            title, o = 'Unordered offsets', np.arange(self.N)

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
        O = self.offset.matrix()
        ax.imshow(O[o][:,o])
        ax.set_xlim(0, self.N - 1)
        ax.set_ylim(0, self.N - 1)
        ax.set_title(title)
        plt.setp([ax.get_xticklabels(), ax.get_yticklabels()],
                 visible = False)
        plt.show()

    def show_degree_histograms(self):
        # Messy since otherwise row/column sums can overflow...
        r = np.array(self.network.asfptype().sum(1),dtype=np.int).flatten()
        c = np.array(self.network.asfptype().sum(0),dtype=np.int).flatten()
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.title('out-degree')
        plt.hist(r, bins = max(r))
        plt.subplot(2,1,2)
        plt.title('in-degree')
        plt.hist(c, bins = max(c))
        plt.show()

# Some "tests"
if __name__ == '__main__':
    net = Network()
    net.network_from_file_gexf('data/test.gexf')
    net.new_node_covariate('x_0')
    net.node_covariates['x_0'].from_pairs([str(i) for i in range(10)],
                                          [i**2 for i in range(10)])
    net.new_node_covariate('x_1')
    net.node_covariates['x_1'].data[:] = np.random.normal(2,1,net.N)
    def f_self(n_1, n_2):
        return n_1 == n_2
    net.new_edge_covariate('self_edge').from_binary_function_name(f_self)
    def f_first_half_dir(n_1, n_2):
        return (n_1 < n_2) and (n_2 in ['0','1','2','3','4'])
    net.new_edge_covariate('ec_2').from_binary_function_name(f_first_half_dir)

    print net.node_covariates['x_0']
    print net.node_covariates['x_1']
    print net.edge_covariates['self_edge']
    print net.edge_covariates['ec_2']
    
    print net.adjacency_matrix()
    print net.nodes()
    net.show()

    net_2 = net.subnetwork(np.array([5,0,1,6]))
    print net_2.adjacency_matrix()
    print net_2.node_covariates['x_0']
    print net_2.node_covariates['x_1']
    print net_2.edge_covariates['self_edge']
    print net_2.edge_covariates['ec_2']
    net_2.show()
            
    net_3 = Network(10)
    ord = np.arange(10)
    np.random.shuffle(ord)
    for i in range(10):
        for j in range(i,10):
            net_3.network[ord[i],ord[j]] = True
    net_3.offset_extremes()
    print net_3.offset.matrix()
