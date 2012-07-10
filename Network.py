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
            
    def new_node_covariate(self, name):
        self.node_covariates[name] = NodeCovariate(self.names)
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
    def generate(self, model, *opts):
        self.network = model.generate(self, *opts)

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
            title = 'Ordered by node covariate "%s"' % order_by
            o = np.argsort(self.node_covariates[order_by][:])
        else:
            title, o = 'Unordered', np.arange(self.N)

        plt.figure()
        A = self.adjacency_matrix()
        plt.imshow(A[o][:,o])
        plt.set_cmap('binary')
        plt.title(title)
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
    net.network_from_file_gexf('test.gexf')
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
            
