#!/usr/bin/env python

# Network representation and basic operations
# Daniel Klein, 5/10/2012

from os import system, unlink
import numpy as np
import scipy.sparse as sparse
import networkx as nx
import matplotlib.pyplot as plt

from Array import Array
from Covariate import NodeCovariate

class Network(Array):
    def __init__(self, N = 0, names = None):
        Array.__init__(self, N, N)
        if names is None:
            self.names = np.array(['%d' % n for n in range(self.N)])
        else:
            self.names = names
        self.rnames = self.names
        self.cnames = self.names
        self.node_covariates = {}

    def new_node_covariate(self, name, as_int = False):
        if as_int:
            node_cov = NodeCovariate(self.names, dtype = np.int)
        else:
            node_cov = NodeCovariate(self.names)
        self.node_covariates[name] = node_cov
        self.row_covariates[name] = node_cov
        self.col_covariates[name] = node_cov
        return node_cov

    def new_node_covariate_int(self, name):
        return self.new_node_covariate(name, as_int = True)

    def subnetwork(self, inds):
        sub_array = self.subarray(inds, inds)

        sub = Network(len(inds), self.names[inds])
        sub.array = sub_array.array
        sub.row_covariates = sub_array.row_covariates
        sub.col_covariates = sub_array.col_covariates
        if sub_array.offset:
            sub.offset = sub_array.offset
        sub.edge_covariates = sub_array.edge_covariates
        for node_covariate in self.node_covariates:
            src = self.node_covariates[node_covariate]
            sub.node_covariates[node_covariate] = src.subset(inds)

        return sub

    def nodes(self):
        return self.names

    def edges(self):
        if self.is_sparse():
            nz_i, nz_j = self.array.nonzero()
            for n in range(self.array.nnz):
                yield (self.names[nz_i[n]], self.names[nz_j[n]])
        else:
            for i in range(self.N):
                for j in range(self.N):
                    if self.array[i,j]:
                        yield (self.names[i], self.names[j])

    def show(self):
        graph = nx.DiGraph()

        for n in self.nodes():
            graph.add_node(n)

        for n_i, n_j in self.edges():
            graph.add_edge(n_i, n_j)

        pos = nx.nx_pydot.graphviz_layout(graph, prog = 'neato')
        nx.draw(graph, pos)
        plt.show()

    def show_graphviz(self, file = 'out.pdf', splines = True, labels = True):
        outfile = open('temp_graphviz.dot', 'w')
        outfile.write('digraph G {\n')
        outfile.write('size="12,16";\n')
        outfile.write('orientation=landscape;\n')
        outfile.write('overlap=none;\n')
        outfile.write('repulsiveforce=12;\n')
        if splines:
            outfile.write('splines=true;\n')

        for name in self.nodes():
            outfile.write('%s [label=""];\n' % name)

        for edge in self.edges():
            outfile.write('%s -> %s;\n' % edge)
                        
        outfile.write('}\n')
        outfile.close()
        system('fdp -Tps2 temp_graphviz.dot -o temp_graphviz.ps')
        unlink('temp_graphviz.dot')
        system('ps2pdf temp_graphviz.ps %s' % file)
        unlink('temp_graphviz.ps')

    def show_heatmap(self, order_by = None,
                     order_by_row = None, order_by_col = None):
        if order_by:
            title = 'Network ordered by node covariate\n"%s"' % order_by
            o = np.argsort(self.node_covariates[order_by][:])
        elif order_by_row:
            title = 'Network ordered by row covariate\n"%s"' % order_by_row
            o = np.argsort(self.row_covariates[order_by_row][:])
        elif order_by_col:
            title = 'Network ordered by column covariate\n"%s"' % order_by_col
            o = np.argsort(self.col_covariates[order_by_col][:])
        else:
            title, o = 'Unordered adjacency matrix', np.arange(self.N)

        f, (ax_im, ax_ord) = plt.subplots(2, sharex = True)
        f.set_figwidth(3)
        f.set_figheight(6)
        A = self.as_dense()
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
        O = self.initialize_offset().matrix()
        ax.imshow(O[o][:,o])
        ax.set_xlim(0, self.N - 1)
        ax.set_ylim(0, self.N - 1)
        ax.set_title(title)
        plt.setp([ax.get_xticklabels(), ax.get_yticklabels()],
                 visible = False)
        plt.show()

    def show_degree_histograms(self):
        # Messy since otherwise row/column sums can overflow...
        r = np.array(self.array.asfptype().sum(1),dtype=np.int).flatten()
        c = np.array(self.array.asfptype().sum(0),dtype=np.int).flatten()
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.title('out-degree')
        plt.hist(r, bins = max(r))
        plt.subplot(2,1,2)
        plt.title('in-degree')
        plt.hist(c, bins = max(c))
        plt.show()

def network_from_networkx(g, cov_names = []):
    N = g.number_of_nodes()
    names = np.array(g.nodes())
    network = Network(N, names)

    name_to_index = {}
    for i, n in enumerate(names):
        name_to_index[n] = i
    for s, t in g.edges():
        network.array[name_to_index[s],name_to_index[t]] = True

    for cov_name in cov_names:
        nodes = g.nodes()
        covs = [g.node[n][cov_name] for n in nodes]
        network.new_node_covariate(cov_name).from_pairs(nodes, covs)

    return network

def network_from_file_gexf(path, cov_names = []):
    in_network = nx.read_gexf(path)
    return network_from_networkx(in_network, cov_names)

def network_from_file_gml(path, cov_names = []):
    in_network = nx.read_gml(path)
    in_network = nx.DiGraph(in_network)
    return network_from_networkx(in_network, cov_names)

def network_from_edges(edges):
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
    network = Network(N, np.array(list(names)))
    name_to_index = {}
    for i, n in enumerate(names):
        name_to_index[n] = i

    # Second pass over edges to populate network
    for n_1, n_2 in edges:
        network[name_to_index[n_1],name_to_index[n_2]] = True

    return network


# Some "tests"
if __name__ == '__main__':
    net = network_from_file_gexf('data/test.gexf')
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
    
    print net.as_dense()
    print net.nodes()
    net.show()

    net_2 = net.subnetwork(np.array([5,0,1,6]))
    print net_2.as_dense()
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
            net_3.array[ord[i],ord[j]] = True
    net_3.offset_extremes()
    print net_3.offset.matrix()
    print net_3.subnetwork(np.array([2,1,0])).offset.matrix()

    net_4 = network_from_file_gml('data/polblogs/polblogs.gml', ['value'])
    print net_4.node_covariates['value']
