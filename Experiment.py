#!/usr/bin/env python

# Code used for testing and quantifying network inference
# Daniel Klein, 5/15/2012

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# Convenience object for consistent generation of test data
class Seed:
    def __init__(self, seed = 0):
        self.seed = seed
        np.random.seed(self.seed)

    def next(self):
        self.seed += 1
        np.random.seed(self.seed)

    def final(self):
        self.next()
        print 'URN from Seed:', np.random.random()

class RandomSubnetworks:
    def __init__(self, network, train_size, test_size = 0, method = 'node'):
        self.network = network
        self.train_size = train_size
        self.test_size = test_size
        self.method = method

        if self.method == 'node':
            if type(self.train_size) == tuple:
                self.rinds = np.arange(self.network.N)
                self.cinds = np.arange(self.network.N)
            else:
                self.inds = np.arange(self.network.N)
        else:
            self.rinds = np.arange(self.network.M)
            self.cinds = np.arange(self.network.N)
            if self.method == 'edge':
                edges = self.network.array.nonzero()
                edges_i, edges_j = edges[0], edges[1]
                self.edges_i = edges_i
                self.edges_j = edges_j
                self.num_edges = len(edges_i)
            elif self.method in ('link', 'link_f') :
                edges = self.network.array.nonzero()
                self.num_edges = len(edges[0])
                neighbors = { n: set() for n in range(self.network.N) }
                for i, j in zip(edges[0], edges[1]):
                    neighbors[i].add(j)
                    if self.method == 'link':
                        neighbors[j].add(i)
                self.neighbors = { n: list(neighbors[n]) for n in neighbors }
                self.num_nodes = len(self.neighbors)

    def sample(self, as_network = False):
        M = self.network.M
        N = self.network.N
        s = self.train_size

        def produce(r, c):
            if as_network:
                inds = np.unique(np.array(list(r) + list(c)))
                return self.network.subnetwork(inds)
            else:
                r_a = np.array(list(r))
                r_c = np.array(list(c))
                return self.network.subarray(r_a, r_c)

        def fallback():
            s_r, s_c = s
            np.random.shuffle(self.rinds)
            np.random.shuffle(self.cinds)
            if as_network:
                return produce(self.rinds[0:s_r], self.rinds[0:s_r])
            else:
                return produce(self.rinds[0:s_r], self.cinds[0:s_c])

        if self.method == 'node':
            return fallback()
        elif self.method == 'row':
            np.random.shuffle(self.rinds)
            return produce(self.rinds[0:s], self.cinds)
        elif self.method == 'col':
            np.random.shuffle(self.cinds)
            return produce(self.rinds, self.cinds[0:s])
        elif self.method in ('edge', 'link', 'link_f'):
            s_r, s_c = s

            if (self.num_edges < s_r) or (self.num_edges < s_c):
                return fallback()

            added_r = set()
            added_c = set()

            if self.method == 'edge':
                while (len(added_r) < s_r) and (len(added_c) < s_c):
                    e = np.random.randint(self.num_edges)
                    edge_i = self.edges_i[e]
                    edge_j = self.edges_j[e]
                    added_r.add(edge_i)
                    added_c.add(edge_j)
            elif self.method in ('link', 'link_f'):
                # The difference between 'link' and 'link_f' is in the
                # construction of neighbors, not the algorithm used to
                # follow links
                loc = np.random.randint(self.num_nodes)
                while (len(added_r) < s_r) and (len(added_c) < s_c):
                    added_r.add(loc)
                    if added_r.issuperset(self.neighbors[loc]):
                        loc = np.random.randint(self.num_nodes)
                        continue
                    pick_j = np.random.randint(len(self.neighbors[loc]))
                    loc_j = self.neighbors[loc][pick_j]
                    added_c.add(loc_j)
                    loc = loc_j

            remain_r = list(set(range(M)).difference(added_r))
            np.random.shuffle(remain_r)
            added_r.update(set(remain_r[0:(s_r - len(added_r))]))

            remain_c = list(set(range(N)).difference(added_c))
            np.random.shuffle(remain_c)
            added_c.update(set(remain_c[0:(s_c - len(added_c))]))

            return produce(added_r, added_c)

        if self.test_size == 0:
            if self.method in ('row', 'col'):
                return self.network.subnetwork(sub_train, self.method)
            else:
                return self.network.subnetwork(sub_train)
        else:
            sub_full = self.inds[0:(self.train_size + self.test_size)]
            return (self.network.subnetwork(sub_train),
                    self.network.subnetwork(sub_full))

# A major purpose of this package is studying consistency and other
# distributional properties of estimators. Hence, a natural
# experimental setup will be repeating an inference procedure on a
# range of network sizes. This class is designed to expedite such
# experiments.

class Results:
    def __init__(self, M_sizes, N_sizes, num_reps, title = None,
                 interactive = True):
        self.M_sizes = M_sizes
        self.N_sizes = N_sizes

        len_M = len(self.M_sizes)
        len_N = len(self.N_sizes)
        if not (len_M == len_N):
            print 'Warning: mismatch in M-dims and N-dims lengths.'
            num_both = min(len_M, len_N)
            self.M_sizes = self.M_sizes[0:num_both]
            self.N_sizes = self.N_sizes[0:num_both]

        # Remove redundant conditions
        sizes_seen = set()
        nonredundant = np.zeros_like(M_sizes, dtype=np.bool)
        for i, size in enumerate(zip(M_sizes, N_sizes)):
            if not size in sizes_seen:
                nonredundant[i] = True
            sizes_seen.add(size)
        self.M_sizes = self.M_sizes[nonredundant]
        self.N_sizes = self.N_sizes[nonredundant]

        self.num_conditions = len(self.M_sizes)
        self.num_reps = num_reps
        self.title = title
        self.interactive = interactive
        self.results = {}

        self.size_to_ind = {}
        for i, size in enumerate(zip(self.M_sizes, self.N_sizes)):
            self.size_to_ind[size] = i

    def merge(self, other):
        if not (np.all(self.M_sizes == other.M_sizes) and
                np.all(self.N_sizes == other.N_sizes)):
            print 'Warning: mismatched conditions in Results to merge.'

        self.num_reps = self.num_reps + other.num_reps
        for result_name in self.results:
            old_data = self.results[result_name]['data']
            other_data = other.results[result_name]['data']
            merged_data = np.hstack([old_data, other_data])
            self.results[result_name]['data'] = merged_data

    # Return a copy of the result structure, with new allocated storage
    def copy(self):
        dup = Results(self.M_sizes, self.N_sizes,
                      self.num_reps,
                      title = self.title, interactive = self.interactive)

        for result_name in self.results:
            result = self.results[result_name]
            f, f_type = result['f'], result['f_type']

            dup.new(result_name, f_type, f)

        return dup

    # Values expected for "f_type": 'a' (adjacency)
    #                               'n' (network)
    #                               'm' (models),
    #                               'nm' (network and models)
    def new(self, name, f_type, f):
        assert(f_type in ['a', 'n', 'm', 'nm', 'dof'])
        self.results[name] = { 'f': f, 'f_type': f_type,
                               'data': np.empty((self.num_conditions,
                                                 self.num_reps)) }

        return name

    def record(self, size, rep, network,
               data_model = None, fit_model = None):
        for result in self.results:
            f = self.results[result]['f']
            f_type = self.results[result]['f_type']
            if f_type == 'a':
                val = f(network.as_dense())
            elif f_type == 'n':
                val = f(network)
            elif f_type == 'm':
                val = f(data_model, fit_model)
            elif f_type == 'nm':
                val = f(network, data_model, fit_model)
            elif f_type == 'dof':
                val = f(network.M, network.N, len(fit_model.beta))

            data = self.results[result]['data']
            data[self.size_to_ind[size], rep] = val

    # To be called after all results have been recorded...
    def sweep(self, name, func, *args):
        self.results[name] = {'data': np.empty((self.num_conditions,1))}

        data = []
        for arg in args:
            data.append(self.results[arg]['data'])

        for n in range(self.num_conditions):
            self.results[name]['data'][n,0] = func(n, *data)

    def estimate_mse(self, name, true, estimate):
        self.sweep(name, lambda n, t, e: np.mean((t[n]-e[n])**2),
                   true, estimate)

    def estimate_mae(self, name, true, estimate):
        self.sweep(name, lambda n, t, e: np.median(np.abs(t[n]-e[n])),
                   true, estimate)

    def summary(self):
        for field in self.results:
            data = self.results[field]['data']
            average_data = np.mean(data, 1)
            print '%s: %s' % (field, repr(average_data))

    def plot(self, requests = None, general = {'xaxis': 'c'}):
        if general['xaxis'] == 'c':
            sizes = self.N_sizes
            sizes_label = 'N'
        elif general['xaxis'] == 'r':
            sizes = self.M_sizes
            sizes_label = 'M'
        else:
            print 'Warning: expecting \'xaxis\' to be \'r\' or \'c\'.'

        if requests is None:
            requests = self.results.keys()
        num_plots = len(requests)

        f, axarr = plt.subplots(num_plots, sharex = True)
        if num_plots == 1:
            axarr = [axarr]
            
        if self.title:
            axarr[0].set_title(self.title)

        for i, request in enumerate(requests):
            ax = axarr[i]

            if type(request) != str:
                names, options = request
            else:
                names, options = request, {}

            if type(names) == str:
                names = [names, names]
            plot_name, names = names[0], names[1:]
            for name in names:
                result = self.results[name]
                data = result['data']
                if 'plot_mean' in options and options['plot_mean']:
                    ax.plot(sizes, data.mean(1))
                else:
                    for rep in range(data.shape[1]):
                        ax.scatter(sizes, data[:,rep])

            if 'baseline' in options:
                ax.plot(sizes, np.repeat(options['baseline'], len(sizes)),
                        'k:')

            ax.set_ylabel(plot_name)
            if 'ymin' in options and 'ymax' in options:
                ax.set_ylim(options['ymin'], options['ymax'])
            elif 'ymin' in options:
                ax.set_ylim(ymin = options['ymin'])
            elif 'ymax' in options:
                ax.set_ylim(ymax = options['ymax'])

            if 'loglog' in options:
                ax.set_xscale('log')
                ax.set_yscale('log')
            elif 'semilogx' in options:
                ax.set_xscale('log')
            elif 'semilogy' in options:
                ax.set_yscale('log')

        axarr[-1].set_xlabel(sizes_label)
        f.subplots_adjust(hspace = 0)

        if self.interactive:
            plt.show()

# Add a suite of standard array statistics to a Results instance
def add_array_stats(results, network = False):
    results.new('Density', 'a',
                lambda a: 1.0 * np.sum(a) / (a.shape[0] * a.shape[1]))
    results.new('Average row-sum', 'a',
                lambda a: 1.0 * np.sum(a) / a.shape[0])
    results.new('Average col-sum', 'a',
                lambda a: 1.0 * np.sum(a) / a.shape[1])
    results.new('Max row-sum', 'a', lambda a: np.max(a.sum(1)))
    results.new('Min row-sum', 'a', lambda a: np.min(a.sum(1)))
    results.new('Max col-sum', 'a', lambda a: np.max(a.sum(0)))
    results.new('Min col-sum', 'a', lambda a: np.min(a.sum(0)))
    if network:
        results.new('Self-loop density', 'a',
                    lambda a: np.mean(np.diagonal(a)))

# Find the minimum of a disagreement function from true class labels
# over distinct relabelings of the estimated class labels
#
# Eventually should add option to use Hungarian algorithm, although
# this is probably unnecessary for, say, K <= 6.
def minimum_disagreement(z_true, z_est, f = None, normalized = True):
    from itertools import permutations

    assert(len(z_true) == len(z_est))
    N = len(z_true)

    if not f:
        def f(x, y):
            return np.sum(x != y) / N

    true_classes = list(set(z_true))
    est_classes = list(set(z_est))
    K_true = len(true_classes)
    K_est = len(est_classes)

    if K_est < K_true:
        est_classes += [-1] * (K_true - K_est)
    best = np.inf
    for est_permutation in permutations(est_classes, K_true):
        z_est_perm = np.repeat(-1, N)
        for s, t in zip(est_permutation, true_classes):
            z_est_perm[z_est == s] = t
        best = min(best, f(z_true, z_est_perm))

    if normalized:
        best_constant = np.inf
        for z in set(z_true):
            z_constant = np.repeat(z, N)
            best_constant = min(best_constant, f(z_true, z_constant))
        best /= best_constant
        
    return best

# Differences of infinities make sense in this context...
def robust_mse(x, y):
    diff = x - y
    equal_cells = (x == y)
    diff[equal_cells] = 0
    return np.mean(diff ** 2)
    
def rel_mse(est_1, est_2, truth):
    return robust_mse(est_1, truth) / robust_mse(est_2, truth)
