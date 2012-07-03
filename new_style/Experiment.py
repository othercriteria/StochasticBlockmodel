#!/usr/bin/env python

# Code used for testing and quantifying network inference
# Daniel Klein, 5/15/2012

import numpy as np
import matplotlib.pyplot as plt

class RandomSubnetworks:
    def __init__(self, network, train_size, test_size = 0):
        self.network = network
        self.train_size = train_size
        self.test_size = test_size

        self.inds = np.arange(self.network.N)

    def sample(self):
        np.random.shuffle(self.inds)
        sub_train = self.inds[0:self.train_size]
        if self.test_size == 0:
            return self.network.subnetwork(sub_train)
        else:
            sub_full = self.inds[0:(self.train_size + self.test_size)]
            return (self.network.subnetwork(sub_train),
                    self.network.subnetwork(sub_full))

# A major purpose of this package is studying the consistency. Hence,
# a natural experimental setup will be repeating an inference
# procedure on a range of network sizes. This class is designed to
# expedite such experiments.

class Results:
    def __init__(self, sub_sizes, num_reps, title = None):
        self.sub_sizes = sub_sizes
        self.N_subs = len(sub_sizes)
        self.N_reps = num_reps
        self.title = title
        self.results = {}

        self.sub_size_to_ind = {}
        for i, sub_size in enumerate(sub_sizes):
            self.sub_size_to_ind[sub_size] = i

    # Values expected for "f_type": 'a' (adjacency)
    #                               'm' (models),
    #                               'nm' (network and models)
    def new(self, name, f_type, f):
        self.results[name] = { 'f': f, 'f_type': f_type,
                               'data': np.empty((self.N_subs, self.N_reps)) }

        return name

    def record(self, sub_size, rep, network, data_model, fit_model):
        for result in self.results:
            f = self.results[result]['f']
            f_type = self.results[result]['f_type']
            if f_type == 'a':
                val = f(network.adjacency_matrix())
            elif f_type == 'm':
                val = f(data_model, fit_model)
            elif f_type == 'nm':
                val = f(network, data_model, fit_model)

            data = self.results[result]['data']
            data[self.sub_size_to_ind[sub_size], rep] = val

    # To be called after all results have been recorded...
    def estimate_mse(self, name, true, estimate):
        self.results[name] = {'data': np.empty((self.N_subs,1))}

        t = self.results[true]['data']
        e = self.results[estimate]['data']

        for n in range(self.N_subs):
            self.results[name]['data'][n,0] = np.mean((t[n]-e[n])**2)
                
    def plot(self, requests = None):
        if requests == None:
            requests = self.results.keys()
        
        plt.figure()
        if self.title:
            plt.title(self.title)

        num_plots = len(requests)
        for i, request in enumerate(requests):
            plt.subplot(num_plots, 1, (i+1))

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
                    plt.plot(self.sub_sizes, data.mean(1), hold = True)
                else:
                    for rep in range(data.shape[1]):
                        plt.plot(self.sub_sizes, data[:,rep],
                                 'k.', hold = True)

            plt.ylabel(plot_name)
            if 'ymin' in options and 'ymax' in options:
                plt.ylim(options['ymin'], options['ymax'])
            elif 'ymin' in options:
                plt.ylim(ymin = options['ymin'])
            elif 'ymax' in options:
                plt.ylim(ymax = options['ymax'])

            if (i+1) == num_plots:
                plt.xlabel('N_sub')

        plt.show()

# Add a suite of standard network statistics to a Results instance
def add_network_stats(results):
    results.new('Average degree', 'a', lambda a: 1.0 * np.sum(a) / a.shape[0])
    results.new('Max out-degree', 'a', lambda a: np.max(a.sum(1)))
    results.new('Min out-degree', 'a', lambda a: np.min(a.sum(1)))
    results.new('Max in-degree', 'a', lambda a: np.max(a.sum(0)))
    results.new('Min in-degree', 'a', lambda a: np.min(a.sum(0)))
    results.new('Self-loop density', 'a', lambda a: np.mean(np.diagonal(a)))
