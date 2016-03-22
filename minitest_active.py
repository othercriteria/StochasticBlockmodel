#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from Network import Network
from Models import NonstationaryLogistic
from Models import alpha_zero, alpha_norm, alpha_gamma, alpha_unif
from Experiment import RandomSubnetworks

# Parameters
N = 300
reps = 10
sub_sizes = range(10, 110, 10)
kappa_target = ('row_sum', 2)

net = Network(N)
model = NonstationaryLogistic()

num_sizes = len(sub_sizes)
data_none = np.empty((num_sizes,reps))
data_het = np.empty((3,3,num_sizes,reps))
for i, degree_het in enumerate(['Normal', 'Gamma', 'Uniform', 'None']):
    if degree_het == 'None':
        alpha_zero(net)
    for j, het_sd in enumerate([1.0, 2.0, 3.0, 0.0]):
        if degree_het == 'None' and het_sd != 0.0: continue
        if degree_het != 'None' and het_sd == 0.0: continue
        
        if degree_het == 'Normal':
            alpha_norm(net, het_sd)
        if degree_het == 'Gamma':
            alpha_gamma(net, 4.0, het_sd)
        if degree_het == 'Uniform':
            alpha_unif(net, het_sd)
                
        for k, sub_size in enumerate(sub_sizes):
            size = (sub_size, sub_size)
            gen = RandomSubnetworks(net, size)
            
            for l in range(reps):
                print degree_het, het_sd, size, l

                subnet = gen.sample()
                model.match_kappa(subnet, kappa_target)
                subnet.generate(model)
                
                subnet.offset_extremes()

                active = np.isfinite(subnet.offset.matrix()).sum()

                if degree_het == 'None':
                    data_none[k, l] = active
                else:
                    data_het[i, j, k, l] = active

plt.figure()
plt.title('None')
plt.xlabel('Network size')
plt.ylabel('#Active')
for l in range(reps):
    plt.plot(sub_sizes, data_none[:,l], 'b.')

plt.figure()
a = [None] * 3
plt.title('Normal')
plt.xlabel('Network size')
plt.ylabel('#Active')
for j, het_sd in enumerate([1.0, 2.0, 3.0]):
    colors = ['g', 'c', 'm']
    for l in range(reps):
        a[j], = plt.plot(sub_sizes, data_het[0,j,:,l], colors[j] + '.')
plt.legend(a, ['SD = 1.0', 'SD = 2.0', 'SD = 3.0'], loc = 2)

plt.figure()
for i, degree_het in enumerate(['Normal', 'Gamma', 'Uniform']):
    if i == 0:
        continue
    plt.subplot(2, 1, i)
    plt.title(degree_het)
    plt.xlabel('Network size')
    plt.ylabel('#Active')
    a = [None] * 3
    for j, het_sd in enumerate([1.0, 2.0, 3.0]):
        colors = ['g', 'c', 'm']
        for l in range(reps):
            a[j], = plt.plot(sub_sizes, data_het[i,j,:,l], colors[j] + '.')
    plt.legend(a, ['SD = 1.0', 'SD = 2.0', 'SD = 3.0'], loc = 2)
plt.tight_layout()

plt.show()
