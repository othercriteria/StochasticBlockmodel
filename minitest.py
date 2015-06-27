#!/usr/bin/env python

from Models import Stationary, Blockmodel, FixedMargins
from Network import Network

net = Network(100)
net.new_node_covariate_int('r')[:] = 20
net.new_node_covariate_int('c')[:] = 20
net.new_node_covariate_int('z')[:] = ([0] * 50) + ([1] * 50)

base_model = Blockmodel(Stationary(),2)
base_model.Theta[0,0] = 3.0
base_model.Theta[0,1] = -1.0
base_model.Theta[1,0] = -2.0
base_model.Theta[1,1] = 0.0
model = FixedMargins(base_model)

net.generate(base_model)
net.show_heatmap('z')

net.generate(model)
net.show_heatmap('z')

