#!/usr/bin/env python

# Representing networks and models for interactive visualization on the web
# Daniel Klein, 8/1/2012

import numpy as np
import json

def dump_to_json(network):
    s = { 'nodes': [], 'links': [] }

    # Nodes
    raw = {}
    for name in network.names:
        raw[name] = { 'name': name }
    for cov_name in network.node_covariates:
        cov = network.node_covariates[cov_name]
        for name, val in zip(cov.names, cov.data):
            raw[name][cov_name] = val
    for name in network.names:
        s['nodes'].append(raw[name])

    # Edges
    raw = {}
    nnz_i, nnz_j = network.network.nonzero()
    for i, j in zip(nnz_i, nnz_j):
        i, j = int(i), int(j)
        s['links'].append({ 'source': i, 'target': j })

    return json.dumps(s)
