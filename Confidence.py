#!/usr/bin/env python

# Generic procedures used in the construction of confidence intervals
# Daniel Klein, 2015-11-20

import numpy as np

from Utility import logsumexp

def invert_test(theta_grid, test_val, crit):
    theta_l_min, theta_l_max = min(theta_grid), max(theta_grid)

    C_alpha = theta_grid[test_val > crit]
    if len(C_alpha) == 0:
        return 0, 0

    C_alpha_l, C_alpha_u = np.min(C_alpha), np.max(C_alpha)
    if C_alpha_l == theta_l_min:
        C_alpha_l = -np.inf
    if C_alpha_u == theta_l_max:
        C_alpha_u = np.inf

    return C_alpha_l, C_alpha_u

