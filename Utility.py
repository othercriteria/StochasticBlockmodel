#!/usr/bin/env python

# Utility functions.
# Daniel Klein, 5/21/2012

import numpy as np
from scipy.stats import norm
from scipy.special import logit as logit
from scipy.special import expit as inv_logit

import pickle
from collections import defaultdict

def l2(x):
    return np.sqrt(np.sum(x ** 2))

def logsumexp(x):
    return reduce(np.logaddexp, x)

def logabsdiffexp(x, y):
    def f(x, y):
        if x < y: x, y = y, x
        return y + np.log(np.expm1(x - y))
    return (np.vectorize(f))(x,y)

# Get the "mean" log-odds from a collection of log-odds
def logit_mean(x):
    return logit(np.mean(inv_logit(x)))

# Convenience functions for (un)pickling
pick = lambda x: pickle.dumps(x, protocol = 0)
unpick = lambda x: pickle.loads(x)

# Autovivifying nested dictionary
def tree():
    return defaultdict(tree)

# Initialize LaTeX rendering in matplotlib
def init_latex_rendering():
    import os
    os.environ['PATH'] += ':/usr/texbin'
    from matplotlib import rc
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex = True)

# Convenience functions for plotting ellipses, e.g., 2-d confidence regions
#
# Finding the right settings for Ellipse is surprisingly tricky so I follow:
#   http://scikit-learn.org/stable/auto_examples/plot_lda_qda.html
try:
    from matplotlib.patches import Ellipse
except:
    print 'Failed import of matplotlib.patches.Ellipse.'
def draw_confidence(a, m, S, levels = [0.95]):
    # Convert levels into ellipse scale multipliers
    d = norm()
    multipliers = [d.ppf(l / 2.0 + 0.5) for l in levels]
    multipliers.sort()
    multipliers.reverse()
    alphas = np.linspace(1.0, 0.0, len(multipliers) + 2)[1:-1]

    v, w = np.linalg.eigh(S)
    u = w[0] / np.linalg.norm(w[0])
    angle = (180.0 / np.pi) * np.arctan(u[1] / u[0])
    for alpha, multiplier in zip(alphas, multipliers):
        e = Ellipse(m, multiplier * np.sqrt(v[0]), multiplier * np.sqrt(v[1]),
                    180.0 + angle, color = 'k')
        a.add_artist(e)
        e.set_clip_box(a.bbox)
        e.set_alpha(alpha)
