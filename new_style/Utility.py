#!/usr/bin/env python

# Utility functions.
# Daniel Klein, 5/21/2012

import numpy as np

def logit(x):
    return np.log(x / (1.0 - x))

def inv_logit(x):
    return 1.0 / (np.exp(-x) + 1.0)

# Convenience functions for plotting ellipses, e.g., 2-d confidence regions
#
# Finding the right settings for Ellipse is surprisingly tricky so I follow:
#   http://scikit-learn.org/stable/auto_examples/plot_lda_qda.html
try:
    from matplotlib.patches import Ellipse
catch:
    print 'Failed import of matplotlib.patches.Ellipse.'
def draw_ellipse(a, m, S):
    v, w = np.linalg.eigh(S)
    u = w[0] / np.linalg.norm(w[0])
    angle = (180.0 / np.pi) * np.arctan(u[1] / u[0])
    e = Ellipse(m, 2.0 * np.sqrt(v[0]), 2.0 * np.sqrt(v[1]),
                180.0 + angle, color = 'k')
    a.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(0.5)
