Networks seems like a somewhat propitious environment for learning as, in some sense, we have O(N^2) observations for networks on N individuals. However, model complexity may also grow as O(N) so we are potentially faced with a nuisance parameter problem.

Development Environment
=======================

I'm working in Python 2.7. I make deep use of the Numpy library and more incidental use of the Scipy and scikit-learn libraries. I'm doing hacky stuff for graph visualization using Graphviz but I should probably switch to networkx at some point.

References
==========

I don't make use of (and haven't necessarily worked through) all of these papers, but this list should be roughly indicative of the approach I'm taking.

[1]	K. Rohe, S. Chatterjee, and B. Yu, “Spectral clustering and the high-dimensional stochastic blockmodel,” The Annals of Statistics, vol. 39, no. 4, pp. 1878–1915, Aug. 2011.
[2]	E. M. Airoldi, D. S. Choi, and P. J. Wolfe, “Confidence sets for network structure,” arXiv.org, vol. stat.ME. 31-May-2011.
[3]	B. Karrer and M. Newman, “Stochastic blockmodels and community structure in networks,” Phys. Rev. E, vol. 83, no. 1, p. 016107, 2011.
[4]	D. S. Choi, P. J. Wolfe, and E. M. Airoldi, “Stochastic blockmodels with growing number of classes,” arXiv.org, vol. math.ST. 21-Nov.-2010.
[5]	A. Goldenberg, A. X. Zheng, S. E. Fienberg, and E. M. Airoldi, “A survey of statistical network models,” arXiv, vol. stat.ME, Jan. 2009.
[6]	D. R. Hunter and M. S. Handcock, “Inference in Curved Exponential Family Models for Networks,” Journal of computational and graphical statistics, vol. 15, no. 3, pp. 565–583, Sep. 2006.
[7]	M. Stephens, “Dealing with label switching in mixture models,” Journal of the Royal Statistical Society: Series B (Statistical Methodology), vol. 62, no. 4, pp. 795–809, 2000.
[8]	Y. J. Wang and G. Y. Wong, “Stochastic blockmodels for directed graphs,” Journal of the American Statistical Ass, pp. 8–19, 1987.
[9]	“A nonparametric view of network models and Newman–Girvan and other modularities.”