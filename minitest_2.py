from Network import Network
from Models import NonstationaryLogistic
from Models import alpha_zero, alpha_norm, alpha_unif, alpha_gamma
n = Network(60)
nsl = NonstationaryLogistic()
nsl.kappa = -0.5
nsl.beta['x'] = 1.0
ec = n.new_edge_covariate('x')
ec.from_binary_function_ind(lambda i, j: random.normal(0,1))
n.edge_covariates['x']
alpha_unif(n, 2.0)
n.generate(nsl)
n_copy = n.subnetwork(range(60))
nsf = NonstationaryLogistic()
nsf.beta['x'] = 0.0
alpha_zero(n)
nsf.fit_mh(n)
