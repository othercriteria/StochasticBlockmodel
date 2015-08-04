#!/usr/bin/env python

# Generate collection of JSON parameter files that can be used to
# easily spawn simulations over a range of settings.
#
# Daniel Klein, 2015-08-04

import json
import pickle
import numpy as np

# Parameters
generations = 5
reps = 3
c_shape = ('sub_sizes_c', np.floor(np.logspace(1.2, 2.1, 30)))


pick = lambda x: pickle.dumps(x, protocol = 0)
params = {}
def register(x):
    params[x[0]] = x[1]

register(c_shape)
register(('N', int(np.floor(np.max(c_shape[1])) + 100)))

generation_delta = reps * len(c_shape)

index = 0
info = ''
for alpha_dist in (('alpha_unif_sd', 1.0),
                   ('alpha_norm_sd', 1.0),
                   ('alpha_gamma_sd', 1.0)):
    register(alpha_dist)
    for cov_dist in (('cov_unif_sd', 1.0),
                     ('cov_norm_sd', 1.0),
                     ('cov_disc_sd', 1.0)):
        register(cov_dist)
        for density in (('kappa_target', ('row_sum', 2)),
                        ('kappa_target', ('density', 0.1))):
            register(density)
            for r_shape in (('sub_sizes_r', np.repeat(2, 30)),
                            ('sub_sizes_r', np.floor(np.log(c_shape[1]))),
                            ('sub_sizes_r', np.floor(0.2 * c_shape[1]))):
                register(r_shape)
                for method in  (('fit_method', 'convex_opt'),
                                ('fit_method', 'c_conditional'),
                                ('fit_method', 'irls'),
                                ('fit_method', 'logistic_l2'),
                                ('fit_method', 'conditional'),
                                ('fit_method', 'conditional_is')):
                    register(method)

                    if method[1] in ('convex_opt', 'irls', 'logistic_l2'):
                        register(('pre_offset', True))
                        register(('post_fit', False))
                    else:
                        register(('pre_offset', False))
                        register(('post_fit', True))
                    
                    info += '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' % \
                      (index, alpha_dist, cov_dist, density,
                       r_shape, c_shape, method)

                    seed = 0
                    for generation in range(generations):
                        stem = 'runs/%s_%s' % (index, generation)
                        params['random_seed'] = seed

                        dump_filename = stem + '__completed.json'
                        for action in ('dump', 'load'):
                            if action == 'dump':
                                register(('dump_fits', dump_filename))
                                register(('load_fits', None))
                                params_filename = stem + '__dump.json'
                            elif action == 'load':
                                register(('dump_fits', None))
                                register(('load_fits', dump_filename))
                                params_filename = stem + '__load.json'

                            with open(params_filename, 'w') as outfile:
                                json.dump([(p, pick(params[p]))
                                           for p in params],
                                          outfile)
                        seed += generation_delta
                    index += 1

with open('runs/info.txt', 'w') as outfile:
    outfile.write(info)

