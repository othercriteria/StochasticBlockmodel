#!/usr/bin/env python

# Grab runs from S3 and do analysis
#
# Daniel Klein, 2015-08-14

import sys
import subprocess
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('remote_dir', help = 'S3 directory with completed runs')
parser.add_argument('output_dir', help = 'Local destination for output')
parser.add_argument('run_ids', help = 'IDs of runs to analyze',
                    nargs = '+')
parser.add_argument('--leave', help = 'don\'t delete downloaded files',
                    action = 'store_true')
                    
args = parser.parse_args()

for run_id in args.run_ids:
    print run_id
    
    match_run = '%s_*__completed.json' % run_id

    subprocess.call(['aws', 's3', 'sync', args.remote_dir, args.output_dir,
                     '--exclude', '*',
                     '--include', match_run])

    runs = glob.glob(args.output_dir + match_run)
    print runs

    run_stems = [run.split('completed')[0] for run in runs]

    subprocess.call(['python', 'test.py'] + \
                    [run_stem + 'load.json' for run_stem in run_stems])

    subprocess.call(['mv', 'out.pdf',
                     '%s/%s_figs.pdf' % (args.output_dir, run_id)])

    if not args.leave:
        for run in runs:
            subprocess.call(['rm', run])
