#!/usr/bin/env python

# Grab runs from S3 and do analysis
#
# Daniel Klein, 2015-08-14

import sys
import subprocess
import glob

# Putting this in front of expensive imports
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('remote_dir', help = 'S3 directory with completed runs')
parser.add_argument('run_id', help = 'ID of run to analyze')
parser.add_argument('--leave', help = 'don\'t delete downloaded files',
                    action = 'store_true')
                    
args = parser.parse_args()

match_run = '%s_*__completed.json' % args.run_id

subprocess.call(['aws', 's3', 'sync', args.remote_dir, 'runs/',
                 '--exclude', '*',
                 '--include', match_run])

runs = glob.glob('runs/' + match_run)
print runs

run_stems = [run.split('completed')[0] for run in runs]

subprocess.call(['python', 'test.py'] + \
                [run_stem + 'load.json' for run_stem in run_stems])
    
subprocess.call(['mv', 'out.pdf', 'runs/%s_figs.pdf' % args.run_id])

if not args.leave:
    for run in runs:
        subprocess.call(['rm', run])
