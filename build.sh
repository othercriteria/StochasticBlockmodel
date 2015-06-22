#!/usr/bin/env bash

# Compiles C code until I get around to setting up a proper build system.
# For now, this is OS X-centric.
# Daniel Klein, 12/4/2012

gcc -O3 -fPIC -c support_BinaryMatrix.c -o support_BinaryMatrix.o
ld -dylib -macosx_version_min 10.7 -o support.so -lc -lm -lgmp support_BinaryMatrix.o outside_code/exact_v0.21/code/C_version/exact_library.o

