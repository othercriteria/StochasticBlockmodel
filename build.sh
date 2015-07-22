#!/usr/bin/env bash

# Compiles C code until I get around to setting up a proper build system.
# Daniel Klein, 12/4/2012

gcc -O3 -fPIC -c support_BinaryMatrix.c -o support_BinaryMatrix.o
gcc -shared -o support.so -lc -lm -lgmp support_BinaryMatrix.o

