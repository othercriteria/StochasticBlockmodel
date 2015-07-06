#!/bin/bash

# Script for compiling executables.

FLAGS="-std=c99 -g -Wall -O2 -fomit-frame-pointer"
INCLUDES=""
LIBRARIES="-lgmp -lm"

gcc count.c  -o count.exe  $FLAGS $INCLUDES $LIBRARIES
gcc sample.c -o sample.exe $FLAGS $INCLUDES $LIBRARIES
gcc Darwin.c -o Darwin.exe $FLAGS $INCLUDES $LIBRARIES
gcc Patterson.c -o Patterson.exe $FLAGS $INCLUDES $LIBRARIES

chmod 755 count.exe
chmod 755 sample.exe
chmod 755 Darwin.exe
chmod 755 Patterson.exe


