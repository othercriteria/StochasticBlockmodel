
===============================================================================
CONTENTS
===============================================================================
Python_version

    exact_pure.py
    A pure Python implementation of the counting and sampling algorithm, in the case of binary matrices.

    examples.py
    Examples illustrating how to use exact_pure.py.


C_version

    *.c, *.h
    C code implementing the counting and sampling algorithm, for both binary and nonnegative integer matrices.
    
    make.bat
    make.sh
    Scripts for compiling the "count" and "sample" programs (.bat for Windows, .sh for *nix)
    
    test.bat
    test.sh
    Scripts running "count" and "sample" on a simple test example (.bat for Windows, .sh for *nix).
    
    input.dat
    Example input file for "count" and "sample".
    
    Darwin.c
    Patterson.c
    p_value.c
    These are examples of how to write a C program that uses count.c and sample.c.
    They show how to compute a p-value and a histogram for a test statistic.
    
    GMP
    This folder contains the GMP (http://gmplib.org) headers and library files for WINDOWS.
    
    
===============================================================================
COMPILING INSTRUCTIONS
===============================================================================

-------
WINDOWS
-------
If you are running Windows, you probably don't need to compile this code, since precompiled versions are already in the folder "executables". If you want to compile it, continue reading.

Install the GCC compiler (if you haven't already), and add it to the Windows environment variable PATH.
I recommend using the MinGW GCC compiler, available for free at: http://www.mingw.org/

You should be able to just click on make.bat and then test.bat.


-------
*NIX
-------
Make sure you have the GCC compiler and the basic C equipment installed (e.g. stdio.h, libm, etc.). On Ubuntu, for example, you can call:

    sudo apt-get install build-essential

Install the GMP library (http://gmplib.org). Again, on Ubuntu, you can call:

    sudo apt-get install libgmp3-dev

CD to the folder code/C_version and call:

    chmod 755 *.sh
    ./make.sh

If all goes well, you will not get any errors. Then you can call ./test.sh to try it out.


If you get an error like:

    /usr/bin/ld: cannot find -lgmp
    
then, assuming you have GMP installed, it's probably installed to someplace that's not on the include path and library path for GCC. You will need to modify make.sh appropriately, e.g.

    INCLUDES="-I/path/to/gmp/includes/"
    LIBRARIES="-L/path/to/gmp/libs/ -lgmp -lm"

where /path/to/gmp/includes/ contains gmp.h and /path/to/gmp/libs/ contains libgmp.*.



===============================================================================
GMP library
===============================================================================

The code uses the Gnu Multiple Precision (GMP) library (http://gmplib.org) for handling integers of arbitrary size. I've included a precompiled copy of the GMP 4.1 libraries for Windows, downloaded from  http://www.cs.nyu.edu/exact/core/gmp/.
GMP is distributed under the LGPL license.


===============================================================================
Other code that's not mine
===============================================================================

khash.h
A very nice hash table implementation.
http://klib.sourceforge.net/
See also attractivechaos.wordpress.com
It is distributed under the MIT license (see khash.h).

lookup3.c
A hash function for integer arrays.
By Bob Jenkins 
http://burtleburtle.net/bob/c/lookup3.c
See also http://en.wikipedia.org/wiki/Jenkins_hash_function
It is in the Public Domain (see lookup3.c).


















