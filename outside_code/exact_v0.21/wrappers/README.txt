

===============================================================================
CONTENTS
===============================================================================

    Matlab
    Python
    R

These contain scripts for calling the executables count.exe and sample.exe from within these respective programming environments. They are very simple wrappers that interact with the executables through input and output files. These wrappers are platform-independent.


===============================================================================
SETUP INSTRUCTIONS
===============================================================================

(1) Put count.exe and sample.exe somewhere on the system path.

    WINDOWS
    The folder "executables" contains precompiled versions of count.exe and sample.exe for Windows. You can either copy count.exe and sample.exe to someplace already on the path, e.g. C:\Windows, or add the folder "executables" to the environment variable "Path".

    *NIXes
    If on a Unix-like system, follow the instructions in code/README.txt to compile the executables. After compiling them, you can copy them to /usr/bin/ for example.
    
    
    Note: If all else fails, you can always modify the wrapper scripts to use the full path of count.exe and sample.exe (e.g. C:\path\to\exact\executables\count.exe).

    
(2) If you want to be able to call the wrapper functions from anywhere, then add the folder containing them to your programming environment. For example, for Matlab:

    Add the folder "wrappers/Matlab" folder to the Matlab path, and restart Matlab.


    

===============================================================================
USAGE
===============================================================================

See the files:

    simple_example.m
    simple_example.py
    simple_example.r
    












