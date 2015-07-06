: Script for compiling executables.

set FLAGS= -std=c99 -g -Wall -O2 -fomit-frame-pointer -Wl,--large-address-aware
set INCLUDES= -I./GMP/
set LIBRARIES= -L./GMP/ -lgmp -lm

gcc count.c  -o count.exe  %FLAGS% %INCLUDES% %LIBRARIES%

gcc sample.c -o sample.exe %FLAGS% %INCLUDES% %LIBRARIES%

gcc Darwin.c -o Darwin.exe %FLAGS% %INCLUDES% %LIBRARIES%

gcc Patterson.c -o Patterson.exe %FLAGS% %INCLUDES% %LIBRARIES%

pause