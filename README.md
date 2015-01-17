MAPS: Device-Level GPU Memory Abstraction Library
=================================================

MAPS is a header-only C++ CUDA template library for automatic optimization of 
GPU kernels. It uses common memory access patterns to provide near-optimal 
performance. 

For more information, see the library website at:
http://www.cs.huji.ac.il/~talbn/maps/


Requirements
------------

CUDA.


Installation
------------

To compile code with MAPS, use the includes under the "include" directory.

You can either include specific header files, or include MAPS using the 
all-inclusive header (from .cu files only):

``` cpp
#include <maps/maps.cuh>
```


Samples
-------

Code samples are available under the "samples" directory. To compile,
either use Visual Studio on Windows or CMake on other platforms (http://www.cmake.org/)
