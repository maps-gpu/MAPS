#!/bin/bash

set -e

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..

make -j $NUM_THREADS -k
