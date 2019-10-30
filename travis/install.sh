#!/bin/bash
# This script must be run with sudo.

set -e

# Set GCC 4.9 as the default compiler
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.6 10
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 20
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.6 10
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 20
update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30
update-alternatives --set cc /usr/bin/gcc
update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30
update-alternatives --set c++ /usr/bin/g++
update-alternatives --set gcc /usr/bin/gcc-4.9
update-alternatives --set g++ /usr/bin/g++-4.9

# gtest
wget --no-check-certificate https://github.com/google/googletest/archive/release-1.7.0.zip -O gtest.zip
unzip gtest.zip
mv googletest-release-1.7.0/* win/gtest
chmod -R a+rw win/gtest
rm -rf googletest-release-1.7.0 gtest.zip

# CMake
wget --no-check-certificate http://www.cmake.org/files/v3.2/cmake-3.2.3-Linux-x86_64.sh -O cmake3.sh
chmod +x cmake3.sh
./cmake3.sh --prefix=/usr/ --skip-license --exclude-subdir
rm -f ./cmake3.sh
