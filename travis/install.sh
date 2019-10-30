#!/bin/bash
# This script must be run with sudo.

# Adapted from Caffe: https://github.com/BVLC/caffe/blob/master/scripts/travis/travis_install.sh

set -e

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

# Install CUDA 10.1, if needed
CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-10-1_10.1.243-1_amd64.deb
CUDA_FILE=/tmp/cuda_install.deb
curl $CUDA_URL -o $CUDA_FILE
dpkg -i $CUDA_FILE
rm -f $CUDA_FILE
apt-get -y update
apt-get -y install cuda-toolkit-10-1 cuda-core-10-1 cuda-cublas-10-1 cuda-cublas-dev-10-1 cuda-cudart-10-1 cuda-cudart-dev-10-1
