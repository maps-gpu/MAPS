#!/bin/bash
# This script must be run with sudo.

# Adapted from Caffe: https://github.com/BVLC/caffe/blob/master/scripts/travis/travis_install.sh

set -e

# This ppa is for gflags and glog
add-apt-repository -y ppa:tuleu/precise-backports
apt-get -y update
apt-get install \
    wget git curl \
    libgflags-dev libgoogle-glog-dev \
    bc

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

# Install CUDA 7.5, if needed
CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
CUDA_FILE=/tmp/cuda_install.deb
curl $CUDA_URL -o $CUDA_FILE
dpkg -i $CUDA_FILE
rm -f $CUDA_FILE
apt-get -y update
# Install the minimal CUDA subpackages required to test the build.
# For a full CUDA installation, add 'cuda' to the list of packages.
apt-get -y install cuda-core-7-5 cuda-cublas-7-5 cuda-cublas-dev-7-5 cuda-cudart-7-5 cuda-cudart-dev-7-5

# Create CUDA symlink at /usr/local/cuda
# (This would normally be created by the CUDA installer, but we create it
# manually since we did a partial installation.)
ln -s /usr/local/cuda-7.5 /usr/local/cuda
