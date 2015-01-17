cmake_minimum_required (VERSION 2.6)
project (maps-samples)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_20,code=sm_21;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35)

include_directories(${PROJECT_SOURCE_DIR}/include)

cuda_add_executable(convolution samples/Convolution/Convolution.cu)
cuda_add_executable(histogram samples/Histogram/Histogram.cu)
cuda_add_executable(conv2d samples/ImageConvolution/ImageConvolution.cu)
cuda_add_executable(matmul samples/MatrixMultiplication/MatrixMultiplication.cu)
cuda_add_executable(nbody samples/N-Body/N-Body.cu)
