cmake_minimum_required (VERSION 2.6)
project (maps)

############## DEPENDENCIES ##############

find_package(CUDA 7 REQUIRED)

find_package(GTest QUIET)
if(NOT GTEST_FOUND)
  add_subdirectory(win/gtest)
endif()



############## BUILD ##############

include_directories(${PROJECT_SOURCE_DIR}/include)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-g;-lineinfo;-Xcompiler;-ggdb;-std=c++11)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG;-std=c++11)
endif()

list(APPEND CMAKE_CXX_FLAGS "-std=c++11")

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

cuda_add_executable(gol samples/game-of-life/main.cpp samples/game-of-life/golsample.cu)
target_link_libraries(gol gflags)

cuda_add_executable(sgemm samples/sgemm/main.cpp samples/sgemm/sgemm.cu)
cuda_add_cublas_to_target(sgemm)
target_link_libraries(sgemm gflags)

cuda_add_executable(histogram samples/histogram/main.cpp samples/histogram/histogram.cu)
target_link_libraries(histogram gflags)

# Unit tests
enable_testing()

# GTest directory settings
if(NOT GTEST_FOUND)
  include_directories(
    ${gtest_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR})
  link_directories(${gtest_BINARY_DIR}/src)
endif()

cuda_add_executable(unit-tests test/unit-tests/main.cpp
                    test/unit-tests/block_perftest.cu
                    test/unit-tests/block_test.cu
                    test/unit-tests/input_common_test.cu
                    test/unit-tests/segmentation_test.cu
                    test/unit-tests/system_test.cu
                    test/unit-tests/window_perftest.cu
                    test/unit-tests/window_test.cu
                    test/unit-tests/invoke_multithreading_test.cu)

target_link_libraries(unit-tests gtest)
cuda_add_cublas_to_target(unit-tests)

GTEST_ADD_TESTS(unit-tests test/unit-tests/main.cpp
                test/unit-tests/block_perftest.cu
                test/unit-tests/block_test.cu
                test/unit-tests/input_common_test.cu
                test/unit-tests/segmentation_test.cu
                test/unit-tests/system_test.cu
                test/unit-tests/window_perftest.cu
                test/unit-tests/window_test.cu
                test/unit-tests/invoke_multithreading_test.cu)
