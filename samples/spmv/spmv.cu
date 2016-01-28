// MAPS - Memory Access Pattern Specification Framework
// http://maps-gpu.github.io/
// Copyright (c) 2015, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include <algorithm>
#include <vector>
#include <unordered_set>
#include <random>
#include <map>
#include <memory>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <gflags/gflags.h>

#include <maps/maps.cuh>
#include <maps/multi/multi.cuh>

#include "mmio.h"
#include "stopwatch.h"

#ifndef NDEBUG

#define BS 32
#define VERBOSE true

#define REPETITIONS_DEFAULT 1
#define ROWS_DEFAULT 100
#define COLS_DEFAULT 100
#define NNZ_DEFAULT 200

#else

// Depends on the architecture
#define BS 256
#define VERBOSE false

#define REPETITIONS_DEFAULT 200
#define ROWS_DEFAULT 62451
#define COLS_DEFAULT 62451
#define NNZ_DEFAULT 2034917

#endif

DEFINE_string(matrixfile, "", "A file with a matrix in matrix-market format");
DEFINE_bool(verbose, VERBOSE, "Be verbose with error details");

DEFINE_int32(repetitions, REPETITIONS_DEFAULT, "Repetitions of GPU tests");
DEFINE_int32(rows, ROWS_DEFAULT, "Number of rows in matrix if no file is specified");
DEFINE_int32(cols, COLS_DEFAULT, "Number of columns in matrix if no file is specified");
DEFINE_int32(nnz, NNZ_DEFAULT, "Number of sparse values (randomized) in matrix if no file is specified (nnz << rows * cols)");

__global__ void SpMVMapsMultiKernel MAPS_MULTIDEF(
    maps::Adjacency<float, float> inGraph,
    maps::StructuredInjectiveOutput<float, 1, BS, 1, 1> outVector)
{
    MAPS_MULTI_INITVARS(inGraph, outVector);

    // Return only after participating in shared memory load
    if (outVector.Items() == 0)
        return;

    float res = 0.f;
    for (auto it = inGraph.begin(); it != inGraph.end(); ++it)
    {
        res +=
            (*it).edge_weight *            // The matrix value 
            (*it).adjacent_node_value;     // The vector value
    }

    *outVector.begin() = res;
    outVector.commit();
}

bool CompareHostBuffers(
    const float *host_data1, const float *host_data2, const unsigned int size, bool verbose = false)
{
    float meanDiff = 0.0f;
    int numDiffs = 0;

    for (size_t i = 0; i < size; ++i)
    {
        float diff = host_data1[i] - host_data2[i];
        if (fabs(1.0f - (host_data1[i] / host_data2[i])) > 1e-2)
        {
            if (verbose)
                printf("Difference in index %d: %f != %f\n", (int)i, host_data1[i], host_data2[i]);
            numDiffs++;
        }
        meanDiff += fabs(diff);
    }
    meanDiff /= size;

    bool res = !(numDiffs > 0 || meanDiff > 1e-3);
    if (!res || verbose)
        printf("SUMMARY: %d/%d large differences, total mean diff: %f\n", numDiffs, (int)size, meanDiff);
    return res;
}

bool CompareDeviceHostBuffers(
    const float *dev_data, const float *host_data, const unsigned int size, bool verbose = false)
{
    std::vector<float> dev_to_host_copy(size);
    MAPS_CUDA_CHECK(cudaMemcpy(&dev_to_host_copy[0], dev_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    return CompareHostBuffers(&dev_to_host_copy[0], host_data, size, verbose);
}

template<typename T>
struct MatrixCell
{
    int i, j;
    T val;
};

template<typename T>
struct SparseMatrix
{
    int rows;
    int cols;
    int NNZ;
    bool symmetric;

    std::vector< MatrixCell<T> > cells;

    SparseMatrix(int rows, int cols, int nnz, bool symmetric = false)
        : rows(rows), cols(cols), NNZ(nnz), cells(nnz), symmetric(symmetric)
    {

    }
};

template<typename T>
struct CSRSparseMatrix
{
    int rows;
    int cols;
    int NNZ;

    bool symmetric;
    unsigned int maxRowRank;

    std::vector<T>        valueArray;
    std::vector<int>    columnPerValueArray;
    std::vector<int>    rowIndexArray;

    CSRSparseMatrix(const SparseMatrix<T>& source)
        :
        rows(source.rows), cols(source.cols), NNZ(source.NNZ), maxRowRank(0), symmetric(source.symmetric),
        valueArray(source.NNZ), columnPerValueArray(source.NNZ), rowIndexArray(source.rows + 1)
    {
        // Note: relying on source to be sorted by row/column

        int counter = 0;
        int row = source.cells[0].i;

        unsigned int row_size = 0;
        rowIndexArray[0] = 0;

        for (auto&& cell : source.cells)
        {
            if (row != cell.i)
            {
                maxRowRank = std::max(maxRowRank, row_size);
                row_size = 0;

                for (int j = row; j < cell.i; j++) {
                    rowIndexArray[j + 1] = counter;
                }

                row = cell.i;
            }

            row_size++;

            valueArray[counter] = cell.val;
            columnPerValueArray[counter] = cell.j;

            counter++;
        }

        maxRowRank = std::max(maxRowRank, row_size);
        for (int j = row; j < rows; j++) {
            rowIndexArray[j + 1] = counter;
        }
    }
};

std::shared_ptr< SparseMatrix<float> > CreateRandomizedSparseMatrix(int rows, int cols, int nnz)
{
    printf("\nMatrix file was not specified, creating randomized matrix\n");
    printf("Matrix has %d NNZ, %d rows, and %d columns\n", nnz, rows, cols);

    std::shared_ptr< SparseMatrix<float> > matrix = std::make_shared< SparseMatrix<float> >(rows, cols, nnz, false);

    std::default_random_engine generator;

    std::uniform_int_distribution<int> r_distribution(0, rows - 1);
    std::uniform_int_distribution<int> c_distribution(0, cols - 1);
    std::uniform_real_distribution<float> nnz_distribution(0.0, 100.0);

    struct hash {
        size_t operator()(const std::pair<int, int>& pair) const
        {
            long long chain = (((int64_t)pair.first) << sizeof(int) * 8) | pair.second;
            std::hash<int64_t> h;
            return h(chain);
        }
    };

    std::unordered_set< std::pair<int, int>, hash> indexes;

    for (size_t i = 0; i < nnz; ++i)
    {
        int r, c;

        do {
            r = r_distribution(generator);
            c = c_distribution(generator);
        } while (indexes.find(std::make_pair(r, c)) != indexes.end());

        indexes.insert(std::make_pair(r, c));

        auto& cell = matrix->cells[i];
        cell.i = r;
        cell.j = c;
        cell.val = nnz_distribution(generator);
    }

    std::sort(
        matrix->cells.begin(),
        matrix->cells.end(),
        [](const MatrixCell<float>& c1, const MatrixCell<float>& c2)
    {
        if (c1.i == c2.i)
            return (c1.j < c2.j);

        return (c1.i < c2.i);
    }
    );

    return matrix;
}

std::shared_ptr< SparseMatrix<float> > ReadSparseMatrix(const std::string& matrixfile)
{
    if (matrixfile == "")
        return CreateRandomizedSparseMatrix(FLAGS_rows, FLAGS_cols, FLAGS_nnz);

    MM_typecode matcode;
    FILE *f;
    int rows, cols, nz;
    int i, *I, *J;
    double *val;
    bool symmetric = false;

    if ((f = fopen(matrixfile.c_str(), "r")) == NULL)
    {
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Failed to process Matrix Market banner in the provided matrix file.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
        mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    symmetric = mm_is_symmetric(matcode);

    /* find out size of sparse matrix .... */

    if ((mm_read_mtx_crd_size(f, &rows, &cols, &nz)) != 0)
        exit(1);

    printf("\nLoading matrix %s\n", matrixfile.substr(matrixfile.find_last_of('\\') + 1).c_str());
    printf("Matrix has %d NNZ, %d rows, and %d columns\n", nz, rows, cols);

    /* reserve memory for matrices */

    I = (int *)malloc(nz * sizeof(int));
    J = (int *)malloc(nz * sizeof(int));
    val = (double *)malloc(nz * sizeof(double));

    /*
    Matrix market format: (from http://math.nist.gov/MatrixMarket/formats.html)

    %%MatrixMarket matrix coordinate real general
    %=================================================================================
    %
    % This ASCII file represents a sparse MxN matrix with L
    % nonzeros in the following Matrix Market format:
    %
    % +----------------------------------------------+
    % |%%MatrixMarket matrix coordinate real general | <--- header line
    % |%                                             | <--+
    % |% comments                                    |    |-- 0 or more comment lines
    % |%                                             | <--+
    % |    M  N  L                                   | <--- rows, columns, entries
    % |    I1  J1  A(I1, J1)                         | <--+
    % |    I2  J2  A(I2, J2)                         |    |
    % |    I3  J3  A(I3, J3)                         |    |-- L lines
    % |        . . .                                 |    |
    % |    IL JL  A(IL, JL)                          | <--+
    % +----------------------------------------------+
    %
    % Indices are 1-based, i.e. A(1,1) is the first element.
    %
    %=================================================================================
    5  5  8
    1     1   1.000e+00
    2     2   1.050e+01
    3     3   1.500e-02
    1     4   6.000e+00
    4     2   2.505e+02
    4     4  -2.800e+02
    4     5   3.332e+01
    5     5   1.200e+01

    This produces the following matrix:
    1    0      0       6      0
    0   10.5    0       0      0
    0    0    .015      0      0
    0  250.5    0     -280    33.32
    0    0      0       0     12
    */

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i = 0; i < nz; i++)
    {
        if(!fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]))
            continue;
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f != stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/


    int symNZcount = 0;
    if (symmetric)
    {
        for (i = 0; i < nz; i++)
        {
            if (I[i] != J[i])
                symNZcount++;
        }
    }

    std::shared_ptr< SparseMatrix<float> > matrix = std::make_shared< SparseMatrix<float> >(rows, cols, nz + symNZcount, symmetric);

    int ii = 0;

    for (i = 0; i < nz; i++)
    {
        matrix->cells[i].i = I[i];
        matrix->cells[i].j = J[i];
        matrix->cells[i].val = (float)val[i];
        if (symmetric && I[i] != J[i])
        {
            matrix->cells[nz + ii].i = J[i];
            matrix->cells[nz + ii].j = I[i];
            matrix->cells[nz + ii].val = (float)val[i];
            ii++;
        }
    }

    if (symmetric)
    {
        printf("Filling symmetric matrix, new size is %d\n", nz + symNZcount);
    }

    std::sort(
        matrix->cells.begin(),
        matrix->cells.end(),
        [](const MatrixCell<float>& c1, const MatrixCell<float>& c2)
    {
        if (c1.i == c2.i)
            return (c1.j < c2.j);

        return (c1.i < c2.i);
    });

    return matrix;
}

std::map < std::string, std::shared_ptr< SparseMatrix<float> > > g_cachedMatrixes;

SparseMatrix<float>& GetCachedSparseMatrix(const std::string& matrixfile)
{
    auto it = g_cachedMatrixes.find(matrixfile);
    if (it == g_cachedMatrixes.end())
    {
        auto sparseMatrix_ptr = ReadSparseMatrix(matrixfile);
        g_cachedMatrixes[matrixfile] = sparseMatrix_ptr;
        return *(sparseMatrix_ptr);
    }
    else
    {
        return *(it->second);
    }
}

bool TestSpMVMAPSMulti(int ngpus)
{
    SparseMatrix<float>& sparseMatrix = GetCachedSparseMatrix(FLAGS_matrixfile);

    // Create input vector
    std::vector<float> host_inVector(sparseMatrix.cols, 1.0f);
    srand(static_cast <unsigned> (123));
    for (int c = 0; c < sparseMatrix.cols; c++)
    {
        host_inVector[c] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    // Run on CPU for regression
    std::vector<float> host_regression(sparseMatrix.rows, 0.0f);
    for (auto&& cell : sparseMatrix.cells)
    {
        host_regression[cell.i] +=
            host_inVector[cell.j] *
            cell.val;
    }

    // Build graph datum
    maps::multi::Graph<float, float> graph;

    // The SpMV input vector can be viewed as per node values in a graph
    graph.AddNodes(host_inVector);

    // Fill the graph with zero value nodes (needed if cols < rows)
    graph.AddNodes(std::max(sparseMatrix.rows, sparseMatrix.cols) - host_inVector.size());

    // The SpMV input matrix can be viewed as per edge weights in a graph
    for (auto&& cell : sparseMatrix.cells) {
        graph.AddEdge(cell.i, cell.j, cell.val);
    }

    // 
    // The SpMV multiplication process can be viewed as a graph adjacency traversal, where each node is operating
    // on its neighbor edges and nodes (multiplying each edge weight with its other node's value, and summing up)
    // 

    // Out Datum and vector
    maps::multi::Vector<float> O(sparseMatrix.rows);
    std::vector<float> host_outVector(sparseMatrix.rows, 0.0f);

    // Work dims
    dim3 block_dims(BS, 1, 1);
    dim3 grid_dims(maps::RoundUp(sparseMatrix.rows, block_dims.x), 1, 1);

    if (ngpus > grid_dims.x) { // Sanity check
        printf("\nSkipping SpMV MAPS-Multi over %d GPUs. Data is too small.\n\n", ngpus);
        return true;
    }

    // Create GPU list
    int num_gpus;
    MAPS_CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    std::vector<unsigned int> gpuids;
    for (int i = 0; i < ngpus; ++i)
        gpuids.push_back(i % num_gpus);

    // Create scheduler
    maps::multi::Scheduler sched(gpuids);

    Stopwatch host_pre_sw(true);
    // Create a graph datum from the raw graph
    // Also triggers any required pre-processing
    maps::multi::GraphDatum<float, float, BS, maps::multi::ADJACENCY> datum(graph);
    host_pre_sw.stop();

    // Analyze the memory access patterns for allocation purposes
    maps::multi::AnalyzeCall(
        sched, grid_dims, block_dims,
        maps::multi::Adjacency<float, float, BS>(datum),
        maps::multi::StructuredInjectiveVectorO<float>(O)
        );

    for (int i = 0; i < num_gpus; i++)
    {
        MAPS_CUDA_CHECK(cudaSetDevice(i));
        MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    }
    MAPS_CUDA_CHECK(cudaSetDevice(0));

    // Invoke once to force memory allocations
    maps::multi::Invoke(
        sched, SpMVMapsMultiKernel, grid_dims, block_dims,
        maps::multi::Adjacency<float, float, BS>(datum),
        maps::multi::StructuredInjectiveVectorO<float>(O));

    Stopwatch maps_sw(true);

    printf("\nLaunching MAPS-Multi kernel with %d GPUs\n", ngpus);

    // Invoke the kernels (data exchanges are performed implicitly)
    for (int i = 0; i < FLAGS_repetitions; ++i)
    {
        maps::multi::Invoke(
            sched, SpMVMapsMultiKernel, grid_dims, block_dims,
            maps::multi::Adjacency<float, float, BS>(datum),
            maps::multi::StructuredInjectiveVectorO<float>(O)
            );
    }

    sched.WaitAll();
    for (int i = 0; i < num_gpus; i++)
    {
        MAPS_CUDA_CHECK(cudaSetDevice(i));
        MAPS_CUDA_CHECK(cudaDeviceSynchronize());
    }
    maps_sw.stop();

    O.Bind(&host_outVector[0]);
    maps::multi::Gather(sched, O);

    bool res = CompareHostBuffers(
        &host_outVector[0], (float*)&(*host_regression.begin()), host_regression.size(), FLAGS_verbose
        );

    printf("MAPS Multi SpMV Host pre processing: %f ms\n", host_pre_sw.ms());
    printf("MAPS Multi SpMV: %f ms\n\n", maps_sw.ms() / FLAGS_repetitions);

    return res;
}
