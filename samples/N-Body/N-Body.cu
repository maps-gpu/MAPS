/*
 *	MAPS: GPU device level memory abstraction library
 *	Based on the paper: "MAPS: Optimizing Massively Parallel Applications 
 *	Using Device-Level Memory Abstraction"
 *	Copyright (C) 2014  Amnon Barak, Eri Rubin, Tal Ben-Nun
 *
 *	This program is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *	
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *	
 *	You should have received a copy of the GNU General Public License
 *	along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <maps/maps.cuh>

#define NUM_BODIES 16384
#define TIMESTEPS 100

#define DAMPING_FACTOR 0.5f
#define SOFTENING_FACTOR 0.001f
#define DELTA_TIME 0.01f

#define BW 256
#define BH 1

enum ComputationType
{
	CT_NAIVE,
	CT_MAPS
};

template<typename T>
__device__ T rsqrt_T(T x)
{
    return rsqrt(x);
}

template<>
__device__ float rsqrt_T<float>(float x)
{
    return rsqrtf(x);
}

// Function taken from the NVIDIA CUDA NBody code sample
template <typename T>
__device__ float3
bodyBodyInteraction(float3 ai,
                    float4 bi,
                    float4 bj)
{
    float3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    T distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += (SOFTENING_FACTOR*SOFTENING_FACTOR);

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist = rsqrt_T(distSqr);
    T invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__device__ float3 ComputeAccelerationNaive(float4 pos, const float4 *inBodies, int numBodies)
{
	float3 acc = {0.0f, 0.0f, 0.0f};

	for (unsigned int counter = 0; counter < numBodies; counter++)
	{
	    acc = bodyBodyInteraction<float>(acc, pos, inBodies[counter]);
	}

    return acc;
}

template<int BLOCK_WIDTH, int BLOCK_HEIGHT>
__device__ __forceinline__ void ComputeAccelerationMAPS(float4 pos, const float4 *inBodies, 
														int numBodies, float3& accel)
{
	typedef maps::Block1D<float4, BLOCK_WIDTH, BLOCK_HEIGHT> block1Dtype;
	__shared__ block1Dtype blk;

	blk.init(inBodies, numBodies);
	
	do
	{	
		blk.nextChunk();
		
		typename block1Dtype::iterator eiter = blk.end();

		#pragma unroll 32
		for(typename block1Dtype::iterator iter = blk.begin(); iter != eiter; ++iter)
			accel = bodyBodyInteraction<float>(accel, pos, *iter);

	} while(!blk.isDone());
}

template<int BLOCK_WIDTH, int BLOCK_HEIGHT, ComputationType type>
__global__ void NBodyTimeStep(const float4 *inBodies, float4 *outBodies,
							  float4 *velocity,
							  int numBodies, float deltaTime)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= numBodies)
		return;

	float4 pos = inBodies[idx];
	float3 accel = {0.0f, 0.0f, 0.0f};

	if(type == CT_NAIVE)
		accel = ComputeAccelerationNaive(pos, inBodies, numBodies);
	else if(type == CT_MAPS)
		ComputeAccelerationMAPS<BLOCK_WIDTH, BLOCK_HEIGHT>(pos, inBodies, numBodies, accel);

	// Integrate: Compute new positions and velocities from acceleration
	float4 vel = velocity[idx];

    vel.x += accel.x * deltaTime;
    vel.y += accel.y * deltaTime;
    vel.z += accel.z * deltaTime;

    vel.x *= DAMPING_FACTOR;
    vel.y *= DAMPING_FACTOR;
    vel.z *= DAMPING_FACTOR;

    pos.x += vel.x * deltaTime;
    pos.y += vel.y * deltaTime;
    pos.z += vel.z * deltaTime;

	velocity[idx]  = vel;
	outBodies[idx] = pos;
}

template <typename T>
inline T FPRandom(const T& min, const T& max)
{
	return (((T)rand() / (T)RAND_MAX) * (max - min)) + min;
}

template <typename T>
inline T sq(const T& val) 
{ 
	return val*val; 
}

int main(int argc, char **argv)
{
	size_t numBodies = NUM_BODIES;
	std::vector<float4> bodies (numBodies);
	
	srand((unsigned int)time(NULL));

	// Create input data by randomly placing bodies in the unit cube	
	for(size_t i = 0; i < numBodies; i++)
	{
		// Bodies are organized as follows: [x, y, z, inverse mass]
		bodies[i] = make_float4(
			FPRandom(-1.0f, 1.0f), FPRandom(-1.0f, 1.0f), FPRandom(-1.0f, 1.0f), 1.0f);
	}
	
	// Allocate GPU buffers (using double buffering)
	float4 *dev_bodiesA = NULL, *dev_bodiesB = NULL, *dev_velocity = NULL;
	float4 *dev_MAPSbodiesA = NULL, *dev_MAPSbodiesB = NULL, *dev_MAPSvelocity = NULL;

	MAPS_CUDA_CHECK(cudaMalloc(&dev_bodiesA, numBodies * sizeof(float4)));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_bodiesB, numBodies * sizeof(float4)));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_velocity, numBodies * sizeof(float4)));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_MAPSbodiesA, numBodies * sizeof(float4)));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_MAPSbodiesB, numBodies * sizeof(float4)));
	MAPS_CUDA_CHECK(cudaMalloc(&dev_MAPSvelocity, numBodies * sizeof(float4)));
	
	// Copy input data to GPU
	MAPS_CUDA_CHECK(cudaMemset(dev_velocity, 0, numBodies * sizeof(float4)));
	MAPS_CUDA_CHECK(cudaMemset(dev_MAPSvelocity, 0, numBodies * sizeof(float4)));
	MAPS_CUDA_CHECK(cudaMemcpy(dev_bodiesA,     &bodies[0], numBodies * sizeof(float4), cudaMemcpyHostToDevice));
	MAPS_CUDA_CHECK(cudaMemcpy(dev_MAPSbodiesA, &bodies[0], numBodies * sizeof(float4), cudaMemcpyHostToDevice));

	dim3 block_dims (BW, BH, 1);
	dim3 grid_dims (maps::RoundUp(numBodies, block_dims.x), 1, 1);

	printf("Computing Naive N-Body simulation\n");

	// Run N-Body simulation (Naive)
	for(int i = 0; i < TIMESTEPS; i++)
	{
		NBodyTimeStep<BW, BH, CT_NAIVE> <<<grid_dims, block_dims>>>(
			(i % 2 == 0) ? dev_bodiesA : dev_bodiesB,
			(i % 2 == 0) ? dev_bodiesB : dev_bodiesA,
			dev_velocity, numBodies, DELTA_TIME);
	}

	MAPS_CUDA_CHECK(cudaDeviceSynchronize());
	printf("Computing MAPS N-Body simulation\n");

	// Run N-Body simulation (MAPS)
	for(int i = 0; i < TIMESTEPS; i++)
	{
		NBodyTimeStep<BW, BH, CT_MAPS> <<<grid_dims, block_dims>>>(
			((i % 2) == 0) ? dev_MAPSbodiesA : dev_MAPSbodiesB,
			((i % 2) == 0) ? dev_MAPSbodiesB : dev_MAPSbodiesA,
			dev_MAPSvelocity, numBodies, DELTA_TIME);
	}

	MAPS_CUDA_CHECK(cudaDeviceSynchronize());

	// Copy and compare the results
	std::vector<float4> finalPos (numBodies), finalPosMAPS (numBodies);
	MAPS_CUDA_CHECK(cudaMemcpy(&finalPos[0], 
							   ((TIMESTEPS % 2) == 0) ? dev_bodiesA : dev_bodiesB,
							   numBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	MAPS_CUDA_CHECK(cudaMemcpy(&finalPosMAPS[0], 
							   ((TIMESTEPS % 2) == 0) ? dev_MAPSbodiesA : dev_MAPSbodiesB,
							   numBodies * sizeof(float4), cudaMemcpyDeviceToHost));
	
	// NOTE: There will always be a difference between the two different methods
	//       due to floating-point errors.
	//       These errors will also aggregate with the time-steps.
	float meanError = 0.0f;
	for(size_t i = 0; i < numBodies; i++)
	{
		float error = fabs(sq(finalPos[i].x - finalPosMAPS[i].x) +
						   sq(finalPos[i].y - finalPosMAPS[i].y) +
						   sq(finalPos[i].z - finalPosMAPS[i].z));

		meanError += error;
	}
	meanError /= numBodies;
	printf("Mean error for %d time-steps: %f\n", TIMESTEPS, meanError);

	// Free allocated data
	MAPS_CUDA_CHECK(cudaFree(dev_bodiesA));
	MAPS_CUDA_CHECK(cudaFree(dev_bodiesB));
	MAPS_CUDA_CHECK(cudaFree(dev_velocity));
	MAPS_CUDA_CHECK(cudaFree(dev_MAPSbodiesA));
	MAPS_CUDA_CHECK(cudaFree(dev_MAPSbodiesB));
	MAPS_CUDA_CHECK(cudaFree(dev_MAPSvelocity));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

	printf("Done!\n");
    
    return 0;
}
