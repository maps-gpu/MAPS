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

#ifndef __MAPS_MAPS_CUH_
#define __MAPS_MAPS_CUH_

// Common headers
#include "internal/cuda_utils.hpp"
#include "internal/common.cuh"

// Input Containers
#include "input_containers/adjacency.cuh"
#include "input_containers/block1D.cuh"
#include "input_containers/block2D.cuh"
#include "input_containers/permutation.cuh"
#include "input_containers/traversal.cuh"
#include "input_containers/window1D.cuh"
#include "input_containers/window2D.cuh"

// Algorithms
#include "algorithms/histogram.cuh"
#include "algorithms/reduction.cuh"

#endif  // __MAPS_MAPS_CUH_
