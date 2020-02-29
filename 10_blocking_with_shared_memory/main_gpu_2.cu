
#include <stdlib.h>
#include <ctime>
#include <iostream>
#include <cuda.h>

struct float3_t
{
	float x, y, z;
};

__device__ const int blockSize = 640;


__global__
void FindClosetGPU2 (float3_t* points, int* indices, int count)
{

__shared__ float3_t sharedPoints[blockSize];

	if (count <= 1)
		return;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int indexOfClosest = -1;

	float3_t thisPoint;

	float distanceToCloset = 3.40282e38f;

	if (idx < count) thisPoint = points[idx];

	// iterate blocks
	for (int currentBlockOfPoints = 0; currentBlockOfPoints < gridDim.x; currentBlockOfPoints++) {
		// copy GPU memory to block cache (shared memory).
		if (threadIdx.x + currentBlockOfPoints * blockSize < count) {
			sharedPoints[threadIdx.x] = points[threadIdx.x + currentBlockOfPoints * blockSize];
		}

		__syncthreads();

		if (idx < count) {
			// Using reference is faster the using indexing.
			float *ptr = &sharedPoints[0].x;

			// K-Neariest-Neighbor wthin a block, i.e. K=blockSize.
			for (int i = 0; i < blockSize; i++) {
				float dist = (thisPoint.x - ptr[0]) * (thisPoint.x - ptr[0]) +
					(thisPoint.y - ptr[1]) * (thisPoint.y - ptr[1]) +
					(thisPoint.z - ptr[2]) * (thisPoint.z - ptr[2]);

				ptr += 3;

				if ((dist < distanceToCloset) &&
					(i + currentBlockOfPoints * blockSize < count) &
					(i + currentBlockOfPoints * blockSize != idx)) {

					distanceToCloset = dist;
					indexOfClosest = i + currentBlockOfPoints * blockSize;
				}
			}
		}

		__syncthreads();
	}

	if (idx < count) {
		indices[idx] = indexOfClosest;
	}
}

int main ()
{
	std::srand(std::time(NULL));

	const int count = 10000;

	float3_t* h_points = new float3_t[count];
	int* h_indices = new int[count];
	float3_t* d_points;
	int* d_indices;
	
	// alloc memory on device 
	cudaMalloc(&d_points, count*sizeof(float3_t));
	if (d_points == NULL) {
		std::cout << "Failed to alloc GPU mem\n";
		delete[] h_points;
		delete[] h_indices;
		return -1;
	}

	cudaMalloc(&d_indices, count*sizeof(int));
	if (d_indices == NULL) {
		std::cout << "Failed to alloc GPU mem" << std::endl;
		cudaFree(d_points);
		delete[] h_points;
		delete[] h_indices;
		return -1;
	}

	// init host memory
	for (int i=0; i < count; ++i) {
		h_points[i].x = std::rand() % 10000 - 5000;
		h_points[i].y = std::rand() % 10000 - 5000;
		h_points[i].z = std::rand() % 10000 - 5000;
	}

	// move data from host to device
	cudaMemcpy(d_points, h_points, count*sizeof(float3_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, h_indices, count*sizeof(int), cudaMemcpyHostToDevice);
	
	long fastest = 2^31 - 1;

	for (int q=0; q < 20; ++q) {
		long startTime = std::clock();

		// execute KNN
		FindClosetGPU2<<<count/640+1, 640>>>(d_points, d_indices, count);

		long endTime = std::clock();
		long runTime = endTime - startTime;

		fastest = max(fastest, runTime);
		std::cout << q << ". Run time: " << runTime << std::endl;

		
	}

	std::cout << "Fastest time: " << fastest << std::endl;

	// move result from device to host
	cudaMemcpy(h_points, d_points, count*sizeof(float3_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_indices, d_indices, count*sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << "Final results:" << std::endl;
	for (int i=0; i < 20; ++i) {
		std::cout << i << "." << h_indices[i] << std::endl;
	}

	// free memory
	cudaFree(d_points);
	cudaFree(d_indices);

	delete[] h_points;
	delete[] h_indices;

	return 0;
}

