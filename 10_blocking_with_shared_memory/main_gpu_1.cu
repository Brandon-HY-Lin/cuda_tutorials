
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <cuda.h>

struct float3_t
{
	float x, y, z;
};


__global__
void FindClosetGPU (float3_t* points, int* indices, int count)
{
	if (count <= 1)
		return;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < count) {
		float3_t thisPoint = points[idx];
		float smallestSoFar = 3.40282e38f;

		for (int i=0; i < count; ++i) {
			if (i ==  idx) continue;

			float dist = (thisPoint.x - points[i].x) * (thisPoint.x - points[i].x);
			dist += (thisPoint.y - points[i].y) * (thisPoint.y - points[i].y);
			dist += (thisPoint.z - points[i].z) * (thisPoint.z - points[i].z);

			if (dist < smallestSoFar) {
				smallestSoFar = dist;
				indices[idx] =i;
			}
		}
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

	cudaMalloc(&d_points, count*sizeof(float3_t));
	if (d_points == NULL) {
		std::cout << "Failed to alloc mem in GPU" << std::endl;
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
		FindClosetGPU<<< (count/640 + 1), 640 >>>(d_points, d_indices, count);
	
		long endTime = std::clock();
		long runTime = endTime - startTime;

		if (runTime < fastest) {
			fastest = runTime;
		}

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

	cudaFree(d_points);
	cudaFree(d_indices);

	delete[] h_points;
	delete[] h_indices;

	return 0;
}
