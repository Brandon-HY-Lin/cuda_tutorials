
#include <iostream>
#include <cuda.h>
#include <ctime>
#include <stdlib.h>

__global__
void AddInts (int* a, int* b, const int count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < count)
		a[i] += b[i];
}

int main ()
{
	const int count = 1000;
	int *h_a = new int[count];
	int *h_b = new int[count];

	int *d_a;
	int *d_b;

	// init host memory
	std::srand(time(NULL));

	for (int i=0; i<count; ++i) {
		h_a[i] = rand() % 1000;
		h_b[i] = rand() % 1000;
	}

	std::cout << "Prior to calculation" << std::endl;
	for (int i = 0; i < 5; ++i) {
		std::cout << h_a[i] << " + " << h_b[i] << std::endl;
	}

	// alloc GPU memory
	cudaMalloc(&d_a, sizeof(int)*count);

	if (d_a == NULL) {
		std::cout << "Fail to alloc GPU mem\n";
		return -1;
	}

	cudaMalloc(&d_b, sizeof(int)*count);
	if (d_b == NULL) {
		std::cout << "Fail to alloc GPU mem\n";
		return -1;
	}

	// copy initialized host memory to device
	if (cudaMemcpy(d_a, h_a, count*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cout << "Failed to move mem from host to device\n";
		cudaFree(d_a);
		cudaFree(d_b);
	}

	if (cudaMemcpy(d_b, h_b, count*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
		std::cout << "Failed to move mem from host to device\n";
		cudaFree(d_a);
		cudaFree(d_b);
	}

	// Parrellel calculation
	AddInts<<< count/256+1, 256>>>(d_a, d_b, count);

	// copy result from device to host 
	if (cudaMemcpy(h_a, d_a, count*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
		std::cout << "Failed to move data from device to host\n";
		cudaFree(d_a);
		cudaFree(d_b);
		return -1;
	}

	for (int i = 0; i < 5; ++i) {
		std::cout << "It's " << h_a[i] << std::endl;
	}

	cudaFree(d_a);
	cudaFree(d_b);

	return 0;
}
