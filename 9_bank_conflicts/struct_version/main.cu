#include <iostream>
#include <ctime>
#include <cuda.h>


struct Pehw
{
	float x, y, z, w;
  float padding;
};

__global__
void MyKernel (unsigned long long *time)
{
	__shared__ Pehw shared[1024];
	unsigned long long startTime = std::clock();

	shared[threadIdx.x].x++;

	unsigned long long finishTime = std::clock();

	*time = (finishTime - startTime);
}


int main ()
{
	unsigned long long time;
	unsigned long long *d_time;

	cudaMalloc(&d_time, sizeof(unsigned long long));

	for (int i = 0; i < 10; ++i) {
		MyKernel<<<1, 32>>>(d_time);

		cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

		// 14 is the overhead for calling clock
		std::cout << "Time: " << (time - 14) / 32 << std::endl;

		std::cout << std::endl;
	}

	cudaFree(d_time);

	std::cin.get();

	cudaDeviceReset();

	return 0;
}
