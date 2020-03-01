#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

#include <stdlib.h>
#include <ctime>


int main ()
{
	srand(time(NULL));

	thrust::device_vector<int> dv(0);
	thrust::host_vector<int> hv(0);

	for (int i = 0; i < 5; ++i) {
		hv.push_back(rand() % 101);
	}

	dv = hv;
	
	thrust::sort(dv.begin(), dv.end());

	float sum = thrust::reduce(dv.begin(), dv.end());

	std::cout << "Average is " << sum / 5.0f  << std::endl;

	return 0;
}
