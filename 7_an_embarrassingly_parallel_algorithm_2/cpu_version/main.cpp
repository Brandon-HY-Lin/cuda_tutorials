
#include <iostream>
#include <cmath>
#include <stdlib.h>  // library of rand()
#include <ctime>

struct float3
{
	float x, y, z;
};


void FindClosetCPU(float3* points, int* indices, int count)
{
	// Base case, if there's 1 point don't do anything
	if (count <= 1)
		return;

	// Loop through every point
	for (int curPoint = 0; curPoint < count; curPoint++) {
		// This variable is nearest so far, set it to float.max
		float distToCloset = 3.40282e38f;

		// See how far it is from every other point
		for (int i = 0; i < count; ++i) {

			// Don't check distance to itself
			if (i == curPoint)
				continue;

			/*
			float dist = std::sqrt(
				(points[curPoint].x - points[i].x) * (points[curPoint].x - points[i].x) +
				(points[curPoint].y - points[i].y) * (points[curPoint].y - points[i].y) +
				(points[curPoint].z - points[i].z) * (points[curPoint].z - points[i].z));
			*/
			float dist =
				(points[curPoint].x - points[i].x) * (points[curPoint].x - points[i].x) +
				(points[curPoint].y - points[i].y) * (points[curPoint].y - points[i].y) +
				(points[curPoint].z - points[i].z) * (points[curPoint].z - points[i].z);

			if (dist < distToCloset) {
				distToCloset = dist;
				indices[curPoint] = i;
			}
		}
	}
}


int main()
{
	srand(std::time(NULL));

	// Number of points
	const int count = 10000;

	// Arrays of points
	int* indexOfClosest = new int[count];
	float3* points = new float3[count];

	// Create a list of random points
	for (int i = 0; i < count; ++i) {
		points[i].x = (float)((rand() % 10000) - 5000);
		points[i].y = (float)((rand() % 10000) - 5000);
		points[i].z = (float)((rand() % 10000) - 5000);
	}

	// This variable is used to keep track of the fastest time so far
	long fastest = 1000000;

	// Run the algorithm 20 times
	for (int q = 0; q < 20; ++q) {
		long startTime = std::clock();

		// Run the algorithm
		FindClosetCPU(points, indexOfClosest, count);

		long finishTime = std::clock();

		long runtime = (finishTime - startTime);
		std::cout << "Run " << q << " took " << runtime << " millis" << std::endl;

		// If that run was faster, update the fastest time so far
		if (runtime < fastest) {
			fastest = runtime;
		}
	}

	// Print out the fastest time
	std::cout << "Fastest time: " << fastest << std::endl;

	// Print the final results to screen
	std::cout << "Final results: " << std::endl;
	for (int i = 0; i < 10; ++i) {
		std::cout << i << "." << indexOfClosest[i] << std::endl;
	}

	// Deallocate ram
	delete[] indexOfClosest;
	delete[] points;

	std::cin.get();

	return 0;
}
