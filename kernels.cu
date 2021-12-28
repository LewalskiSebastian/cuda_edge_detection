#include "kernels.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#define GRID_SIZE 512

using namespace std;

__global__ void Laplace(int width, int height, unsigned char *d_wsk, unsigned char *device_img)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x+1;
	int j_base = blockIdx.y*blockDim.y + threadIdx.x+1;

	// Dla każdego piksela (i, j) w siatce o sizeze GRID_SIZE
	// oblicz filtr Laplace'a 8-krotnym wzmocnieniem środkowego piksela

	for (; i < width-1; i = i + GRID_SIZE)
	{
		for (int j = j_base; j < height-1; j = j + GRID_SIZE)
		{
			unsigned int suma = 0;

			unsigned char r = d_wsk[width*j+i];
			unsigned char r1 = d_wsk[width*(j-1) + i-1];
			unsigned char r2 = d_wsk[width*(j-1) + i];
			unsigned char r3 = d_wsk[width*(j-1) + i+1];
			unsigned char r4 = d_wsk[width*j + i-1];
			unsigned char r5 = d_wsk[width*j + i+1];
			unsigned char r6 = d_wsk[width*(j+1) + i-1];
			unsigned char r7 = d_wsk[width*(j+1) + i];
			unsigned char r8 = d_wsk[width*(j+1) + i+1];

			suma = (unsigned int)((float)(8 * r - r1 - r2 - r3 - r4 - r5 - r6 - r7 - r8));

			device_img[j*width + i] = suma;
		}
	}
}

void filter(unsigned char *host_pointer, unsigned char *host_img, int width, int height, long int size)
{
	// Pomiar czasu
	chrono::steady_clock::time_point begin_all = chrono::steady_clock::now();

	//cudaDeviceSynchronize();

	// Stworzenie wskaźników na piksele na device
	unsigned char *d_wsk;
	unsigned char *device_img;

	// Alokacja pamięci na device
	cudaMalloc((void**)&d_wsk, size);
	cudaMalloc((void**)&device_img, width * height * sizeof(unsigned char));

	// Skopiowanie danych z host na device
	cudaMemcpy(d_wsk, host_pointer, size, cudaMemcpyHostToDevice);

	// Uruchomienie kernela (razem z pomiarem czasu)
	dim3 threads(2, 2, 1);
	dim3 grid(GRID_SIZE, GRID_SIZE, 1);
	// dim3 grid((unsigned int) ceil((double)((width+1) / 2)), (unsigned int) ceil((double)((height+1) / 2)), 1);

	// Pomiar czasu
	chrono::steady_clock::time_point begin = chrono::steady_clock::now();

	Laplace <<<grid, threads>>> (width, height, d_wsk, device_img);
	cudaDeviceSynchronize();

	// Pomiar czasu
	chrono::steady_clock::time_point end = chrono::steady_clock::now();

	// Skopiowanie danych z device na host
	cudaMemcpy(host_img, device_img, width*height, cudaMemcpyDeviceToHost);

	// Zwolnienie pamięci na device
	cudaFree(device_img);
	cudaFree(d_wsk);

	// Pomiar czasu ciąg dalszy
	chrono::steady_clock::time_point end_all = chrono::steady_clock::now();

	cout << "Czas kernela = " << chrono::duration_cast<chrono::nanoseconds> (end - begin).count() << " ns" << std::endl;
	cout << "Czas całości = " << chrono::duration_cast<chrono::nanoseconds> (end_all - begin_all).count() << " ns" << std::endl;
}