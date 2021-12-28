#include "kernels.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

using namespace std;

__global__ void Laplace(int width, int height, unsigned char *d_wsk, unsigned char *d_bin)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x+1;
	int j = blockIdx.y*blockDim.y + threadIdx.x+1;

	if (i < width-1 && j < height-1)
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

		d_bin[j*width + i] = suma;
	}
}

void filter(unsigned char *h_wsk, unsigned char *h_bin, int szerokosc, int wysokosc, long int rozmiar)
{
	// Pomiar czasu
	chrono::steady_clock::time_point begin_all = chrono::steady_clock::now();

	//cudaDeviceSynchronize();

	// Stworzenie wskaźników na piksele na device
	unsigned char *d_wsk;
	unsigned char *d_bin;

	// Alokacja pamięci na device
	cudaMalloc((void**)&d_wsk, rozmiar);
	cudaMalloc((void**)&d_bin, szerokosc * wysokosc * sizeof(unsigned char));

	// Skopiowanie danych z host na device
	cudaMemcpy(d_wsk, h_wsk, rozmiar, cudaMemcpyHostToDevice);

	// Uruchomienie kernela (razem z pomiarem czasu)
	dim3 bloki(2, 2, 1);
	dim3 siatka((unsigned int) ceil((double)((szerokosc+1) / 2)), (unsigned int) ceil((double)((wysokosc+1) / 2)), 1);

	// Pomiar czasu
	chrono::steady_clock::time_point begin = chrono::steady_clock::now();

	Laplace <<<siatka, bloki>>> (szerokosc, wysokosc, d_wsk, d_bin);
	cudaDeviceSynchronize();

	// Pomiar czasu
	chrono::steady_clock::time_point end = chrono::steady_clock::now();

	// Skopiowanie danych z device na host
	cudaMemcpy(h_bin, d_bin, szerokosc*wysokosc, cudaMemcpyDeviceToHost);

	// Zwolnienie pamięci na device
	cudaFree(d_bin);
	cudaFree(d_wsk);

	// Pomiar czasu ciąŋ dalszy
	chrono::steady_clock::time_point end_all = chrono::steady_clock::now();

	cout << "Czas kernela = " << chrono::duration_cast<chrono::nanoseconds> (end - begin).count() << " ns" << std::endl;
	cout << "Czas całości = " << chrono::duration_cast<chrono::microseconds> (end_all - begin_all).count() << " µs" << std::endl;
}