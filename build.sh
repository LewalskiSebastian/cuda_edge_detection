#! /bin/bash

rm laplace -f
rm kernels.o -f
nvcc -c kernels.cu
nvcc -ccbin g++ -Xcompiler "-std=c++11" kernels.o main.cpp -lcuda -lcudart -o laplace -lm -lpthread -lX11