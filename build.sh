#! /bin/bash

rm bin/laplace -f
rm bin/kernels.o -f
nvcc -c kernels.cu -o bin/kernels.o
nvcc -ccbin g++ -Xcompiler "-std=c++11" bin/kernels.o main.cpp -lcuda -lcudart -o bin/laplace -lm -lpthread -lX11