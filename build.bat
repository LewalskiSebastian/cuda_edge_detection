del laplace
del laplace.exp
del laplace.lib
del kernels.obj
nvcc -c kernels.cu
nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe" -rdc=true kernels.obj main.cpp -o laplace.exe -luser32 -lGdi32