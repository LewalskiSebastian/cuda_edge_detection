cd bin
if exist laplace.exe del laplace.exe
if exist laplace.exp del laplace.exp
if exist laplace.lib del laplace.lib
if exist kernels.obj del kernels.obj
nvcc -c ..\kernels.cu
nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe" -rdc=true kernels.obj ..\main.cpp -o laplace.exe -luser32 -lGdi32
cd ..
