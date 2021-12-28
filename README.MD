# Laplace

## Wymagania

nvcc, gcc wspierający C++11, karta graficzna nVidia obłsugująca CUDA

Testowano na GTX 960 i Ubuntu 20.04.2 LTS x64

## Kompilacja

```bash
sh build.sh
```

## Uruchomienie

Program współpracuje z plikami o rozszerzeniach BMP i PPM.

```bash
./laplace /path/to/input/file.bmp path/to/output/file.ppm
```
