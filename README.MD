# Laplace

## Wymagania

nvcc, kompilator wspierający C++11, karta graficzna nVidia obłsugująca CUDA

Testowano w następujących warunkach:

- Ubuntu 20.04.2 LTS x64, GTX 960, g++

- Windows 10 x64, GTX 1050 Ti, MSVC 14.29.30133, nvcc V11.3.58

## Kompilacja

```bash
sh build.sh
```

lub

```bat
build.bat
```

Może być potrzebne zmienienie ścieżki do używanego kompilatora w skrypcie budującym.

## Uruchomienie

Program współpracuje z plikami o rozszerzeniach BMP i PPM. Formaty JPG oraz PNG wymagałyby użycia dodatkowej biblioteki.

```bash
./laplace /path/to/input/file.bmp /path/to/output/file.ppm
```

lub

```bat
laplace /path/to/input/file.bmp /path/to/output/file.ppm
```
