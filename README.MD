# Laplace

## Wymagania

nvcc, kompilator wspierający C++11, karta graficzna nVidia obłsugująca CUDA

Testowano w następujących warunkach:

- Ubuntu 20.04.2 LTS x64, GTX 960, g++ 7.5.0-6ubuntu2, nvcc V10.0.130

- Windows 10 x64, GTX 1050 Ti, MSVC 14.29.30133, nvcc V11.3.58

## Kompilacja

```bash
sh build.sh
```

lub (Windows)

```bat
build
```

Może być potrzebne zmienienie ścieżki do używanego kompilatora w skrypcie budującym.

Program zostanie umieszczony w katalogu bin.

## Uruchomienie

Program współpracuje z plikami o rozszerzeniach BMP i PPM. Formaty JPG oraz PNG wymagałyby użycia dodatkowej biblioteki.

```bash
./bin/laplace /path/to/input/file.bmp /path/to/output/file.ppm
```

lub (Windows)

```bat
bin\laplace path\to\input\file.bmp path\to\output\file.ppm
```
